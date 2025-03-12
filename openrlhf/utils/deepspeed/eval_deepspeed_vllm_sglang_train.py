#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
import time
from datetime import timedelta

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from typing import List, Tuple, Union

# 假设 DeepspeedStrategy 已经定义（如你的 deepspeed demo 代码中所示），本例中从 openrlhf 导入
from openrlhf.utils.deepspeed.deepspeed_strategy import DeepspeedStrategy

import torch.distributed as dist

def create_sub_group(group_size: int):
    """创建 TP/PP 分组，如果 world_size 可以整除 group_size，则生成所有组并测试通信"""
    world_size = dist.get_world_size()
    if world_size % group_size != 0:
        raise ValueError(f"world_size ({world_size}) % group_size ({group_size}) != 0 ")

    num_groups = world_size // group_size
    all_group_ranks = []
    for i in range(num_groups):
        start_rank = i * group_size
        group_ranks = list(range(start_rank, start_rank + group_size))
        all_group_ranks.append(group_ranks)
    group, _ = dist.new_subgroups_by_enumeration(all_group_ranks)

    if dist.get_rank() == 0:
        print(
            f"Finished create TP/PP group, groupsize={torch.distributed.get_world_size(group)}, "
            "start testing communication...",
            flush=True,
        )
    dist.barrier()
    tmp = torch.tensor(1.1, device="cuda")
    dist.all_reduce(tmp, group=group, op=dist.ReduceOp.AVG)
    dist.barrier()
    assert abs(tmp.item() - 1.1) < 1e-4
    if dist.get_rank() == 0:
        print("Finished testing comm!", flush=True)

    return group


# -------------------------------
# vLLM 初始化接口
# -------------------------------

def get_vllm_engine(args):
    """
    根据参数构建 vLLM 引擎，并返回引擎对象以及分组
    """
    # 建立张量并行组，确保 world_size 可以整除 engine_tp_size
    vllm_tp_group = create_sub_group(args.engine_tp_size)

    # 延迟导入 vllm，仅在真正初始化时导入
    from vllm import LLM

    # 确保设置 task="generate"
    vllm_engine = LLM(
        model=args.pretrain,
        task="generate",
        tensor_parallel_size=args.engine_tp_size,
        gpu_memory_utilization=args.engine_mem_util,
        distributed_executor_backend="external_launcher",
        worker_cls="lightrlhf.strategy.vllm_utils.vllm_worker_wrap_no_ray.WorkerWrap",
        enable_sleep_mode=args.enable_engine_sleep,
    )

    return vllm_engine, vllm_tp_group

# -------------------------------
# 新增 get_sglang_engine 接口（仿照 get_vllm_engine）
# -------------------------------
def get_sglang_engine(args):
    """
    根据参数构建 sglang 引擎，并返回引擎对象以及分组
    """
    sglang_tp_group = create_sub_group(args.engine_tp_size)
    import sglang as sgl
    sglang_engine = sgl.Engine(model_path=args.pretrain)
    return sglang_engine, sglang_tp_group

# -------------------------------
# 原有 BaseGenerationBackend 和 Samples 保持不变
# -------------------------------

class BaseGenerationBackend:
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List:
        raise NotImplementedError

class Samples:
    def __init__(
        self,
        sequences,
        attention_mask,
        action_mask,
        num_actions,
        packed_seq_lens,
        response_length,
        total_length,
        prompts,
        labels,
        pad_len,
    ):
        self.sequences = sequences
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.num_actions = num_actions
        self.packed_seq_lens = packed_seq_lens
        self.response_length = response_length
        self.total_length = total_length
        self.prompts = prompts
        self.labels = labels
        self.pad_len = pad_len

# -------------------------------
# 修改后的 VLLMBackend
# -------------------------------

class VLLMBackend(BaseGenerationBackend):
    def __init__(self, model_name: str, tokenizer, prompt_max_len: int = 1024, tensor_parallel_size=1, **kwargs):
        """
        model_name: 模型名称或路径（实际初始化在 get_vllm_engine 内处理）
        tokenizer: 用于 tokenization 的对象
        prompt_max_len: prompt 最大长度
        tensor_parallel_size: 张量并行大小
        kwargs: 传给 vllm.LLM 的其他参数
        """
        self.tokenizer = tokenizer
        from vllm import LLM
        print(f"Initializing vLLM LLM with tensor_parallel_size: {tensor_parallel_size}")
        # 指定 task="generate"
        self.llm = LLM(model=model_name, task="generate", tensor_parallel_size=tensor_parallel_size, **kwargs)
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        outputs = self.llm.generate(prompts, **generate_kwargs)
        samples_list = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label, output in zip(prompts, labels, outputs):
            output_token_ids = list(output.outputs[0].token_ids)
            if not output_token_ids or output_token_ids[-1] != eos_token_id:
                output_token_ids.append(eos_token_id)
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False, max_length=self.prompt_max_len, truncation=True
            )["input_ids"]
            all_ids = prompt_ids + output_token_ids
            sequences = torch.tensor([all_ids])
            attention_mask = (sequences != pad_token_id).float()
            action_mask = torch.zeros_like(sequences, dtype=torch.bool)
            action_mask[:, len(prompt_ids):] = 1

            sequences = sequences.to("cuda")
            attention_mask = attention_mask.to("cuda")
            action_mask = action_mask.to("cuda")

            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask[:, 1:].float(),
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=torch.tensor([action_mask.float().sum().item()]),
                total_length=torch.tensor([attention_mask.float().sum().item()]),
                prompts=[prompt],
                labels=[label],
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

# -------------------------------
# 修改后的 SglangBackend（使用 get_sglang_engine 初始化）
# -------------------------------
class SglangBackend(BaseGenerationBackend):
    def __init__(self, sgl_engine, tokenizer, prompt_max_len: int = 1024, sampling_params=None):
        """
        sgl_engine: 通过 get_sglang_engine 得到的 sglang 引擎对象
        tokenizer: 用于 tokenization 的对象
        prompt_max_len: prompt 最大长度
        sampling_params: 采样参数，例如 {"temperature": 0.8, "top_p": 0.95}
        """
        self.tokenizer = tokenizer
        self.sgl_engine = sgl_engine
        self.prompt_max_len = prompt_max_len
        self.sampling_params = sampling_params if sampling_params is not None else {"temperature": 0.8, "top_p": 0.95}

    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        samples_list = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label in zip(prompts, labels):
            # sglang 的 generate 接口接收 prompt 列表，返回的是列表结果
            outputs = self.sgl_engine.generate([prompt], sampling_params=self.sampling_params)
            output = outputs[0]
            output_text = output.get("text", "无文本输出")
            output_token_ids = self.tokenizer(output_text, add_special_tokens=False)["input_ids"]
            if not output_token_ids or output_token_ids[-1] != eos_token_id:
                output_token_ids.append(eos_token_id)
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False, max_length=self.prompt_max_len, truncation=True
            )["input_ids"]
            all_ids = prompt_ids + output_token_ids

            sequences = torch.tensor([all_ids])
            attention_mask = (sequences != pad_token_id).float()
            action_mask = torch.zeros_like(sequences, dtype=torch.bool)
            action_mask[:, len(prompt_ids):] = 1

            sequences = sequences.to("cuda")
            attention_mask = attention_mask.to("cuda")
            action_mask = action_mask.to("cuda")

            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask[:, 1:].float(),
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=torch.tensor([action_mask.float().sum().item()]),
                total_length=torch.tensor([attention_mask.float().sum().item()]),
                prompts=[prompt],
                labels=[label],
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

# -------------------------------
# 简单模型示例（普通前向传播和生成模式）
# -------------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1, use_vllm=False, use_sglang=False):
        super(SimpleModel, self).__init__()
        self.use_vllm = use_vllm
        self.use_sglang = use_sglang
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x=None, prompts=None, labels=None, generate_kwargs={},
                vllm_backend=None, sglang_backend=None):
        import torch.distributed as dist
        
        if self.use_vllm and vllm_backend is not None:
            print(f'rank {dist.get_rank()}: ========= Using vLLM backend =========')
            samples = vllm_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences
        elif self.use_sglang and sglang_backend is not None:
            print(f'rank {dist.get_rank()}: ========= Using sglang backend =========')
            samples = sglang_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences
        else:
            return self.net(x)

###############################################################################
# 主函数，展示如何在 DeepSpeed 训练过程中使用生成后端进行推理以及普通训练模式
###############################################################################
def main():
    # 模拟 DeepSpeed 的命令行参数，一般由 launcher 传入
    class Args:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        adam_offload = False
        ring_attn_size = 1
        pretrain_data = None
        zero_stage = 2
        bf16 = False
        micro_train_batch_size = 4
        train_batch_size = 32
        max_norm = 1.0

        # 以下为引擎参数，共 vLLM 和 sglang 均使用
        engine_tp_size = 2      # tensor parallel 大小
        engine_mem_util = 0.5   # GPU 内存利用率限制（示例）
        pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
        enable_engine_sleep = True

    args = Args()

    # 初始化 Deepspeed strategy，并设置分布式环境
    strategy = DeepspeedStrategy(
        seed=42,
        max_norm=args.max_norm,
        micro_train_batch_size=args.micro_train_batch_size,
        train_batch_size=args.train_batch_size,
        zero_stage=args.zero_stage,
        bf16=args.bf16,
        args=args,
    )
    strategy.setup_distributed()

    # 构造一个简单的 dummy tokenizer，实际请使用 Hugging Face tokenizer
    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1

        def __call__(self, text, add_special_tokens=True, max_length=None, truncation=False):
            # 简单示例：将每个字符转为其 ascii 模 100 值
            token_ids = [ord(c) % 100 for c in text]
            if max_length:
                token_ids = token_ids[:max_length]
            return {"input_ids": token_ids}

        def save_pretrained(self, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "tokenizer.txt"), "w", encoding="utf-8") as f:
                f.write("Dummy tokenizer parameters.")

    tokenizer = DummyTokenizer()

    import torch.distributed as dist

    ###############
    # 方式一：使用 vLLM 推理模式（仅生成调用，不参与梯度更新）
    ###############
    vllm_backend = VLLMBackend(
        model_name=args.pretrain,
        tokenizer=tokenizer,
        prompt_max_len=1024,
        tensor_parallel_size=args.engine_tp_size,
        gpu_memory_utilization=args.engine_mem_util,
        distributed_executor_backend="external_launcher",
        worker_cls="lightrlhf.strategy.vllm_utils.vllm_worker_wrap_no_ray.WorkerWrap",
        enable_sleep_mode=args.enable_engine_sleep,
    )
    
    model_vllm = SimpleModel(use_vllm=True)
    model_vllm, optimizer_vllm, scheduler_vllm = strategy.prepare(
        (
            model_vllm, 
            optim.Adam(model_vllm.parameters(), lr=1e-3), 
            StepLR(optim.Adam(model_vllm.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )
    prompts = ["Hello, how are you?", "What is the weather today?"]
    labels = ["I am fine", "Sunny and warm"]

    model_vllm.eval()
    with torch.no_grad():
        generated_vllm = model_vllm(prompts=prompts, labels=labels, generate_kwargs={}, vllm_backend=vllm_backend)
        strategy.print("vLLM 模式下生成的序列 tensor:", generated_vllm)

    ###############
    # 方式二：使用 sglang 推理模式（采用 get_sglang_engine 初始化）
    ###############
    sgl_engine, sgl_tp_group = get_sglang_engine(args)
    sglang_backend = SglangBackend(
        sgl_engine=sgl_engine,
        tokenizer=tokenizer,
        prompt_max_len=1024,
        sampling_params={"temperature": 0.8, "top_p": 0.95}
    )
    model_sglang = SimpleModel(use_sglang=True)
    model_sglang, optimizer_sglang, scheduler_sglang = strategy.prepare(
        (
            model_sglang,
            optim.Adam(model_sglang.parameters(), lr=1e-3),
            StepLR(optim.Adam(model_sglang.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )
    model_sglang.eval()
    with torch.no_grad():
        generated_sglang = model_sglang(prompts=prompts, labels=labels, generate_kwargs={}, sglang_backend=sglang_backend)
        strategy.print("sglang 模式下生成的序列 tensor:", generated_sglang)

    ###############
    # 方式三：普通前向传播训练（不使用生成后端）
    ###############
    model_training = SimpleModel(use_vllm=False, use_sglang=False)
    optimizer_training = optim.Adam(model_training.parameters(), lr=1e-3)
    scheduler_training = StepLR(optimizer_training, step_size=10, gamma=0.1)

    model_training, optimizer_training, scheduler_training = strategy.prepare(
        (model_training, optimizer_training, scheduler_training),
        is_rlhf=False
    )

    inputs = torch.randn(1000, 10).cuda()
    targets = torch.randn(1000, 1).cuda()
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=strategy.micro_train_batch_size, shuffle=True)

    model_training.train()
    num_epochs = 2
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):
            outputs = model_training(x)
            loss = ((outputs - y) ** 2).mean()
            model_training.backward(loss)
            if (i + 1) % strategy.accumulated_gradient == 0:
                model_training.step()
                optimizer_training.zero_grad()
                strategy.print(f"Epoch {epoch}, Step {i}: loss = {loss.item()}")

    if strategy.is_rank_0():
        save_dir = "./saved_model_debug"
        os.makedirs(save_dir, exist_ok=True)
        model_training.save_checkpoint(save_dir, tag="final_checkpoint_debug")
        strategy.print("Training completed and model saved.")

if __name__ == '__main__':
    # torchrun --nnodes=1 --nproc-per-node 2 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_vllm_sglang_train.py 
    main()