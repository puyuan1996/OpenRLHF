#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
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

# 假设 DeepspeedStrategy 已经在 openrlhf 内部定义
from openrlhf.utils.deepspeed.deepspeed_strategy import DeepspeedStrategy

import torch.distributed as dist

# -------------------------------
# 分布式子分组（如果需要）
# -------------------------------
def create_sub_group(group_size: int):
    """创建 TP/PP 分组，如果 world_size 可以整除 group_size，则生成所有组并测试通信"""
    world_size = dist.get_world_size()
    if world_size % group_size != 0:
        raise ValueError(f"world_size ({world_size}) % group_size ({group_size}) != 0")

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
# vLLM 中的 get_vllm_engine 类似，这里新增 get_sglang_engine
# -------------------------------
def get_sglang_engine(args):
    """
    根据参数构建 sglang 引擎，并返回引擎对象以及子分组（如果有需要）。
    目前示例中直接初始化 sglang 引擎，无需额外分组，返回 None。
    """
    import sglang as sgl

    # 如有需要可以根据 args 创建 tensor parallel group：
    sglang_tp_group = create_sub_group(args.engine_tp_size)  # 示例：如果 args 中有该参数

    print(f'rank {dist.get_rank()}: =========debug: pos 1 =========')

    # 目前未使用额外的分组
    if dist.get_rank() == 0:
        sglang_engine = sgl.Engine(model_path=args.pretrain)
        # sglang_engine = sgl.Engine(model_path=args.pretrain, distributed_executor_backend="external_launcher")
    else:
        sglang_engine = None

    print(f'rank {dist.get_rank()}: =========debug: pos 2 =========')

    return sglang_engine, None

# -------------------------------
# Samples 和 BaseGenerationBackend 定义
# -------------------------------
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

class BaseGenerationBackend:
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List:
        raise NotImplementedError

# -------------------------------
# SGLangBackend 基于 sglang 工具包实现文本生成接口，
# 采用 get_sglang_engine 获取引擎并赋值给 self.engine
# -------------------------------
class SGLangBackend(BaseGenerationBackend):
    def __init__(self, args, tokenizer, prompt_max_len: int = 1024):
        """
        args: 包含 sglang 引擎参数的对象，其中 pretrain 为模型路径
        tokenizer: 用于 tokenization 的 tokenizer 对象
        prompt_max_len: prompt 最大长度
        """
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        print(f"Initializing sglang engine via get_sglang_engine with model path: {args.pretrain}")
        self.engine, _ = get_sglang_engine(args)
    
    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        # 采样参数：可通过 generate_kwargs 传入采样配置，否则使用默认值
        sampling_params = generate_kwargs.get("sampling_params", {"temperature": 0.8, "top_p": 0.95})
        samples_list = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label in zip(prompts, labels):
            # 调用 sglang 引擎生成，注意 sglang 的 generate 接口要求传入 prompt 列表
            outputs = self.engine.generate([prompt], sampling_params=sampling_params)
            # 获取 sglang 返回的第一个结果
            output_dict = outputs[0]
            # 从结果中获取生成文本
            generated_text = output_dict.get("text", "")
            # 可以检查是否存在 eos_token, 此处示例简单拼接
            if len(generated_text) == 0 or generated_text[-1] != chr(eos_token_id):
                # 此处仅为示例，实际请根据 tokenizer 的 eos_token_id 处理
                generated_text += chr(eos_token_id)

            # Tokenize prompt 和生成文本
            prompt_tokens = self.tokenizer(
                prompt, add_special_tokens=False, max_length=self.prompt_max_len, truncation=True
            )["input_ids"]
            generated_tokens = self.tokenizer(
                generated_text, add_special_tokens=False
            )["input_ids"]
            all_token_ids = prompt_tokens + generated_tokens
            sequences = torch.tensor([all_token_ids])
            attention_mask = (sequences != pad_token_id).float()
            action_mask = torch.zeros_like(sequences, dtype=torch.bool)
            # 指定 prompt 之外部分为可操作区域
            action_mask[:, len(prompt_tokens):] = 1

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
# 示例模型：支持普通前向传播和 sglang 推理模式
# -------------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1, use_sglang=False):
        super(SimpleModel, self).__init__()
        self.use_sglang = use_sglang
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x=None, prompts=None, labels=None, generate_kwargs={}, sglang_backend=None):
        import torch.distributed as dist
        
        if self.use_sglang and sglang_backend is not None:
            print(f'rank {dist.get_rank()}: =========debug: sglang generation mode =========')
            samples = sglang_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences
        elif self.use_sglang:
            # 非推理进程返回 dummy 结果
            return torch.zeros((1, 10), device='cuda')
        else:
            return self.net(x)

###############################################################################
# 主函数，展示如何在 DeepSpeed 训练过程中使用 sglang 推理以及普通训练模式
###############################################################################
def main():
    # 模拟 DeepSpeed 的命令行参数，通常由 launcher 传入
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

        # sglang 引擎参数：模型路径，请确保该路径正确
        pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
        # 如有需要可以增加 engine_tp_size 等其他参数

        engine_tp_size = 1      # tensor parallel 的大小，可根据需要调整


    args = Args()

    # 初始化 DeepSpeed strategy 并设置分布式环境
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

    # 构造一个简单的 dummy tokenizer（实际工程建议使用 Hugging Face tokenizer）
    class DummyTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            # 为便于示例，这里将 eos_token_id 与一个字符关联（例如 1 对应 ASCII SOH）
            self.eos_token_id = 1

        def __call__(self, text, add_special_tokens=True, max_length=None, truncation=False):
            # 简单示例：将每个字符转换为其 ASCII 码的末两位，并截断至 max_length
            token_ids = [ord(c) % 100 for c in text][:max_length]
            return {"input_ids": token_ids}

        def save_pretrained(self, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "tokenizer.txt"), "w", encoding="utf-8") as f:
                f.write("Dummy tokenizer parameters.")

    tokenizer = DummyTokenizer()

    print(f'rank {dist.get_rank()}: =========debug: pos 0 =========')

    # 方式一：使用 sglang 推理模式（生成调用，不用于梯度更新）
    sglang_backend = SGLangBackend(
        args=args,
        tokenizer=tokenizer,
        prompt_max_len=1024
    )
    
    print(f'rank {dist.get_rank()}: =========debug: starting sglang generation test =========')

    model_sgl = SimpleModel(use_sglang=True)
    model_sgl, optimizer_sgl, scheduler_sgl = strategy.prepare(
        (
            model_sgl, 
            optim.Adam(model_sgl.parameters(), lr=1e-3), 
            StepLR(optim.Adam(model_sgl.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )
    print(f'rank {dist.get_rank()}: =========debug: prepared model_sgl =========')

    # 设置生成时的输入示例
    prompts = ["你好，今天天气如何？", "请问量子计算的基本原理是什么？"]
    labels = ["今天天气晴朗", "量子计算基于量子力学原理"]

    model_sgl.eval()
    with torch.no_grad():
        generated = model_sgl(
            prompts=prompts, 
            labels=labels, 
            generate_kwargs={"sampling_params": {"temperature": 0.8, "top_p": 0.95}},
            sglang_backend=sglang_backend
        )
        print(f'rank {dist.get_rank()}: =========debug: sglang generation output =========')
        print(f"rank {dist.get_rank()}: sglang 模式下生成的序列 tensor:", generated)

    print(f'rank {dist.get_rank()}: =========debug: sglang generation test finished =========')

    # 方式二：普通前向传播训练（用于梯度更新）的示例（代码已注释，可按需启用）
    # model_training = SimpleModel(use_sglang=False)
    # optimizer_training = optim.Adam(model_training.parameters(), lr=1e-3)
    # scheduler_training = StepLR(optimizer_training, step_size=10, gamma=0.1)
    #
    # model_training, optimizer_training, scheduler_training = strategy.prepare(
    #     (model_training, optimizer_training, scheduler_training),
    #     is_rlhf=False
    # )
    #
    # inputs = torch.randn(1000, 10).cuda()
    # targets = torch.randn(1000, 1).cuda()
    # dataset = TensorDataset(inputs, targets)
    # dataloader = DataLoader(dataset, batch_size=strategy.micro_train_batch_size, shuffle=True)
    #
    # model_training.train()
    # num_epochs = 2
    # for epoch in range(num_epochs):
    #     for i, (x, y) in enumerate(dataloader):
    #         outputs = model_training(x)
    #         loss = ((outputs - y) ** 2).mean()
    #         model_training.backward(loss)
    #         if (i + 1) % strategy.accumulated_gradient == 0:
    #             model_training.step()
    #             optimizer_training.zero_grad()
    #             strategy.print(f"Epoch {epoch}, Step {i}: loss = {loss.item()}")
    #
    # if strategy.is_rank_0():
    #     save_dir = "./saved_model_sgl_debug"
    #     os.makedirs(save_dir, exist_ok=True)
    #     model_training.save_checkpoint(save_dir, tag="final_checkpoint_sgl_debug")
    #     strategy.print("Training completed and model saved.")

if __name__ == '__main__':
    # 示例运行命令：
    # torchrun --nnodes=1 --nproc-per-node 2 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_sglang_train.py
    main()