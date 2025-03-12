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

from openrlhf.utils.deepspeed.deepspeed_strategy import DeepspeedStrategy

import torch.distributed as dist

def create_sub_group(group_size: int):
    """
    创建 TP/PP 分组，只在初始化阶段调用一次；通信测试部分可以在调试时启用，生产环境中建议禁用
    """
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
    assert abs(tmp.item() - 1.1) < 1e-4, "通信测试失败！"
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
    # 建立张量并行组，此处仅调用一次分组创建（建议在初始化时）
    vllm_tp_group = create_sub_group(args.engine_tp_size)

    # 延迟导入 vllm，仅在真正初始化时导入
    from vllm import LLM

    vllm_engine = LLM(
        model=args.pretrain,
        task="generate",
        tensor_parallel_size=args.engine_tp_size,
        gpu_memory_utilization=args.engine_mem_util,
        distributed_executor_backend="external_launcher",  # TODO： 非常重要
        # worker_cls="lightrlhf.strategy.vllm_utils.vllm_worker_wrap_no_ray.WorkerWrap",
        enable_sleep_mode=args.enable_engine_sleep,
    )

    return vllm_engine, vllm_tp_group

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
    def __init__(self, args, tokenizer, prompt_max_len: int = 1024):
        """
        args: 包含 vLLM 引擎参数的对象
        tokenizer: 用于 tokenization 的对象
        prompt_max_len: prompt 最大长度
        """
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

        # if dist.get_rank() == 0:
        print(f"Initializing vLLM engine via get_vllm_engine with tensor_parallel_size: {args.engine_tp_size}")
        self.llm, self.tp_group = get_vllm_engine(args)

    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        # 调用 vLLM.generate 接口进行生成
        outputs = self.llm.generate(prompts, **generate_kwargs)
        samples_list = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label, output in zip(prompts, labels, outputs):
            output_token_ids = list(output.outputs[0].token_ids)
            if output_token_ids[-1] != eos_token_id:
                output_token_ids.append(eos_token_id)
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False, max_length=self.prompt_max_len, truncation=True
            )["input_ids"]
            all_ids = prompt_ids + output_token_ids
            sequences = torch.tensor([all_ids], device="cuda")
            attention_mask = (sequences != pad_token_id).float()
            action_mask = torch.zeros_like(sequences, dtype=torch.bool)
            # 假设 action 从 prompt 长度开始生效
            action_mask[:, len(prompt_ids):] = 1

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
# 简单模型示例（普通前向传播和 vllm 模式）
# -------------------------------

class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1, use_vllm=False):
        super(SimpleModel, self).__init__()
        self.use_vllm = use_vllm
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x=None, prompts=None, labels=None, generate_kwargs={}, vllm_backend=None):
        # 注意：仅当 use_vllm 为 True 且传入 vllm_backend 时调用生成接口
        if self.use_vllm and vllm_backend is not None:
            # 为确保调试信息顺序，建议只在 rank 0 输出日志
            # if dist.get_rank() == 0:
            print(f'rank {dist.get_rank()}: =========vLLM generate invoked=========')
            samples = vllm_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences
        elif self.use_vllm:
            # 非推理进程返回 dummy 结果，确保所有进程均返回结果，避免分布式 hang
            return torch.zeros((1, 10), device='cuda')
        else:
            return self.net(x)

###############################################################################
# 主函数，展示如何在 DeepSpeed 训练过程中使用 vllm 推理以及普通训练模式
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

        # 以下为 vLLM 引擎参数
        engine_mem_util = 0.5   # GPU 内存利用率限制（示例）
        # engine_tp_size = 8      # tensor parallel 的大小，根据实际环境调整
        # pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--OpenRLHF--Llama-3-8b-sft-mixture/snapshots/03334dc4a796d9d72850ead46956c33da22e6d7b"
        engine_tp_size = 4      # tensor parallel 的大小，根据实际环境调整
        pretrain = "Qwen/Qwen2.5-7B-Instruct"
        enable_engine_sleep = True

    args = Args()

    # DeepSpeed Strategy 初始化与分布式环境设置
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

    # 尝试加载预训练模型对应的 tokenizer，如果加载失败则从 Hugging Face 下载 Qwen2.5-7B 的 tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        print("成功加载预训练tokenizer:", args.pretrain)
    except Exception as e:
        print("加载预训练的tokenizer失败，尝试从 Hugging Face 下载 Qwen2.5-7B 的tokenizer。错误信息：", e)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # 方式一：vLLM 推理模式（仅用于生成不参与梯度更新）
    vllm_backend = VLLMBackend(
        args=args,
        tokenizer=tokenizer,
        prompt_max_len=1024
    )
    
    if dist.get_rank() == 0:
        print(f'rank {dist.get_rank()}: =========Starting vLLM mode inference=========')

    # 包装 vLLM 模式下的模型和优化器（注意：vLLM 模式下梯度不参与更新，但依然需调用 prepare 保持进程同步）
    model_vllm = SimpleModel(use_vllm=True)
    model_vllm, optimizer_vllm, scheduler_vllm = strategy.prepare(
        (
            model_vllm, 
            optim.Adam(model_vllm.parameters(), lr=1e-3), 
            StepLR(optim.Adam(model_vllm.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )
    
    if dist.get_rank() == 0:
        print(f'rank {dist.get_rank()}: =========vLLM model prepared=========')

    # 设置生成输入（这里仅取两个示例文本）
    prompts = ["Hello, how are you?", "What is the weather today?"]
    labels = ["I am fine", "Sunny and warm"]

    model_vllm.eval()
    with torch.no_grad():
        generated = model_vllm(prompts=prompts, labels=labels, generate_kwargs={}, vllm_backend=vllm_backend)
        # if dist.get_rank() == 0:
        print(f"rank {dist.get_rank()}: vLLM 模式下生成的序列 tensor:\n", generated)

    if dist.get_rank() == 0:
        print(f'rank {dist.get_rank()}: =========Finished vLLM inference=========')


if __name__ == '__main__':
    # 示例启动命令：
    # torchrun --nnodes=1 --nproc-per-node 8 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_vllm_train.py
    main()