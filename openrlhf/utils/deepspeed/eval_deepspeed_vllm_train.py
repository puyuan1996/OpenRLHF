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
    # vllm_tp_group = None

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
# BaseGenerationBackend 与 Samples 保持不变
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

        print(f"Initializing vLLM engine via get_vllm_engine with tensor_parallel_size: {args.engine_tp_size}")
        self.llm, self.tp_group = get_vllm_engine(args)
        # 用于记录一次 batch 的生成耗时（单位：秒）
        self.latest_gen_time = None

    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        # 仅统计 engine.generate 的耗时
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = self.llm.generate(prompts, **generate_kwargs)
        torch.cuda.synchronize()
        self.latest_gen_time = time.time() - t0
        print(f"[VLLMBackend] Engine generation time for batch of {len(prompts)} prompt(s): {self.latest_gen_time*1000:.2f} ms")

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
# 简单模型示例（支持普通前向传播和 vllm 模式）
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
            if dist.get_rank() == 0:
                print(f'rank {dist.get_rank()}: =========vLLM generate invoked=========')
            samples = vllm_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences  # 返回第一个 sample 的序列（作为示例）
        elif self.use_vllm:
            # 非推理进程返回 dummy 结果，确保所有进程均返回结果，避免分布式 hang
            return torch.zeros((1, 10), device='cuda')
        else:
            return self.net(x)

###############################################################################
# 主函数，展示如何在 DeepSpeed 训练过程中使用 vLLM 推理以及普通训练模式，并进行性能指标分析
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
        engine_tp_size = 4      # tensor parallel 的大小，根据实际环境调整
        pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--OpenRLHF--Llama-3-8b-sft-mixture/snapshots/03334dc4a796d9d72850ead46956c33da22e6d7b"
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

    # 尝试加载预训练模型对应的 tokenizer；若失败则尝试使用 Hugging Face 上的 Qwen2.5-7B 的 tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        print("成功加载预训练tokenizer:", args.pretrain)
    except Exception as e:
        print("加载预训练的tokenizer失败，尝试从 Hugging Face 下载 Qwen2.5-7B 的tokenizer。错误信息：", e)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # 方式一：vLLM 推理模式（仅用于生成，不参与梯度更新）
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
    # prompts = ["Hello, how are you?", "What is the weather today?"]
    # labels = ["I am fine", "Sunny and warm"]

    prompts = [
    "Hello, my name is John Doe.",
    "请详细描述一下人工智能在医疗领域的应用前景。",
    "What are the benefits and drawbacks of remote work?",
    "简述一下区块链技术的基本原理。",
    "Explain the process of photosynthesis in simple terms.",
    "请给出一个简单易懂的量子计算介绍。",
    "How does the economy react in times of deflation?",
    "请写一首关于春天的诗，要求语言清新。",
    "Describe the cultural impact of social media on modern society.",
    "请简单介绍一下机器学习和深度学习的区别。"
    ]
    labels = ["I am fine" for i in range(10)]

    model_vllm.eval()

    # -------------------------------
    # 进行多次推理测试，统计性能指标
    # -------------------------------
    num_iterations = 10  # 可根据需要增加迭代次数
    durations = []
    throughputs = []
    per_token_times = []
    
    for i in range(num_iterations):
        # 每次 iteration 开始前重置 GPU 显存统计
        torch.cuda.reset_peak_memory_stats()
        
        # 预热（首轮调用不计入统计）
        if i == 0:
            _ = model_vllm(prompts=prompts, labels=labels, generate_kwargs={}, vllm_backend=vllm_backend)

        # 调用 vLLM 推理（调用内部仅统计 engine.generate 耗时）
        samples = model_vllm(prompts=prompts, labels=labels, generate_kwargs={}, vllm_backend=vllm_backend)
        # 使用 vllm_backend.latest_gen_time 作为引擎调用耗时（不计入 tokenization 等数据处理耗时）
        gen_time = vllm_backend.latest_gen_time
        durations.append(gen_time)
        
        # 获取生成结果，并统计生成 token 数量（假设 batch 中每个 sample 的 token 数一致）
        # samples = model_vllm(prompts=prompts, labels=labels, generate_kwargs={}, vllm_backend=vllm_backend)
        # print(f"samples:{samples}")
       
        token_count =  sum(vllm_backend.token_count)

        # token_count = samples.shape[1]
        # token_count =  len(samples[0].outputs[0].token_ids)

        throughput = token_count / gen_time
        throughputs.append(throughput)
        per_token_times.append(gen_time / token_count)
        
        if dist.get_rank() == 0:
            print(f'Iteration {i+1}/{num_iterations}: Generation Time={gen_time:.4f}s, '
                  f'Tokens={token_count}, Throughput={throughput:.2f} tokens/s, '
                  f'Per-token time={(gen_time / token_count)*1000:.2f} ms')
    
    # 获取此次推理过程中 GPU 显存峰值（单位：MB）
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # 计算统计指标
    avg_duration = np.mean(durations)
    std_duration = np.std(durations)
    avg_throughput = np.mean(throughputs)
    avg_per_token = np.mean(per_token_times)
    
    # 生成分析报告内容
    report_lines = [
        "Performance Analysis Report",
        "=============================",
        f"Number of iterations: {num_iterations}",
        "",
        f"Average Engine Generation Time: {avg_duration:.4f} seconds (std: {std_duration:.4f} sec)",
        "",
        f"Average Throughput: {avg_throughput:.2f} tokens/second",
        f"Average Per-token Time: {avg_per_token*1000:.2f} ms",
        "",
        f"Max GPU Memory Allocated during inference: {max_gpu_memory:.2f} MB",
        "",
        "Raw generation times per iteration (s): " + ", ".join([f"{d:.4f}" for d in durations]),
        "Raw throughputs per iteration (tokens/s): " + ", ".join([f"{t:.2f}" for t in throughputs]),
        "Raw per-token times per iteration (ms): " + ", ".join([f"{p*1000:.2f}" for p in per_token_times]),
    ]
    report_content = "\n".join(report_lines)
    
    # 保存性能分析报告（仅 rank 0 保存即可）
    if dist.get_rank() == 0:
        report_filename = "performance_report.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Performance analysis report saved to {report_filename}")
        print(report_content)
    
    if dist.get_rank() == 0:
        print(f'rank {dist.get_rank()}: =========Finished vLLM inference=========')

if __name__ == '__main__':
    # 示例启动命令：
    # torchrun --nnodes=1 --nproc-per-node 4 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_vllm_train.py
    main()