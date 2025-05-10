#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
import time
import datetime
import sys
from datetime import timedelta

from torch.distributed.device_mesh import init_device_mesh

from sglang.srt.entrypoints.verl_engine import VerlEngine
import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from typing import List, Tuple, Union

# 添加工程路径
sys.path.insert(0, "/fs-computility/ai-shen/puyuan/code/OpenRLHF")

from openrlhf.utils.deepspeed.deepspeed_strategy import DeepspeedStrategy

import torch.distributed as dist


# -------------------------------
# 修改后的 get_sglang_engine 使用 VerlEngine 初始化推理引擎
# -------------------------------
def get_sglang_engine(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def _log(text):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] [rank={rank}] {text}")

    _log(
        f'start {local_rank=} {rank=} {world_size=} {sys.argv=} {os.environ.get("CUDA_VISIBLE_DEVICES")}'
    )
    tp_size = args.engine_tp_size
    dp_size = world_size // tp_size

    assert world_size == tp_size * dp_size

    device_mesh_kwargs = dict(
        mesh_shape=(tp_size, dp_size, 1), mesh_dim_names=["tp", "dp", "pp"]
    )
    device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
    _log(f"device_mesh_cpu={device_mesh_cpu}")

    tp_rank = device_mesh_cpu.get_local_rank("tp")
    dp_rank = device_mesh_cpu.get_local_rank("dp")
    _log(f"{tp_rank=}, tp_size={tp_size}; {dp_rank=}, dp_size={dp_size}")

    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]

    verl_engine = VerlEngine(
        model_path=args.pretrain,
        mem_fraction_static=args.mem_fraction_static,
        device_mesh_cpu=device_mesh_cpu["tp"],
        base_gpu_id=dp_rank,
        gpu_id_step=dp_size,
        port=args.port,
    )

    return verl_engine, device_mesh_cpu

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
# SGLangBackend 基于 VerlEngine 实现文本生成接口
# -------------------------------
class SGLangBackend(BaseGenerationBackend):
    def __init__(self, args, tokenizer, prompt_max_len: int = 1024):
        """
        args: 包含 VerlEngine 引擎参数的对象，其中 pretrain 为模型路径等
        tokenizer: 用于 tokenization 的 tokenizer 对象
        prompt_max_len: prompt 最大长度
        """
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        print(f"Initializing VerlEngine with model path: {args.pretrain}")
        # 针对所有 rank 均初始化 VerlEngine
        self.engine, _ = get_sglang_engine(args)
        # 用于记录每个 prompt 的 engine.generate 耗时（单位：秒）
        self.latest_gen_times = []
    
    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        # 设置采样参数，默认值或者通过 generate_kwargs 传入（例如 temperature、top_p）
        sampling_params = generate_kwargs.get("sampling_params", {"temperature": 0.8, "top_p": 0.95})
        samples_list = []
        # 清空上一次记录的生成耗时
        self.latest_gen_times = []
        self.token_count = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label in zip(prompts, labels):
            # 仅统计 engine.generate 的耗时
            torch.cuda.synchronize()
            t0 = time.time()
            generated = self.engine.generate(prompt=prompt, sampling_params=sampling_params)
            torch.cuda.synchronize()
            gen_time = time.time() - t0
            self.latest_gen_times.append(gen_time)
            print(f"[SGLangBackend] Engine generation time for prompt: {gen_time*1000:.2f} ms")
            
            print(f"generated:{generated}")
            generated_text = generated['text']

            # token_count =  sglang_backend.token_count
            # token_count =  generated[0]["meta_info"]['completion_tokens']
            token_count =  generated["meta_info"]['completion_tokens']

            self.token_count.append(token_count)

            # 如果返回文本为空或末尾没有 eos 标记，则进行简单补全
            if len(generated_text) == 0 or generated_text[-1] != chr(eos_token_id):
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
        # 打印该次生成的平均耗时
        if self.latest_gen_times:
            avg_gen_time = sum(self.latest_gen_times) / len(self.latest_gen_times)
            print(f"[SGLangBackend] Average engine generation time for {len(prompts)} prompt(s): {avg_gen_time*1000:.2f} ms")
        if self.token_count:
            avg_token_count = sum(self.token_count) / len(self.token_count)

        return samples_list

# -------------------------------
# 示例模型：支持普通前向传播和 VerlEngine 推理模式
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
        if self.use_sglang and sglang_backend is not None:
            # 使用 VerlEngine 进行文本生成
            samples = sglang_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences  # 返回第一个 sample 的序列（作为示例）
        elif self.use_sglang:
            # 非推理进程返回 dummy 结果，避免 hang
            return torch.zeros((1, 10), device='cuda')
        else:
            return self.net(x)

###############################################################################
# 主函数，展示如何在 DeepSpeed 训练过程中使用 VerlEngine 推理，并进行性能指标分析
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

        # VerlEngine 参数：模型路径、tensor parallel 大小、显存比例等
        engine_tp_size = 4      # tensor parallel 的大小，可根据实际环境调整
        pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--OpenRLHF--Llama-3-8b-sft-mixture/snapshots/03334dc4a796d9d72850ead46956c33da22e6d7b"
        mem_fraction_static = 0.1
        port = 30000

    args = Args()

    # 显式设置当前进程使用的 GPU
    torch.cuda.set_device(args.local_rank)

    # 初始化 DeepSpeed Strategy 并设置分布式环境
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

    # 方式一：使用 VerlEngine 推理模式（生成接口，仅用于生成，不参与梯度更新）
    sglang_backend = SGLangBackend(
        args=args,
        tokenizer=tokenizer,
        prompt_max_len=1024
    )
    
    # 使用 VerlEngine 模式下的模型及优化器（注意：推理模式下的梯度不更新，但需同步所有进程）
    model_sgl = SimpleModel(use_sglang=True)
    model_sgl, optimizer_sgl, scheduler_sgl = strategy.prepare(
        (
            model_sgl, 
            optim.Adam(model_sgl.parameters(), lr=1e-3), 
            StepLR(optim.Adam(model_sgl.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )

    # 设置生成时的输入示例，保持与 vLLM 例子一致
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


    model_sgl.eval()

    # -------------------------------
    # 多次推理测试及性能指标统计
    # -------------------------------
    num_iterations = 10  # 可根据实际需要修改
    durations = []
    throughputs = []
    per_token_times = []
    gpu_memories = []

    for i in range(num_iterations):
        # 在每个 iteration 开始前重置 Peak Memory 统计
        torch.cuda.reset_peak_memory_stats()

        # 预热（第一次调用）不计入性能指标
        if i == 0:
            _ = model_sgl(prompts=prompts, labels=labels, generate_kwargs={"sampling_params": {"temperature": 0.8, "top_p": 0.95}}, sglang_backend=sglang_backend)

        # 调用 model_sgl.forward（内部只统计 engine.generate 的耗时）
        samples = model_sgl(prompts=prompts, labels=labels, generate_kwargs={"sampling_params": {"temperature": 0.8, "top_p": 0.95}}, sglang_backend=sglang_backend)
        # 此时，sglang_backend.latest_gen_times 中保存了每个 prompt 的 engine.generate 耗时（单位：秒）
        gen_time = sum(sglang_backend.latest_gen_times)
        durations.append(gen_time)

        # 统计生成的 tokens 数量（假设每个 sample 的 tokens 数量一致）
        # samples = model_sgl(prompts=prompts, labels=labels, generate_kwargs={"sampling_params": {"temperature": 0.8, "top_p": 0.95}}, sglang_backend=sglang_backend)
        
        print(f"samples:{samples}")
        # token_count = samples.shape[1]
        # token_count =  outputs[0]["meta_info"]['completion_tokens']
        token_count =  sum(sglang_backend.token_count)

        throughput = token_count / gen_time
        throughputs.append(throughput)
        per_token_times.append(gen_time / token_count)

        # 获取当前 iteration 的 Peak GPU 内存（单位：MB）
        iteration_peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        gpu_memories.append(iteration_peak_memory)

        if dist.get_rank() == 0:
            print(f"Iteration {i+1}/{num_iterations}: Generation Time={gen_time:.4f}s, "
                  f"Tokens={token_count}, Throughput={throughput:.2f} tokens/s, "
                  f"Per-token time={ (gen_time / token_count)*1000:.2f} ms, "
                  f"Peak GPU Memory={iteration_peak_memory:.2f} MB")

    # 计算统计指标
    avg_duration = np.mean(durations)
    std_duration = np.std(durations)
    avg_throughput = np.mean(throughputs)
    avg_per_token = np.mean(per_token_times)
    avg_gpu_memory = np.mean(gpu_memories)

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
        f"Average Peak GPU Memory Allocated during inference: {avg_gpu_memory:.2f} MB",
        "",
        "Raw generation times per iteration (s): " + ", ".join([f"{d:.4f}" for d in durations]),
        "Raw throughputs per iteration (tokens/s): " + ", ".join([f"{t:.2f}" for t in throughputs]),
        "Raw per-token times per iteration (ms): " + ", ".join([f"{p*1000:.2f}" for p in per_token_times]),
        "Raw Peak GPU Memory per iteration (MB): " + ", ".join([f"{m:.2f}" for m in gpu_memories]),
    ]
    report_content = "\n".join(report_lines)

    # 保存报告到本地文件（仅 rank 0 保存即可）
    if dist.get_rank() == 0:
        report_filename = "./performance_report_sglang.txt"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Performance analysis report saved to {report_filename}")
        print(report_content)

    if dist.get_rank() == 0:
        print(f"rank {dist.get_rank()}: =========VerlEngine generation test finished =========")


if __name__ == '__main__':
    # 示例运行命令：
    # export CUDA_VISIBLE_DEVICES=0,1,2,3
    # torchrun --master_port=29502 --nnodes=1 --nproc-per-node 4 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_sglang_train.py
    main()