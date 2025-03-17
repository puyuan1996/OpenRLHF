#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
import sys
import time
import datetime
from datetime import timedelta
from collections import defaultdict
from typing import List

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist

# -------------------------------------------------------------------
# 分布式辅助函数等（包括创建通信子组）
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# DeepSpeed Strategy（假设已在 openrlhf 内部定义，这里给出一个简化版）
# -------------------------------------------------------------------
class DeepspeedStrategy:
    def __init__(self, seed, max_norm, micro_train_batch_size, train_batch_size, zero_stage, bf16, args):
        self.seed = seed
        self.max_norm = max_norm
        self.micro_train_batch_size = micro_train_batch_size
        self.train_batch_size = train_batch_size
        self.zero_stage = zero_stage
        self.bf16 = bf16
        self.args = args

    def setup_distributed(self):
        if not dist.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if dist.get_rank() == 0:
            print("Distributed environment setup complete.")

    def prepare(self, components, is_rlhf=False):
        # 此处主要返回模型、优化器与 scheduler（注意：vLLM/VerlEngine 模式梯度不参与更新，但需调用 prepare 保持进程同步）
        model, optimizer, scheduler = components
        model.cuda()
        return model, optimizer, scheduler

# -------------------------------------------------------------------
# 通用的 Samples 与 GenerationBackend 定义
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# vLLM 后端实现
# -------------------------------------------------------------------
def get_vllm_engine(args):
    """
    根据参数构建 vLLM 引擎，并返回引擎对象以及分组
    """
    vllm_tp_group = create_sub_group(args.engine_tp_size)
    # 延迟导入 vLLM，仅在真正初始化时导入
    from vllm import LLM

    vllm_engine = LLM(
        model=args.pretrain,
        task="generate",
        tensor_parallel_size=args.engine_tp_size,
        gpu_memory_utilization=args.engine_mem_util,
        distributed_executor_backend="external_launcher",  # 非常重要
        enable_sleep_mode=args.enable_engine_sleep,
    )
    return vllm_engine, vllm_tp_group

class VLLMBackend(BaseGenerationBackend):
    def __init__(self, args, tokenizer, prompt_max_len: int = 1024):
        """
        args: 包含 vLLM 引擎参数的对象
        tokenizer: 用于 tokenization 的对象
        prompt_max_len: prompt 最大长度
        """
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        if dist.get_rank() == 0:
            print(f"Initializing vLLM engine with tensor_parallel_size: {args.engine_tp_size}")
        self.llm, self.tp_group = get_vllm_engine(args)

    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
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

# -------------------------------------------------------------------
# VerlEngine（SGLang） 后端实现
# -------------------------------------------------------------------
def get_sglang_engine(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    def _log(text):
        t = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{t}] [rank={rank}] {text}")

    _log(f'start {local_rank=} {rank=} {world_size=} {sys.argv=} {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    tp_size = args.engine_tp_size
    dp_size = world_size // tp_size
    assert world_size == tp_size * dp_size

    from torch.distributed.device_mesh import init_device_mesh
    device_mesh_kwargs = dict(
        mesh_shape=(tp_size, dp_size, 1), mesh_dim_names=["tp", "dp", "pp"]
    )
    device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
    _log(f"device_mesh_cpu: {device_mesh_cpu}")

    tp_rank = device_mesh_cpu.get_local_rank("tp")
    dp_rank = device_mesh_cpu.get_local_rank("dp")
    _log(f"tp_rank={tp_rank}, tp_size={tp_size} ; dp_rank={dp_rank}, dp_size={dp_size}")

    # 清理部分环境变量
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]

    from sglang.srt.entrypoints.verl_engine import VerlEngine
    verl_engine = VerlEngine(
        model_path=args.pretrain,
        mem_fraction_static=args.mem_fraction_static,
        device_mesh_cpu=device_mesh_cpu["tp"],
        base_gpu_id=dp_rank,
        gpu_id_step=dp_size,
        port=30000,
    )
    return verl_engine, device_mesh_cpu

class SGLangBackend(BaseGenerationBackend):
    def __init__(self, args, tokenizer, prompt_max_len: int = 1024):
        """
        args: 包含 VerlEngine 引擎参数的对象，其中 pretrain 为模型路径等
        tokenizer: 用于 tokenization 的对象
        prompt_max_len: prompt 最大长度
        """
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        if dist.get_rank() == 0:
            print(f"Initializing VerlEngine with model path: {args.pretrain}")
        self.engine, self.device_mesh = get_sglang_engine(args)

    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        sampling_params = generate_kwargs.get("sampling_params", {"temperature": 0.8, "top_p": 0.95})
        samples_list = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label in zip(prompts, labels):
            generated = self.engine.generate(prompt=prompt, sampling_params=sampling_params)
            print(f"generated: {generated}")
            generated_text = generated['text']
            # 简单补全 EOS
            if len(generated_text) == 0 or generated_text[-1] != chr(eos_token_id):
                generated_text += chr(eos_token_id)

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

# -------------------------------------------------------------------
# 通用简单模型（支持两种后端选择）
# -------------------------------------------------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1, use_inference_backend=False, backend_type=None):
        """
        use_inference_backend: 若为 True，则在 forward 中调用后端的 generate 接口
        backend_type: 字符串 'vllm' 或 'sglang' 用于指定使用哪个推理后端
        """
        super(SimpleModel, self).__init__()
        self.use_inference_backend = use_inference_backend
        self.backend_type = backend_type  # 'vllm' or 'sglang'
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x=None, prompts=None, labels=None, generate_kwargs={}, backend=None):
        if self.use_inference_backend and backend is not None:
            if self.backend_type == 'vllm':
                if dist.get_rank() == 0:
                    print(f'rank {dist.get_rank()}: vLLM generate invoked')
                samples = backend.generate(prompts, labels, generate_kwargs)
                return samples[0].sequences
            elif self.backend_type == 'sglang':
                if dist.get_rank() == 0:
                    print(f'rank {dist.get_rank()}: VerlEngine (SGLang) generate invoked')
                samples = backend.generate(prompts, labels, generate_kwargs)
                return samples[0].sequences
        else:
            return self.net(x)

# -------------------------------------------------------------------
# 主函数，分别对 vLLM 与 VerlEngine 后端进行测试，并生成性能分析报告
# -------------------------------------------------------------------
def main():
    # 模拟从 launcher 传入的 DeepSpeed 命令行参数
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

        # vLLM 参数
        engine_mem_util = 0.5   # GPU 内存利用率限制
        engine_tp_size = 4      # 根据实际环境调整
        # pretrain = "Qwen/Qwen2.5-7B-Instruct"
        pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--OpenRLHF--Llama-3-8b-sft-mixture/snapshots/03334dc4a796d9d72850ead46956c33da22e6d7b"
        enable_engine_sleep = True

        # SGLang（VerlEngine） 参数
        mem_fraction_static = 0.1
        port = 30000

    args = Args()

    # 初始化 DeepSpeed Strategy 与分布式环境
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

    # 设置 GPU
    torch.cuda.set_device(args.local_rank)

    # 尝试加载预训练模型的 tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        if dist.get_rank() == 0:
            print("成功加载预训练 tokenizer:", args.pretrain)
    except Exception as e:
        print("加载预训练的 tokenizer 失败，尝试下载 Qwen2.5-7B 的 tokenizer。错误信息：", e)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    # 设定统一输入
    prompts = ["Hello, how are you?", "What is the weather today?"]
    labels = ["I am fine", "Sunny and warm"]

    # 记录结果信息
    report_lines = []
    report_lines.append("推理后端对比分析报告")
    report_lines.append("="*60)
    report_lines.append("测试环境：")
    report_lines.append(f"  - 使用 GPU: {torch.cuda.get_device_name(args.local_rank)}")
    report_lines.append(f"  - 分布式世界大小: {dist.get_world_size()}")
    report_lines.append("")

    # -------------------------------------------------------------------
    # 对比一：vLLM 后端测试
    # -------------------------------------------------------------------
    if dist.get_rank() == 0:
        print("\n========== 开始 vLLM 后端测试 ==========")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")
    start_time = time.time()

    vllm_backend = VLLMBackend(args=args, tokenizer=tokenizer, prompt_max_len=1024)

    model_vllm = SimpleModel(use_inference_backend=True, backend_type='vllm')
    model_vllm, optimizer_vllm, scheduler_vllm = strategy.prepare(
        (
            model_vllm, 
            optim.Adam(model_vllm.parameters(), lr=1e-3), 
            StepLR(optim.Adam(model_vllm.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )
    model_vllm.eval()
    with torch.no_grad():
        generated_vllm = model_vllm(prompts=prompts, labels=labels, generate_kwargs={}, backend=vllm_backend)
        # 同步保证准确计时
        torch.cuda.synchronize()
    end_time = time.time()
    vllm_time = end_time - start_time
    vllm_mem = torch.cuda.max_memory_allocated("cuda") / (1024 ** 2)  # 单位 MB

    if dist.get_rank() == 0:
        print("vLLM 模式下生成的序列 tensor:", generated_vllm)
        print(f"vLLM 后端耗时: {vllm_time:.4f}s, GPU 最大显存占用: {vllm_mem:.2f} MB")
    
    report_lines.append("vLLM 后端:")
    report_lines.append(f"  - 总耗时 (s): {vllm_time:.4f}")
    report_lines.append(f"  - GPU 最大显存占用 (MB): {vllm_mem:.2f}")
    report_lines.append("")

    # -------------------------------------------------------------------
    # 对比二：VerlEngine (SGLang) 后端测试
    # -------------------------------------------------------------------
    if dist.get_rank() == 0:
        print("\n========== 开始 VerlEngine (SGLang) 后端测试 ==========")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")
    start_time = time.time()

    sglang_backend = SGLangBackend(args=args, tokenizer=tokenizer, prompt_max_len=1024)

    model_sgl = SimpleModel(use_inference_backend=True, backend_type='sglang')
    model_sgl, optimizer_sgl, scheduler_sgl = strategy.prepare(
        (
            model_sgl, 
            optim.Adam(model_sgl.parameters(), lr=1e-3), 
            StepLR(optim.Adam(model_sgl.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )
    model_sgl.eval()
    with torch.no_grad():
        generated_sgl = model_sgl(
            prompts=["你好，今天天气如何？", "请问量子计算的基本原理是什么？"],
            labels=["今天天气晴朗", "量子计算基于量子力学原理"],
            generate_kwargs={"sampling_params": {"temperature": 0.8, "top_p": 0.95}},
            backend=sglang_backend
        )
        torch.cuda.synchronize()
    end_time = time.time()
    sglang_time = end_time - start_time
    sglang_mem = torch.cuda.max_memory_allocated("cuda") / (1024 ** 2)  # 单位 MB

    if dist.get_rank() == 0:
        print("VerlEngine 模式下生成的序列 tensor:", generated_sgl)
        print(f"VerlEngine 后端耗时: {sglang_time:.4f}s, GPU 最大显存占用: {sglang_mem:.2f} MB")
    
    report_lines.append("VerlEngine (SGLang) 后端:")
    report_lines.append(f"  - 总耗时 (s): {sglang_time:.4f}")
    report_lines.append(f"  - GPU 最大显存占用 (MB): {sglang_mem:.2f}")
    report_lines.append("")

    # -------------------------------------------------------------------
    # 综合对比及分析报告生成
    # -------------------------------------------------------------------
    if dist.get_rank() == 0:
        report_lines.append("对比分析:")
        if vllm_time < sglang_time:
            report_lines.append("  - 推理速度: vLLM 后端较快。")
        elif vllm_time == sglang_time:
            report_lines.append("  - 推理速度: 二者耗时相近。")
        else:
            report_lines.append("  - 推理速度: VerlEngine (SGLang) 后端较快。")
            
        if vllm_mem < sglang_mem:
            report_lines.append("  - 显存占用: vLLM 后端占用显存较低。")
        elif vllm_mem == sglang_mem:
            report_lines.append("  - 显存占用: 二者占用显存相近。")
        else:
            report_lines.append("  - 显存占用: VerlEngine (SGLang) 后端占用显存较低。")

        report_lines.append("="*60)
        report_lines.append("详细报告已保存至 inference_comparison_report.txt")
        report = "\n".join(report_lines)
        print("\n" + report)

        with open("inference_comparison_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

if __name__ == '__main__':
    # torchrun --nnodes=1 --nproc-per-node 4 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_vllm_train.py
    main()