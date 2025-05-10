#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
from datetime import timedelta
import datetime
import os
import sys

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
import sys
sys.path.insert(0, "/fs-computility/ai-shen/puyuan/code/OpenRLHF")

# 假设 DeepspeedStrategy 已经在 openrlhf 内部定义
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
    _log(f"{device_mesh_cpu=}")

    tp_rank = device_mesh_cpu.get_local_rank("tp")
    dp_rank = device_mesh_cpu.get_local_rank("dp")
    _log(f"{tp_rank=} {tp_size=} ; {dp_rank=} {dp_size=}")

    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]

    # import torch.distributed as dist
    # # 显式同步所有进程
    # group = dist.new_group(backend='gloo', timeout=timedelta(seconds=30)) # 使用 gloo 后端支持 CPU 张量的广播

    verl_engine = VerlEngine(
        model_path=args.pretrain,
        mem_fraction_static=args.mem_fraction_static,
        device_mesh_cpu=device_mesh_cpu["tp"],
        base_gpu_id=dp_rank,
        gpu_id_step=dp_size,
        port=30000,
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
    
    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        # 采样参数：可通过 generate_kwargs 传入采样配置，否则使用默认值

        # 生成前同步各进程，避免部分进程提前进入生成逻辑
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()
        #     print(f"rank {torch.distributed.get_rank()} has passed the barrier in generate.")
            
        sampling_params = generate_kwargs.get("sampling_params", {"temperature": 0.8, "top_p": 0.95})
        samples_list = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label in zip(prompts, labels):
            # 调用 VerlEngine 进行生成，此处直接传入字符串 prompt
            # print(f"prompts:{prompts}, ")
            generated = self.engine.generate(prompt=prompt, sampling_params=sampling_params)
            # generated = self.engine.generate(prompt=prompt, sampling_params=dict(max_new_tokens=16, temperature=0.0))
            print(f"generated:{generated}")
            generated_text = generated['text']
            # 如果返回文本为空或末尾没有 eos 标记，则进行简单补全（实际使用时请根据 tokenizer 配置处理）
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
        import torch.distributed as dist
        
        if self.use_sglang and sglang_backend is not None:
            # print(f'rank {dist.get_rank()}: =========debug: VerlEngine generation mode =========')
            samples = sglang_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences
        elif self.use_sglang:
            # 非推理进程返回 dummy 结果
            return torch.zeros((1, 10), device='cuda')
        else:
            return self.net(x)

###############################################################################
# 主函数，展示如何在 DeepSpeed 训练过程中使用 VerlEngine 推理以及普通训练模式
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

        # VerlEngine（原 sglang 引擎）参数：模型路径、tensor parallel 大小等
        engine_tp_size = 4      # tensor parallel 的大小，可根据需要调整
        pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--OpenRLHF--Llama-3-8b-sft-mixture/snapshots/03334dc4a796d9d72850ead46956c33da22e6d7b"
        mem_fraction_static = 0.1
        port = 30000

        # 如有需要，也可以调整其他参数
        # engine_tp_size = 2
        # pretrain = "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"


    args = Args()

    # 显式设置当前进程使用的 GPU
    torch.cuda.set_device(args.local_rank)

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

    # 尝试加载预训练模型对应的 tokenizer，如果加载失败则从 Hugging Face 下载 Qwen2.5-7B 的 tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        print("成功加载预训练tokenizer:", args.pretrain)
    except Exception as e:
        print("加载预训练的tokenizer失败，尝试从 Hugging Face 下载 Qwen2.5-7B 的tokenizer。错误信息：", e)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


    # print(f'rank {dist.get_rank()}: =========debug: pos 0 =========')

    # 方式一：使用 VerlEngine 推理模式（生成调用，不用于梯度更新）
    sglang_backend = SGLangBackend(
        args=args,
        tokenizer=tokenizer,
        prompt_max_len=1024
    )
    
    # print(f'rank {dist.get_rank()}: =========debug: starting VerlEngine generation test =========')

    model_sgl = SimpleModel(use_sglang=True)

    # print(f'rank {dist.get_rank()}: =========debug: pos 3 =========')

    model_sgl, optimizer_sgl, scheduler_sgl = strategy.prepare(
        (
            model_sgl, 
            optim.Adam(model_sgl.parameters(), lr=1e-3), 
            StepLR(optim.Adam(model_sgl.parameters(), lr=1e-3), step_size=10, gamma=0.1)
        ),
        is_rlhf=False
    )
    # print(f'rank {dist.get_rank()}: =========debug: prepared model_sgl =========')

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
        # print(f'rank {dist.get_rank()}: =========debug: VerlEngine generation output =========')
        print(f"rank {dist.get_rank()}: VerlEngine 模式下生成的序列 tensor:", generated)

    print(f'rank {dist.get_rank()}: =========VerlEngine generation test finished =========')


if __name__ == '__main__':
    # 示例运行命令：
    # export CUDA_VISIBLE_DEVICES=0,1,2,3
    # torchrun --master_port=29502 --nnodes=1 --nproc-per-node 4 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_sglang_train.py
    main()