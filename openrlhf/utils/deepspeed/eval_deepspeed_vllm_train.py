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

# 假设 DeepspeedStrategy 已经定义（如你的 deepspeed demo 代码中所示），本例中从 openrlhf 导入
from openrlhf.utils.deepspeed.deepspeed_strategy import DeepspeedStrategy

###############################################################################
# vllm 后端相关代码（仅为示例）
###############################################################################
# 假设存在一个基类（也可以没有，仅为了提示接口一致性）
class BaseGenerationBackend:
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List:
        raise NotImplementedError

# Samples 类用于封装生成结果（可根据自己的需求调整）
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

class VLLMBackend(BaseGenerationBackend):
    def __init__(self, model_name: str, tokenizer, prompt_max_len: int = 1024, tensor_parallel_size=1, **kwargs):
        """
        model_name: vllm 模型名称或路径
        tokenizer: 用于 tokenization 的对象，要求接口与 Hugging Face 的一致
        prompt_max_len: prompt 最大长度
        tensor_parallel_size: 张量并行大小
        kwargs: 传给 vllm.LLM 的其他参数
        """
        from vllm import LLM  # 注意确保已安装 vllm
        print(f"Initializing vllm LLM with tensor_parallel_size: {tensor_parallel_size}")
        self.tokenizer = tokenizer
        self.llm = LLM(model=model_name, task="generate", tensor_parallel_size=tensor_parallel_size, **kwargs)
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate(self, prompts: List[str], labels: List[str], generate_kwargs: dict = {}) -> List[Samples]:
        # 调用 vllm 接口进行生成，假设返回的是一个列表，每个元素包含 token_ids 信息
        outputs = self.llm.generate(prompts, **generate_kwargs)
        samples_list = []
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else 1

        for prompt, label, output in zip(prompts, labels, outputs):
            output_token_ids = list(output.outputs[0].token_ids)
            # 保证生成的 token 序列以 eos 结束
            if output_token_ids[-1] != eos_token_id:
                output_token_ids.append(eos_token_id)
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=False, max_length=self.prompt_max_len, truncation=True
            )["input_ids"]
            all_ids = prompt_ids + output_token_ids
            sequences = torch.tensor([all_ids])
            attention_mask = (sequences != pad_token_id).float()
            # 标记生成的部分（从 prompt 长度开始设为1）
            action_mask = torch.zeros_like(sequences, dtype=torch.bool)
            action_mask[:, len(prompt_ids):] = 1

            # 移动到 GPU（假设训练在 CUDA 上）
            sequences = sequences.to("cuda")
            attention_mask = attention_mask.to("cuda")
            action_mask = action_mask.to("cuda")

            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask[:, 1:].float(),  # 此处仅为示例，依实际场景调整
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

###############################################################################
# 定义支持 vllm 的简单模型
###############################################################################
class SimpleModel(nn.Module):
    """
    如果 use_vllm 为 True，则 forward 中调用 vllm 后端生成方法（用于测试生成能力）；
    否则，使用普通的前向前馈神经网络进行训练（支持反向传播）。
    """
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1, use_vllm=False, vllm_backend=None):
        super(SimpleModel, self).__init__()
        self.use_vllm = use_vllm
        self.vllm_backend = vllm_backend
        # if not self.use_vllm:
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x=None, prompts=None, labels=None, generate_kwargs={}):
        """
        当 use_vllm 为 True 时，以 prompts, labels 为输入调用 vllm 后端生成回复；
        否则执行常规前向传播。  
        注意：vllm 的生成过程通常不可微，不适合直接参与梯度反传。
        """
        import torch.distributed as dist
        # if self.use_vllm and self.vllm_backend is not None and dist.get_rank() == 0:
        if self.use_vllm and self.vllm_backend is not None:

            # rank 0 使用 vllm
            if prompts is None or labels is None:
                raise ValueError("在使用 vllm 模式时，必须提供 prompts 和 labels。")
            samples = self.vllm_backend.generate(prompts, labels, generate_kwargs)
            return samples[0].sequences
        else:
            # 非rank0或未配置 vllm backend 时，使用普通前向传播（这里 x 需要有效输入）
            return self.net(x)

###############################################################################
# 主函数，展示如何在 deepspeed 训练过程中使用 vllm 模式及普通模式
###############################################################################
def main():
    # 模拟 DeepSpeed 的命令行参数，一般由 launcher 传入
    class Args:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        adam_offload = False
        ring_attn_size = 1  # 可根据需要设置 >1 使用 ring attn
        pretrain_data = None  # 用于特定 rlhf 情景
        zero_stage = 2
        bf16 = False
        micro_train_batch_size = 4
        train_batch_size = 32
        max_norm = 1.0

    args = Args()

    # 初始化 deepspeed strategy，并设置分布式环境
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
            # 简单的示例：将每个字符的 Unicode 码 % 100 作为 token id（仅用于演示）
            token_ids = [ord(c) % 100 for c in text][:max_length]
            return {"input_ids": token_ids}

        def save_pretrained(self, output_dir):
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "tokenizer.txt"), "w", encoding="utf-8") as f:
                f.write("Dummy tokenizer parameters.")

    tokenizer = DummyTokenizer()

    import torch.distributed as dist
    vllm_backend = VLLMBackend(
        # model_name="Qwen/Qwen2.5-0.5B",  # 替换为实际模型名称或路径
        model_name="/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987",
        tokenizer=tokenizer,
        prompt_max_len=1024,
        # tensor_parallel_size=2,
        tensor_parallel_size=1,
    )
    # if dist.get_rank() == 0:
    #     # 初始化 vllm 后端
    #     vllm_backend = VLLMBackend(
    #         # model_name="Qwen/Qwen2.5-0.5B",  # 替换为实际模型名称或路径
    #         model_name="/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987",
    #         tokenizer=tokenizer,
    #         prompt_max_len=1024,
    #         tensor_parallel_size=1,
    #     )
    # else:
    #     vllm_backend = None

    print(f'rank {dist.get_rank()}: =========debug: pos 1=========')
    ###############################################################################
    # 方式一：使用 vllm 模式（仅生成调用，不进行梯度更新）
    ###############################################################################
    model_vllm = SimpleModel(use_vllm=True, vllm_backend=vllm_backend)
    # 注意：在 vllm 模式下，模型的 forward 结果通常为推理或生成内容，无法直接计算反向传播梯度

    print(f'rank {dist.get_rank()}: =========debug: pos 2=========')

    # 此处仅为了演示如何用 DeepSpeed 包装模型（vllm 模式下可以用于测试生成分布式调用流程）
    model_vllm, optimizer_vllm, scheduler_vllm = strategy.prepare((model_vllm, optim.Adam(model_vllm.parameters(), lr=1e-3), StepLR(optim.Adam(model_vllm.parameters(), lr=1e-3), step_size=10, gamma=0.1)), is_rlhf=False)

    print(f'rank {dist.get_rank()}: =========debug: pos 3=========')

    # 设置生成输入（注意，为了方便展示，这里只取两个示例文本）
    prompts = ["Hello, how are you?", "What is the weather today?"]
    labels = ["I am fine", "Sunny and warm"]

    model_vllm.eval()
    with torch.no_grad():
        generated = model_vllm(prompts=prompts, labels=labels, generate_kwargs={})
        strategy.print("vllm 模式下生成的序列 tensor:", generated)

    print(f'rank {dist.get_rank()}: =========debug: pos 4=========')

    ###############################################################################
    # 方式二：使用普通前向传播进行训练
    ###############################################################################
    model_training = SimpleModel(use_vllm=False)
    optimizer_training = optim.Adam(model_training.parameters(), lr=1e-3)
    scheduler_training = StepLR(optimizer_training, step_size=10, gamma=0.1)

    model_training, optimizer_training, scheduler_training = strategy.prepare((model_training, optimizer_training, scheduler_training), is_rlhf=False)

    # 构造简单的训练数据（随机数据示例）
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
            # 反向传播（调用 DeepSpeed 包装过的 backward）
            model_training.backward(loss)
            if (i + 1) % strategy.accumulated_gradient == 0:
                model_training.step()
                optimizer_training.zero_grad()
                strategy.print(f"Epoch {epoch}, Step {i}: loss = {loss.item()}")

    # 保存训练好的模型（仅在 rank 0 上操作）
    if strategy.is_rank_0():
        save_dir = "./saved_model_debug"
        os.makedirs(save_dir, exist_ok=True)
        model_training.save_checkpoint(save_dir, tag="final_checkpoint_debug")
        strategy.print("Training completed and model saved.")

if __name__ == '__main__':
    main()
    # deepspeed --num_gpus=2 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_vllm_train.py 