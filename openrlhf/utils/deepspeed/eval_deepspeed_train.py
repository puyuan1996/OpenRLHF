import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from datetime import timedelta


# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    # 模拟 DeepSpeed 的命令行参数，一般由 launcher 传入
    class Args:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        adam_offload = False
        ring_attn_size = 1  # 如果需要 ring attn 设置成 >1，比如2，4等

    args = Args()

    from openrlhf.utils.deepspeed.deepspeed_strategy import DeepspeedStrategy

    # strategy = DeepspeedStrategy(
    #     seed=getattr(args, "seed", 42),
    #     max_norm=getattr(args, "max_norm", 1.0),
    #     micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
    #     train_batch_size=getattr(args, "train_batch_size", 128),
    #     zero_stage=args.zero_stage,
    #     bf16=getattr(args, "bf16", True),
    #     args=args,
    # )
    
    # 实例化训练策略，并初始化分布式环境
    strategy = DeepspeedStrategy(
        seed=42,
        max_norm=1.0,
        micro_train_batch_size=4,
        train_batch_size=32,
        zero_stage=2,
        bf16=False,
        args=args,
    )
    strategy.setup_distributed()

    # 创建一个简单的模型，优化器，调度器
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 使用 DeepSpeed 封装模型
    model, optimizer, scheduler = strategy.prepare((model, optimizer, scheduler), is_rlhf=False)

    # 构造一个简单的训练数据：这里用随机数据替代真实数据，实际使用中需加载真实数据集
    inputs = torch.randn(1000, 10).cuda()
    targets = torch.randn(1000, 1).cuda()
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=strategy.micro_train_batch_size, shuffle=True)

    model.train()
    num_epochs = 2

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):
            # 前向传播
            outputs = model(x)
            loss = ((outputs - y) ** 2).mean()
            # 反向传播与梯度累积处理
            model.backward(loss)
            if (i + 1) % strategy.accumulated_gradient == 0:
                model.step()
                optimizer.zero_grad()
                strategy.print(f"Epoch {epoch}, Step {i}: loss = {loss.item()}")

    # 保存训练好的模型（仅在 rank 0 上执行保存操作）
    if strategy.is_rank_0():
        save_dir = "./saved_model_debug"
        os.makedirs(save_dir, exist_ok=True)
        model.save_checkpoint(save_dir, tag="final_checkpoint_debug")
        strategy.print("Training completed and model saved.")

if __name__ == '__main__':
    # torchrun --nnodes=1 --nproc-per-node 8 /fs-computility/ai-shen/puyuan/code/OpenRLHF/openrlhf/utils/deepspeed/eval_deepspeed_train.py 
    main()