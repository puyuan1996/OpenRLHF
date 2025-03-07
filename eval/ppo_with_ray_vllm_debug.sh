#!/bin/bash
# 停止当前所有已启动的 ray 服务（如果有）
ray stop

# 等待 2 秒，确保所有进程停止
sleep 2

# 启动 head 节点，使用 8 个 GPU
echo "启动 Ray head 节点..."
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8

# 等待 5 秒，确保 Ray head 节点和 GCS 服务正常启动
sleep 5

# 可以通过以下命令查看 Ray 集群状态（可选）
ray status

# 提交作业，运行指定的 Python 模块
# 请注意这里--address后面的 IP 地址和端口需要对应上你的集群配置，
# 此外确保 runtime-env-json 中的 working_dir 路径存在且正确
echo "提交 Ray 作业..."


ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json='{
    "working_dir": "/fs-computility/ai-shen/puyuan/code/OpenRLHF",
    "excludes": [
      "checkpoint/**",
      "hub/**",
      "datasets/**"
    ]
  }' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --backend vllm \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 2 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 2 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2 \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --pretrain /fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --reward_pretrain /fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
  --micro_train_batch_size 8 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 1024 \
  --max_samples 100000 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb 968275bc822c87ac741ecce2f06cdfb54dbc1608

# 脚本结束后不退出，方便查看输出日志
echo "Ray 作业提交完成。"

  # --prompt_data /fs-computility/ai-shen/puyuan/model/huggingface/hub/datasets--OpenRLHF--prompt-collection-v0.1/snapshots/1d3be64c51aa57fa16aa5dc70d1bfc26e9847e12 \
