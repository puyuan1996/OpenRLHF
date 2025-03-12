# TODO: no-ray sglang
deepspeed --module openrlhf.cli.train_ppo \
  --backend sglang \
  --pretrain  /fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --reward_pretrain  /fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
  --save_path ./checkpoint/qwen25-0.5b-rlhf-sglang \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data /fs-computility/ai-shen/puyuan/model/huggingface/hub/datasets--OpenRLHF--prompt-collection-v0.1/snapshots/1d3be64c51aa57fa16aa5dc70d1bfc26e9847e12 \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb 968275bc822c87ac741ecce2f06cdfb54dbc1608

# Support remote reward model (HTTP)
# --remote_rm_url http://localhost:5000/get_reward


