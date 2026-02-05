# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

python3 -m verl.trainer.main_trapo \
    algorithm.adv_estimator=grpo \
    data.train_files=/ossfs/workspace/aml0/484999/code/LUFFY/data/ID_domain/openr1_2048_fixed.parquet \
    data.val_files=/ossfs/workspace/aml0/484999/code/LUFFY/data/ID_domain/openr1_1024_fixed.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/ossfs/workspace/aml0/Qwen3/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_TraPO_test' \
    trainer.train_mode="semi-supervised" \
    trainer.experiment_name='qwen3_06b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20000 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@



python /ossfs/workspace/aml0/485004/test.py --gpus 0,1,2,3