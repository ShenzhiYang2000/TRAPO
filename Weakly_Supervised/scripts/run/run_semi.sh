# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x


ROOT=/ossfs/workspace/aml0
MODEL_DATABASE=/ossfs/workspace/aitech_aidata/models/Qwen
MODEL_NAME=Qwen3-4B-Base
EXP_NAME=TraPO-semi
SAVE_ROOT=/ossfs/workspace/aitech_aidata
CKPT_DIR=$SAVE_ROOT/ckpt/$MODEL_NAME/$EXP_NAME



python3 -m Weakly_Supervised.wsl_main \
    algorithm.adv_estimator=grpo \
    data.train_files=/ossfs/workspace/aml0/code/TraPO/data/ID_data/processed/id_l_1k_u_1k_fixed.parquet \
    data.val_files=/ossfs/workspace/aml0/code/TraPO/data/valid_fixed.parquet\
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_DATABASE/$MODEL_NAME \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=1 \
    trainer.logger='["console"]' \
    trainer.project_name=$EXP_NAME \
    trainer.experiment_name=$EXP_NAME \
    +trainer.warm_up=8 \
    +trainer.update_db=True \
    +trainer.topk=0.1 \
    +trainer.thres=0.4 \
    trainer.train_mode="semi-supervised" \
    trainer.output_log_path=/ossfs/workspace/aitech_aidata/log/TraPO-semi.log \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20000 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
