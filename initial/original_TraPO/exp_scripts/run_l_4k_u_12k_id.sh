set -x
export HF_ENDPOINT=https://hf-mirror.com

eval "$(conda shell.bash hook)"
# conda activate trapo

# NOTE: change to your root dir
ROOT=/ossfs/workspace/aitech_aidata/chuwei/code/TraPO

ray stop 

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export WANDB_DIR='/ossfs/workspace/aitech_aidata/chuwei/code/TraPO/wandb'
# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

export MODEL_PATH=/ossfs/workspace/aml0/484999/1-models/Qwen2.5-Math-7B-16k-think

export DATA_DIR=$ROOT/data/
export EXP_NAME=labeled_4k_unlabeled_12k_id
export WANDB_PROJECT="semi-grpo-indomain"

cd $ROOT/trapo/verl/

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/ID_domain/openr1_4096_fixed.parquet \
    +data.unlabel_train_files=$DATA_DIR/ID_domain/openr1_4096_12288_fixed.parquet \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_batch_size=32 \
    +data.unlabel_train_batch_size=96 \
    data.val_batch_size=512 \
    data.max_prompt_length=3072 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=4096 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=True \
    +trainer.mode='semi-supervised' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="no_reshape" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    +actor_rollout_ref.labeled_is_label=True \
    +actor_rollout_ref.unlabeled_is_label=False \
    +actor_rollout_ref.use_unlabel_loss=True \
    +actor_rollout_ref.output_log_path=/ossfs/workspace/aitech_aidata/chuwei/code/TraPO/log/trapo_l_1k_u_3k_id.log \
    +actor_rollout_ref.start_semi_epoch=8 \
    +actor_rollout_ref.tau=0.4 \
    +actor_rollout_ref.use_trend_match=True \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=15 "${@:1}"



python /ossfs/workspace/aml0/485004/test.py --gpus 0,1,2,3,4,5,6,7
