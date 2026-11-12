#!/usr/bin/env bash
# Full-weight Qwen-Image-2.1 base T2I, FlowGRPO, VeOmni actor and vllm-omni rollout.
# Run from the repository root. MODEL_PATH is a local diffusers checkpoint directory.
# TRAIN_FILES/VAL_FILES are parquet files; see README.md for data and reward setup.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen-Image-2.1 base checkpoint}"
: "${TRAIN_FILES:?Set TRAIN_FILES to the training parquet file}"
: "${VAL_FILES:?Set VAL_FILES to the validation parquet file}"
NUM_GPUS=${NUM_GPUS:-4}
NUM_NODES=${NUM_NODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-4}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-40}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-pkg://verl_omni.utils.reward_score.jpeg_compressibility}
REWARD_FUNCTION_NAME=${REWARD_FUNCTION_NAME:-compute_score}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen_image_21_flowgrpo_veomni}
for value in "$NUM_GPUS" "$NUM_NODES"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_GPUS (total GPUs) and NUM_NODES must be positive integers." >&2
        exit 1
    fi
done
if (( NUM_GPUS % NUM_NODES != 0 )); then
    echo "NUM_GPUS must be divisible by NUM_NODES." >&2
    exit 1
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROMPT_TEMPLATE=$(cat "$SCRIPT_DIR/prompt_template.jinja")

python3 -m verl_omni.trainer.main_diffusion_v1 \
    diffusion/model_engine=veomni_diffusion \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    "data.train_files=$TRAIN_FILES" \
    "data.val_files=$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length=512 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "actor_rollout_ref.model.tokenizer_path=$MODEL_PATH/processor" \
    "actor_rollout_ref.model.custom_chat_template=\"$PROMPT_TEMPLATE\"" \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.strategy=veomni \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.veomni_config.attn_implementation=eager \
    actor_rollout_ref.actor.veomni_config.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.veomni_config.param_offload=true \
    actor_rollout_ref.actor.veomni_config.optimizer_offload=true \
    actor_rollout_ref.actor.veomni_config.mixed_precision_cast_forward_inputs=false \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.agent.num_workers="$NUM_GPUS" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.step_execution=false \
    actor_rollout_ref.rollout.enable_prompt_embed_cache=false \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=true \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height="$IMAGE_RESOLUTION" \
    actor_rollout_ref.rollout.pipeline.width="$IMAGE_RESOLUTION" \
    actor_rollout_ref.rollout.pipeline.num_inference_steps="$NUM_INFERENCE_STEPS" \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=512 \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type=sde \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,5]' \
    actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps="$NUM_INFERENCE_STEPS" \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.veomni_config.attn_implementation=eager \
    actor_rollout_ref.ref.veomni_config.mixed_precision_cast_forward_inputs=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.num_workers=1 \
    reward.reward_model.enable=false \
    "reward.custom_reward_function.path=$REWARD_FUNCTION_PATH" \
    "reward.custom_reward_function.name=$REWARD_FUNCTION_NAME" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    'trainer.logger=[console]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_21_veomni \
    "trainer.default_local_dir=$OUTPUT_DIR" \
    trainer.n_gpus_per_node="$((NUM_GPUS / NUM_NODES))" \
    trainer.nnodes="$NUM_NODES" \
    trainer.val_before_train=false \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" "$@"
