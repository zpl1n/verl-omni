#!/usr/bin/env bash
# Two optimizer steps cover rollout -> rule reward -> replay -> update -> weight sync.
# Requires the VeOmni qwen-image-21-adaptation branch in the training environment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export NUM_GPUS=${NUM_GPUS:-4}
export MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/qwen-image-21}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_qwen_image_21}
export TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
export VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
export TRAIN_BATCH_SIZE=$NUM_GPUS
export PPO_MINI_BATCH_SIZE=$NUM_GPUS
export ROLLOUT_N=2
export IMAGE_RESOLUTION=64
export NUM_INFERENCE_STEPS=4
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

python3 tests/special_e2e/build_qwen_image_21_tiny_random.py --output-dir "$MODEL_PATH"
python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "$DATA_DIR" --train_size "$((NUM_GPUS * 2))" --val_size "$NUM_GPUS"
bash examples/flowgrpo_trainer/qwen_image_21/run_qwen_image_21_veomni.sh \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,3]' \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    trainer.save_freq=-1 trainer.test_freq=-1 "$@"
