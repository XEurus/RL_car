#!/usr/bin/env bash
# Sequential curriculum training: pass previous stage model to the next stage
# Usage:
#   scripts/train_many.sh
# Env overrides:
#   TOTAL_STEPS   (default: 50000)
#   BASE_MODEL_DIR (default: <repo>/models/<timestamp>)
#
# This version does NOT use a loop. You can manually configure each stage's arguments below.

set -euo pipefail

# Resolve paths
TRAIN_PY="/root/workspace/RL_car2/rosbot_navigation/train_stage1.py"

echo "==> Manual curriculum run"

# ==========================
# User manual configuration
# Edit the ARGS arrays per stage as needed.
# You can override TOTAL_STEPS per stage with STAGE_TOTAL_STEPS env or inline.
# Examples of useful flags:
#   --distributed true
#   --device auto
#   --learning_rate 3e-4
#   --enable_speed_smoothing true
#   --lr_schedule_type cosine --lr_final 1e-5
#   --exploration_noise_type linear --exploration_noise_init 0.2 --exploration_noise_final 0.05
# ==========================
BASE_MODEL_DIR="/root/workspace/RL_car2/rosbot_navigation/results/vertical_2"
prev_model_path=""
train_id="1"
experiment_name="Vertical_NoNoiseDecay_NoScale"
remark="任务不变变环境,vertical测试"

# ---------- Stage: start ----------
stage=end
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env1.wbt"
total_steps=30000
STAGE_DIR="$BASE_MODEL_DIR/${stage}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
echo "\n==== Training stage: $stage | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --models_dir "$STAGE_DIR"
  --curriculum_stage "$stage"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
  --world_2 "$WORLD_PATH"
  # add more start-specific args below
  # --learning_rate 3e-4
)

/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $stage finished | $(date) ====\n" | tee -a "$LOG_FILE"
ls "$STAGE_DIR"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip model found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi

# ---------- Stage: easy ----------
STAGE=end
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env2.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${STAGE}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=500000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --models_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --prev_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
  --world_2 "$WORLD_PATH"
  # add more easy-specific args below
)
/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $STAGE finished | $(date) ====\n" | tee -a "$LOG_FILE"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip model found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi

# ---------- Stage: medium ----------
STAGE=end
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env3.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${STAGE}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=100000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --models_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --prev_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
  --world_2 "$WORLD_PATH"
  # add more medium-specific args below
)
/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $STAGE finished | $(date) ====\n" | tee -a "$LOG_FILE"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip model found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi

# ---------- Stage: hard ----------
STAGE=end
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env4.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${STAGE}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=100000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --models_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --prev_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
  --world_2 "$WORLD_PATH"
  # add more hard-specific args below
)
/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $STAGE finished | $(date) ====\n" | tee -a "$LOG_FILE"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip model found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi


STAGE=end
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env5.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${STAGE}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=100000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --models_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --prev_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
  --world_2 "$WORLD_PATH"
  # add more hard-specific args below
)
/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $STAGE finished | $(date) ====\n" | tee -a "$LOG_FILE"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip model found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi


STAGE=end
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env6.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${STAGE}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=100000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --models_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --prev_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
  --world_2 "$WORLD_PATH"
  # add more hard-specific args below
)
/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $STAGE finished | $(date) ====\n" | tee -a "$LOG_FILE"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip model found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi



# ---------- Stage: end ----------
STAGE=end
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env_end1.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${STAGE}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=300000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --models_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --prev_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
  --world_2 "$WORLD_PATH"
  # add more end-specific args below
)
/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $STAGE finished | $(date) ====\n" | tee -a "$LOG_FILE"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip model found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi

echo "All stages completed. Models saved under: $BASE_MODEL_DIR"
