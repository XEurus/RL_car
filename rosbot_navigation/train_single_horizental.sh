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
TRAIN_PY="/root/workspace/RL_car2/rosbot_navigation/train_single.py"

echo "==> Manual curriculum run"

BASE_MODEL_DIR="/root/workspace/RL_car2/rosbot_navigation/results/single_vertical_7"
prev_model_path=""
train_id="1"
experiment_name="Single-Vertical-7"

# ---------- 无障碍物 ----------
stage=end
remark="env1"
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/warehouse5_end1.wbt"
total_steps=60000
STAGE_DIR="$BASE_MODEL_DIR/${remark}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
echo "\n==== Training stage: $stage | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --pretrained_model_path "$prev_model_path"
  --total_steps "$total_steps"
  --model_dir "$STAGE_DIR"
  --curriculum_stage "$stage"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
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


# ----------四个障碍物 ----------
STAGE=end
remark="env2"
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env2.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${remark}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=80000
prev_model_path="/root/workspace/RL_car2/rosbot_navigation/results/single_vertical_4/env1_1/td3_env1_normal_80000.zip"
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --model_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --pretrained_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
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


# ---------- 六个障碍物 ----------
STAGE=end
remark="env3"
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env3.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${remark}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=100000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --model_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --pretrained_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
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


# ---------- 八个障碍物 ----------
STAGE=end
remark="env4"
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env4.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${remark}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=120000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --model_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --pretrained_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
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


# ---------- 十个障碍物 ----------
STAGE=end
remark="env5"
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env5.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${remark}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=150000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --model_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --pretrained_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
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


# ---------- 12个障碍物 ----------
STAGE=end
remark="env6"
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env6.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${remark}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=170000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --model_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --pretrained_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
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



# ---------- 13个障碍物 ----------
STAGE=end
remark="env_end"
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/vertical/warehouse5_env_end3.wbt"
STAGE_DIR="$BASE_MODEL_DIR/${remark}_${train_id}"
mkdir -p "$STAGE_DIR"
LOG_FILE="$STAGE_DIR/train.log"
total_steps=500000
echo "\n==== Training stage: $STAGE | $(date) ====" | tee -a "$LOG_FILE"
ARGS=(
  --total_steps "$total_steps"
  --model_dir "$STAGE_DIR"
  --curriculum_stage "$STAGE"
  --pretrained_model_path "$prev_model_path"
  --remark "$remark"
  --experiment_name "$experiment_name"
  --world_1 "$WORLD_PATH"
)
/root/miniconda3/envs/rl_car/bin/python -u "$TRAIN_PY" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "==== Stage $STAGE finished | $(date) ====\n" | tee -a "$LOG_FILE"
# Set prev_model_path to latest .zip inside this stage dir (fallback to dir if none)
latest_zip=$(ls -1t "$STAGE_DIR"/*.zip 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_zip:-}" ]]; then
  echo "WARNING: No .zip modStage: hard2el found in $STAGE_DIR; using directory as prev_model_path"
  prev_model_path="$STAGE_DIR"
else
  echo "Found previous model: $latest_zip"
  prev_model_path="$latest_zip"
fi

echo "All stages completed. Models saved under: $BASE_MODEL_DIR"
