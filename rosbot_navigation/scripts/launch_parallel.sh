#!/usr/bin/env bash
set -euo pipefail

# 并行启动多个 Webots 实例（FAST/无渲染）并运行训练脚本
# 使用方式：
#   ./scripts/launch_parallel.sh <num_envs> <cargo_type> <total_steps>
# 例如：
#   ./scripts/launch_parallel.sh 4 normal 200000

NUM_ENVS=${1:-8}
CARGO=${2:-normal}
STEPS=${3:-200000}

# 使用 CUDA 可选，默认禁用 GPU 对 Webots 渲染的影响
export LIBGL_ALWAYS_SOFTWARE=1

python3 rosbot_navigation/train_stage1.py \
  --cargo_type "${CARGO}" \
  --total_steps "${STEPS}" \
  --num_envs "${NUM_ENVS}" \
  --world warehouse/worlds/warehouse.wbt \
  --headless \
  --fast_mode \
  --control_period_ms 200 \
  --device auto


