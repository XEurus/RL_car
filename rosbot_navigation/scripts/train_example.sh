#!/bin/bash
# 训练参数示例脚本 - 已修复参数传递问题

# 确保脚本执行时显示执行的命令（便于调试）
set -x

# 设置环境变量以支持无头模式
export LIBGL_ALWAYS_SOFTWARE=1
export WEBOTS_DISABLE_GPU=1

# 选择示例（取消注释一个）

# 1. 无头快速模式（无GUI/服务器训练）- 使用4个并行环境
python rosbot_navigation/train_stage1.py \
  --cargo_type normal \
  --total_steps 20000 \
  --learning_rate 1e-4 \
  --learning_starts 5000 \
  --buffer_size 250000 \
  --batch_size 1024 \
  --gamma 0.995 \
  --policy_delay 3 \
  --target_noise 0.15 \
  --num_envs 4 \
  --headless \
  --fast_mode \
  --batch \
  --no-rendering \
  --minimize \
  --stdout \
  --stderr \
  --control_period_ms 200

# 2. 有界面模式（可视化调试）- 单环境
# python rosbot_navigation/train_stage1.py \
#   --cargo_type normal \
#   --total_steps 10000 \
#   --learning_rate 3e-4 \
#   --learning_starts 1000 \
#   --buffer_size 50000 \
#   --batch_size 256 \
#   --num_envs 1 \
#   --fast_mode \
#   --control_period_ms 200

# 高学习率/快速探索训练示例 - 快速原型开发
# python rosbot_navigation/train_stage1.py \
#   --cargo_type normal \
#   --total_steps 20000 \
#   --learning_rate 5e-4 \
#   --learning_starts 2000 \
#   --buffer_size 100000 \
#   --batch_size 512 \
#   --gamma 0.98 \
#   --num_envs 4 \
#   --headless \
#   --control_period_ms 200

# 危险货物训练示例（更稳健的参数） - 稳定性优先场景
# python rosbot_navigation/train_stage1.py \
#   --cargo_type dangerous \
#   --total_steps 50000 \
#   --learning_rate 8e-5 \
#   --learning_starts 8000 \
#   --buffer_size 500000 \
#   --batch_size 2048 \
#   --gamma 0.998 \
#   --policy_delay 4 \
#   --target_noise 0.15 \
#   --noise_clip 0.4 \
#   --num_envs 8 \
#   --headless \
#   --control_period_ms 200

# 超快速测试（小步数，快速验证）
# python rosbot_navigation/train_stage1.py \
#   --cargo_type normal \
#   --total_steps 1000 \
#   --learning_rate 3e-4 \
#   --learning_starts 500 \
#   --buffer_size 10000 \
#   --batch_size 128 \
#   --num_envs 2 \
#   --control_period_ms 200
