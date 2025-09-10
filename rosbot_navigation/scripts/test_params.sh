#!/bin/bash
# 测试命令行参数是否生效的脚本

# 创建实验目录
mkdir -p param_test_logs

# 设置基本参数
TOTAL_STEPS=1000
WORLD_PATH="/root/workspace/RL_car2/warehouse/worlds/warehouse4.wbt"

echo -e "\n=== 测试命令行参数生效情况 ==="
echo "本测试将运行3个简短的测试来验证参数是否生效"
echo "结果将输出到 param_test_logs/ 目录"
echo -e "===========================\n"

# 测试1：完全有界面模式（最小步数）
echo -e "\n\n测试1：有界面模式 - 1个环境"
echo "参数：--fast_mode"
python rosbot_navigation/train_stage1.py \
  --cargo_type normal \
  --total_steps $TOTAL_STEPS \
  --learning_rate 3e-4 \
  --buffer_size 10000 \
  --batch_size 256 \
  --learning_starts 500 \
  --num_envs 1 \
  --world $WORLD_PATH \
  --fast_mode \
  --control_period_ms 200 \
  > param_test_logs/test1_gui.log 2>&1 &
  
echo "已在后台启动测试1，日志: param_test_logs/test1_gui.log"
echo "你应该可以看到一个Webots窗口启动"

# 等待3秒
sleep 3

# 测试2：最小化模式
echo -e "\n\n测试2：最小化模式 - 1个环境"
echo "参数：--fast_mode --minimize"
python rosbot_navigation/train_stage1.py \
  --cargo_type normal \
  --total_steps $TOTAL_STEPS \
  --learning_rate 3e-4 \
  --buffer_size 10000 \
  --batch_size 256 \
  --learning_starts 500 \
  --num_envs 1 \
  --world $WORLD_PATH \
  --fast_mode \
  --minimize \
  --control_period_ms 200 \
  > param_test_logs/test2_minimize.log 2>&1 &
  
echo "已在后台启动测试2，日志: param_test_logs/test2_minimize.log"
echo "你应该可以看到一个最小化的Webots窗口启动"

# 等待3秒
sleep 3

# 测试3：完全无头模式
echo -e "\n\n测试3：无头模式 - 2个环境"
echo "参数：--headless --fast_mode --batch --no-rendering"
export LIBGL_ALWAYS_SOFTWARE=1
export WEBOTS_DISABLE_GPU=1

python rosbot_navigation/train_stage1.py \
  --cargo_type normal \
  --total_steps $TOTAL_STEPS \
  --learning_rate 3e-4 \
  --buffer_size 10000 \
  --batch_size 256 \
  --learning_starts 500 \
  --num_envs 2 \
  --world $WORLD_PATH \
  --headless \
  --fast_mode \
  --batch \
  --no-rendering \
  --control_period_ms 200 \
  > param_test_logs/test3_headless.log 2>&1 &
  
echo "已在后台启动测试3，日志: param_test_logs/test3_headless.log"
echo "这个测试应该是真正的无界面模式，没有窗口显示"
echo -e "\n你可以用以下命令查看运行日志："
echo "tail -f param_test_logs/test3_headless.log"

echo -e "\n===========测试结束============"
echo "请检查各个运行日志，验证参数是否生效"
