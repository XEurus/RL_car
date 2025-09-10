# Webots并行多实例训练指南

本指南介绍如何使用并行多实例训练来提升强化学习训练效率。通过并行运行多个Webots实例，每个实例在单独的进程中运行，可以更充分地利用多核CPU资源，提高采样效率。

## 功能概述

- 支持在单机多核CPU上并行运行多个Webots实例
- 自动处理端口分配、进程管理和资源清理
- 适配OpenAI Gym/Gymnasium接口，可与常见RL框架无缝集成
- 支持无头模式(headless)运行，适合服务器环境

## 快速开始

使用脚本启动2个并行实例进行训练：

```bash
# 运行2个环境、normal货物、20000步训练
./rosbot_navigation/scripts/launch_parallel.sh 2 normal 20000
```

或者直接使用Python API：

```bash
python rosbot_navigation/train_stage1.py \
  --cargo_type normal \
  --total_steps 20000 \
  --num_envs 2 \
  --headless \
  --fast_mode
```

## 参数说明

并行训练主要参数：

- `--num_envs`: 并行环境数量，建议设置为CPU逻辑核心数的50-80%
- `--headless`: 无界面模式，服务器训练必选
- `--fast_mode`: 使用Webots的FAST模式（无图形加速）
- `--control_period_ms`: 控制周期，越大则物理模拟更快但精度降低

## 性能调优

1. **调整环境数量**:
   - 在本地开发时用2-4个环境进行测试
   - 在服务器上可以根据CPU核心数增加到8-16个环境
   - 注意监控CPU利用率，避免过度订阅导致性能下降

2. **降低渲染与物理负担**:
   - 使用`--headless`和`--fast_mode`选项
   - 增大控制周期`--control_period_ms`(默认200ms)
   - 确保设置了`LIBGL_ALWAYS_SOFTWARE=1`环境变量

3. **避免内存不足**:
   - 每个Webots实例约需要200-500MB内存
   - 监控总内存使用，避免交换分区使用

## 常见问题

1. **端口冲突**
   - 每个实例使用不同的端口范围(10000 + instance_id * 100)
   - 如果发生端口冲突，尝试修改基础端口

2. **OpenGL/图形问题**
   - 使用软件渲染模式(`LIBGL_ALWAYS_SOFTWARE=1`)
   - 在服务器上确保安装了mesa软件渲染库

3. **进程管理**
   - 训练脚本会自动清理所有Webots进程
   - 如果异常退出，可使用`pkill webots`手动清理

## 调试技巧

若并行训练出现问题：

1. 先测试单实例训练
2. 使用小规模测试脚本验证：
```bash
./rosbot_navigation/scripts/launch_parallel_test.sh 2 normal 1000
```
3. 检查实例启动日志中的错误信息
4. 监视CPU和内存使用情况

## 实现细节

并行实现基于以下组件：
- `webots_launcher.py`: 管理Webots实例启动与URL连接
- `SubprocVecEnv`: Stable-Baselines3提供的并行环境包装器
- 自定义外部控制器连接机制，支持TCP和IPC通信
