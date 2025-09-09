#!/usr/bin/env python3
"""
测试ROSbot的平滑运动控制
模拟强化学习模型的输出，生成平滑的动作序列
"""

import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    # 导入环境
    from src.environments.navigation_env import ROSbotNavigationEnv
    print("成功导入ROSbotNavigationEnv")
    
    # 创建环境实例
    env = ROSbotNavigationEnv(cargo_type='normal')
    print("成功创建环境实例")
    
    # 重置环境
    observation, info = env.reset()
    print(f"环境重置成功")
    print(f"初始信息: {info}")
    
    # 等待3秒让车辆稳定
    print("等待3秒让车辆稳定...")
    for _ in range(30):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        time.sleep(0.1)
    
    # 定义测试动作序列
    test_sequence = [
        ("右转", np.array([0.5, -0.8], dtype=np.float32), 20),  # 右转2秒
        ("直行", np.array([0.8, 0.0], dtype=np.float32), 30)    # 直行3秒
    ]
    
    print("\n开始运动学测试...")
    
    for name, action, duration in test_sequence:
        print(f"\n执行动作: {name}")
        for i in range(duration):
            observation, reward, terminated, truncated, info = env.step(action)
            
            print(f"步骤 {i+1}:")
            print(f"  动作: 线速度={action[0]:.2f}, 角速度={action[1]:.2f}")
            print(f"  奖励: {reward:.4f}")
            print(f"  距离目标: {info['distance_to_target']:.4f}m")
            print(f"  最小障碍物距离: {info['min_obstacle_distance']:.4f}m")
            
            # 如果终止或截断，退出循环
            if terminated or truncated:
                print(f"\n检测到终止条件! 终止={terminated}, 截断={truncated}")
                break
            
            # 等待一小段时间
            time.sleep(0.1)
    
    print("\n运动学测试完成")
    
except ImportError as e:
    print(f"导入失败: {e}")
except Exception as e:
    print(f"测试失败: {e}")
