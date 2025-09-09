#!/usr/bin/env python3
"""
测试ROSbot的碰撞检测功能
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
    
    # 测试碰撞检测
    print("\n开始碰撞检测测试...")
    
    # 执行一系列动作，尝试触发碰撞
    for i in range(50):  # 增加步数
        # 执行前进动作 - 使用较小的速度
        if i < 10:  # 前10步慢慢加速
            speed = (i + 1) / 10.0  # 归一化到[0-1]范围
            action = np.array([speed, 0.0], dtype=np.float32)  # 逐渐加速
        elif i >= 40:  # 最后10步慢慢减速
            speed = (50 - i) / 10.0  # 归一化到[0-1]范围
            action = np.array([speed, 0.0], dtype=np.float32)  # 逐渐减速
        else:
            action = np.array([1.0, 0.0], dtype=np.float32)  # 匀速前进
        
        # 每10步尝试一次转向，测试转向功能
        if i % 10 == 0 and i > 0:
            action = np.array([0.8, 0.8], dtype=np.float32)  # 右转
        elif i % 10 == 5 and i > 0:
            action = np.array([0.8, -0.8], dtype=np.float32)  # 左转
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"步骤 {i+1}:")
        print(f"  动作: 线速度={action[0]:.2f}, 角速度={action[1]:.2f}")
        print(f"  奖励: {reward:.4f}")
        print(f"  终止: {terminated}")
        print(f"  截断: {truncated}")
        print(f"  距离目标: {info['distance_to_target']:.4f}m")
        print(f"  最小障碍物距离: {info['min_obstacle_distance']:.4f}m")
        
        # 如果终止或截断，退出循环
        if terminated or truncated:
            print(f"\n检测到终止条件! 终止={terminated}, 截断={truncated}")
            break
        
        # 等待一小段时间
        time.sleep(0.2)  # 增加等待时间，让物理引擎有更多时间模拟
    
    print("\n碰撞检测测试完成")
    
except ImportError as e:
    print(f"导入失败: {e}")
except Exception as e:
    print(f"测试失败: {e}")
