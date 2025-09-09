#!/usr/bin/env python3
"""
测试ROSbotNavigationEnv环境
"""

import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 尝试导入navigation_env
    from rosbot_navigation.src.environments.navigation_env import ROSbotNavigationEnv
    print("成功导入ROSbotNavigationEnv")
    
    # 创建环境实例
    env = ROSbotNavigationEnv(cargo_type='normal')
    print("成功创建环境实例")
    
    # 测试重置环境
    try:
        observation, info = env.reset()
        print(f"环境重置成功，观察空间维度: {observation.shape}")
        print(f"初始信息: {info}")
    except Exception as e:
        print(f"环境重置失败: {e}")
    
    # 测试执行动作
    try:
        action = np.array([0.5, 0.0], dtype=np.float32)  # 前进，不转向
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"执行动作成功，奖励: {reward}")
        print(f"观察空间: {observation.shape}")
        print(f"终止状态: {terminated}, 截断状态: {truncated}")
        print(f"信息: {info}")
    except Exception as e:
        print(f"执行动作失败: {e}")
        
except ImportError as e:
    print(f"导入失败: {e}")
except Exception as e:
    print(f"测试失败: {e}")
