#!/usr/bin/env python3
"""
测试ROSbot的重置位置功能
"""

import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 导入环境
    from rosbot_navigation.src.environments.navigation_env import ROSbotNavigationEnv
    print("成功导入ROSbotNavigationEnv")
    
    # 创建环境实例
    env = ROSbotNavigationEnv(cargo_type='normal')
    print("成功创建环境实例")
    
    # 测试多次重置，验证是否总是重置到固定位置
    for i in range(3):
        print(f"\n===== 测试重置 #{i+1} =====")
        
        # 重置环境
        observation, info = env.reset()
        
        # 获取并打印起始位置
        start_pos = info['start_position']
        print(f"起始位置: {start_pos}")
        
        # 验证是否为预期的固定位置(-5, 3, 0)
        expected_pos = np.array([-5.0, 3.0, 0.0], dtype=np.float32)
        is_correct = np.allclose(start_pos, expected_pos, atol=1e-4)
        print(f"位置正确: {is_correct}")
        
        if not is_correct:
            print(f"错误: 预期位置={expected_pos}, 实际位置={start_pos}")
        
        # 执行一些动作，使机器人移动
        for j in range(5):
            action = np.array([1.0, 0.0], dtype=np.float32)  # 前进
            observation, reward, terminated, truncated, info = env.step(action)
            
            # 获取当前位置
            if hasattr(env, 'robot_node') and env.robot_node:
                try:
                    current_pos = env._get_sup_position()
                    print(f"  步骤 {j+1} - 当前位置: {current_pos}")
                except Exception as e:
                    print(f"  步骤 {j+1} - 获取位置错误: {e}")
            
            time.sleep(0.1)
    
    print("\n测试完成!")
    
except ImportError as e:
    print(f"导入失败: {e}")
except Exception as e:
    print(f"测试失败: {e}")
