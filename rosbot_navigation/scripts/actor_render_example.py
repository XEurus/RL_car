#!/usr/bin/env python3
"""
Actor渲染示例
展示如何在actor中调用局部地图渲染函数
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.navigation_env import ROSbotNavigationEnv
from src.environments.local_map_obs import LocalMapNavigationEnv


def actor_with_map_rendering():
    """Actor中使用地图渲染的示例"""
    print("Actor地图渲染示例")
    print("=" * 50)
    
    try:
        # 创建局部地图环境
        base_env = ROSbotNavigationEnv(
            cargo_type='normal',
            instance_id=0,
            controller_url=None,
            fast_mode=True,
            control_period_ms=200,
            debug=False
        )
        
        local_map_env = LocalMapNavigationEnv(
            base_env=base_env,
            map_size=200,
            resolution=0.05,
            max_range=10.0
        )
        
        # 重置环境
        obs, info = local_map_env.reset()
        print(f"初始观测形状: {obs.shape}")
        
        # 模拟actor循环
        step_count = 0
        max_steps = 20
        
        print("开始actor循环...")
        print("按键说明:")
        print("  'q' - 退出")
        print("  's' - 保存当前地图")
        print("  其他键 - 继续下一步")
        
        while step_count < max_steps:
            # 1. 获取当前观测 (局部地图)
            current_obs = obs
            print(f"步骤 {step_count}: 观测形状 {current_obs.shape}")
            
            # 2. 渲染地图为OpenCV图像
            map_image = local_map_env.render_map()
            print(f"渲染地图形状: {map_image.shape}")
            
            # 3. 显示地图
            cv2.imshow("Actor Local Map", map_image)
            
            # 4. 等待用户输入
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('s'):
                # 保存地图
                filename = f"actor_map_step_{step_count}.png"
                cv2.imwrite(filename, map_image)
                print(f"地图已保存为: {filename}")
            
            # 5. 执行动作 (这里使用随机动作作为示例)
            action = local_map_env.action_space.sample()
            obs, reward, terminated, truncated, info = local_map_env.step(action)
            
            print(f"执行动作: {action[:4]}... (显示前4维)")
            print(f"奖励: {reward:.4f}, 终止: {terminated}, 截断: {truncated}")
            
            # 6. 检查是否结束
            if terminated or truncated:
                print("Episode结束，重置环境")
                obs, info = local_map_env.reset()
            
            step_count += 1
            print("-" * 30)
        
        cv2.destroyAllWindows()
        local_map_env.close()
        
        print("Actor循环完成!")
        
    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def simple_render_function_example():
    """简单的渲染函数使用示例"""
    print("简单渲染函数示例")
    print("=" * 50)
    
    try:
        # 创建环境
        base_env = ROSbotNavigationEnv(
            cargo_type='normal',
            instance_id=0,
            controller_url=None,
            fast_mode=True,
            control_period_ms=200,
            debug=False
        )
        
        local_map_env = LocalMapNavigationEnv(
            base_env=base_env,
            map_size=200,
            resolution=0.05,
            max_range=10.0
        )
        
        # 重置环境
        obs, info = local_map_env.reset()
        
        # 简单的渲染函数
        def render_current_map():
            """渲染当前地图的简单函数"""
            return local_map_env.render_map()
        
        # 使用渲染函数
        print("使用简单渲染函数...")
        for i in range(5):
            # 渲染地图
            map_image = render_current_map()
            
            # 显示地图
            cv2.imshow("Simple Render Example", map_image)
            
            # 等待按键
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                break
            
            # 执行一步动作
            action = local_map_env.action_space.sample()
            obs, reward, terminated, truncated, info = local_map_env.step(action)
            
            print(f"步骤 {i+1}: 奖励={reward:.4f}")
        
        cv2.destroyAllWindows()
        local_map_env.close()
        
        print("简单渲染函数示例完成!")
        
    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("Actor渲染示例程序")
    print("=" * 50)
    
    print("选择运行模式:")
    print("1. Actor地图渲染示例")
    print("2. 简单渲染函数示例")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        actor_with_map_rendering()
    elif choice == '2':
        simple_render_function_example()
    else:
        print("无效选择，运行默认示例...")
        simple_render_function_example()
    
    print("程序结束")


if __name__ == "__main__":
    main()
