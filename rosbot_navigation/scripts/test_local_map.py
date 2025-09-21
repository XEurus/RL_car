#!/usr/bin/env python3
"""
测试局部地图功能
验证激光雷达数据转换为局部地图的正确性
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.navigation_env import ROSbotNavigationEnv
from src.environments.local_map_obs import LocalMapNavigationEnv


def test_local_map_with_real_env():
    """使用真实环境测试局部地图"""
    print("测试局部地图功能...")
    
    try:
        # 创建原始环境
        print("创建原始导航环境...")
        base_env = ROSbotNavigationEnv(
            cargo_type='normal',
            instance_id=0,
            controller_url=None,
            fast_mode=True,
            control_period_ms=200,
            debug=False
        )
        
        # 创建局部地图环境包装器
        print("创建局部地图环境包装器...")
        local_map_env = LocalMapNavigationEnv(
            base_env=base_env,
            map_size=200,
            resolution=0.05,  # 5cm分辨率
            max_range=10.0
        )
        
        print(f"观测空间形状: {local_map_env.observation_space.shape}")
        print(f"动作空间形状: {local_map_env.action_space.shape}")
        
        # 重置环境
        print("重置环境...")
        obs, info = local_map_env.reset()
        print(f"观测形状: {obs.shape}")
        
        # 测试渲染功能
        print("测试地图渲染...")
        cv_image = local_map_env.render_map()
        print(f"渲染图像形状: {cv_image.shape}")
        
        # 显示地图
        cv2.imshow("Local Map Test", cv_image)
        print("按任意键继续...")
        cv2.waitKey(0)
        
        # 测试几步动作
        print("测试几步动作...")
        for i in range(5):
            # 随机动作
            action = local_map_env.action_space.sample()
            obs, reward, terminated, truncated, info = local_map_env.step(action)
            
            print(f"步骤 {i+1}: 奖励={reward:.4f}, 终止={terminated}, 截断={truncated}")
            
            # 渲染并显示地图
            cv_image = local_map_env.render_map()
            cv2.imshow("Local Map Test", cv_image)
            
            # 等待按键
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        local_map_env.close()
        
        print("测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def test_local_map_standalone():
    """独立测试局部地图生成"""
    print("独立测试局部地图生成...")
    
    from src.environments.local_map_obs import LocalMapObservation
    
    # 创建局部地图观测对象
    local_map_obs = LocalMapObservation(
        map_size=200,
        resolution=0.05,
        max_range=10.0
    )
    
    # 创建测试激光雷达数据 (模拟8-399范围的数据)
    # 假设激光雷达有392个有效数据点 (8-399)
    num_lidar_points = 392
    test_lidar_data = np.random.uniform(0.1, 8.0, num_lidar_points)  # 米为单位
    
    # 设置一些特定的障碍物
    test_lidar_data[100:110] = 1.0  # 前方近距离障碍物
    test_lidar_data[200:210] = 2.0  # 右侧障碍物
    test_lidar_data[300:310] = 0.5  # 左侧近距离障碍物
    
    # 机器人位姿
    robot_pose = (0.0, 0.0, 0.0)  # 在地图中心，朝向0度
    
    # 生成局部地图
    local_map = local_map_obs.create_observation(test_lidar_data, robot_pose)
    print(f"生成的地图形状: {local_map.shape}")
    print(f"地图值范围: {local_map.min():.3f} - {local_map.max():.3f}")
    
    # 渲染为OpenCV图像
    cv_image = local_map_obs.render_map_to_cv_image(local_map)
    print(f"渲染图像形状: {cv_image.shape}")
    
    # 显示图像
    cv2.imshow("Standalone Local Map Test", cv_image)
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("独立测试完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试局部地图功能')
    parser.add_argument('--mode', type=str, default='real', 
                       choices=['real', 'standalone'],
                       help='测试模式: real=真实环境, standalone=独立测试')
    
    args = parser.parse_args()
    
    print("局部地图功能测试")
    print("=" * 50)
    
    if args.mode == 'real':
        test_local_map_with_real_env()
    elif args.mode == 'standalone':
        test_local_map_standalone()
    
    print("所有测试完成!")


if __name__ == "__main__":
    main()
