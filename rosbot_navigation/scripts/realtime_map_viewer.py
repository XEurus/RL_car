#!/usr/bin/env python3
"""
实时局部地图查看器
连接到Webots仿真环境，实时显示机器人的局部地图
每3秒更新一次地图显示
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import sys
import threading
import queue
import math

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environments.navigation_env import ROSbotNavigationEnv
from src.environments.local_map_obs import LocalMapObservation


class RealtimeMapViewer:
    """实时地图查看器"""
    
    def __init__(self, 
                 map_size=200, 
                 resolution=0.05, 
                 max_range=10.0,
                 update_interval=3.0,
                 scale=3):
        """
        初始化实时地图查看器
        
        Args:
            map_size: 地图大小 (像素)
            resolution: 分辨率 (米/像素)
            max_range: 最大探测距离 (米)
            update_interval: 更新间隔 (秒)
            scale: 显示缩放倍数
        """
        self.map_size = map_size
        self.resolution = resolution
        self.max_range = max_range
        self.update_interval = update_interval
        self.scale = scale
        
        # 创建局部地图观测对象
        self.local_map_obs = LocalMapObservation(
            map_size=map_size,
            resolution=resolution,
            max_range=max_range
        )
        
        # 数据队列
        self.data_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # 显示窗口
        self.window_name = "Real-time Local Map"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        print(f"实时地图查看器初始化完成")
        print(f"地图大小: {map_size}x{map_size}")
        print(f"分辨率: {resolution}m/pixel")
        print(f"物理尺寸: {map_size * resolution}m x {map_size * resolution}m")
        print(f"更新间隔: {update_interval}秒")
    
    def visualize_map(self, local_map, robot_pos, robot_orient, step_count):
        """
        可视化局部地图
        
        Args:
            local_map: 局部地图 (1, H, W) 或 (H, W)
            robot_pos: 机器人位置 (x, y, z)
            robot_orient: 机器人朝向 (roll, pitch, yaw)
            step_count: 步数计数
        """
        # 确保地图是2D的
        if len(local_map.shape) == 3:
            map_2d = local_map[0]  # 取第一个通道
        else:
            map_2d = local_map
        
        # 转换为0-255范围用于显示
        map_display = (map_2d * 255).astype(np.uint8)
        
        # 应用颜色映射
        colored_map = cv2.applyColorMap(map_display, cv2.COLORMAP_JET)
        
        # 特殊处理：机器人位置显示为红色
        robot_mask = (map_2d >= 0.4) & (map_2d <= 0.6)
        colored_map[robot_mask] = [0, 0, 255]  # 红色
        
        # 缩放图像
        height, width = colored_map.shape[:2]
        scaled_map = cv2.resize(colored_map, (width * self.scale, height * self.scale), 
                               interpolation=cv2.INTER_NEAREST)
        
        # 添加网格线
        grid_size = 20 * self.scale  # 每20像素一条网格线
        for i in range(0, scaled_map.shape[0], grid_size):
            cv2.line(scaled_map, (0, i), (scaled_map.shape[1], i), (128, 128, 128), 1)
        for i in range(0, scaled_map.shape[1], grid_size):
            cv2.line(scaled_map, (i, 0), (i, scaled_map.shape[0]), (128, 128, 128), 1)
        
        # 添加中心十字线
        center_x = scaled_map.shape[1] // 2
        center_y = scaled_map.shape[0] // 2
        cv2.line(scaled_map, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
        cv2.line(scaled_map, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)
        
        # 添加信息文本
        info_text = [
            f"Step: {step_count}",
            f"Robot Pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})",
            f"Robot Yaw: {math.degrees(robot_orient[2]):.1f}°",
            f"Map Size: {self.map_size}x{self.map_size}",
            f"Resolution: {self.resolution}m/pixel",
            f"Physical Size: {self.map_size * self.resolution:.1f}m x {self.map_size * self.resolution:.1f}m"
        ]
        
        # 在图像上绘制文本
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(scaled_map, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(scaled_map, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 添加图例
        legend_y = scaled_map.shape[0] - 100
        cv2.putText(scaled_map, "Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(scaled_map, "Red: Robot", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(scaled_map, "White: Obstacles", (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(scaled_map, "Dark: Free Space", (10, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return scaled_map
    
    def data_collection_thread(self, env):
        """数据收集线程"""
        step_count = 0
        last_update_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # 检查是否到了更新时间
                if current_time - last_update_time >= self.update_interval:
                    # 获取当前观测
                    obs = env._get_observation()
                    
                    # 提取LiDAR数据
                    lidar_data = obs[0:20]
                    
                    # 获取机器人位姿
                    robot_pos = env._get_sup_position()
                    robot_orient = env._get_sup_orientation()
                    robot_pose = (robot_pos[0], robot_pos[1], robot_orient[2])
                    
                    # 生成局部地图
                    local_map = self.local_map_obs.create_observation(lidar_data, robot_pose)
                    
                    # 将数据放入队列
                    data = {
                        'local_map': local_map,
                        'robot_pos': robot_pos,
                        'robot_orient': robot_orient,
                        'step_count': step_count,
                        'timestamp': current_time
                    }
                    
                    try:
                        self.data_queue.put_nowait(data)
                    except queue.Full:
                        # 队列满了，移除最旧的数据
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(data)
                        except queue.Empty:
                            pass
                    
                    last_update_time = current_time
                    step_count += 1
                    
                    print(f"地图更新 #{step_count} - 机器人位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), 朝向: {math.degrees(robot_orient[2]):.1f}°")
                
                # 短暂休眠避免过度占用CPU
                time.sleep(0.1)
                
            except Exception as e:
                print(f"数据收集线程错误: {e}")
                time.sleep(1.0)
    
    def display_thread(self):
        """显示线程"""
        while self.running:
            try:
                # 从队列获取数据
                data = self.data_queue.get(timeout=1.0)
                
                # 可视化地图
                display_map = self.visualize_map(
                    data['local_map'],
                    data['robot_pos'],
                    data['robot_orient'],
                    data['step_count']
                )
                
                # 显示图像
                cv2.imshow(self.window_name, display_map)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户按下'q'键，退出程序")
                    self.running = False
                    break
                elif key == ord('s'):
                    # 保存当前地图
                    filename = f"realtime_map_step_{data['step_count']}.png"
                    cv2.imwrite(filename, display_map)
                    print(f"地图已保存为: {filename}")
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"显示线程错误: {e}")
                time.sleep(1.0)
    
    def start_viewing(self, env):
        """开始实时查看"""
        print("开始实时地图查看...")
        print("按键说明:")
        print("  'q' - 退出程序")
        print("  's' - 保存当前地图")
        print(f"地图将每 {self.update_interval} 秒更新一次")
        print("=" * 50)
        
        self.running = True
        
        # 启动数据收集线程
        data_thread = threading.Thread(target=self.data_collection_thread, args=(env,))
        data_thread.daemon = True
        data_thread.start()
        
        # 启动显示线程
        display_thread = threading.Thread(target=self.display_thread)
        display_thread.daemon = True
        display_thread.start()
        
        try:
            # 主线程等待
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n收到中断信号，正在退出...")
            self.running = False
        
        # 等待线程结束
        data_thread.join(timeout=2.0)
        display_thread.join(timeout=2.0)
        
        cv2.destroyAllWindows()
        print("实时地图查看器已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时局部地图查看器')
    parser.add_argument('--cargo_type', type=str, default='normal',
                       choices=['normal', 'fragile', 'dangerous'],
                       help='货物类型')
    parser.add_argument('--map_size', type=int, default=200,
                       help='地图大小 (像素)')
    parser.add_argument('--resolution', type=float, default=0.05,
                       help='分辨率 (米/像素)')
    parser.add_argument('--max_range', type=float, default=10.0,
                       help='最大探测距离 (米)')
    parser.add_argument('--update_interval', type=float, default=3.0,
                       help='更新间隔 (秒)')
    parser.add_argument('--scale', type=int, default=3,
                       help='显示缩放倍数')
    parser.add_argument('--world', type=str, 
                       default='/root/workspace/RL_car2/warehouse/worlds/warehouse4.wbt',
                       help='Webots world 文件路径')
    parser.add_argument('--headless', action='store_true',
                       help='以无头模式启动Webots')
    
    args = parser.parse_args()
    
    print("实时局部地图查看器")
    print("=" * 50)
    print(f"货物类型: {args.cargo_type}")
    print(f"地图大小: {args.map_size}x{args.map_size}")
    print(f"分辨率: {args.resolution}m/pixel")
    print(f"物理尺寸: {args.map_size * args.resolution}m x {args.map_size * args.resolution}m")
    print(f"最大探测距离: {args.max_range}m")
    print(f"更新间隔: {args.update_interval}秒")
    print(f"显示缩放: {args.scale}x")
    print("=" * 50)
    
    try:
        # 创建环境
        print("正在启动Webots仿真环境...")
        env = ROSbotNavigationEnv(
            cargo_type=args.cargo_type,
            instance_id=0,
            controller_url=None,
            fast_mode=True,
            control_period_ms=200,
            debug=False
        )
        
        print("环境初始化完成，重置环境...")
        obs, info = env.reset()
        print("环境重置完成")
        
        # 创建实时地图查看器
        viewer = RealtimeMapViewer(
            map_size=args.map_size,
            resolution=args.resolution,
            max_range=args.max_range,
            update_interval=args.update_interval,
            scale=args.scale
        )
        
        # 开始查看
        viewer.start_viewing(env)
        
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass
        print("程序结束")


if __name__ == "__main__":
    main()
