#!/usr/bin/env python3
"""
摄像头查看器 - 专门用于显示Astra摄像头图像
此脚本仅用于显示摄像头图像，不控制机器人
"""

from controller import Robot
import numpy as np
import time

def main():
    # 创建机器人实例
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # 列出所有设备
    print("📋 列出所有设备:")
    n = robot.getNumberOfDevices()
    for i in range(n):
        device = robot.getDeviceByIndex(i)
        print(f"   设备 {i}: {device.getName()} - 类型: {device.getNodeType()}")
    
    # 尝试不同的相机名称
    color_camera_names = ['camera color', 'camera rgb', 'camera', 'rgb']
    depth_camera_names = ['camera depth', 'camera range', 'depth']
    
    # 彩色相机
    camera_color = None
    for name in color_camera_names:
        print(f"尝试获取彩色相机: {name}")
        camera_color = robot.getDevice(name)
        if camera_color:
            print(f"✅ 找到彩色相机: {name}")
            break
    
    # 深度相机
    camera_depth = None
    for name in depth_camera_names:
        print(f"尝试获取深度相机: {name}")
        camera_depth = robot.getDevice(name)
        if camera_depth:
            print(f"✅ 找到深度相机: {name}")
            break
    
    # 启用相机
    if camera_color:
        camera_color.enable(timestep)
        print(f"彩色相机已启用，分辨率: {camera_color.getWidth()}x{camera_color.getHeight()}")
    else:
        print("❌ 未找到彩色相机")
    
    if camera_depth:
        camera_depth.enable(timestep)
        print(f"深度相机已启用，分辨率: {camera_depth.getWidth()}x{camera_depth.getHeight()}")
    else:
        print("❌ 未找到深度相机")
    
    print("\n🔍 正在尝试获取相机图像...")
    print("如果看到图像，请在Webots菜单中选择:")
    print("Tools > Camera Devices > [相机名称]")
    
    # 主循环
    counter = 0
    while robot.step(timestep) != -1:
        counter += 1
        
        if counter % 100 == 0:  # 每100步打印一次
            print(f"\n⏱️ 运行时间: {counter * timestep / 1000:.1f}秒")
            
            # 获取彩色图像
            if camera_color:
                image = camera_color.getImage()
                if image:
                    width = camera_color.getWidth()
                    height = camera_color.getHeight()
                    try:
                        np_image = np.frombuffer(image, np.uint8).reshape((height, width, 4))
                        mean_brightness = np.mean(np_image)
                        print(f"📸 彩色图像已获取，平均亮度: {mean_brightness:.2f}")
                        if mean_brightness < 5:
                            print("⚠️ 图像可能全黑，请检查相机位置和方向")
                    except Exception as e:
                        print(f"⚠️ 处理彩色图像时出错: {e}")
                else:
                    print("❌ 无法获取彩色图像")
                    camera_color.enable(timestep)  # 尝试重新启用
            
            # 获取深度图像
            if camera_depth:
                depth_image = camera_depth.getRangeImage()
                if depth_image:
                    print("📸 深度图像已获取")
                else:
                    print("❌ 无法获取深度图像")
                    camera_depth.enable(timestep)  # 尝试重新启用

if __name__ == "__main__":
    main()

