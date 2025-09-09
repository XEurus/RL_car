#!/usr/bin/env python3
"""
辅助脚本：自动打开摄像头窗口
此脚本用于在Webots GUI中自动打开摄像头窗口
"""

from controller import Robot, Display

def main():
    # 创建机器人实例
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # 获取摄像头
    camera_color = robot.getDevice('camera color')
    camera_depth = robot.getDevice('camera depth')
    
    # 启用摄像头
    if camera_color:
        camera_color.enable(timestep)
        print(f"彩色相机已启用，分辨率: {camera_color.getWidth()}x{camera_color.getHeight()}")
    else:
        print("未找到彩色相机")
    
    if camera_depth:
        camera_depth.enable(timestep)
        print(f"深度相机已启用，分辨率: {camera_depth.getWidth()}x{camera_depth.getHeight()}")
    else:
        print("未找到深度相机")
    
    # 创建显示窗口
    display_color = Display("Color Camera")
    display_depth = Display("Depth Camera")
    
    # 主循环
    while robot.step(timestep) != -1:
        if camera_color:
            # 获取彩色图像
            image = camera_color.getImage()
            if image:
                # 图像已成功获取，Webots会自动显示
                pass
        
        if camera_depth:
            # 获取深度图像
            depth_image = camera_depth.getRangeImage()
            if depth_image:
                # 深度图像已成功获取，Webots会自动显示
                pass

if __name__ == "__main__":
    main()

