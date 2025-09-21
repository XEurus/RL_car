#!/usr/bin/env python3
"""
发现Webots机器人上的所有可用设备
"""

import os
import sys

try:
    from controller import Supervisor
    WEBOTS_AVAILABLE = True
except ImportError:
    print("Webots not available")
    sys.exit(1)

def discover_devices():
    """发现所有可用设备"""
    try:
        # 设置环境变量
        webots_url = "tcp://localhost:1237"
        if webots_url.startswith('tcp://'):
            import urllib.parse
            parsed = urllib.parse.urlparse(webots_url)
            host = parsed.hostname or 'localhost'
            port = parsed.port or 1236
            os.environ['WEBOTS_SERVER'] = host
            os.environ['WEBOTS_PORT'] = str(port)
            os.environ['WEBOTS_CONTROLLER_URL'] = str(webots_url)
        
        # 设置用户环境变量
        if 'USER' not in os.environ:
            os.environ['USER'] = 'webots_user'
        if 'USERNAME' not in os.environ:
            os.environ['USERNAME'] = 'webots_user'
        
        # 初始化Supervisor
        print("🤖 连接到Webots...")
        robot = Supervisor()
        print("✅ Supervisor初始化成功")
        
        # 等待Webots完全加载
        robot.step(1)
        
        print("\n🔍 发现设备:")
        print("=" * 50)
        
        # 尝试常见的设备名称
        device_names = [
            # 电机设备
            'fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint',
            # 摄像头设备
            'camera', 'Camera', 'CAMERA', 'cam', 'rgb_camera', 'color_camera',
            'astra', 'Astra', 'ASTRA', 'astra_camera', 'depth_camera',
            # 激光雷达
            'laser', 'Laser', 'LASER', 'lidar', 'Lidar', 'LIDAR', 'rplidar',
            # 传感器
            'gps', 'GPS', 'compass', 'Compass', 'imu', 'IMU',
            # 其他常见设备
            'distance_sensor', 'proximity_sensor', 'light_sensor'
        ]
        
        found_devices = []
        
        for name in device_names:
            device = robot.getDevice(name)
            if device:
                device_type = type(device).__name__
                found_devices.append((name, device_type))
                print(f"✅ 找到设备: {name} (类型: {device_type})")
        
        if not found_devices:
            print("❌ 未找到任何设备")
        else:
            print(f"\n📊 总共找到 {len(found_devices)} 个设备")
            
        # 尝试获取机器人节点信息
        print("\n🤖 机器人信息:")
        print("=" * 50)
        
        # 获取机器人节点
        robot_node = robot.getSelf()
        if robot_node:
            print(f"机器人节点ID: {robot_node.getId()}")
            print(f"机器人类型: {robot_node.getTypeName()}")
            
            # 获取所有子节点
            field_count = robot_node.getNumberOfFields()
            print(f"字段数量: {field_count}")
            
            for i in range(field_count):
                field = robot_node.getField(i)
                if field:
                    field_name = field.getFieldName()
                    field_type = field.getType()
                    print(f"  字段 {i}: {field_name} (类型: {field_type})")
        
        print("\n✅ 设备发现完成")
        
    except Exception as e:
        print(f"❌ 设备发现失败: {e}")

if __name__ == "__main__":
    discover_devices()
