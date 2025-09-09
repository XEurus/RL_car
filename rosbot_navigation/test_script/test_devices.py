#!/usr/bin/env python3
"""
测试ROSbot设备是否可以正确获取
"""

from controller import Supervisor

def main():
    """主函数"""
    # 初始化机器人
    robot = Supervisor()
    print("Robot initialized successfully")
    
    # 测试激光雷达
    try:
        lidar = robot.getDevice('laser')
        print(f"Lidar found: {lidar is not None}")
    except Exception as e:
        print(f"Lidar error: {e}")
    
    # 测试IMU
    try:
        imu = robot.getDevice('imu')
        print(f"IMU found: {imu is not None}")
    except Exception as e:
        print(f"IMU error: {e}")
    
    # 测试电机
    motors = ['fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint']
    for motor_name in motors:
        try:
            motor = robot.getDevice(motor_name)
            print(f"Motor {motor_name} found: {motor is not None}")
        except Exception as e:
            print(f"Motor {motor_name} error: {e}")
    
    # 测试距离传感器
    sensors = ['fl_range', 'fr_range', 'rl_range', 'rr_range']
    for sensor_name in sensors:
        try:
            sensor = robot.getDevice(sensor_name)
            print(f"Sensor {sensor_name} found: {sensor is not None}")
        except Exception as e:
            print(f"Sensor {sensor_name} error: {e}")

if __name__ == "__main__":
    main()
