#!/usr/bin/env python3
"""
测试控制器 - 验证机器人保持静止
这个控制器确保机器人在仿真开始时完全静止
"""

from controller import Robot

def main():
    # 创建机器人实例
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # 获取所有电机
    motor_names = [
        'front left wheel motor',
        'rear left wheel motor', 
        'front right wheel motor',
        'rear right wheel motor'
    ]
    
    motors = []
    for name in motor_names:
        motor = robot.getDevice(name)
        if motor:
            # 关键：设置位置为无穷大以启用速度控制
            motor.setPosition(float('inf'))
            # 关键：设置速度为0
            motor.setVelocity(0.0)
            motors.append(motor)
            print(f"✅ 找到并初始化电机: {name}")
        else:
            print(f"❌ 未找到电机: {name}")
    
    print("\n机器人应该保持完全静止")
    print("如果机器人仍在移动，请检查物理设置")
    
    # 主循环 - 保持所有电机速度为0
    step_count = 0
    while robot.step(timestep) != -1:
        # 每100步再次确认速度为0
        if step_count % 100 == 0:
            for motor in motors:
                motor.setVelocity(0.0)
            
        step_count += 1
        
        # 每秒打印一次状态
        if step_count % (1000 // timestep) == 0:
            print(f"⏱️ 运行时间: {step_count * timestep / 1000:.1f}秒 - 机器人应保持静止")

if __name__ == "__main__":
    main()
