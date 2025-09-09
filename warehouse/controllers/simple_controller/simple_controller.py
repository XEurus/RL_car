#!/usr/bin/env python3
"""
Rosbot控制器 - 符合Webots R2023b标准
功能：
1. 正确初始化Rosbot的电机
2. 确保机器人初始时保持静止
3. 提供键盘控制功能
"""

from controller import Robot, Motor, Camera, Lidar, GPS, Compass
import numpy as np

class RosbotController:
    def __init__(self):
        # 初始化机器人
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # 初始化电机 - 使用Webots文档中的正确名称
        self.init_motors()
        
        # 初始化传感器
        self.init_sensors()
        
        # 运动参数
        self.max_speed = 20.0  # 增加最大速度（Rosbot PROTO中定义的最大速度为26.0）
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        
        print("RosbotController initialized!")
        print("==========================================")
        print("控制说明:")
        print("W: 前进")
        print("S: 后退")
        print("A: 左转")
        print("D: 右转")
        print("空格: 停止")
        print("Q: 退出")
        print("==========================================")
        
    def init_motors(self):
        """根据Rosbot PROTO文件初始化电机"""
        try:
            # Rosbot的电机名称（根据PROTO文件中的定义）
            motor_names = {
                'front_left': 'fl_wheel_joint',
                'rear_left': 'rl_wheel_joint',
                'front_right': 'fr_wheel_joint',
                'rear_right': 'rr_wheel_joint'
            }
            
            # 获取电机设备
            self.motors = {}
            for key, name in motor_names.items():
                motor = self.robot.getDevice(name)
                if motor is None:
                    print(f"❌ 无法找到电机: {name}")
                else:
                    # 设置位置为无穷大以启用速度控制
                    motor.setPosition(float('inf'))
                    # 初始速度设为0
                    motor.setVelocity(0.0)
                    self.motors[key] = motor
                    print(f"✅ 电机初始化成功: {name}")
                    
            if len(self.motors) == 4:
                print("✅ 所有电机初始化完成")
            else:
                print(f"⚠️ 只有 {len(self.motors)}/4 个电机初始化成功")
                
        except Exception as e:
            print(f"❌ 电机初始化失败: {e}")
    
    def init_sensors(self):
        """初始化传感器"""
        try:
            # 根据PROTO文件初始化传感器
            
            # 首先列出所有设备，帮助调试
            print("📋 列出所有可用设备:")
            n = self.robot.getNumberOfDevices()
            for i in range(n):
                device = self.robot.getDeviceByIndex(i)
                print(f"   设备 {i}: {device.getName()} - 类型: {device.getNodeType()}")
            
            # 彩色相机 - 根据Astra PROTO文件，RGB相机设备名称应该是"camera color"
            self.camera_color = self.robot.getDevice('camera color')
            if not self.camera_color:
                # 尝试其他可能的名称
                possible_names = ['camera color', 'camera rgb', 'camera', 'rgb']
                for name in possible_names:
                    print(f"   尝试获取相机: {name}")
                    self.camera_color = self.robot.getDevice(name)
                    if self.camera_color:
                        print(f"✅ 找到彩色相机: {name}")
                        break
            
            if self.camera_color:
                self.camera_color.enable(self.timestep)
                print(f"✅ 彩色相机已启用")
                print(f"   分辨率: {self.camera_color.getWidth()}x{self.camera_color.getHeight()}")
                print(f"   视场角: {self.camera_color.getFov()}")
                print(f"   近平面: {self.camera_color.getNear()}")
            else:
                print("❌ 未找到彩色相机")
                
            # 深度相机 - 根据Astra PROTO文件，深度相机设备名称应该是"camera depth"
            self.camera_depth = self.robot.getDevice('camera depth')
            if not self.camera_depth:
                # 尝试其他可能的名称
                possible_names = ['camera depth', 'camera range', 'depth']
                for name in possible_names:
                    print(f"   尝试获取深度相机: {name}")
                    self.camera_depth = self.robot.getDevice(name)
                    if self.camera_depth:
                        print(f"✅ 找到深度相机: {name}")
                        break
            
            if self.camera_depth:
                self.camera_depth.enable(self.timestep)
                print(f"✅ 深度相机已启用")
                print(f"   分辨率: {self.camera_depth.getWidth()}x{self.camera_depth.getHeight()}")
                print(f"   视场角: {self.camera_depth.getFov()}")
            else:
                print("❌ 未找到深度相机")
                
            # 激光雷达 (RpLidar A2)
            self.lidar = self.robot.getDevice('lidar')
            if not self.lidar:
                # 尝试其他可能的名称
                possible_names = ['lidar', 'laser', 'rplidar', 'scanner']
                for name in possible_names:
                    print(f"   尝试获取激光雷达: {name}")
                    self.lidar = self.robot.getDevice(name)
                    if self.lidar:
                        print(f"✅ 找到激光雷达: {name}")
                        break
            
            if self.lidar:
                self.lidar.enable(self.timestep)
                self.lidar.enablePointCloud()
                print("✅ 激光雷达已启用")
                print(f"   水平分辨率: {self.lidar.getHorizontalResolution()}")
                print(f"   视场角: {self.lidar.getFov()} 弧度")
            else:
                print("❌ 未找到激光雷达")
                
            # GPS传感器
            self.gps = self.robot.getDevice('gps')
            if not self.gps:
                # 尝试其他可能的名称
                possible_names = ['gps', 'GPS']
                for name in possible_names:
                    print(f"   尝试获取GPS: {name}")
                    self.gps = self.robot.getDevice(name)
                    if self.gps:
                        print(f"✅ 找到GPS: {name}")
                        break
                        
            if self.gps:
                self.gps.enable(self.timestep)
                print("✅ GPS已启用")
            else:
                print("❌ 未找到GPS")
                
            # IMU/指南针传感器
            self.imu = self.robot.getDevice('imu')
            if not self.imu:
                # 尝试其他可能的名称
                possible_names = ['imu', 'compass', 'inertial unit']
                for name in possible_names:
                    print(f"   尝试获取IMU/指南针: {name}")
                    self.imu = self.robot.getDevice(name)
                    if self.imu:
                        print(f"✅ 找到IMU/指南针: {name}")
                        break
                        
            if self.imu:
                self.imu.enable(self.timestep)
                print("✅ IMU/指南针已启用")
            else:
                print("❌ 未找到IMU/指南针")
                
            # 触摸传感器（碰撞检测）
            self.touch_sensor = self.robot.getDevice('touch sensor')
            if not self.touch_sensor:
                # 尝试其他可能的名称
                possible_names = ['touch sensor', 'bumper', 'collision']
                for name in possible_names:
                    print(f"   尝试获取触摸传感器: {name}")
                    self.touch_sensor = self.robot.getDevice(name)
                    if self.touch_sensor:
                        print(f"✅ 找到触摸传感器: {name}")
                        break
                        
            if self.touch_sensor:
                self.touch_sensor.enable(self.timestep)
                print("✅ 触摸传感器已启用")
            else:
                print("❌ 未找到触摸传感器")
                
            # 距离传感器
            distance_sensor_names = ['fl_range', 'fr_range', 'rl_range', 'rr_range']
            self.distance_sensors = {}
            
            for name in distance_sensor_names:
                sensor = self.robot.getDevice(name)
                if sensor:
                    sensor.enable(self.timestep)
                    self.distance_sensors[name] = sensor
                    print(f"✅ 距离传感器已启用: {name}")
            
            # 位置传感器
            position_sensor_names = {
                'front_left': 'front left wheel motor sensor',
                'rear_left': 'rear left wheel motor sensor',
                'front_right': 'front right wheel motor sensor',
                'rear_right': 'rear right wheel motor sensor'
            }
            
            self.position_sensors = {}
            for key, name in position_sensor_names.items():
                sensor = self.robot.getDevice(name)
                if sensor:
                    sensor.enable(self.timestep)
                    self.position_sensors[key] = sensor
                    print(f"✅ 位置传感器已启用: {name}")
                
        except Exception as e:
            print(f"⚠️ 传感器初始化警告: {e}")
    
    def set_wheel_speeds(self, left_speed, right_speed):
        """设置轮子速度"""
        # 限制速度范围
        left_speed = max(min(left_speed, self.max_speed), -self.max_speed)
        right_speed = max(min(right_speed, self.max_speed), -self.max_speed)
        
        # 设置左侧轮子速度
        if 'front_left' in self.motors:
            self.motors['front_left'].setVelocity(left_speed)
        if 'rear_left' in self.motors:
            self.motors['rear_left'].setVelocity(left_speed)
            
        # 设置右侧轮子速度
        if 'front_right' in self.motors:
            self.motors['front_right'].setVelocity(right_speed)
        if 'rear_right' in self.motors:
            self.motors['rear_right'].setVelocity(right_speed)
            
        self.current_left_speed = left_speed
        self.current_right_speed = right_speed
    
    def stop_robot(self):
        """完全停止机器人"""
        self.set_wheel_speeds(0.0, 0.0)
        print("🛑 机器人已停止")
    
    def handle_keyboard(self):
        """处理键盘输入"""
        key = self.robot.getKeyboard().getKey()
        
        if key == ord('W') or key == ord('w'):  # 前进
            self.set_wheel_speeds(10.0, 10.0)  # 增加速度
            print("⬆️ 前进")
        elif key == ord('S') or key == ord('s'):  # 后退
            self.set_wheel_speeds(-10.0, -10.0)  # 增加速度
            print("⬇️ 后退")
        elif key == ord('A') or key == ord('a'):  # 左转
            self.set_wheel_speeds(-5.0, 5.0)  # 增加转弯速度
            print("⬅️ 左转")
        elif key == ord('D') or key == ord('d'):  # 右转
            self.set_wheel_speeds(5.0, -5.0)  # 增加转弯速度
            print("➡️ 右转")
        elif key == ord(' '):  # 停止
            self.stop_robot()
        elif key == ord('Q') or key == ord('q'):  # 退出
            return False
            
        return True
    
    def print_status(self):
        """打印状态信息（每秒一次）"""
        if not hasattr(self, 'status_counter'):
            self.status_counter = 0
            
        self.status_counter += 1
        if self.status_counter >= 1000 // self.timestep:  # 每秒
            self.status_counter = 0
            
            # 打印IMU数据
            if hasattr(self, 'imu') and self.imu:
                try:
                    values = self.imu.getRollPitchYaw()
                    if values:
                        print(f"🧭 姿态: Roll={values[0]:.2f}, Pitch={values[1]:.2f}, Yaw={values[2]:.2f}")
                except:
                    pass
                
            # 打印轮子位置传感器数据
            if hasattr(self, 'position_sensors') and len(self.position_sensors) > 0:
                try:
                    fl_pos = self.position_sensors.get('front_left', None)
                    if fl_pos:
                        print(f"🔄 前左轮位置: {fl_pos.getValue():.2f}")
                except:
                    pass
                
            # 打印速度
            print(f"🚗 速度: 左={self.current_left_speed:.2f}, 右={self.current_right_speed:.2f}")
            
            # 打印距离传感器数据
            if hasattr(self, 'distance_sensors') and len(self.distance_sensors) > 0:
                try:
                    distances = []
                    for name, sensor in self.distance_sensors.items():
                        if sensor:
                            distances.append(f"{name[-2:]}:{sensor.getValue():.2f}")
                    if distances:
                        print(f"📏 距离: {', '.join(distances)}")
                except:
                    pass
    
    def display_camera_images(self):
        """显示摄像头图像"""
        try:
            # 显示彩色相机图像
            if hasattr(self, 'camera_color') and self.camera_color:
                # 强制获取图像，即使不处理也会触发Webots显示
                image = self.camera_color.getImage()
                if image:
                    # 图像已成功获取，Webots会自动显示
                    print("📸 彩色相机图像已获取")
                    # 检查图像是否全黑
                    import numpy as np
                    width = self.camera_color.getWidth()
                    height = self.camera_color.getHeight()
                    np_image = np.frombuffer(image, np.uint8).reshape((height, width, 4))
                    if np.mean(np_image) < 5:  # 如果平均亮度很低，可能是全黑图像
                        print("⚠️ 彩色相机图像可能全黑，请检查相机位置和方向")
                else:
                    # 如果获取失败，重新启用相机
                    print("⚠️ 无法获取彩色相机图像，尝试重新启用")
                    self.camera_color.enable(self.timestep)
                    
            # 显示深度相机图像
            if hasattr(self, 'camera_depth') and self.camera_depth:
                # 强制获取深度图像，即使不处理也会触发Webots显示
                depth_image = self.camera_depth.getRangeImage()
                if depth_image:
                    # 深度图像已成功获取，Webots会自动显示
                    print("📸 深度相机图像已获取")
                else:
                    # 如果获取失败，重新启用相机
                    print("⚠️ 无法获取深度相机图像，尝试重新启用")
                    self.camera_depth.enable(self.timestep)
                    
        except Exception as e:
            print(f"⚠️ 显示相机图像时出错: {e}")
            # 尝试重新初始化相机
            try:
                # 尝试重新获取相机设备
                if not hasattr(self, 'camera_color') or not self.camera_color:
                    # 尝试不同的相机名称
                    for name in ['camera color', 'camera rgb', 'camera', 'rgb']:
                        self.camera_color = self.robot.getDevice(name)
                        if self.camera_color:
                            self.camera_color.enable(self.timestep)
                            print(f"🔄 彩色相机已重新初始化: {name}")
                            break
                
                if not hasattr(self, 'camera_depth') or not self.camera_depth:
                    # 尝试不同的深度相机名称
                    for name in ['camera depth', 'camera range', 'depth']:
                        self.camera_depth = self.robot.getDevice(name)
                        if self.camera_depth:
                            self.camera_depth.enable(self.timestep)
                            print(f"🔄 深度相机已重新初始化: {name}")
                            break
            except Exception as e2:
                print(f"⚠️ 重新初始化相机时出错: {e2}")
    
    def run(self):
        """主循环"""
        # 启用键盘
        keyboard = self.robot.getKeyboard()
        keyboard.enable(self.timestep)
        
        # 确保开始时机器人停止
        self.stop_robot()
        
        print("\n🚀 控制器已启动！")
        print("机器人处于静止状态，请使用键盘控制移动")
        print("摄像头图像应该在Webots窗口中显示")
        
        # 主控制循环
        while self.robot.step(self.timestep) != -1:
            # 处理键盘输入
            if not self.handle_keyboard():
                break
                
            # 显示摄像头图像
            self.display_camera_images()
                
            # 打印状态
            self.print_status()
        
        # 退出前停止机器人
        self.stop_robot()
        print("\n👋 控制器已退出")

# 主程序
if __name__ == "__main__":
    controller = RosbotController()
    controller.run()