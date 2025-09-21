#!/usr/bin/env python3
"""
机器人TCP服务器 - 在Docker容器中运行，接收键盘控制命令
支持ROS和直接控制两种模式
"""

import sys
import os
import json
import time
import socket
import threading
import signal
from typing import Optional, Dict, Any
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import rospy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Float64MultiArray
    ROS_AVAILABLE = True
except ImportError:
    print("ROS not available, using direct control mode")
    ROS_AVAILABLE = False

# 尝试导入Webots控制器
try:
    from controller import Robot, Motor
    WEBOTS_AVAILABLE = True
except ImportError:
    print("Webots not available, using simulation mode")
    WEBOTS_AVAILABLE = False

class RobotTCPServer:
    """机器人TCP服务器类"""
    
    def __init__(self, 
                 port: int = 8888,
                 use_ros: bool = True,
                 use_webots: bool = True,
                 robot_name: str = "rosbot"):
        """
        初始化TCP服务器
        
        Args:
            port: 监听端口
            use_ros: 是否使用ROS发布控制命令
            use_webots: 是否使用Webots直接控制
            robot_name: 机器人名称
        """
        self.port = port
        self.use_ros = use_ros and ROS_AVAILABLE
        self.use_webots = use_webots and WEBOTS_AVAILABLE
        self.robot_name = robot_name
        
        # 服务器状态
        self.running = True
        self.clients = []
        self.server_socket = None
        
        # 控制参数
        self.max_wheel_speed = 26.0  # rad/s
        self.wheel_radius = 0.043    # m
        self.wheel_base = 0.2        # m
        
        # 当前状态（ROSbot轮速百分比）
        self.current_left_percent = 0.0
        self.current_right_percent = 0.0
        self.last_command_time = time.time()
        self.command_timeout = 1.0  # 命令超时时间
        
        # 初始化控制接口
        self._init_control_interface()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"机器人TCP服务器初始化完成")
        print(f"端口: {self.port}")
        print(f"ROS模式: {self.use_ros}")
        print(f"Webots模式: {self.use_webots}")
    
    def _init_control_interface(self):
        """初始化控制接口"""
        if self.use_ros:
            self._init_ros_interface()
        
        if self.use_webots:
            self._init_webots_interface()
    
    def _init_ros_interface(self):
        """初始化ROS接口"""
        try:
            rospy.init_node(f'{self.robot_name}_tcp_server', anonymous=True)
            
            # 创建发布者
            self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
            self.wheel_cmd_pub = rospy.Publisher('/wheel_commands', Float64MultiArray, queue_size=10)
            
            print("ROS接口初始化成功")
            
        except Exception as e:
            print(f"ROS接口初始化失败: {e}")
            self.use_ros = False
    
    def _init_webots_interface(self):
        """初始化Webots接口"""
        try:
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())
            
            # 获取电机设备
            self.left_motor = self.robot.getDevice('left wheel motor')
            self.right_motor = self.robot.getDevice('right wheel motor')
            
            # 设置电机模式
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            
            print("Webots接口初始化成功")
            
        except Exception as e:
            print(f"Webots接口初始化失败: {e}")
            self.use_webots = False
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在关闭服务器...")
        self.running = False
    
    def start_server(self):
        """启动TCP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)
            
            print(f"TCP服务器启动成功，监听端口: {self.port}")
            print("等待客户端连接...")
            
            # 启动命令超时检查线程
            timeout_thread = threading.Thread(target=self._check_command_timeout)
            timeout_thread.daemon = True
            timeout_thread.start()
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"客户端连接: {client_address}")
                    
                    # 为每个客户端创建处理线程
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                    self.clients.append(client_socket)
                    
                except socket.error as e:
                    if self.running:
                        print(f"接受连接时出错: {e}")
                        
        except Exception as e:
            print(f"启动服务器失败: {e}")
        
        finally:
            self.cleanup()
    
    def _handle_client(self, client_socket: socket.socket, client_address):
        """处理客户端连接"""
        try:
            buffer = ""
            
            while self.running:
                try:
                    data = client_socket.recv(1024).decode('utf-8')
                    if not data:
                        break
                    
                    buffer += data
                    
                    # 处理完整的JSON消息（以换行符分隔）
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            self._process_command(line.strip())
                            
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"处理客户端数据时出错: {e}")
                    break
                    
        except Exception as e:
            print(f"客户端处理错误: {e}")
        
        finally:
            print(f"客户端断开连接: {client_address}")
            try:
                client_socket.close()
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
            except:
                pass
    
    def _process_command(self, command_str: str):
        """处理控制命令（ROSbot轮速协议）"""
        try:
            command = json.loads(command_str)
            
            # 检查命令类型
            if command.get('type') == 'wheel_control':
                # ROSbot轮速控制命令
                action = command.get('action', [0.0, 0.0])
                left_percent = float(action[0]) if len(action) > 0 else 0.0
                right_percent = float(action[1]) if len(action) > 1 else 0.0
                
                # 限制范围到 0.0-1.0
                left_percent = max(0.0, min(1.0, left_percent))
                right_percent = max(0.0, min(1.0, right_percent))
                
                # 更新当前状态
                self.current_left_percent = left_percent
                self.current_right_percent = right_percent
                self.last_command_time = time.time()
                
                # 执行轮速控制命令
                self._execute_wheel_command(left_percent, right_percent)
                
                print(f"执行轮速命令 - 左轮: {left_percent:.2f}, 右轮: {right_percent:.2f}")
                
            else:
                # 兼容旧的线速度/角速度命令
                linear = float(command.get('linear', 0.0))
                angular = float(command.get('angular', 0.0))
                
                # 转换为轮速百分比
                left_percent, right_percent = self._convert_to_wheel_percent(linear, angular)
                
                # 更新当前状态
                self.current_left_percent = left_percent
                self.current_right_percent = right_percent
                self.last_command_time = time.time()
                
                # 执行轮速控制命令
                self._execute_wheel_command(left_percent, right_percent)
                
                print(f"执行转换命令 - 线速度: {linear:.2f} m/s, 角速度: {angular:.2f} rad/s -> 左轮: {left_percent:.2f}, 右轮: {right_percent:.2f}")
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        except Exception as e:
            print(f"命令处理错误: {e}")
    
    def _execute_wheel_command(self, left_percent: float, right_percent: float):
        """执行轮速控制命令（ROSbot协议）"""
        # 计算实际轮速（rad/s）
        left_speed = left_percent * self.max_wheel_speed
        right_speed = right_percent * self.max_wheel_speed
        
        # 计算等效线速度和角速度（用于ROS发布）
        linear_vel = self.wheel_radius * (left_speed + right_speed) / 2.0
        angular_vel = self.wheel_radius * (right_speed - left_speed) / self.wheel_base
        
        # ROS控制
        if self.use_ros:
            self._send_ros_command(linear_vel, angular_vel, left_speed, right_speed)
        
        # Webots直接控制
        if self.use_webots:
            self._send_webots_command(left_speed, right_speed)
    
    def _convert_to_wheel_percent(self, linear: float, angular: float) -> tuple:
        """
        将线速度和角速度转换为轮速百分比
        
        Args:
            linear: 线速度 (m/s)
            angular: 角速度 (rad/s)
            
        Returns:
            (left_percent, right_percent) 轮速百分比 (0.0-1.0)
        """
        # 差分驱动运动学
        left_wheel_speed = (linear - angular * self.wheel_base / 2.0) / self.wheel_radius
        right_wheel_speed = (linear + angular * self.wheel_base / 2.0) / self.wheel_base
        
        # 转换为百分比（只支持正向速度）
        left_percent = max(0.0, min(1.0, left_wheel_speed / self.max_wheel_speed))
        right_percent = max(0.0, min(1.0, right_wheel_speed / self.max_wheel_speed))
        
        return left_percent, right_percent
    
    def _calculate_wheel_speeds(self, linear: float, angular: float) -> tuple:
        """
        计算左右轮速度
        
        Args:
            linear: 线速度 (m/s)
            angular: 角速度 (rad/s)
            
        Returns:
            (left_speed, right_speed) in rad/s
        """
        # 差分驱动运动学
        left_wheel_speed = (linear - angular * self.wheel_base / 2.0) / self.wheel_radius
        right_wheel_speed = (linear + angular * self.wheel_base / 2.0) / self.wheel_radius
        
        # 限制轮速
        left_wheel_speed = np.clip(left_wheel_speed, -self.max_wheel_speed, self.max_wheel_speed)
        right_wheel_speed = np.clip(right_wheel_speed, -self.max_wheel_speed, self.max_wheel_speed)
        
        return left_wheel_speed, right_wheel_speed
    
    def _send_ros_command(self, linear: float, angular: float, 
                         left_speed: float, right_speed: float):
        """发送ROS命令"""
        try:
            # 发布Twist消息
            twist = Twist()
            twist.linear.x = linear
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = angular
            
            self.cmd_vel_pub.publish(twist)
            
            # 发布轮速命令
            wheel_cmd = Float64MultiArray()
            wheel_cmd.data = [left_speed, right_speed]
            self.wheel_cmd_pub.publish(wheel_cmd)
            
        except Exception as e:
            print(f"ROS命令发送失败: {e}")
    
    def _send_webots_command(self, left_speed: float, right_speed: float):
        """发送Webots命令"""
        try:
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
            
        except Exception as e:
            print(f"Webots命令发送失败: {e}")
    
    def _check_command_timeout(self):
        """检查命令超时"""
        while self.running:
            try:
                current_time = time.time()
                
                # 如果超时，停止机器人
                if current_time - self.last_command_time > self.command_timeout:
                    if abs(self.current_left_percent) > 0.01 or abs(self.current_right_percent) > 0.01:
                        print("命令超时，停止机器人")
                        self.current_left_percent = 0.0
                        self.current_right_percent = 0.0
                        self._execute_wheel_command(0.0, 0.0)
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"超时检查错误: {e}")
    
    def get_server_ip(self) -> str:
        """获取服务器IP地址"""
        try:
            # 连接到外部地址以获取本机IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    
    def print_connection_info(self):
        """打印连接信息"""
        ip = self.get_server_ip()
        print("\n" + "="*50)
        print("机器人TCP服务器连接信息")
        print("="*50)
        print(f"服务器IP: {ip}")
        print(f"监听端口: {self.port}")
        print(f"连接地址: {ip}:{self.port}")
        print("\n客户端连接命令:")
        print(f"python keyboard_control.py --tcp --ip {ip} --port {self.port}")
        print("="*50 + "\n")
    
    def cleanup(self):
        """清理资源"""
        print("正在清理服务器资源...")
        
        self.running = False
        
        # 停止机器人
        if self.use_ros or self.use_webots:
            self._execute_wheel_command(0.0, 0.0)
        
        # 关闭客户端连接
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        
        # 关闭服务器socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("服务器清理完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='机器人TCP服务器')
    parser.add_argument('--port', type=int, default=8888,
                       help='监听端口 (默认: 8888)')
    parser.add_argument('--no-ros', action='store_true',
                       help='禁用ROS接口')
    parser.add_argument('--no-webots', action='store_true',
                       help='禁用Webots接口')
    parser.add_argument('--robot-name', type=str, default='rosbot',
                       help='机器人名称 (默认: rosbot)')
    parser.add_argument('--max-wheel-speed', type=float, default=26.0,
                       help='最大轮速 (默认: 26.0 rad/s)')
    parser.add_argument('--wheel-radius', type=float, default=0.043,
                       help='轮半径 (默认: 0.043 m)')
    parser.add_argument('--wheel-base', type=float, default=0.2,
                       help='轮距 (默认: 0.2 m)')
    
    args = parser.parse_args()
    
    # 创建服务器
    server = RobotTCPServer(
        port=args.port,
        use_ros=not args.no_ros,
        use_webots=not args.no_webots,
        robot_name=args.robot_name
    )
    
    # 设置参数
    server.max_wheel_speed = args.max_wheel_speed
    server.wheel_radius = args.wheel_radius
    server.wheel_base = args.wheel_base
    
    # 打印连接信息
    server.print_connection_info()
    
    # 启动服务器
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭服务器...")
    finally:
        server.cleanup()


if __name__ == '__main__':
    main()
