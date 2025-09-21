#!/usr/bin/env python3
"""
ROSbot容器适配器 - 在Docker容器中运行，接收键盘控制命令并控制ROSbot
基于navigation_env.py的控制协议，支持Webots外部控制器模式
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

# 尝试导入Webots控制器
try:
    from controller import Supervisor, Robot
    WEBOTS_AVAILABLE = True
except ImportError:
    print("Webots not available, using simulation mode")
    WEBOTS_AVAILABLE = False

class ROSbotContainerAdapter:
    """ROSbot容器适配器类 - 在容器中运行并接收控制命令"""
    
    def __init__(self, 
                 port: int = 8888,
                 webots_url: Optional[str] = None,
                 robot_name: str = "rosbot"):
        """
        初始化ROSbot容器适配器
        
        Args:
            port: 监听端口
            webots_url: Webots外部控制器URL
            robot_name: 机器人名称
        """
        self.port = port
        self.webots_url = webots_url
        self.robot_name = robot_name
        
        # 服务器状态
        self.running = True
        self.clients = []
        self.server_socket = None
        
        # ROSbot控制参数（匹配navigation_env.py）
        self.max_motor_speed = 26.0  # rad/s
        self.wheel_radius = 0.043    # m
        self.wheel_base = 0.22       # m
        
        # 当前轮速百分比状态
        self.current_left_percent = 0.0
        self.current_right_percent = 0.0
        self.last_command_time = time.time()
        self.command_timeout = 1.0  # 命令超时时间
        
        # Webots控制器
        self.robot = None
        self.fl_motor = None
        self.fr_motor = None
        self.rl_motor = None
        self.rr_motor = None
        
        # 轮速平滑缓存
        self._prev_left_speed = 0.0
        self._prev_right_speed = 0.0
        
        # 初始化Webots控制器
        self._init_webots_controller()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"ROSbot容器适配器初始化完成")
        print(f"监听端口: {self.port}")
        print(f"Webots URL: {self.webots_url or '默认'}")
    
    def _init_webots_controller(self):
        """初始化Webots控制器"""
        if not WEBOTS_AVAILABLE:
            print("Webots不可用，使用模拟模式")
            return
            
        try:
            # 设置Webots连接URL
            if self.webots_url:
                os.environ['WEBOTS_CONTROLLER_URL'] = self.webots_url
                print(f"设置Webots URL: {self.webots_url}")
            
            # 初始化Supervisor
            print("初始化Webots Supervisor...")
            self.robot = Supervisor()
            
            # 获取电机设备（匹配navigation_env.py的设备名称）
            self.fl_motor = self.robot.getDevice('fl_wheel_joint')
            self.fr_motor = self.robot.getDevice('fr_wheel_joint')
            self.rl_motor = self.robot.getDevice('rl_wheel_joint')
            self.rr_motor = self.robot.getDevice('rr_wheel_joint')
            
            # 设置电机模式
            for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
                if motor:
                    motor.setPosition(float('inf'))
                    motor.setVelocity(0.0)
            
            # 获取时间步长
            self.timestep = int(self.robot.getBasicTimeStep())
            
            print("Webots控制器初始化成功")
            
        except Exception as e:
            print(f"Webots控制器初始化失败: {e}")
            self.robot = None
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在关闭适配器...")
        self.running = False
    
    def start_server(self):
        """启动TCP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)
            
            print(f"TCP服务器启动成功，监听端口: {self.port}")
            print("等待键盘控制客户端连接...")
            
            # 启动命令超时检查线程
            timeout_thread = threading.Thread(target=self._check_command_timeout)
            timeout_thread.daemon = True
            timeout_thread.start()
            
            # 启动Webots步进线程
            if self.robot:
                webots_thread = threading.Thread(target=self._webots_step_loop)
                webots_thread.daemon = True
                webots_thread.start()
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    print(f"键盘控制客户端连接: {client_address}")
                    
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
    
    def _webots_step_loop(self):
        """Webots步进循环"""
        while self.running and self.robot:
            try:
                if self.robot.step(self.timestep) == -1:
                    break
            except Exception as e:
                print(f"Webots步进错误: {e}")
                break
    
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
                print(f"未知命令类型: {command.get('type', 'unknown')}")
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
        except Exception as e:
            print(f"命令处理错误: {e}")
    
    def _execute_wheel_command(self, left_percent: float, right_percent: float):
        """执行轮速控制命令（匹配navigation_env.py的_execute_action方法）"""
        if not self.robot:
            print("Webots控制器未初始化，跳过命令执行")
            return
            
        try:
            # 转换为实际电机速度（匹配navigation_env.py的实现）
            left_speed = left_percent * self.max_motor_speed
            right_speed = right_percent * self.max_motor_speed
            
            # 平滑限速：限制单步变化，避免瞬时大扭矩引发不稳定
            max_delta = self.max_motor_speed * 0.6  # 60%/step
            
            # 限制变化范围，但确保不会低于0
            left_speed = float(np.clip(left_speed, 
                                     max(0.0, self._prev_left_speed - max_delta), 
                                     self._prev_left_speed + max_delta))
            right_speed = float(np.clip(right_speed, 
                                      max(0.0, self._prev_right_speed - max_delta), 
                                      self._prev_right_speed + max_delta))
            
            # 设置四个电机速度 - 左侧两个轮子相同速度，右侧两个轮子相同速度
            if self.fl_motor and self.fr_motor and self.rl_motor and self.rr_motor:
                # 左侧电机
                self.fl_motor.setVelocity(left_speed)
                self.rl_motor.setVelocity(left_speed)
                
                # 右侧电机
                self.fr_motor.setVelocity(right_speed)
                self.rr_motor.setVelocity(right_speed)
                
                # 记录本次轮速用于下次平滑
                self._prev_left_speed = left_speed
                self._prev_right_speed = right_speed
                
                # 计算等效线速度和角速度（用于调试输出）
                linear_vel = self.wheel_radius * (left_speed + right_speed) / 2.0
                angular_vel = self.wheel_radius * (right_speed - left_speed) / self.wheel_base
                
                print(f"设置电机速度 - 左: {left_speed:.2f} rad/s, 右: {right_speed:.2f} rad/s "
                      f"(等效: 线速度 {linear_vel:.2f} m/s, 角速度 {angular_vel:.2f} rad/s)")
            else:
                print("电机设备未正确初始化")
                
        except Exception as e:
            print(f"执行轮速命令失败: {e}")
    
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
        print("\n" + "="*60)
        print("ROSbot容器适配器连接信息")
        print("="*60)
        print(f"服务器IP: {ip}")
        print(f"监听端口: {self.port}")
        print(f"连接地址: {ip}:{self.port}")
        print(f"Webots状态: {'已连接' if self.robot else '未连接'}")
        print("\n客户端连接命令:")
        print(f"python keyboard_control.py --ip {ip} --port {self.port}")
        print("="*60 + "\n")
    
    def cleanup(self):
        """清理资源"""
        print("正在清理适配器资源...")
        
        self.running = False
        
        # 停止机器人
        if self.robot:
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
        
        print("适配器清理完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ROSbot容器适配器')
    parser.add_argument('--port', type=int, default=8888,
                       help='监听端口 (默认: 8888)')
    parser.add_argument('--webots-url', type=str, default=None,
                       help='Webots外部控制器URL (可选)')
    parser.add_argument('--robot-name', type=str, default='rosbot',
                       help='机器人名称 (默认: rosbot)')
    parser.add_argument('--max-motor-speed', type=float, default=26.0,
                       help='最大电机速度 (默认: 26.0 rad/s)')
    parser.add_argument('--wheel-radius', type=float, default=0.043,
                       help='轮半径 (默认: 0.043 m)')
    parser.add_argument('--wheel-base', type=float, default=0.22,
                       help='轮距 (默认: 0.22 m)')
    
    args = parser.parse_args()
    
    # 创建适配器
    adapter = ROSbotContainerAdapter(
        port=args.port,
        webots_url=args.webots_url,
        robot_name=args.robot_name
    )
    
    # 设置参数
    adapter.max_motor_speed = args.max_motor_speed
    adapter.wheel_radius = args.wheel_radius
    adapter.wheel_base = args.wheel_base
    
    # 打印连接信息
    adapter.print_connection_info()
    
    # 启动适配器
    try:
        adapter.start_server()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭适配器...")
    finally:
        adapter.cleanup()


if __name__ == '__main__':
    main()
