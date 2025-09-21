#!/usr/bin/env python3
"""
键盘控制程序 - 控制另一个Docker容器中的ROSbot
基于Webots外部控制器协议，发送轮速百分比控制命令
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Optional, Tuple

# 检查pygame可用性
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("pygame not available, keyboard input only")
    PYGAME_AVAILABLE = False

# 检查Webots可用性
try:
    from controller import Supervisor
    WEBOTS_AVAILABLE = True
except ImportError:
    print("Webots not available")
    WEBOTS_AVAILABLE = False
    sys.exit(1)

# 检查OCR客户端可用性
try:
    from ocr_service_client import SimpleOCRClient
    OCR_CLIENT_AVAILABLE = True
except ImportError:
    print("OCR service client not available")
    OCR_CLIENT_AVAILABLE = False

except ImportError:
    print("Pygame not available, using keyboard input")
    PYGAME_AVAILABLE = False

class KeyboardController:
    """键盘控制器类 - 支持Webots外部控制器协议的ROSbot控制"""
    
    def __init__(self, 
                 robot_ip: str = "localhost",
                 robot_port: int = 8888,
                 webots_url: Optional[str] = None,
                 use_direct_webots: bool = False):
        """
        初始化键盘控制器
        
        Args:
            robot_ip: 机器人容器IP地址
            robot_port: TCP通信端口
            webots_url: Webots外部控制器URL（可选）
            use_direct_webots: 是否直接连接Webots（而非通过TCP服务器）
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.webots_url = webots_url
        self.use_direct_webots = use_direct_webots and WEBOTS_AVAILABLE
        
        # ROSbot控制参数（基于navigation_env.py）
        self.max_motor_speed = 26.0   # 最大电机速度 rad/s
        self.wheel_base = 0.22        # 轮距 m
        self.wheel_radius = 0.043     # 轮半径 m
        self.speed_increment = 0.1    # 轮速百分比增量
        
        # 当前轮速百分比（0.0-1.0）
        self.left_wheel_percent = 0.0
        self.right_wheel_percent = 0.0
        
        # 控制状态
        self.running = True
        self.connected = False
        
        # 初始化通信
        self._init_communication()
        
        # Webots控制器（直接连接模式）
        self.robot = None
        self.fl_motor = None
        self.fr_motor = None
        self.rl_motor = None
        self.rr_motor = None
        
        # ROS OCR服务客户端
        self.ocr_client = None
        # 延迟OCR客户端初始化，避免多个Robot实例冲突
        if not self.use_direct_webots:
            self._init_ocr_client()
        
        # 初始化输入方式
        if PYGAME_AVAILABLE:
            self._init_pygame()
        
        print("键盘控制器初始化完成")
        self._print_help()
    
    def _init_communication(self):
        """初始化通信方式"""
        if self.use_direct_webots:
            self._init_webots()
        else:
            self._init_tcp()
    
    def _init_webots(self):
        """初始化Webots直接连接"""
        try:
            # 设置用户名环境变量
            if 'USER' not in os.environ and 'USERNAME' not in os.environ:
                os.environ['USER'] = 'webots_user'
                os.environ['USERNAME'] = 'webots_user'
            
            if self.webots_url:
                print(f"🔌 连接到 Webots: {self.webots_url}")
                # 解析URL并设置环境变量
                if self.webots_url.startswith('tcp://'):
                    # TCP连接格式：tcp://localhost:1236
                    import urllib.parse
                    parsed = urllib.parse.urlparse(self.webots_url)
                    host = parsed.hostname or 'localhost'
                    port = parsed.port or 1234
                    
                    # 设置Webots的连接参数
                    os.environ['WEBOTS_SERVER'] = host
                    os.environ['WEBOTS_PORT'] = str(port)
                    os.environ['WEBOTS_CONTROLLER_URL'] = str(self.webots_url)
                    print(f"   设置连接: {host}:{port}")
                elif self.webots_url.startswith('http://'):
                    # HTTP连接格式：http://localhost:1236
                    import urllib.parse
                    parsed = urllib.parse.urlparse(self.webots_url)
                    host = parsed.hostname or 'localhost'
                    port = parsed.port or 1234
                    
                    # 转换为TCP格式
                    tcp_url = f"tcp://{host}:{port}"
                    os.environ['WEBOTS_SERVER'] = host
                    os.environ['WEBOTS_PORT'] = str(port)
                    os.environ['WEBOTS_CONTROLLER_URL'] = tcp_url
                    print(f"   设置连接: {host}:{port} (TCP)")
                else:
                    # 其他格式直接设置
                    os.environ['WEBOTS_CONTROLLER_URL'] = str(self.webots_url)
            else:
                print("🔌 使用默认 Webots 连接")
            
            print("🤖 初始化 Supervisor...")
            # 初始化Webots控制器
            self.robot = Supervisor()
            print("✅ Supervisor 初始化成功")
            
            # 等待Webots完全加载
            timestep = int(self.robot.getBasicTimeStep())
            self.robot.step(timestep)
            
            # 获取电机设备
            self.fl_motor = self.robot.getDevice('fl_wheel_joint')
            self.fr_motor = self.robot.getDevice('fr_wheel_joint')
            self.rl_motor = self.robot.getDevice('rl_wheel_joint')
            self.rr_motor = self.robot.getDevice('rr_wheel_joint')
            
            # 设置电机模式
            for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
                if motor:
                    motor.setPosition(float('inf'))
                    motor.setVelocity(0.0)
            
            self.connected = True
            print("Webots直接连接成功")
            
            # 在Webots初始化成功后初始化OCR客户端
            self._init_ocr_client()
            
        except Exception as e:
            print(f"Webots直接连接初始化失败: {e}")
            self.connected = False
    
    def _init_ocr_client(self):
        """初始化ROS OCR服务客户端"""
        if OCR_CLIENT_AVAILABLE and (ROS_AVAILABLE or WEBOTS_AVAILABLE):
            try:
                # 如果使用直接Webots模式，传递已存在的Robot实例
                robot_instance = self.robot if self.use_direct_webots else None
                
                self.ocr_client = SimpleOCRClient(
                    service_name="/ocr_service",
                    webots_camera_name="camera",
                    robot_instance=robot_instance
                )
                print("ROS OCR服务客户端初始化成功")
            except Exception as e:
                print(f"ROS OCR服务客户端初始化失败: {e}")
                self.ocr_client = None
        else:
            print("ROS OCR服务不可用")
            self.ocr_client = None
    
    def _init_tcp(self):
        """初始化TCP通信"""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(5.0)  # 设置连接超时
            self.tcp_socket.connect((self.robot_ip, self.robot_port))
            self.connected = True
            print(f"TCP连接成功: {self.robot_ip}:{self.robot_port}")
            print("使用ROSbot轮速控制协议（左右轮速百分比 0.0-1.0）")
            
        except Exception as e:
            print(f"TCP连接失败: {e}")
            self.connected = False
    
    def _init_pygame(self):
        """初始化pygame（如果可用）"""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((400, 300))
            pygame.display.set_caption("ROSbot键盘控制")
            self.clock = pygame.time.Clock()
            print("Pygame初始化成功，可以使用游戏手柄或键盘")
        except Exception as e:
            print(f"Pygame初始化失败: {e}")
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print("ROSbot键盘控制帮助 (Webots外部控制器协议)")
        print("="*60)
        print("移动控制:")
        print("  W/↑    - 前进 (增加两轮速度)")
        print("  S/↓    - 后退 (减少两轮速度)") 
        print("  A/←    - 左转 (左轮慢，右轮快)")
        print("  D/→    - 右转 (左轮快，右轮慢)")
        print("  Q      - 左前 (增加右轮更多)")
        print("  E      - 右前 (增加左轮更多)")
        print("  Z      - 左后 (减少左轮更多)")
        print("  C      - 右后 (减少右轮更多)")
        print("\n轮速控制:")
        print("  +/=    - 增加速度增量")
        print("  -/_    - 减少速度增量")
        print("  空格   - 紧急停止")
        print("\n其他:")
        print("  H      - 显示帮助")
        print("  R      - 重置速度")
        print("  I      - 获取摄像头图像并OCR识别")
        print("  ESC    - 退出程序")
        print("="*60)
        print(f"当前连接状态: {'已连接' if self.connected else '未连接'}")
        print(f"通信方式: {'Webots直接' if self.use_direct_webots else 'TCP服务器'}")
        if not self.use_direct_webots:
            print(f"目标地址: {self.robot_ip}:{self.robot_port}")
        print(f"轮速范围: 0.0-1.0 (对应 0-{self.max_motor_speed} rad/s)")
        print(f"当前轮速: 左={self.left_wheel_percent:.2f}, 右={self.right_wheel_percent:.2f}")
        print(f"OCR服务: {'可用' if self.ocr_client and self.ocr_client.is_available() else '不可用'}")
        print("="*60 + "\n")
    
    def send_wheel_command(self, left_percent: float, right_percent: float):
        """
        发送轮速命令
        
        Args:
            left_percent: 左轮速度百分比 (0.0-1.0)
            right_percent: 右轮速度百分比 (0.0-1.0)
        """
        if not self.connected:
            print("未连接到机器人")
            return
        
        # 限制轮速范围
        left_percent = max(0.0, min(1.0, left_percent))
        right_percent = max(0.0, min(1.0, right_percent))
        
        try:
            if self.use_direct_webots:
                # 直接Webots控制
                return self._send_webots_command(left_percent, right_percent)
            else:
                # TCP控制
                return self._send_tcp_command(left_percent, right_percent)
        except Exception as e:
            print(f"发送命令失败: {e}")
            return False
    
    def _send_webots_command(self, left_percent: float, right_percent: float) -> bool:
        """通过Webots直接发送命令"""
        try:
            # 转换为实际电机速度
            left_speed = left_percent * self.max_motor_speed
            right_speed = right_percent * self.max_motor_speed
            
            # 设置四个电机速度
            if self.fl_motor and self.fr_motor and self.rl_motor and self.rr_motor:
                # 左侧电机
                self.fl_motor.setVelocity(left_speed)
                self.rl_motor.setVelocity(left_speed)
                
                # 右侧电机
                self.fr_motor.setVelocity(right_speed)
                self.rr_motor.setVelocity(right_speed)
                
                return True
            else:
                print("电机设备未初始化")
                return False
                
        except Exception as e:
            print(f"Webots命令发送失败: {e}")
            return False
    
    def _send_tcp_command(self, left_percent: float, right_percent: float) -> bool:
        """通过TCP发送命令（ROSbot轮速协议）"""
        try:
            # 构造ROSbot动作命令（匹配navigation_env.py的_execute_action方法）
            # 发送2维动作：[左轮速度百分比, 右轮速度百分比]
            action = [float(left_percent), float(right_percent)]
            
            # 构造命令数据包
            command = {
                'type': 'wheel_control',
                'action': action,
                'left_wheel_percent': float(left_percent),
                'right_wheel_percent': float(right_percent),
                'timestamp': time.time()
            }
            
            # 发送JSON格式数据
            data = json.dumps(command).encode('utf-8')
            self.tcp_socket.send(data + b'\n')
            return True
        except Exception as e:
            print(f"TCP命令发送失败: {e}")
            return False
    
    def update_wheel_speeds(self, left_delta: float = 0.0, right_delta: float = 0.0):
        """更新轮速百分比"""
        self.left_wheel_percent = np.clip(
            self.left_wheel_percent + left_delta, 
            0.0, 
            1.0
        )
        self.right_wheel_percent = np.clip(
            self.right_wheel_percent + right_delta,
            0.0,
            1.0
        )
    
    def stop(self):
        """停止机器人"""
        self.left_wheel_percent = 0.0
        self.right_wheel_percent = 0.0
        self.send_wheel_command(0.0, 0.0)
        print("机器人已停止")
    
    def reset_speed(self):
        """重置速度参数"""
        self.left_wheel_percent = 0.0
        self.right_wheel_percent = 0.0
        print("轮速已重置")
    
    def capture_and_process_image(self, language: str = "ch_en", use_angle_cls: bool = True):
        """获取摄像头图像并发送OCR识别"""
        if not self.ocr_client:
            print("❌ OCR服务客户端未初始化")
            return
        
        if not self.ocr_client.is_available():
            print("❌ OCR服务不可用")
            print(f"   状态: {self.ocr_client.get_status()}")
            return
        
        print(f"📸 正在调用OCR服务进行图像识别 (language={language}, use_angle_cls={use_angle_cls})...")
        
        try:
            # 调用OCR服务
            result = self.ocr_client.call_ocr_service(language, use_angle_cls)
            
            if result['success']:
                print(f"✅ OCR识别成功!")
                print(f"   结果: {result['message']}")
                print(f"   识别文字: {result['total_text']}")
                if result['texts']:
                    print(f"   详细文字: {', '.join(result['texts'])}")
                if result['confidences']:
                    avg_confidence = sum(result['confidences']) / len(result['confidences'])
                    print(f"   平均置信度: {avg_confidence:.2f}")
                if result['bboxes']:
                    print(f"   检测到 {len(result['bboxes'])} 个文字区域")
            else:
                print(f"❌ OCR识别失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            print(f"❌ OCR识别异常: {e}")
    
    def handle_keyboard_input(self):
        """处理键盘输入（非pygame模式）"""
        if PYGAME_AVAILABLE:
            return self._handle_pygame_input()
        else:
            return self._handle_terminal_input()
    
    def _handle_pygame_input(self) -> bool:
        """处理pygame输入"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                key = event.key
                
                # 移动控制（轮速百分比控制）
                if key == pygame.K_w or key == pygame.K_UP:
                    # 前进 - 增加两轮速度
                    self.update_wheel_speeds(self.speed_increment, self.speed_increment)
                elif key == pygame.K_s or key == pygame.K_DOWN:
                    # 后退 - 减少两轮速度
                    self.update_wheel_speeds(-self.speed_increment, -self.speed_increment)
                elif key == pygame.K_a or key == pygame.K_LEFT:
                    # 左转 - 左轮慢，右轮快
                    self.update_wheel_speeds(-self.speed_increment, self.speed_increment)
                elif key == pygame.K_d or key == pygame.K_RIGHT:
                    # 右转 - 左轮快，右轮慢
                    self.update_wheel_speeds(self.speed_increment, -self.speed_increment)
                elif key == pygame.K_q:
                    # 左前 - 右轮增加更多
                    self.update_wheel_speeds(self.speed_increment, self.speed_increment * 1.5)
                elif key == pygame.K_e:
                    # 右前 - 左轮增加更多
                    self.update_wheel_speeds(self.speed_increment * 1.5, self.speed_increment)
                elif key == pygame.K_z:
                    # 左后 - 左轮减少更多
                    self.update_wheel_speeds(-self.speed_increment * 1.5, -self.speed_increment)
                elif key == pygame.K_c:
                    # 右后 - 右轮减少更多
                    self.update_wheel_speeds(-self.speed_increment, -self.speed_increment * 1.5)
                
                # 轮速控制
                elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
                    self.speed_increment = min(self.speed_increment + 0.05, 0.5)
                    print(f"轮速增量: {self.speed_increment:.2f}")
                elif key == pygame.K_MINUS or key == pygame.K_UNDERSCORE:
                    self.speed_increment = max(self.speed_increment - 0.05, 0.05)
                    print(f"轮速增量: {self.speed_increment:.2f}")
                elif key == pygame.K_SPACE:
                    self.stop()
                
                # 其他控制
                elif key == pygame.K_h:
                    self._print_help()
                elif key == pygame.K_r:
                    self.reset_speed()
                elif key == pygame.K_i:
                    # OCR识别功能
                    self.capture_and_process_image("ch_en", True)
                elif key == pygame.K_ESCAPE:
                    return False
        
        return True
    
    def _handle_terminal_input(self) -> bool:
        """处理终端输入"""
        try:
            import select
            import sys
            import tty
            import termios
            
            # 设置非阻塞输入
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            if select.select([sys.stdin], [], [], 0.1):
                key = sys.stdin.read(1)
                
                # 移动控制（轮速百分比控制）
                if key.lower() == 'w':
                    # 前进 - 增加两轮速度
                    self.update_wheel_speeds(self.speed_increment, self.speed_increment)
                elif key.lower() == 's':
                    # 后退 - 减少两轮速度
                    self.update_wheel_speeds(-self.speed_increment, -self.speed_increment)
                elif key.lower() == 'a':
                    # 左转 - 左轮慢，右轮快
                    self.update_wheel_speeds(-self.speed_increment, self.speed_increment)
                elif key.lower() == 'd':
                    # 右转 - 左轮快，右轮慢
                    self.update_wheel_speeds(self.speed_increment, -self.speed_increment)
                elif key.lower() == 'q':
                    # 左前 - 右轮增加更多
                    self.update_wheel_speeds(self.speed_increment, self.speed_increment * 1.5)
                elif key.lower() == 'e':
                    # 右前 - 左轮增加更多
                    self.update_wheel_speeds(self.speed_increment * 1.5, self.speed_increment)
                elif key.lower() == 'z':
                    # 左后 - 左轮减少更多
                    self.update_wheel_speeds(-self.speed_increment * 1.5, -self.speed_increment)
                elif key.lower() == 'c':
                    # 右后 - 右轮减少更多
                    self.update_wheel_speeds(-self.speed_increment, -self.speed_increment * 1.5)
                
                # 轮速控制
                elif key == '+' or key == '=':
                    self.speed_increment = min(self.speed_increment + 0.05, 0.5)
                    print(f"轮速增量: {self.speed_increment:.2f}")
                elif key == '-' or key == '_':
                    self.speed_increment = max(self.speed_increment - 0.05, 0.05)
                    print(f"轮速增量: {self.speed_increment:.2f}")
                elif key == ' ':
                    self.stop()
                
                # 其他控制
                elif key.lower() == 'h':
                    self._print_help()
                elif key.lower() == 'r':
                    self.reset_speed()
                elif key.lower() == 'i':
                    # OCR识别功能
                    self.capture_and_process_image("ch_en", True)
                elif key == '\x1b':  # ESC
                    return False
            
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
        except Exception as e:
            print(f"输入处理错误: {e}")
        
        return True
    
    def run(self):
        """运行控制循环"""
        print("开始键盘控制，按H查看帮助...")
        
        try:
            while self.running:
                # 处理输入
                if not self.handle_keyboard_input():
                    break
                
                # 发送当前轮速命令
                if self.connected:
                    self.send_wheel_command(self.left_wheel_percent, self.right_wheel_percent)
                
                # 显示当前状态
                if abs(self.left_wheel_percent) > 0.01 or abs(self.right_wheel_percent) > 0.01:
                    left_rpm = self.left_wheel_percent * self.max_motor_speed
                    right_rpm = self.right_wheel_percent * self.max_motor_speed
                    print(f"\r左轮: {self.left_wheel_percent:.2f}({left_rpm:.1f}rad/s), "
                          f"右轮: {self.right_wheel_percent:.2f}({right_rpm:.1f}rad/s)", end='', flush=True)
                
                # 控制循环频率
                if PYGAME_AVAILABLE:
                    self.clock.tick(30)  # 30 FPS
                else:
                    time.sleep(0.1)  # 10 Hz
                    
        except KeyboardInterrupt:
            print("\n收到中断信号，正在退出...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        
        # 停止机器人
        if self.connected:
            self.stop()
        
        # 关闭连接
        if not self.use_ros and hasattr(self, 'tcp_socket'):
            try:
                self.tcp_socket.close()
            except:
                pass
        
        # 清理pygame
        if PYGAME_AVAILABLE:
            try:
                pygame.quit()
            except:
                pass
        
        print("清理完成，程序退出")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ROSbot键盘控制程序 (Webots外部控制器协议)')
    parser.add_argument('--ip', type=str, default='localhost',
                       help='机器人容器IP地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=1236,
                       help='TCP通信端口 (默认: 1236)')
    parser.add_argument('--webots-url', type=str, default='tcp://localhost:1236',
                       help='Webots外部控制器URL (可选)')
    parser.add_argument('--direct-webots', action='store_true',
                       help='直接连接Webots而非通过TCP服务器')
    parser.add_argument('--max-motor-speed', type=float, default=26.0,
                       help='最大电机速度 (默认: 26.0 rad/s)')
    parser.add_argument('--wheel-base', type=float, default=0.22,
                       help='轮距 (默认: 0.22 m)')
    parser.add_argument('--wheel-radius', type=float, default=0.043,
                       help='轮半径 (默认: 0.043 m)')
    parser.add_argument('--speed-increment', type=float, default=0.1,
                       help='轮速增量 (默认: 0.1)')
    
    args = parser.parse_args()
    
    # 创建控制器
    controller = KeyboardController(
        robot_ip=args.ip,
        robot_port=args.port,
        webots_url=args.webots_url,
        use_direct_webots=args.direct_webots
    )
    
    # 设置控制参数
    controller.max_motor_speed = args.max_motor_speed
    controller.wheel_base = args.wheel_base
    controller.wheel_radius = args.wheel_radius
    controller.speed_increment = args.speed_increment
    
    # 运行控制器
    controller.run()


if __name__ == '__main__':
    main()
