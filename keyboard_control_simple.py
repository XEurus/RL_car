#!/usr/bin/env python3
"""
简化的ROSbot键盘控制程序
只支持直接Webots连接模式，包含OCR功能
"""

import os
import sys
import time
import argparse
import threading
from typing import Optional
from controller import Supervisor

# 检查pygame可用性
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("pygame not available, using keyboard input only")
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
    from ocr_service_client_simple import SimpleOCRClient
    OCR_CLIENT_AVAILABLE = True
except ImportError:
    print("OCR service client not available")
    OCR_CLIENT_AVAILABLE = False

class SimpleKeyboardController:
    """简化的键盘控制器 - 只支持直接Webots连接"""
    
    def __init__(self, webots_url: str = "tcp://localhost:1236"):
        """
        初始化键盘控制器
        
        Args:
            webots_url: Webots外部控制器URL
        """
        self.webots_url = webots_url
        
        # ROSbot控制参数
        self.max_motor_speed = 26.0   # 最大电机速度 rad/s
        self.speed_increment = 0.1    # 轮速百分比增量
        
        # 当前轮速百分比（0.0-1.0）
        self.left_wheel_percent = 0.0
        self.right_wheel_percent = 0.0
        
        # 控制状态
        self.running = True
        self.connected = False
        self.control_active = False  # 是否正在控制
        
        # 线程同步
        self.control_lock = threading.Lock()
        
        # Webots设备
        self.robot = None
        self.fl_motor = None
        self.fr_motor = None
        self.rl_motor = None
        self.rr_motor = None
        self.camera = None
        
        # OCR客户端
        self.ocr_client = None
        
        # 初始化
        self._init_webots()
        self._init_ocr_client()
        self._init_pygame()
        
    def _init_webots(self):
        """初始化Webots连接"""
        try:
            # 设置环境变量
            if self.webots_url.startswith('tcp://'):
                import urllib.parse
                parsed = urllib.parse.urlparse(self.webots_url)
                host = parsed.hostname or 'localhost'
                port = parsed.port or 1236
                os.environ['WEBOTS_SERVER'] = host
                os.environ['WEBOTS_PORT'] = str(port)
                os.environ['WEBOTS_CONTROLLER_URL'] = str(self.webots_url)
                print(f"🔌 连接到 Webots: {self.webots_url}")
                print(f"   设置连接: {host}:{port}")
            
            # 设置用户环境变量（避免错误）
            if 'USER' not in os.environ:
                os.environ['USER'] = 'webots_user'
            if 'USERNAME' not in os.environ:
                os.environ['USERNAME'] = 'webots_user'
            
            # 初始化Supervisor
            print("🤖 初始化 Supervisor...")
            self.robot = Supervisor()
            print("✅ Supervisor 初始化成功")
            
            # 等待Webots完全加载
            self.robot.step(1)
            
            # 初始化设备（电机和摄像头）
            self._init_devices()
                
        except Exception as e:
            print(f"❌ Webots连接失败: {e}")
            self.connected = False
    
    def _init_ocr_client(self):
        """初始化OCR客户端"""
        if OCR_CLIENT_AVAILABLE and self.robot:
            try:
                self.ocr_client = SimpleOCRClient(
                    service_name="/ocr_service",
                    webots_camera_name="camera",
                    robot_instance=self.robot
                )
                print("✅ OCR服务客户端初始化成功")
            except Exception as e:
                print(f"❌ OCR服务客户端初始化失败: {e}")
                self.ocr_client = None
        else:
            print("❌ OCR服务不可用")
            self.ocr_client = None
    
    def _init_pygame(self):
        """初始化pygame"""
        if PYGAME_AVAILABLE:
            try:
                pygame.init()
                pygame.display.set_mode((1, 1))  # 最小窗口
                print("✅ Pygame初始化成功")
            except Exception as e:
                print(f"❌ Pygame初始化失败: {e}")
    
    def send_wheel_command(self, left_percent: float, right_percent: float):
        """发送轮速命令"""
        if not self.connected:
            print("❌ 未连接到机器人")
            return False
        
        # 限制轮速范围
        left_percent = max(0.0, min(1.0, left_percent))
        right_percent = max(0.0, min(1.0, right_percent))
        
        try:
            # 转换为实际电机速度
            left_speed = left_percent * self.max_motor_speed
            right_speed = right_percent * self.max_motor_speed
            
            # 设置四个电机速度
            if all([self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]):
                # 左侧电机
                self.fl_motor.setVelocity(left_speed)
                self.rl_motor.setVelocity(left_speed)
                
                # 右侧电机
                self.fr_motor.setVelocity(right_speed)
                self.rr_motor.setVelocity(right_speed)
                
                # 执行一步仿真
                self.robot.step(1)
                
                # 输出详细的电机状态信息
                print(f"🔧 电机速度: 左={left_speed:.1f} rad/s, 右={right_speed:.1f} rad/s")
                return True
            else:
                print("❌ 电机设备未初始化")
                return False
        except Exception as e:
            print(f"❌ 发送命令失败: {e}")
            return False
    
    def capture_and_ocr(self):
        """捕获摄像头图像并进行OCR识别"""
        if self.ocr_client and self.ocr_client.is_available():
            print("📸 正在捕获摄像头图像并进行OCR识别...")
            try:
                result = self.ocr_client.call_ocr_service(
                    language="ch",
                    use_angle_cls=True
                )
                
                if result and result.get('success', False):
                    texts = result.get('texts', [])
                    confidences = result.get('confidences', [])
                    total_text = result.get('total_text', '')
                    
                    print("🔍 OCR识别结果:")
                    print(f"   总文本: {total_text}")
                    print(f"   识别到 {len(texts)} 个文本区域:")
                    for i, (text, conf) in enumerate(zip(texts, confidences)):
                        print(f"     {i+1}. '{text}' (置信度: {conf:.2f})")
                else:
                    print("❌ OCR识别失败")
                    
            except Exception as e:
                print(f"❌ OCR识别出错: {e}")
        else:
            print("❌ OCR服务不可用")
    
    def handle_keyboard_input(self):
        """处理键盘输入"""
        # 在容器环境中使用实时控制模式
        print("检测到容器环境，使用实时控制模式")
        self._handle_realtime_control()
    
    def _init_devices(self):
        """初始化所有设备（电机和摄像头）"""
        # 初始化电机设备
        motor_names = ['fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint']
        motors = []
        
        for name in motor_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))  # 设置为速度控制模式
                motor.setVelocity(0.0)
                motors.append(motor)
                print(f"✅ 电机设备初始化成功: {name}")
            else:
                print(f"❌ 未找到电机设备: {name}")
        
        # 初始化摄像头设备
        camera = self.robot.getDevice("camera")
        print(camera)
        if camera:
            timestep = int(self.robot.getBasicTimeStep())
            camera.enable(timestep)
            self.camera = camera
            print("✅ 摄像头设备初始化成功: camera")
        else:
            print("❌ 未找到摄像头设备: camera")
            self.camera = None
                
        if len(motors) == 4:
            self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor = motors
            self.connected = True
            print("✅ Webots直接连接成功")
        else:
            print("❌ 电机设备初始化失败")
    
    def _handle_pygame_input(self):
        """使用pygame处理键盘输入"""
        clock = pygame.time.Clock()
        print("开始pygame键盘控制，按键控制机器人...")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._process_key(event.key)
                    print(f"当前轮速: 左={self.left_wheel_percent:.2f}, 右={self.right_wheel_percent:.2f}")
            
            # 发送当前轮速命令
            if self.connected:
                self.send_wheel_command(self.left_wheel_percent, self.right_wheel_percent)
            
            clock.tick(10)  # 10 FPS
    
    def _handle_console_input(self):
        """使用控制台处理键盘输入"""
        print("\n🎮 控制台输入模式已启动")
        print("输入命令来控制机器人，按 Enter 确认")
        print("当前轮速: 左=0.00, 右=0.00")
        print("-" * 50)
        
        while self.running:
            try:
                key = input("\n🕹️  输入命令 (w/s/a/d/q/e/space/r/i/h/esc): ").strip().lower()
                
                if key == 'esc' or key == 'quit' or key == 'exit':
                    print("👋 退出程序...")
                    self.running = False
                elif key:
                    # 处理命令
                    old_left = self.left_wheel_percent
                    old_right = self.right_wheel_percent
                    
                    self._process_console_key(key)
                    
                    # 只有轮速发生变化时才发送命令
                    if (self.left_wheel_percent != old_left or 
                        self.right_wheel_percent != old_right or 
                        key in ['w', 's', 'a', 'd', 'q', 'e', 'space', 'r']):
                        
                        success = self.send_wheel_command(self.left_wheel_percent, self.right_wheel_percent)
                        status = "✅ 命令发送成功" if success else "❌ 命令发送失败"
                        print(f"{status} - 当前轮速: 左={self.left_wheel_percent:.2f}, 右={self.right_wheel_percent:.2f}")
                        
            except KeyboardInterrupt:
                print("\n👋 程序被用户中断")
                self.running = False
            except EOFError:
                print("\n👋 输入结束，退出程序")
                self.running = False
    
    def _handle_realtime_control(self):
        """实时控制模式 - 支持持续运动"""
        print("\n🚀 实时控制模式已启动")
        print("输入命令来控制机器人，支持持续运动")
        print("当前轮速: 左=0.00, 右=0.00")
        print("-" * 60)
        print("💡 提示: 输入命令后机器人会持续运动，直到输入新命令")
        print("💡 输入 'stop' 或 'space' 停止运动")
        print("-" * 60)
        
        # 启动持续控制线程
        control_thread = threading.Thread(target=self._continuous_control_loop, daemon=True)
        control_thread.start()
        
        # 主输入循环
        while self.running:
            try:
                key = input("\n🕹️  输入命令 (w/s/a/d/q/e/stop/r/i/h/esc): ").strip().lower()
                
                if key == 'esc' or key == 'quit' or key == 'exit':
                    print("👋 退出程序...")
                    self.running = False
                    break
                elif key:
                    with self.control_lock:
                        old_left = self.left_wheel_percent
                        old_right = self.right_wheel_percent
                        
                        self._process_console_key(key)
                        
                        # 显示状态变化
                        if (self.left_wheel_percent != old_left or 
                            self.right_wheel_percent != old_right):
                            print(f"🎯 轮速更新: 左={self.left_wheel_percent:.2f}, 右={self.right_wheel_percent:.2f}")
                            
                            # 如果有运动，激活持续控制
                            if self.left_wheel_percent > 0 or self.right_wheel_percent > 0:
                                self.control_active = True
                                print("▶️  机器人开始持续运动...")
                            else:
                                self.control_active = False
                                print("⏹️  机器人停止运动")
                        
            except KeyboardInterrupt:
                print("\n👋 程序被用户中断")
                self.running = False
                break
            except EOFError:
                print("\n👋 输入结束，退出程序")
                self.running = False
                break
    
    def _continuous_control_loop(self):
        """持续控制循环 - 在后台线程中运行"""
        print("🔄 持续控制线程已启动")
        
        while self.running:
            try:
                with self.control_lock:
                    if self.control_active and self.connected:
                        # 持续发送当前轮速命令
                        self.send_wheel_command(self.left_wheel_percent, self.right_wheel_percent)
                
                # 控制频率 - 10Hz
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 持续控制循环错误: {e}")
                time.sleep(0.5)
        
        print("🔄 持续控制线程已退出")
    
    def _process_key(self, key):
        """处理pygame按键"""
        # 移动控制
        if key == pygame.K_w or key == pygame.K_UP:
            self.left_wheel_percent = min(1.0, self.left_wheel_percent + self.speed_increment)
            self.right_wheel_percent = min(1.0, self.right_wheel_percent + self.speed_increment)
        elif key == pygame.K_s or key == pygame.K_DOWN:
            self.left_wheel_percent = max(0.0, self.left_wheel_percent - self.speed_increment)
            self.right_wheel_percent = max(0.0, self.right_wheel_percent - self.speed_increment)
        elif key == pygame.K_a or key == pygame.K_LEFT:
            self.left_wheel_percent = max(0.0, self.left_wheel_percent - self.speed_increment * 0.5)
            self.right_wheel_percent = min(1.0, self.right_wheel_percent + self.speed_increment * 0.5)
        elif key == pygame.K_d or key == pygame.K_RIGHT:
            self.left_wheel_percent = min(1.0, self.left_wheel_percent + self.speed_increment * 0.5)
            self.right_wheel_percent = max(0.0, self.right_wheel_percent - self.speed_increment * 0.5)
        elif key == pygame.K_q:
            self.right_wheel_percent = min(1.0, self.right_wheel_percent + self.speed_increment)
        elif key == pygame.K_e:
            self.left_wheel_percent = min(1.0, self.left_wheel_percent + self.speed_increment)
        
        # 控制命令
        elif key == pygame.K_SPACE:
            self.left_wheel_percent = 0.0
            self.right_wheel_percent = 0.0
        elif key == pygame.K_r:
            self.left_wheel_percent = 0.0
            self.right_wheel_percent = 0.0
        elif key == pygame.K_i:
            self.capture_and_ocr()
        elif key == pygame.K_h:
            self._show_help()
        elif key == pygame.K_ESCAPE:
            self.running = False
    
    def _process_console_key(self, key):
        """处理控制台按键"""
        if key == 'w':
            self.left_wheel_percent = min(1.0, self.left_wheel_percent + self.speed_increment)
            self.right_wheel_percent = min(1.0, self.right_wheel_percent + self.speed_increment)
        elif key == 's':
            self.left_wheel_percent = max(0.0, self.left_wheel_percent - self.speed_increment)
            self.right_wheel_percent = max(0.0, self.right_wheel_percent - self.speed_increment)
        elif key == 'a':
            self.left_wheel_percent = max(0.0, self.left_wheel_percent - self.speed_increment * 0.5)
            self.right_wheel_percent = min(1.0, self.right_wheel_percent + self.speed_increment * 0.5)
        elif key == 'd':
            self.left_wheel_percent = min(1.0, self.left_wheel_percent + self.speed_increment * 0.5)
            self.right_wheel_percent = max(0.0, self.right_wheel_percent - self.speed_increment * 0.5)
        elif key == 'q':
            self.right_wheel_percent = min(1.0, self.right_wheel_percent + self.speed_increment)
        elif key == 'e':
            self.left_wheel_percent = min(1.0, self.left_wheel_percent + self.speed_increment)
        elif key == 'space' or key == 'stop':
            self.left_wheel_percent = 0.0
            self.right_wheel_percent = 0.0
        elif key == 'r':
            self.left_wheel_percent = 0.0
            self.right_wheel_percent = 0.0
        elif key == 'i':
            self.capture_and_ocr()
        elif key == 'h':
            self._show_help()
    
    def _show_help(self):
        """显示帮助信息"""
        print("\n" + "="*50)
        print("ROSbot简化键盘控制帮助")
        print("="*50)
        print("移动控制:")
        print("  W/↑    - 前进")
        print("  S/↓    - 后退")
        print("  A/←    - 左转")
        print("  D/→    - 右转")
        print("  Q      - 增加右轮速度")
        print("  E      - 增加左轮速度")
        print("\n其他:")
        print("  空格   - 紧急停止")
        print("  R      - 重置速度")
        print("  I      - 摄像头OCR识别")
        print("  H      - 显示帮助")
        print("  ESC    - 退出程序")
        print("="*50 + "\n")
    
    def run(self):
        """运行控制器"""
        if not self.connected:
            print("❌ 未连接到Webots，退出程序")
            return
        
        print("\n✅ 键盘控制器初始化完成")
        self._show_help()
        
        try:
            self.handle_keyboard_input()
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            print("正在退出...")
            if self.robot:
                # 停止所有电机
                self.send_wheel_command(0.0, 0.0)
            if PYGAME_AVAILABLE:
                pygame.quit()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化的ROSbot键盘控制程序")
    parser.add_argument("--webots-url", type=str, default="tcp://localhost:1236",
                        help="Webots外部控制器URL (默认: tcp://localhost:1236)")
    
    args = parser.parse_args()
    
    # 创建并运行控制器
    controller = SimpleKeyboardController(webots_url=args.webots_url)
    controller.run()

if __name__ == "__main__":
    main()
