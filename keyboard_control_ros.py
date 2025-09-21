#!/usr/bin/env python3
"""
ROSbot键盘控制程序 - 完全通过ROS服务控制
"""

import sys
import rospy
import threading
import time
from webots_ros.srv import set_float, set_floatRequest, set_int, set_intRequest
from sensor_msgs.msg import Image

# 尝试导入我们自定义的OCR服务
try:
    from srv import ImageProcessing, ImageProcessingRequest
    CUSTOM_SRV_AVAILABLE = True
except ImportError:
    print("警告: 自定义的ImageProcessing.srv未找到或编译。OCR功能将不可用。")
    print("请确保在Catkin工作空间的根目录运行 'catkin_make'。")
    CUSTOM_SRV_AVAILABLE = False

class ROSKeyboardController:
    """通过ROS服务控制机器人的键盘控制器"""

    def __init__(self):
        # 控制参数
        self.max_motor_speed = 10.0  # 电机最大速度 (rad/s) - 需要根据实际情况调整
        self.speed_increment = 1.0   # 速度增量
        self.turn_factor = 0.5       # 转向系数

        # 当前轮速 (rad/s)
        self.left_speed = 0.0
        self.right_speed = 0.0

        # ROS
        self.motor_services = {}
        self.camera_rgb_enable_service = None
        self.camera_depth_enable_service = None
        self.ocr_service_client = None
        self.image_subscriber = None
        self.last_image_msg = None

        # 状态
        self.running = True
        self.control_active = False
        self.control_lock = threading.Lock()

        self._init_ros()

    def _init_ros(self):
        """初始化所有ROS节点、服务和话题"""
        print("正在初始化ROS...")
        try:
            rospy.init_node('ros_keyboard_controller', anonymous=True)

            # 初始化电机服务
            motor_names = ['fl_wheel_joint', 'fr_wheel_joint', 'rl_wheel_joint', 'rr_wheel_joint']
            for name in motor_names:
                service_name = f"/{name}/set_velocity"
                rospy.wait_for_service(service_name, timeout=3.0)
                self.motor_services[name] = rospy.ServiceProxy(service_name, set_float)
            print("✅ 所有电机服务已连接")

            # 初始化摄像头服务
            rgb_cam_srv = "/camera_rgb/enable"
            depth_cam_srv = "/camera_depth/enable"
            rospy.wait_for_service(rgb_cam_srv, timeout=3.0)
            rospy.wait_for_service(depth_cam_srv, timeout=3.0)
            self.camera_rgb_enable_service = rospy.ServiceProxy(rgb_cam_srv, set_int)
            self.camera_depth_enable_service = rospy.ServiceProxy(depth_cam_srv, set_int)
            print(f"✅ 摄像头服务 '{rgb_cam_srv}' 和 '{depth_cam_srv}' 已连接")

            # 启用摄像头
            self._enable_camera()

            # 订阅图像话题
            image_topic = "/camera_rgb/image"
            self.image_subscriber = rospy.Subscriber(image_topic, Image, self._image_callback)
            print(f"✅ 已订阅图像话题 '{image_topic}'")

            # 初始化OCR服务客户端
            if CUSTOM_SRV_AVAILABLE:
                ocr_srv_name = "/ocr_service"
                rospy.wait_for_service(ocr_srv_name, timeout=3.0)
                self.ocr_service_client = rospy.ServiceProxy(ocr_srv_name, ImageProcessing)
                print(f"✅ OCR服务 '{ocr_srv_name}' 已连接")

        except Exception as e:
            print(f"❌ ROS初始化或连接失败: {e}")
            self.running = False

    def _set_wheel_velocities(self, left_vel, right_vel):
        """通过ROS服务设置四个轮子的速度"""
        if not self.running or not self.motor_services:
            return False
        
        try:
            req = set_floatRequest()
            
            # 设置左轮速度
            req.value = left_vel
            self.motor_services['fl_wheel_joint'](req)
            self.motor_services['rl_wheel_joint'](req)

            # 设置右轮速度
            req.value = right_vel
            self.motor_services['fr_wheel_joint'](req)
            self.motor_services['rr_wheel_joint'](req)
            
            print(f"🔧 速度设置: 左={left_vel:.2f}, 右={right_vel:.2f} rad/s")
            return True
        except rospy.ServiceException as e:
            print(f"❌ 设置速度失败: {e}")
            return False

    def _enable_camera(self):
        """调用服务来启用RGB和深度摄像头"""
        if not self.camera_rgb_enable_service or not self.camera_depth_enable_service:
            print("摄像头服务尚未初始化。")
            return

        req = set_intRequest()
        req.value = 32  # 设置32ms的采样周期 (~30 FPS), 8的倍数

        # 启用深度摄像头 (根据您的要求，这将触发RGB图像话题)
        try:
            print(f"正在尝试启用深度摄像头 (/camera_depth/enable) 来获取RGB图像...")
            response_depth = self.camera_depth_enable_service(req)
            if response_depth.success:
                print("✅ 深度摄像头启用成功!")
            else:
                print("⚠️ 深度摄像头启用服务返回False，但这可能是正常的。")
        except rospy.ServiceException as e:
            print(f"❌ 调用深度摄像头启用服务失败: {e}")

        # 同时启用RGB摄像头以确保其处于活动状态
        try:
            print(f"正在尝试启用RGB摄像头 (/camera_rgb/enable)...")
            response_rgb = self.camera_rgb_enable_service(req)
            if response_rgb.success:
                print("✅ RGB摄像头启用成功!")
            else:
                print("⚠️ RGB摄像头启用服务返回False，但这可能是正常的。")
        except rospy.ServiceException as e:
            print(f"❌ 调用RGB摄像头启用服务失败: {e}")

    def _image_callback(self, msg):
        """图像话题的回调函数"""
        with self.control_lock:
            self.last_image_msg = msg
            # print("收到一张图像!") # 取消注释以进行调试

    def _call_ocr_service(self):
        """获取最新图像并调用OCR服务"""
        with self.control_lock:
            if self.last_image_msg is None:
                print("❌ 尚未收到任何图像。请确保摄像头已启用且话题正确。")
                return
            image_to_process = self.last_image_msg
        
        if not self.ocr_service_client:
            print("❌ OCR服务客户端不可用。")
            return

        print("🔍 正在调用OCR服务...")
        try:
            req = ImageProcessingRequest()
            req.input_image = image_to_process
            req.language = "ch"
            req.use_angle_cls = True
            
            res = self.ocr_service_client(req)
            
            if res.success:
                print("\n--- OCR识别结果 ---")
                print(f"  总文本: {res.total_text}")
                for i, text in enumerate(res.texts):
                    print(f"  - '{text}' (置信度: {res.confidences[i]:.2f})")
                print("---------------------")
            else:
                print(f" OCR服务返回失败: {res.message}")

        except rospy.ServiceException as e:
            print(f"❌ 调用OCR服务失败: {e}")

    def _process_key(self, key):
        """处理键盘输入，更新目标速度"""
        with self.control_lock:
            if key == 'w':
                self.left_speed = min(self.max_motor_speed, self.left_speed + self.speed_increment)
                self.right_speed = min(self.max_motor_speed, self.right_speed + self.speed_increment)
            elif key == 's':
                self.left_speed = max(-self.max_motor_speed, self.left_speed - self.speed_increment)
                self.right_speed = max(-self.max_motor_speed, self.right_speed - self.speed_increment)
            elif key == 'a':
                self.left_speed -= self.speed_increment * self.turn_factor
                self.right_speed += self.speed_increment * self.turn_factor
            elif key == 'd':
                self.left_speed += self.speed_increment * self.turn_factor
                self.right_speed -= self.speed_increment * self.turn_factor
            elif key == 'stop' or key == 'space':
                self.left_speed = 0.0
                self.right_speed = 0.0
            elif key == 'r':
                self.left_speed = 0.0
                self.right_speed = 0.0
            elif key == 'i':
                self._call_ocr_service()
            
            # 限制速度范围
            self.left_speed = max(-self.max_motor_speed, min(self.max_motor_speed, self.left_speed))
            self.right_speed = max(-self.max_motor_speed, min(self.max_motor_speed, self.right_speed))

            print(f"🎯 目标速度更新: 左={self.left_speed:.2f}, 右={self.right_speed:.2f}")
            self.control_active = True

    def _continuous_control_loop(self):
        """持续发送速度命令的循环"""
        print("🔄 持续控制线程已启动")
        rate = rospy.Rate(10) # 10 Hz
        while self.running and not rospy.is_shutdown():
            with self.control_lock:
                if self.control_active:
                    self._set_wheel_velocities(self.left_speed, self.right_speed)
            rate.sleep()
        print("🔄 持续控制线程已退出")

    def run(self):
        """运行主循环"""
        if not self.running:
            print("初始化失败，程序退出。")
            return

        print("\n🚀 ROS键盘控制器已启动")
        print("  - 输入 'w/a/s/d' 控制方向")
        print("  - 输入 'stop' 或 'space' 停止")
        print("  - 输入 'i' 进行OCR识别")
        print("  - 输入 'esc' 或 'quit' 退出")
        print("-"*60)

        control_thread = threading.Thread(target=self._continuous_control_loop, daemon=True)
        control_thread.start()

        while self.running and not rospy.is_shutdown():
            try:
                key = input("\n🕹️  输入命令: ").strip().lower()
                if key in ['esc', 'quit', 'exit']:
                    self.running = False
                    break
                elif key:
                    self._process_key(key)
            except (KeyboardInterrupt, EOFError):
                self.running = False
                break
        
        print("👋 正在退出...")
        self.running = False
        time.sleep(0.2)
        # 停止机器人
        self._set_wheel_velocities(0.0, 0.0)
        time.sleep(0.5)

if __name__ == "__main__":
    controller = ROSKeyboardController()
    controller.run()
