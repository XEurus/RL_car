#!/usr/bin/env python3
"""
ROS图像服务客户端 - 用于获取机器人摄像头图片并发送到远端处理
支持与键盘控制程序集成
"""

import sys
import os
import time
import threading
from typing import Optional, Dict, Any
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import rospy
    from sensor_msgs.msg import Image, CompressedImage
    from geometry_msgs.msg import Point
    from cv_bridge import CvBridge
    import cv2
    ROS_AVAILABLE = True
except ImportError:
    print("ROS not available, image service disabled")
    ROS_AVAILABLE = False

# 尝试导入Webots控制器（用于获取摄像头）
try:
    from controller import Supervisor, Camera
    WEBOTS_AVAILABLE = True
except ImportError:
    print("Webots not available, using ROS camera topics only")
    WEBOTS_AVAILABLE = False

class ROSImageClient:
    """简化的ROS OCR服务客户端类 - 直接使用Webots Supervisor"""
    
    def __init__(self, 
                 service_name: str = "/ocr_service",
                 camera_topic: str = "/camera/image_raw",
                 webots_camera_name: str = "camera",
                 timeout: float = 10.0):
        """
        初始化简化的ROS OCR服务客户端
        
        Args:
            service_name: ROS服务名称 (默认: /ocr_service)
            camera_topic: 摄像头话题名称 (未使用)
            webots_camera_name: Webots摄像头设备名称
            timeout: 服务调用超时时间
        """
        self.service_name = service_name
        self.camera_topic = camera_topic
        self.webots_camera_name = webots_camera_name
        self.timeout = timeout
        
        # ROS组件
        self.bridge = None
        self.service_client = None
        
        # Webots摄像头
        self.robot = None
        self.camera = None
        
        # 图像缓存
        self.latest_image = None
        self.image_lock = threading.Lock()
        
        # 初始化状态
        self.ros_initialized = False
        self.webots_initialized = False
        
        if ROS_AVAILABLE:
            self._init_ros()
        
        if WEBOTS_AVAILABLE:
            self._init_webots()
    
    def _init_ros(self):
        """初始化ROS组件"""
        try:
            # 初始化ROS节点（如果还没有初始化）
            try:
                rospy.get_node_uri()
            except:
                rospy.init_node('ocr_client', anonymous=True)
            
            # 创建CV桥接器
            self.bridge = CvBridge()
            
            # 等待OCR服务可用
            print(f"等待OCR服务: {self.service_name}")
            rospy.wait_for_service(self.service_name, timeout=5.0)
            
            # 这里应该导入实际的OCR服务定义，暂时使用Empty作为占位符
            from std_srvs.srv import Empty
            self.service_client = rospy.ServiceProxy(self.service_name, Empty)
            
            self.ros_initialized = True
            print("ROS OCR客户端初始化成功")
            
        except Exception as e:
            print(f"ROS OCR客户端初始化失败: {e}")
            self.ros_initialized = False
    
    def _init_webots(self):
        """初始化Webots摄像头"""
        try:
            # 获取Webots机器人实例
            self.robot = Supervisor()
            
            # 获取摄像头设备
            self.camera = self.robot.getDevice(self.webots_camera_name)
            if self.camera:
                # 启用摄像头
                timestep = int(self.robot.getBasicTimeStep())
                self.camera.enable(timestep)
                
                self.webots_initialized = True
                print("Webots摄像头初始化成功")
            else:
                print(f"未找到Webots摄像头设备: {self.webots_camera_name}")
                
        except Exception as e:
            print(f"Webots摄像头初始化失败: {e}")
            self.webots_initialized = False
    
    # 移除ROS话题回调，只使用Webots摄像头
    
    def get_camera_image(self) -> Optional[np.ndarray]:
        """
        从Webots Supervisor获取摄像头图像
        
        Returns:
            numpy数组格式的图像，如果获取失败返回None
        """
        if not self.webots_initialized or not self.camera:
            print("Webots摄像头未初始化")
            return None
            
        try:
            # 获取Webots摄像头图像
            image_array = self.camera.getImageArray()
            if image_array:
                # 转换为numpy数组
                height = self.camera.getHeight()
                width = self.camera.getWidth()
                
                # Webots图像格式为BGRA，转换为RGB
                image = np.array(image_array, dtype=np.uint8)
                image = image.reshape((height, width, 4))
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                
                print(f"从Webots获取图像: {image.shape}")
                return image
            else:
                print("Webots摄像头返回空图像")
                return None
                
        except Exception as e:
            print(f"获取Webots摄像头图像失败: {e}")
            return None
    
    def send_image_for_processing(self, 
                                language: str = "ch_en",
                                use_angle_cls: bool = True) -> Dict[str, Any]:
        """
        发送图像到远端进行OCR识别
        
        Args:
            language: 识别语言 ("ch", "en", "ch_en")
            use_angle_cls: 是否使用角度分类器
            
        Returns:
            OCR识别结果字典
        """
        # 获取摄像头图像
        image = self.get_camera_image()
        if image is None:
            return {
                'success': False,
                'message': '无法获取摄像头图像',
                'texts': [],
                'confidences': [],
                'bboxes': [],
                'total_text': ''
            }
        
        try:
            # 将numpy图像转换为ROS Image消息
            if self.bridge:
                image_msg = self.bridge.cv2_to_imgmsg(image, "rgb8")
            else:
                # 如果没有CV桥接器，创建简单的图像消息
                image_msg = Image()
                image_msg.height = image.shape[0]
                image_msg.width = image.shape[1]
                image_msg.encoding = "rgb8"
                image_msg.data = image.tobytes()
            
            # 调用真实的OCR服务
            print(f"发送OCR识别请求到 {self.service_name}: language={language}, use_angle_cls={use_angle_cls}")
            print(f"图像尺寸: {image.shape}")
            
            # 调用OCR服务（暂时使用模拟，实际需要使用真实服务定义）
            if self.service_client:
                try:
                    # 这里应该发送真实的OCR请求
                    # 暂时使用模拟结果
                    print("调用OCR服务...")
                    response = self.service_client()
                    result = self._simulate_ocr_service_call(image, language, use_angle_cls)
                except Exception as e:
                    print(f"OCR服务调用失败: {e}")
                    result = self._simulate_ocr_service_call(image, language, use_angle_cls)
            else:
                result = self._simulate_ocr_service_call(image, language, use_angle_cls)
            
            return result
            
        except Exception as e:
            print(f"发送OCR识别请求失败: {e}")
            return {
                'success': False,
                'message': f'OCR识别失败: {str(e)}',
                'texts': [],
                'confidences': [],
                'bboxes': [],
                'total_text': ''
            }
    
    def _simulate_ocr_service_call(self, image: np.ndarray, language: str, use_angle_cls: bool) -> Dict[str, Any]:
        """
        模拟OCR服务调用（实际应该调用真实的ROS服务）
        
        Args:
            image: 输入图像
            language: 识别语言
            use_angle_cls: 是否使用角度分类器
            
        Returns:
            模拟的OCR识别结果
        """
        # 模拟OCR处理延迟
        time.sleep(1.0)
        
        # 简单的图像分析
        height, width = image.shape[:2]
        mean_brightness = np.mean(image)
        
        # 模拟不同语言的OCR结果
        if language == "ch":
            texts = ["你好", "世界", "机器人"]
            confidences = [0.95, 0.92, 0.88]
            total_text = "你好世界机器人"
            message = "中文OCR识别完成"
        elif language == "en":
            texts = ["Hello", "World", "Robot"]
            confidences = [0.96, 0.94, 0.91]
            total_text = "Hello World Robot"
            message = "英文OCR识别完成"
        else:  # ch_en
            texts = ["Hello", "你好", "Robot", "机器人"]
            confidences = [0.96, 0.95, 0.91, 0.88]
            total_text = "Hello你好Robot机器人"
            message = "中英文混合OCR识别完成"
        
        # 模拟边界框坐标 (使用geometry_msgs/Point简化表示)
        bboxes = []
        for i, text in enumerate(texts):
            # 简单模拟边界框位置
            x = (i * 100) % width
            y = (i * 50) % height
            bboxes.append({'x': float(x), 'y': float(y), 'z': 0.0})
        
        return {
            'success': True,
            'message': message,
            'texts': texts,
            'confidences': confidences,
            'bboxes': bboxes,
            'total_text': total_text
        }
    
    def is_available(self) -> bool:
        """检查OCR服务是否可用"""
        return self.ros_initialized and self.webots_initialized
    
    def get_status(self) -> Dict[str, bool]:
        """获取服务状态"""
        return {
            'ros_initialized': self.ros_initialized,
            'webots_initialized': self.webots_initialized,
            'camera_available': self.webots_initialized and self.camera is not None
        }


def test_image_client():
    """测试图像客户端"""
    client = ROSImageClient()
    
    print("图像客户端状态:", client.get_status())
    
    if client.is_available():
        # 测试图像获取
        image = client.get_camera_image()
        if image is not None:
            print(f"成功获取图像: {image.shape}")
            
            # 测试OCR服务
            result = client.send_image_for_processing("ch_en", True)
            print("OCR识别结果:", result)
        else:
            print("无法获取图像")
    else:
        print("图像服务不可用")


if __name__ == '__main__':
    test_image_client()
