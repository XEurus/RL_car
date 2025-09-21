#!/usr/bin/env python3
"""
简化的OCR服务客户端
只支持Webots Supervisor直接读取摄像头并调用OCR服务
"""

import numpy as np
from typing import Dict, List, Optional, Any

# 检查ROS可用性
try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    print("ROS not available, using mock OCR service")
    ROS_AVAILABLE = False

# 检查Webots可用性
try:
    from controller import Supervisor, Camera
    WEBOTS_AVAILABLE = True
except ImportError:
    print("Webots not available")
    WEBOTS_AVAILABLE = False

class SimpleOCRClient:
    """简化的OCR服务客户端"""
    
    def __init__(self, 
                 service_name: str = "/ocr_service",
                 webots_camera_name: str = "camera",
                 robot_instance=None):
        """
        初始化简化的OCR客户端
        
        Args:
            service_name: OCR服务名称
            webots_camera_name: Webots摄像头设备名称
            robot_instance: 已存在的Robot实例（可选）
        """
        self.service_name = service_name
        self.webots_camera_name = webots_camera_name
        
        # ROS服务客户端
        self.service_client = None
        self.ros_initialized = False
        
        # Webots摄像头
        self.robot = robot_instance
        self.camera = None
        self.webots_initialized = False
        
        # CV Bridge（用于图像转换）
        if ROS_AVAILABLE:
            self.bridge = CvBridge()
        
        # 初始化
        if ROS_AVAILABLE:
            self._init_ros()
        if WEBOTS_AVAILABLE:
            self._init_webots()
    
    def _init_ros(self):
        """初始化ROS服务客户端"""
        try:
            # 检查ROS是否运行
            if not rospy.get_node_uri():
                rospy.init_node('ocr_client', anonymous=True)
            
            # 等待OCR服务可用
            print(f"等待OCR服务: {self.service_name}")
            rospy.wait_for_service(self.service_name, timeout=5.0)
            
            # 创建服务客户端
            from srv.ImageProcessing import ImageProcessing
            self.service_client = rospy.ServiceProxy(self.service_name, ImageProcessing)
            
            self.ros_initialized = True
            print("✅ ROS OCR服务连接成功")
            
        except Exception as e:
            print(f"❌ ROS OCR服务初始化失败: {e}")
            self.ros_initialized = False
    
    def _init_webots(self):
        """初始化Webots摄像头"""
        try:
            # 如果没有传入Robot实例，尝试创建新的
            if self.robot is None:
                print("创建新的Supervisor实例...")
                self.robot = Supervisor()
            else:
                print("使用已存在的Robot实例...")
            
            # 获取摄像头设备
            self.camera = self.robot.getDevice(self.webots_camera_name)
            if self.camera:
                # 启用摄像头
                timestep = int(self.robot.getBasicTimeStep())
                self.camera.enable(timestep)
                
                # 等待摄像头准备就绪
                self.robot.step(timestep)
                
                self.webots_initialized = True
                print(f"✅ Webots摄像头初始化成功: {self.webots_camera_name}")
            else:
                print(f"❌ 未找到Webots摄像头设备: {self.webots_camera_name}")
                self.webots_initialized = False
                
        except Exception as e:
            print(f"❌ Webots摄像头初始化失败: {e}")
            self.webots_initialized = False
    
    def _capture_webots_image(self) -> Optional[np.ndarray]:
        """从Webots摄像头捕获图像"""
        if not self.webots_initialized or not self.camera:
            return None
        
        try:
            # 执行一步仿真以更新摄像头
            self.robot.step(1)
            
            # 获取图像数据
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            image_data = self.camera.getImage()
            
            if image_data:
                # 转换为numpy数组
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image_array = image_array.reshape((height, width, 4))  # BGRA格式
                
                # 转换为RGB格式
                image_rgb = image_array[:, :, [2, 1, 0]]  # BGR -> RGB
                
                return image_rgb
            else:
                print("❌ 无法获取摄像头图像数据")
                return None
                
        except Exception as e:
            print(f"❌ 摄像头图像捕获失败: {e}")
            return None
    
    def _create_ros_image_msg(self, image: np.ndarray) -> Optional[Any]:
        """将numpy图像转换为ROS Image消息"""
        if not ROS_AVAILABLE or not hasattr(self, 'bridge'):
            return None
        
        try:
            # 转换为ROS Image消息
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            return ros_image
        except Exception as e:
            print(f"❌ 图像转换失败: {e}")
            return None
    
    def call_ocr_service(self, 
                        language: str = "ch", 
                        use_angle_cls: bool = True) -> Optional[Dict[str, Any]]:
        """
        调用OCR服务
        
        Args:
            language: 识别语言 ("ch", "en", "ch_en")
            use_angle_cls: 是否使用角度分类器
            
        Returns:
            OCR识别结果字典
        """
        # 捕获图像
        image = self._capture_webots_image()
        if image is None:
            print("❌ 无法捕获摄像头图像")
            return None
        
        print(f"📸 已捕获图像: {image.shape}")
        
        # 如果ROS可用，调用真实OCR服务
        if self.ros_initialized and self.service_client:
            return self._call_real_ocr_service(image, language, use_angle_cls)
        else:
            # 使用模拟OCR服务
            return self._call_mock_ocr_service(image, language, use_angle_cls)
    
    def _call_real_ocr_service(self, 
                              image: np.ndarray, 
                              language: str, 
                              use_angle_cls: bool) -> Optional[Dict[str, Any]]:
        """调用真实的ROS OCR服务"""
        try:
            # 转换为ROS Image消息
            ros_image = self._create_ros_image_msg(image)
            if ros_image is None:
                return None
            
            # 创建服务请求
            from srv.ImageProcessing import ImageProcessingRequest
            request = ImageProcessingRequest()
            request.input_image = ros_image
            request.language = language
            request.use_angle_cls = use_angle_cls
            
            # 调用服务
            print("🔍 正在调用OCR服务...")
            response = self.service_client(request)
            
            # 解析响应
            if response.success:
                result = {
                    'success': True,
                    'message': response.message,
                    'texts': response.texts,
                    'confidences': response.confidences,
                    'bboxes': [{'x': bbox.x, 'y': bbox.y, 'z': bbox.z} for bbox in response.bboxes],
                    'total_text': response.total_text
                }
                return result
            else:
                print(f"❌ OCR服务返回错误: {response.message}")
                return None
                
        except Exception as e:
            print(f"❌ OCR服务调用失败: {e}")
            return None
    
    def _call_mock_ocr_service(self, 
                              image: np.ndarray, 
                              language: str, 
                              use_angle_cls: bool) -> Dict[str, Any]:
        """模拟OCR服务（当ROS不可用时）"""
        print("🔍 使用模拟OCR服务...")
        
        # 模拟识别结果
        if language == "ch":
            texts = ["模拟中文文本", "测试识别"]
            message = "模拟中文OCR识别完成"
        elif language == "en":
            texts = ["Mock English Text", "Test Recognition"]
            message = "Mock English OCR recognition completed"
        else:  # ch_en
            texts = ["Mixed Text 混合文本", "Test 测试"]
            message = "Mock mixed language OCR recognition completed"
        
        confidences = [0.95, 0.88]
        total_text = " ".join(texts)
        
        # 模拟边界框
        bboxes = []
        for i, text in enumerate(texts):
            x = float(i * 120 + 50)
            y = float(i * 40 + 30)
            bboxes.append({'x': x, 'y': y, 'z': 0.0})
        
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
        return self.webots_initialized and (self.ros_initialized or True)  # 模拟服务总是可用
    
    def get_status(self) -> Dict[str, bool]:
        """获取服务状态"""
        return {
            'ros_initialized': self.ros_initialized,
            'webots_initialized': self.webots_initialized,
            'camera_available': self.webots_initialized and self.camera is not None,
            'service_available': self.ros_initialized or True  # 模拟服务总是可用
        }

def main():
    """测试OCR客户端"""
    print("测试简化OCR客户端...")
    
    client = SimpleOCRClient()
    
    if client.is_available():
        print("OCR客户端可用，测试识别...")
        result = client.call_ocr_service(language="ch", use_angle_cls=True)
        
        if result:
            print("识别结果:")
            print(f"  成功: {result['success']}")
            print(f"  消息: {result['message']}")
            print(f"  总文本: {result['total_text']}")
            print(f"  文本列表: {result['texts']}")
        else:
            print("识别失败")
    else:
        print("OCR客户端不可用")

if __name__ == "__main__":
    main()
