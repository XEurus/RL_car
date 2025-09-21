#!/usr/bin/env python3
"""
ROS图像处理服务器示例
接收图像处理请求并返回处理结果
"""

import sys
import os
import time
import threading
from typing import Dict, Any
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import rospy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Point
    from std_srvs.srv import Empty, EmptyResponse
    from cv_bridge import CvBridge
    import cv2
    ROS_AVAILABLE = True
except ImportError:
    print("ROS not available")
    ROS_AVAILABLE = False
    sys.exit(1)

class OCRProcessingServer:
    """OCR图像识别服务器类"""
    
    def __init__(self, service_name: str = "/image_processing_service"):
        """
        初始化OCR图像识别服务器
        
        Args:
            service_name: ROS服务名称
        """
        self.service_name = service_name
        self.bridge = CvBridge()
        
        # 初始化ROS节点
        rospy.init_node('ocr_processing_server', anonymous=True)
        
        # 创建服务服务器（使用Empty服务作为示例，实际应该使用自定义服务）
        self.service_server = rospy.Service(
            self.service_name, 
            Empty, 
            self.handle_ocr_processing_request
        )
        
        print(f"OCR识别服务器已启动: {self.service_name}")
        print("等待OCR识别请求...")
    
    def handle_ocr_processing_request(self, req) -> EmptyResponse:
        """
        处理OCR识别请求
        
        Args:
            req: 服务请求
            
        Returns:
            服务响应
        """
        print("收到OCR识别请求")
        
        try:
            # 模拟OCR识别处理
            result = self.process_ocr_simulation()
            
            print(f"OCR识别完成: {result['message']}")
            print(f"识别文字: {result['total_text']}")
            print(f"平均置信度: {np.mean(result['confidences']):.2f}")
            
            # 返回空响应（实际应该返回自定义响应）
            return EmptyResponse()
            
        except Exception as e:
            print(f"OCR识别失败: {e}")
            return EmptyResponse()
    
    def process_ocr_simulation(self) -> Dict[str, Any]:
        """
        模拟OCR识别处理
        
        Returns:
            OCR识别结果
        """
        # 模拟OCR处理延迟
        time.sleep(0.5)
        
        # 模拟不同的OCR识别结果
        import random
        
        ocr_results = [
            {
                'message': '中文OCR识别完成',
                'texts': ['欢迎', '使用', 'ROSbot', '机器人'],
                'confidences': [0.96, 0.94, 0.92, 0.89],
                'total_text': '欢迎使用ROSbot机器人'
            },
            {
                'message': '英文OCR识别完成',
                'texts': ['Welcome', 'to', 'Robot', 'Control'],
                'confidences': [0.98, 0.97, 0.95, 0.93],
                'total_text': 'Welcome to Robot Control'
            },
            {
                'message': '中英文混合OCR识别完成',
                'texts': ['Hello', '你好', 'Robot', '控制系统'],
                'confidences': [0.97, 0.95, 0.94, 0.91],
                'total_text': 'Hello你好Robot控制系统'
            }
        ]
        
        result = random.choice(ocr_results)
        
        # 模拟边界框坐标
        bboxes = []
        for i, text in enumerate(result['texts']):
            x = float(i * 120 + 50)
            y = float(i * 40 + 30)
            bboxes.append({'x': x, 'y': y, 'z': 0.0})
        
        result['success'] = True
        result['bboxes'] = bboxes
        
        return result
    
    def run(self):
        """运行服务器"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("图像处理服务器关闭")


def main():
    """主函数"""
    if not ROS_AVAILABLE:
        print("ROS不可用，无法启动OCR识别服务器")
        return
    
    try:
        server = OCRProcessingServer()
        server.run()
    except Exception as e:
        print(f"启动OCR识别服务器失败: {e}")


if __name__ == '__main__':
    main()
