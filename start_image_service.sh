#!/bin/bash
"""
启动ROS OCR图像识别服务的脚本
"""

# 检查ROS环境
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS环境未设置，请先source ROS setup文件"
    echo "例如: source /opt/ros/noetic/setup.bash"
    exit 1
fi

echo "🚀 启动ROS OCR图像识别服务..."

# 启动roscore（如果还没有运行）
if ! pgrep -f "roscore" > /dev/null; then
    echo "启动roscore..."
    roscore &
    sleep 3
fi

# 启动OCR识别服务器
echo "启动OCR识别服务器..."
python3 image_processing_server.py &
IMAGE_SERVER_PID=$!

echo "✅ OCR识别服务已启动"
echo "服务器PID: $IMAGE_SERVER_PID"
echo ""
echo "现在可以运行键盘控制程序:"
echo "python3 keyboard_control.py --robot-ip <ROBOT_IP> --robot-port 8888"
echo ""
echo "按 Ctrl+C 停止服务"

# 等待中断信号
trap "echo '停止OCR识别服务...'; kill $IMAGE_SERVER_PID 2>/dev/null; exit" INT

wait $IMAGE_SERVER_PID
