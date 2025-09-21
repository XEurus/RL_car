#!/bin/bash

# 实时局部地图查看器启动脚本

echo "启动实时局部地图查看器..."
echo "=================================="

# 设置默认参数
CARGO_TYPE=${1:-normal}
UPDATE_INTERVAL=${2:-3.0}
MAP_SIZE=${3:-200}
RESOLUTION=${4:-0.05}

echo "参数设置:"
echo "  货物类型: $CARGO_TYPE"
echo "  更新间隔: ${UPDATE_INTERVAL}秒"
echo "  地图大小: ${MAP_SIZE}x${MAP_SIZE}"
echo "  分辨率: ${RESOLUTION}m/pixel"
echo "=================================="

# 启动实时地图查看器
python3 /root/workspace/RL_car2/rosbot_navigation/scripts/realtime_map_viewer.py \
    --cargo_type $CARGO_TYPE \
    --update_interval $UPDATE_INTERVAL \
    --map_size $MAP_SIZE \
    --resolution $RESOLUTION \
    --max_range 10.0 \
    --scale 3

echo "实时地图查看器已退出"
