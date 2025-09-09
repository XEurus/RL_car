#!/usr/bin/env python3
"""
测试修改后的导航任务设置
"""

import sys
import os
import numpy as np
from collections import Counter

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    # 导入导航工具类
    from src.utils.navigation_utils import NavigationUtils
    print("成功导入NavigationUtils")
    
    # 创建导航工具实例
    nav_utils = NavigationUtils()
    print("成功创建NavigationUtils实例")
    
    # 打印固定位置坐标
    print("\n===== 固定位置坐标 =====")
    for pos_name, pos_coords in nav_utils.fixed_positions.items():
        print(f"{pos_name}: {pos_coords}")
    
    # 测试基础模型任务（混合训练）
    print("\n===== 测试基础模型任务（混合训练）=====")
    task_counts = Counter()
    for i in range(100):
        start_pos, target_pos = nav_utils.get_navigation_task('normal', task_stage='base')
        
        # 确定任务类型
        if np.array_equal(start_pos, nav_utils.fixed_positions['start']):
            if np.array_equal(target_pos, nav_utils.fixed_positions['dangerous']):
                task_type = "起点->危险点"
            elif np.array_equal(target_pos, nav_utils.fixed_positions['fragile']):
                task_type = "起点->易碎点"
            elif np.array_equal(target_pos, nav_utils.fixed_positions['normal']):
                task_type = "起点->普通点"
            else:
                task_type = "未知任务"
        elif np.array_equal(start_pos, nav_utils.fixed_positions['normal']):
            if np.array_equal(target_pos, nav_utils.fixed_positions['unload']):
                task_type = "普通点->卸货点"
            else:
                task_type = "未知任务"
        else:
            task_type = "未知任务"
        
        task_counts[task_type] += 1
    
    # 打印统计结果
    print("基础模型任务分布（100次采样）:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count}次")
    
    # 测试特定任务
    print("\n===== 测试特定任务 =====")
    
    # 危险货物到卸货点
    start_pos, target_pos = nav_utils.get_navigation_task('dangerous', task_stage='dangerous_to_unload')
    print(f"危险货物到卸货点: 起点={start_pos}, 终点={target_pos}")
    print(f"  起点是危险点: {np.array_equal(start_pos, nav_utils.fixed_positions['dangerous'])}")
    print(f"  终点是卸货点: {np.array_equal(target_pos, nav_utils.fixed_positions['unload'])}")
    
    # 易碎货物到卸货点
    start_pos, target_pos = nav_utils.get_navigation_task('fragile', task_stage='fragile_to_unload')
    print(f"易碎货物到卸货点: 起点={start_pos}, 终点={target_pos}")
    print(f"  起点是易碎点: {np.array_equal(start_pos, nav_utils.fixed_positions['fragile'])}")
    print(f"  终点是卸货点: {np.array_equal(target_pos, nav_utils.fixed_positions['unload'])}")
    
    # 普通货物到卸货点
    start_pos, target_pos = nav_utils.get_navigation_task('normal', task_stage='normal_to_unload')
    print(f"普通货物到卸货点: 起点={start_pos}, 终点={target_pos}")
    print(f"  起点是普通点: {np.array_equal(start_pos, nav_utils.fixed_positions['normal'])}")
    print(f"  终点是卸货点: {np.array_equal(target_pos, nav_utils.fixed_positions['unload'])}")
    
    print("\n测试完成!")
    
except ImportError as e:
    print(f"导入失败: {e}")
except Exception as e:
    print(f"测试失败: {e}")
