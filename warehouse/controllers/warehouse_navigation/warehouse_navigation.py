#!/usr/bin/env python3
"""
Warehouse导航控制器
作为Supervisor控制器运行，用于强化学习训练
"""

import os
import sys
from pathlib import Path

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到Python路径
project_root = str(Path(current_dir).parent.parent.parent)
sys.path.append(project_root)

# 导入环境类
from warehouse_navigation.main import main

# 运行主函数
if __name__ == "__main__":
    # 创建控制器日志目录
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 记录启动信息
    with open(os.path.join(log_dir, "controller_log.txt"), "a") as f:
        f.write("Warehouse导航控制器启动\n")
        f.write(f"工作目录: {os.getcwd()}\n")
        f.write(f"Python路径: {sys.path}\n")
    
    try:
        # 运行主程序
        main()
    except Exception as e:
        # 记录错误
        with open(os.path.join(log_dir, "controller_error.txt"), "a") as f:
            f.write(f"错误: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
