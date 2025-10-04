"""
障碍物随机化功能 - 用于 navigation_env.py
"""
import random
import numpy as np


def _randomize_obstacles(self):
    """
    渐进式障碍物随机化
    - 根据全局训练步数决定障碍物数量（课程学习）
    - 在指定范围内随机分布障碍物位置
    - 避开起点、终点和机器人当前位置
    """
    try:
        # 1. 根据训练步数确定障碍物数量（课程学习）
        current_step = int(getattr(self, '_global_training_step', 0))
        num_obstacles = 0
        for i, step_threshold in enumerate(self.obstacle_curriculum_steps):
            if current_step >= step_threshold:
                num_obstacles = self.obstacle_curriculum_counts[i]
        
        # 限制在最大值
        num_obstacles = min(num_obstacles, self.max_obstacles)
        
        # 2. 初始化障碍物节点列表（仅第一次）
        if not self.obstacle_nodes:
            # 从场景中查找所有 WoodenBox
            root = self.supervisor.getRoot()
            children_field = root.getField('children')
            num_children = children_field.getCount()
            
            for i in range(num_children):
                node = children_field.getMFNode(i)
                if node and node.getTypeName() == 'WoodenBox':
                    self.obstacle_nodes.append(node)
            
            if self.obstacle_nodes:
                print(f"[Obstacle] 找到 {len(self.obstacle_nodes)} 个 WoodenBox 障碍物")
            else:
                print(f"[Obstacle] 警告：未找到任何 WoodenBox 障碍物")
                return
        
        # 3. 定义安全区域（起点、终点周围）
        safe_radius = 0.8  # 安全半径（米）
        safe_zones = []
        
        # 起点安全区
        if hasattr(self, 'task_info') and 'start_pos' in self.task_info:
            start_pos = self.task_info['start_pos']
            if start_pos is not None and len(start_pos) >= 2:
                safe_zones.append((float(start_pos[0]), float(start_pos[1]), safe_radius))
        
        # 终点安全区
        if hasattr(self, 'task_info') and 'target_pos' in self.task_info:
            target_pos = self.task_info['target_pos']
            if target_pos is not None and len(target_pos) >= 2:
                safe_zones.append((float(target_pos[0]), float(target_pos[1]), safe_radius))
        
        # 4. 随机化障碍物位置
        for idx, obstacle_node in enumerate(self.obstacle_nodes):
            translation_field = obstacle_node.getField('translation')
            if translation_field is None:
                continue
            
            if idx < num_obstacles:
                # 激活并随机化位置
                max_attempts = 50
                for attempt in range(max_attempts):
                    # 生成随机位置
                    x = random.uniform(self.obstacle_x_range[0], self.obstacle_x_range[1])
                    y = random.uniform(self.obstacle_y_range[0], self.obstacle_y_range[1])
                    
                    # 检查是否在安全区内
                    in_safe_zone = False
                    for sx, sy, sr in safe_zones:
                        dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                        if dist < sr:
                            in_safe_zone = True
                            break
                    
                    # 如果不在安全区，使用这个位置
                    if not in_safe_zone:
                        translation_field.setSFVec3f([x, y, self.obstacle_z_height])
                        break
                else:
                    # 如果多次尝试失败，使用最后一次生成的位置
                    translation_field.setSFVec3f([x, y, self.obstacle_z_height])
            else:
                # 移到场景外（禁用）
                translation_field.setSFVec3f([100.0, 100.0, 0.3])
        
        # 5. 输出调试信息
        if self.debug or current_step % 10000 == 0:
            print(f"[Obstacle] Step {current_step}: {num_obstacles}/{len(self.obstacle_nodes)} 个障碍物激活")
    
    except Exception as e:
        print(f"[Obstacle] 随机化失败: {e}")
        import traceback
        traceback.print_exc()


def update_global_training_step(self, step: int):
    """
    更新全局训练步数（由训练脚本调用）
    
    参数:
        step: 当前全局训练步数
    """
    self._global_training_step = int(step)
