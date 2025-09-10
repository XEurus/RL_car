"""
导航工具类 - 提供任务配置和导航辅助功能
"""

import numpy as np
from typing import Dict, List, Tuple
import random
import json
from pathlib import Path


class NavigationUtils:
    """导航工具类"""
    
    def __init__(self):
        # 固定的位置坐标
        self.fixed_positions = {
            'start': [-5.0, 3.0, 0.0],       # 固定起点
            'unload': [-5.0, -2.0, 0.0],     # 固定卸货点
            'dangerous': [5.0, 3.0, 0.0],    # 固定危险货物点
            'fragile': [5.0, 1.7, 0.0],      # 固定易碎货物点
            'normal': [5.0, 0.2, 0.0]        # 固定普通货物点
        }
        
        # 位置区域范围 - 用于随机生成目标点
        self.position_areas = {
            'start_area': {
                'x_min': -5.5, 'x_max': -4.5,
                'y_min': 2.5, 'y_max': 3.5,
                'z': 0.0
            },
            'unload_area': {
                'x_min': -5.5, 'x_max': -4.5,
                'y_min': -2.5, 'y_max': -1.5,
                'z': 0.0
            },
            'dangerous_area': {
                'x_min': 4.5, 'x_max': 5.5,
                'y_min': 2.5, 'y_max': 3.5,
                'z': 0.0
            },
            'fragile_area': {
                'x_min': 4.5, 'x_max': 5.5,
                'y_min': 1.2, 'y_max': 2.2,
                'z': 0.0
            },
            'normal_area': {
                'x_min': 4.5, 'x_max': 5.5,
                'y_min': -0.3, 'y_max': 0.7,
                'z': 0.0
            }
        }
        
        # 导航任务配置 - 根据货物类型分不同的取货点和目的地
        self.navigation_config = {
            'normal': self._get_normal_cargo_tasks(),
            'fragile': self._get_fragile_cargo_tasks(),
            'dangerous': self._get_dangerous_cargo_tasks()
        }
        
        # 环境边界
        self.env_bounds = {
            'x_min': -6.0, 'x_max': 6.0,
            'y_min': -4.0, 'y_max': 4.0,
            'z_min': 0, 'z_max': 0
        }
        
        # 是否使用随机目标点
        self.use_random_targets = True
    
    def _generate_random_position_in_area(self, area_name: str) -> List[float]:
        """从指定区域生成随机位置"""
        if area_name not in self.position_areas:
            # 如果区域不存在，返回对应的固定位置
            area_name = area_name.replace('_area', '')
            return self.fixed_positions.get(area_name, [0, 0, 0])
        
        area = self.position_areas[area_name]
        x = random.uniform(area['x_min'], area['x_max'])
        y = random.uniform(area['y_min'], area['y_max'])
        z = area['z']
        
        return [x, y, z]
    
    def get_navigation_task(self, cargo_type: str, task_stage: str = 'base') -> Tuple[List[float], List[float]]:
        """
        根据货物类型和任务阶段获取导航任务
        
        参数:
            cargo_type: 货物类型 ('normal', 'fragile', 'dangerous')
            task_stage: 任务阶段
                - 'base': 基础模型训练，从固定起点到各个货物点，再从普通货物点到卸货点
                - 'dangerous_to_unload': 从危险货物点到卸货点
                - 'fragile_to_unload': 从易碎货物点到卸货点
                - 'normal_to_unload': 从普通货物点到卸货点
        
        返回:
            起点和终点坐标元组
        """
        if cargo_type not in self.navigation_config:
            cargo_type = 'normal'  # 默认为普通货物
        
        # 根据任务阶段选择起点和终点
        if task_stage == 'base':
            if random.random() < 0.5:
                # 从起点到各个货物点
                if self.use_random_targets:
                    start_pos = self._generate_random_position_in_area('start_area')
                    # 随机选择一个货物点类型作为终点
                    target_type = random.choice(['dangerous', 'fragile', 'normal'])
                    target_pos = self._generate_random_position_in_area(f'{target_type}_area')
                else:
                    start_pos = self.fixed_positions['start']
                    target_type = random.choice(['dangerous', 'fragile', 'normal'])
                    target_pos = self.fixed_positions[target_type]
            else:
                # 从普通货物点到卸货点
                if self.use_random_targets:
                    start_pos = self._generate_random_position_in_area('normal_area')
                    target_pos = self._generate_random_position_in_area('unload_area')
                else:
                    start_pos = self.fixed_positions['normal']
                    target_pos = self.fixed_positions['unload']
        
        elif task_stage == 'dangerous_to_unload':
            # 从危险货物点到卸货点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('dangerous_area')
                target_pos = self._generate_random_position_in_area('unload_area')
            else:
                start_pos = self.fixed_positions['dangerous']
                target_pos = self.fixed_positions['unload']
        
        elif task_stage == 'fragile_to_unload':
            # 从易碎货物点到卸货点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('fragile_area')
                target_pos = self._generate_random_position_in_area('unload_area')
            else:
                start_pos = self.fixed_positions['fragile']
                target_pos = self.fixed_positions['unload']
        
        elif task_stage == 'normal_to_unload':
            # 从普通货物点到卸货点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('normal_area')
                target_pos = self._generate_random_position_in_area('unload_area')
            else:
                start_pos = self.fixed_positions['normal']
                target_pos = self.fixed_positions['unload']
        
        else:
            # 默认：从固定起点到普通货物点
            if self.use_random_targets:
                start_pos = self._generate_random_position_in_area('start_area')
                target_pos = self._generate_random_position_in_area('normal_area')
            else:
                start_pos = self.fixed_positions['start']
                target_pos = self.fixed_positions['normal']
        
        return start_pos, target_pos
    
    def _get_normal_cargo_tasks(self) -> List[Dict]:
        """普通货物导航任务 - 使用固定位置"""
        return [
            {
                'start_pos': self.fixed_positions['start'],
                'pickup_pos': self.fixed_positions['normal'],
                'destination': self.fixed_positions['unload'],
                'description': '普通货物：从固定起点到普通货物点'
            },
            {
                'start_pos': self.fixed_positions['normal'],
                'pickup_pos': self.fixed_positions['unload'],
                'destination': self.fixed_positions['start'],
                'description': '普通货物：从普通货物点到卸货点'
            }
        ]
    
    def _get_fragile_cargo_tasks(self) -> List[Dict]:
        """易碎品导航任务 - 使用固定位置"""
        return [
            {
                'start_pos': self.fixed_positions['start'],
                'pickup_pos': self.fixed_positions['fragile'],
                'destination': self.fixed_positions['unload'],
                'description': '易碎品：从固定起点到易碎货物点'
            },
            {
                'start_pos': self.fixed_positions['fragile'],
                'pickup_pos': self.fixed_positions['unload'],
                'destination': self.fixed_positions['start'],
                'description': '易碎品：从易碎货物点到卸货点'
            }
        ]
    
    def _get_dangerous_cargo_tasks(self) -> List[Dict]:
        """危险品导航任务 - 使用固定位置"""
        return [
            {
                'start_pos': self.fixed_positions['start'],
                'pickup_pos': self.fixed_positions['dangerous'],
                'destination': self.fixed_positions['unload'],
                'description': '危险品：从固定起点到危险货物点'
            },
            {
                'start_pos': self.fixed_positions['dangerous'],
                'pickup_pos': self.fixed_positions['unload'],
                'destination': self.fixed_positions['start'],
                'description': '危险品：从危险货物点到卸货点'
            }
        ]
    
    def is_position_valid(self, position: List[float]) -> bool:
        """检查位置是否在有效范围内"""
        x, y, z = position
        return (self.env_bounds['x_min'] <= x <= self.env_bounds['x_max'] and
                self.env_bounds['y_min'] <= y <= self.env_bounds['y_max'] and
                self.env_bounds['z_min'] <= z <= self.env_bounds['z_max'])
    
    def calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两点间欧氏距离"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
    def calculate_manhattan_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算曼哈顿距离"""
        return sum(abs(a - b) for a, b in zip(pos1, pos2))
    
    def calculate_heading(self, from_pos: List[float], to_pos: List[float]) -> float:
        """计算从from_pos到to_pos的航向角（偏航角）"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        return math.atan2(dy, dx)
    
    def normalize_heading(self, heading: float) -> float:
        """归一化航向角到[-π, π]"""
        while heading > math.pi:
            heading -= 2 * math.pi
        while heading < -math.pi:
            heading += 2 * math.pi
        return heading
    
    def interpolate_positions(self, start: List[float], end: List[float], num_points: int) -> List[List[float]]:
        """在两点间插值生成路径点"""
        start = np.array(start)
        end = np.array(end)
        waypoints = []
        
        for i in range(num_points + 1):
            t = i / num_points
            point = start + t * (end - start)
            waypoints.append(point.tolist())
        
        return waypoints
    
    def generate_random_position(self) -> List[float]:
        """生成随机有效位置"""
        return [
            random.uniform(self.env_bounds['x_min'], self.env_bounds['x_max']),
            random.uniform(self.env_bounds['y_min'], self.env_bounds['y_max']),
            random.uniform(self.env_bounds['z_min'], self.env_bounds['z_max'])
        ]
    
    def generate_nearby_position(self, center: List[float], radius: float) -> List[float]:
        """在以center为中心、radius为半径的圆内生成随机位置"""
        while True:
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, radius)
            
            new_pos = [
                center[0] + r * math.cos(angle),
                center[1] + r * math.sin(angle),
                center[2]  # z保持不变
            ]
            
            if self.is_position_valid(new_pos):
                return new_pos
    
    def save_config(self, file_path: str):
        """保存导航配置"""
        config_data = {
            'navigation_config': self.navigation_config,
            'env_bounds': self.env_bounds,
            'created_at': str(np.datetime64('now'))
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def load_config(self, file_path: str):
        """加载导航配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.navigation_config = config_data['navigation_config']
            self.env_bounds = config_data['env_bounds']
            
            print(f"成功加载导航配置文件: {file_path}")
            
        except FileNotFoundError:
            print(f"导航配置文件不存在: {file_path}，使用默认配置")
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
    
    def verify_task_feasibility(self, start: List[float], target: List[float], 
                              obstacles: List[List[float]] = None) -> Dict:
        """验证任务可行性"""
        # 检查边界
        if not (self.is_position_valid(start) and self.is_position_valid(target)):
            return {
                'feasible': False,
                'reason': '位置超出有效范围',
                'start_valid': self.is_position_valid(start),
                'target_valid': self.is_position_valid(target)
            }
        
        # 检查是否可以直线到达
        distance = self.calculate_distance(start, target)
        
        # 简单的障碍物检查
        if obstacles:
            collision_count = 0
            for obstacle in obstacles:
                obstacle_distance_to_path = self._point_to_line_distance(obstacle, start, target)
                if obstacle_distance_to_path < 0.2:  # 20cm安全距离
                    collision_count += 1
            
            if collision_count > 0:
                return {
                    'feasible': False,
                    'reason': f'路径上有{collision_count}个障碍物碰撞',
                    'distance': distance,
                    'collision_count': collision_count
                }
        
        return {
            'feasible': True,
            'distance': distance,
            'estimated_time': distance / 0.8  # 假设平均速度0.8m/s
        }
    
    def _point_to_line_distance(self, point: List[float], line_start: List[float], 
                               line_end: List[float]) -> float:
        """计算点到线段的距离"""
        # 向量计算
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        
        # 投影计算
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.linalg.norm(point_vec)
        
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = np.array(line_start) + t * line_vec
        
        return np.linalg.norm(np.array(point) - projection)


class NavigationTaskGenerator:
    """导航任务生成器 - 用于训练和测试"""
    
    def __init__(self, nav_utils: NavigationUtils):
        self.nav_utils = nav_utils
        self.task_pool = []
        
    def generate_training_tasks(self, cargo_type: str, num_tasks: int) -> List[Dict]:
        """生成训练任务集"""
        tasks = []
        
        base_tasks = self.nav_utils.navigation_config[cargo_type]
        
        for i in range(num_tasks):
            # 基于基础任务生成变体
            base_task = random.choice(base_tasks)
            
            # 添加小的随机扰动
            perturbed_task = self._perturb_task(base_task, perturbation_range=0.5)
            
            tasks.append({
                'task_id': i,
                'cargo_type': cargo_type,
                **perturbed_task,
                'difficulty': random.choice(['easy', 'medium', 'hard'])
            })
        
        return tasks
    
    def generate_test_tasks(self, cargo_type: str, num_tasks: int) -> List[Dict]:
        """生成测试任务集"""
        tasks = self.generate_training_tasks(cargo_type, num_tasks)
        
        # 为测试任务增加难度评估
        for task in tasks:
            difficulty = self._assess_task_difficulty(
                task['start_pos'], task['target_pos']
            )
            task['test_difficulty'] = difficulty
        
        return tasks
    
    def _perturb_task(self, base_task: Dict, perturbation_range: float) -> Dict:
        """对任务添加随机扰动"""
        return {
            'start_pos': [
                base_task['start_pos'][0] + random.uniform(-perturbation_range, perturbation_range),
                base_task['start_pos'][1] + random.uniform(-perturbation_range, perturbation_range),
                base_task['start_pos'][2]  # z坐标保持不变
            ] if self.nav_utils.is_position_valid([
                base_task['start_pos'][0] + random.uniform(-perturbation_range, perturbation_range),
                base_task['start_pos'][1] + random.uniform(-perturbation_range, perturbation_range),
                base_task['start_pos'][2]
            ]) else base_task['start_pos'],
            
            'target_pos': [
                base_task['target_pos'][0] + random.uniform(-perturbation_range, perturbation_range),
                base_task['target_pos'][1] + random.uniform(-perturbation_range, perturbation_range),
                base_task['target_pos'][2]
            ] if self.nav_utils.is_position_valid([
                base_task['target_pos'][0] + random.uniform(-perturbation_range, perturbation_range),
                base_task['target_pos'][1] + random.uniform(-perturbation_range, perturbation_range),
                base_task['target_pos'][2]
            ]) else base_task['target_pos'],
            
            'description': base_task['description'] + ' (扰动后)'
        }
    
    def _assess_task_difficulty(self, start: List[float], target: List[float]) -> str:
        """评估任务难度"""
        distance = self.nav_utils.calculate_distance(start, target)
        manhattan_distance = self.nav_utils.calculate_manhattan_distance(start, target)
        
        # 基于距离评估
        if distance < 5.0:
            distance_difficulty = 'easy'
        elif distance < 10.0:
            distance_difficulty = 'medium'
        else:
            distance_difficulty = 'hard'
        
        # 基于转弯复杂度评估
        start_heading = math.atan2(target[1] - start[1], target[0] - start[0])
        complexity_score = abs(math.sin(start_heading)) * 2.0  # 垂直方向更复杂
        
        if complexity_score < 0.5:
            complexity_difficulty = 'easy'
        elif complexity_score < 1.0:
            complexity_difficulty = 'medium'
        else:
            complexity_difficulty = 'hard'
        
        # 综合难度
        if distance_difficulty == 'easy' and complexity_difficulty == 'easy':
            return 'easy'
        elif distance_difficulty == 'hard' or complexity_difficulty == 'hard':
            return 'hard'
        else:
            return 'medium'


import math  # 确保math模块被导入