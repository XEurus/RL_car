"""
ROSbot导航环境的奖励函数模块
包含各种奖励计算逻辑，用于强化学习训练
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Any, Set


class RewardFunctions:
    """
    奖励函数类，包含各种奖励计算逻辑
    """
    
    def __init__(self):
        """初始化奖励函数类"""
        # 奖励计算状态
        self.prev_distance_to_target = None
        self.reward_previous_position = (0.0, 0.0)
        self.current_rotation = 0.0
        self.visited_locations = set()
        self.same_spot_steps = 0
        self.prev_controls = []
        
        # 原地打转检测变量
        self.previous_yaw_for_spin_check = 0.0
        self.total_rotation_in_place = 0.0
        self.stuck_steps_for_spin_check = 5       # 卡住超过5步开始检测 (更敏感)
        self.spin_termination_threshold = math.pi    # 累计旋转半圈就终止 (更严格)
    
    def reset(self, position: Tuple[float, float, float], orientation: Tuple[float, float, float]):
        """重置奖励状态"""
        self.prev_distance_to_target = None
        self.reward_previous_position = (float(position[0]), float(position[1]))
        self.current_rotation = 0.0
        self.visited_locations = set()
        self.same_spot_steps = 0
        self.prev_controls = []
        
        # 重置原地打转检测变量
        self.total_rotation_in_place = 0.0
        self.previous_yaw_for_spin_check = orientation[2]
    
    def calculate_reward(self, 
                        action: np.ndarray, 
                        observation: np.ndarray, 
                        env_state: Dict[str, Any]) -> float:
        """
        综合奖励函数：碰撞、到达、距离、靠近、时间、探索、移动、贴墙、反空转、角度与一致转向
        
        参数:
            action: 当前执行的动作
            observation: 当前观察
            env_state: 环境状态，包含以下键:
                - get_sup_position: 获取机器人位置的函数
                - get_sup_orientation: 获取机器人朝向的函数
                - calculate_distance_to_target: 计算到目标距离的函数
                - get_lidar_features: 获取LiDAR特征的函数
                - detect_collision_simple: 检测碰撞的函数
                - episode_steps: 当前回合步数
                - cargo_type: 货物类型
                - success_threshold: 成功阈值
                - task_info: 任务信息
        
        返回:
            float: 计算得到的奖励值
        """
        rewards = {}

        # 计算距离与目标
        distance, closest_target = env_state['calculate_distance_to_target']()
        
        # 根据货物类型调整奖励
        if env_state['cargo_type'] == 'fragile':
            rewards['cargo_specific'] = self._fragile_cargo_reward(action, observation)
        elif env_state['cargo_type'] == 'dangerous':
            rewards['cargo_specific'] = self._dangerous_cargo_reward(action, observation)
        
        # 1) 碰撞惩罚（强惩罚）
        try:
            if env_state['detect_collision_simple']():
                rewards['collision_penalty'] = -200.0
        except Exception:
            pass

        # 2) 渐进式到达目标奖励，增加停车奖励
        DIST_THRESHOLD = env_state['success_threshold']
        # 获取当前速度
        linear_vel = abs(observation[23])  # 线速度大小
        angular_vel = abs(observation[40])  # 角速度大小
        
        if distance < DIST_THRESHOLD:
            # 增加停车奖励：速度越小，奖励越大
            stop_bonus = 200.0 * (1.0 - min(1.0, linear_vel + angular_vel))
            rewards['goal_reward'] = 500.0 + stop_bonus  # 完全到达 + 停车奖励
        elif distance < 0.5:  # 50cm内
            close_reward = 200.0 * (0.5 - distance) / 0.5  # 线性递增奖励
            # 在接近目标时，鼓励减速
            if distance < 0.3:  # 30cm内开始鼓励减速
                slow_down_reward = 50.0 * (1.0 - min(1.0, linear_vel))
                rewards['slow_down_reward'] = slow_down_reward
            rewards['close_to_goal'] = close_reward
        elif distance < 1.0:  # 1m内  
            approach_reward = 50.0 * (1.0 - distance) / 0.5  # 接近奖励
            rewards['approach_goal'] = approach_reward

        # 3) 时间惩罚（随步数递增）
        time_penalty = 0.5 + (float(env_state['episode_steps']) / 400.0)
        rewards['time_penalty'] = -time_penalty

        # 4) 距离基奖励（距离越小越好）
        dist_reward = min(3.0, ((1.0 / distance) * 0.3) if distance > 1e-6 else 3.0)
        rewards['distance_reward'] = dist_reward

        # 5) 距离变化（靠近奖励）
        prev_d = self.prev_distance_to_target if self.prev_distance_to_target is not None else distance
        dist_change = (prev_d - distance) * 2.0
        rewards['distance_change_reward'] = dist_change

        # 6) 探索奖励（新位置）
        current_pos = env_state['get_sup_position']()
        robot_position = (float(current_pos[0]), float(current_pos[1]))
        current_position = (round(robot_position[0], 2), round(robot_position[1], 2))
        if current_position not in self.visited_locations:
            self.visited_locations.add(current_position)
            new_position_reward = 0.1
            if self.same_spot_steps > 5:
                new_position_reward *= 2.5
            rewards['exploration_reward'] = new_position_reward

        # 7) 移动奖励（位移）
        distance_moved = float(np.linalg.norm(np.array(robot_position) - np.array(self.reward_previous_position)))
        movement_reward = distance_moved * 0.5
        rewards['movement_reward'] = movement_reward

        # 8) 靠近墙壁惩罚（LiDAR特征）
        lidar_features = env_state['get_lidar_features']()
        wall_prox_penalty = 0.0
        for feature in lidar_features:
            if feature <= 0.2:
                wall_prox_penalty += (0.2 - float(feature)) * 3.0
        rewards['wall_proximity_penalty'] = -wall_prox_penalty

        # 9) 精确的原地停留检测：参考成功代码的逻辑
        distance_moved = float(np.linalg.norm(np.array(robot_position) - np.array(self.reward_previous_position)))
        
        # 参考成功代码：使用0.005的更严格阈值
        if distance_moved < 0.005:  # 单步移动小于0.5cm，视为在原地
            self.same_spot_steps += 1
        else:
            # 一旦有明显移动，重置所有计数器
            self.same_spot_steps = 0
            self.total_rotation_in_place = 0.0
            
        # 参考成功代码：额外的原地停留惩罚机制
        same_spot_penalty = 0.0
        if self.same_spot_steps > 3:  # 参考成功代码的阈值
            same_spot_penalty = self.same_spot_steps * 0.25
            # 靠近墙壁时惩罚减半，鼓励转向
            close_to_wall = any(feature <= 0.12 for feature in lidar_features)
            if close_to_wall:
                same_spot_penalty *= 0.5
            # 距离目标近时惩罚加重
            if distance < 0.3:
                same_spot_penalty *= 1.5
        rewards['same_spot_penalty'] = -same_spot_penalty

        # 如果在原地停留超过阈值，开始累积旋转
        if self.same_spot_steps > self.stuck_steps_for_spin_check:
            current_yaw = env_state['get_sup_orientation']()[2]
            delta_yaw = current_yaw - self.previous_yaw_for_spin_check
            # 处理角度环绕问题
            delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
            self.total_rotation_in_place += abs(delta_yaw)

        # 实时更新上一步的朝向，用于计算下一步的角度变化
        self.previous_yaw_for_spin_check = env_state['get_sup_orientation']()[2]

        # 如果累计旋转超过阈值，给予一个惩罚
        if self.total_rotation_in_place > self.spin_termination_threshold:
             rewards['excessive_spin_penalty'] = -800.0
             
        # 早期原地打转检测：渐进性惩罚
        if self.same_spot_steps > 3:  # 原地停留超过3步开始惩罚
            early_spin_penalty = -10.0 * (self.same_spot_steps - 3)  # 渐进增加惩罚
            rewards['early_spin_penalty'] = early_spin_penalty

        # 10) 角度奖励：首先判断目标方向是否有清晰路径
        angle = abs(self._calculate_angle_to_target(env_state))
        if len(lidar_features) > 0:
            target_angle_index = int((angle + math.pi) / (2.0 * math.pi) * len(lidar_features))
            target_angle_index = max(0, min(target_angle_index, len(lidar_features) - 1))
            clear_path_to_target = lidar_features[target_angle_index] > 0.3
        else:
            clear_path_to_target = True
        if clear_path_to_target:
            angle_reward = min(1.0, 0.2 / angle) if angle > 0.05 else 1.0
        else:
            angle_reward = min(0.3, 0.05 / angle) if angle > 0.05 else 0.3
        rewards['angle_reward'] = angle_reward

        # 更新缓存
        self.prev_distance_to_target = distance
        self.reward_previous_position = robot_position
            
        total_reward = sum(rewards.values())
        return total_reward
    
    def _calculate_angle_to_target(self, env_state: Dict[str, Any]) -> float:
        """
        计算当前航向与目标方向的角度偏差（-π, π）
        
        参数:
            env_state: 环境状态
            
        返回:
            float: 角度偏差
        """
        current_pos = env_state['get_sup_position']()
        current_orient = env_state['get_sup_orientation']()
        target_pos = env_state['task_info']['target_pos']
        
        vec_to_target = target_pos - current_pos
        target_heading = math.atan2(float(vec_to_target[1]), float(vec_to_target[0]))
        current_heading = float(current_orient[2])
        angle = target_heading - current_heading
        angle = math.atan2(math.sin(angle), math.cos(angle))
        return angle
    
    def _fragile_cargo_reward(self, action: np.ndarray, observation: np.ndarray) -> float:
        """
        易碎品专用奖励
        
        参数:
            action: 当前执行的动作
            observation: 当前观察
            
        返回:
            float: 奖励值
        """
        reward = 0.0
        
        # 稳定性奖励 - 惩罚大加速度
        linear_acc = observation[31]  # 线加速度
        angular_acc = observation[29]  # 角加速度
        stability_penalty = -abs(linear_acc) * 5.0 - abs(angular_acc) * 3.0
        reward += stability_penalty
        
        # 速度限制 - 惩罚高速
        linear_vel = observation[23]  # 线速度大小
        velocity_penalty = -max(0, linear_vel - 1.0) * 3.0  # 超过1m/s惩罚
        reward += velocity_penalty
        
        # 角速度平滑性
        angular_vel = abs(observation[40])
        angular_penalty = -angular_vel * 2.0 if angular_vel > 1.0 else 0.0
        reward += angular_penalty
        
        return reward
    
    def _dangerous_cargo_reward(self, action: np.ndarray, observation: np.ndarray) -> float:
        """
        危险品专用奖励
        
        参数:
            action: 当前执行的动作
            observation: 当前观察
            
        返回:
            float: 奖励值
        """
        reward = 0.0
        
        # 安全性奖励 - 保持安全距离
        lidar_data = observation[0:20]
        min_obstacle_dist = min(lidar_data) * 10.0  # 反归一化
        
        if min_obstacle_dist < 0.5:  # 距离障碍物小于0.5m
            safety_penalty = -20.0 * (0.5 - min_obstacle_dist)
            reward += safety_penalty
        
        # 保守速度奖励
        linear_vel = observation[23]
        conservative_reward = -abs(linear_vel) * 2.0 if linear_vel > 0.8 else 0.0
        reward += conservative_reward
        
        return reward
    
    def update_action_history(self, action: np.ndarray) -> None:
        """
        更新动作历史
        
        参数:
            action: 当前执行的动作
        """
        try:
            self.prev_controls.append([float(action[0]), float(action[1])])
            if len(self.prev_controls) > 10:
                self.prev_controls.pop(0)
        except Exception:
            pass
        
    def update_current_rotation(self, angular_vel: float) -> None:
        """
        更新当前旋转
        
        参数:
            angular_vel: 当前角速度
        """
        self.current_rotation = float(angular_vel)
    
    def is_excessive_spin(self) -> bool:
        """
        检查是否过度旋转
        
        返回:
            bool: 是否过度旋转
        """
        return self.total_rotation_in_place > self.spin_termination_threshold
