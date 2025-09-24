"""
ROSbot导航环境的奖励函数模块
包含各种奖励计算逻辑，用于强化学习训练
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Any, Set

def _debug(self, msg: str):
    """条件调试输出"""
    try:
        if getattr(self, 'debug', False) is True:
            step = int(getattr(self, '_global_step', 0))
            print(f"[DEBUG][inst {self.instance_id}][step {step}] {msg}")
    except Exception:
        pass
    
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
        self.stuck_steps_for_spin_check = 3       # 卡住超过5个动作开始检测 (更敏感)
        self.spin_termination_threshold = 4*math.pi    # 累计旋转半圈就终止 (更严格)
        
        # 卡住检测变量
        self.consecutive_stuck_steps = 0
        self.stuck_termination_threshold = 20      # 连续卡住超过15个动作就终止
        self.previous_position = None
        self.stuck_position_threshold = 0.1      # 单个动作位置变化小于此值认为卡住
    
    def reset(self, position: Tuple[float, float, float], orientation: Tuple[float, float, float]):
        """重置奖励状态"""
        self.prev_distance_to_target = None
        self.reward_previous_position = (float(position[0]), float(position[1]))
        self.current_rotation = 0.0
        self.visited_locations.clear()
        self.same_spot_steps = 0
        self.prev_controls.clear()
        
        # 重置原地打转检测变量
        self.previous_yaw_for_spin_check = float(orientation[2])
        self.total_rotation_in_place = 0.0
        
        # 重置卡住检测变量
        self.consecutive_stuck_steps = 0
        self.previous_position = np.array([float(position[0]), float(position[1])])
        
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
        # 从环境状态获取可选的参数对象（args），并提供安全的默认值
        args= env_state.get('args')


        # 计算距离与目标
        distance, closest_target = env_state['calculate_distance_to_target']()
        
        # 根据货物类型调整奖励
        if env_state['cargo_type'] == 'fragile':
            rewards['cargo_specific'] = self._fragile_cargo_reward(action, observation)
        elif env_state['cargo_type'] == 'dangerous':
            rewards['cargo_specific'] = self._dangerous_cargo_reward(action, observation)
        
        # 1) 碰撞惩罚（强惩罚）
        pose=env_state['get_sup_position']()
        collision_terminate = False
        try:
            if env_state['detect_collision_simple']():
                if -2 <= pose[0] <= 2 and -2.7 <= pose[1] <= 2.7:
                    # 在指定区域内碰撞，给予惩罚但不结束回合
                    rewards['collision_penalty'] = -20.0
                    collision_terminate = False
                else:
                    # 在指定区域外碰撞，给予强惩罚并结束回合
                    rewards['collision_penalty'] = -100.0
                    collision_terminate = True
            else:
                rewards['collision_penalty'] = 0.0
                collision_terminate = False
        except Exception:
            collision_terminate = False
            
        # 2) 卡住检测
        stuck_terminate = False
        current_position = np.array(pose[:2])  # 只考虑x, y坐标
        if self.previous_position is not None:
            position_change = np.linalg.norm(current_position - self.previous_position)
            if position_change < self.stuck_position_threshold:
                self.consecutive_stuck_steps += 1
                if self.consecutive_stuck_steps >= self.stuck_termination_threshold:
                    rewards['stuck_penalty'] = -1000.0
                    stuck_terminate = True
                    print(f"[REWARD] 检测到卡住: 连续{self.consecutive_stuck_steps}步位置变化小于{self.stuck_position_threshold}")
            else:
                self.consecutive_stuck_steps = 0
        self.previous_position = current_position.copy()
        
        # 设置终止标志
        env_state['terminate'] = collision_terminate or stuck_terminate

        # 3) 渐进式到达目标奖励，增加停车奖励
        DIST_THRESHOLD = env_state['success_threshold']
        # 获取当前速度（兼容旧版向量obs与新版字典obs）
        try:
            if isinstance(observation, dict):
                odom = env_state['get_odometry_data']()
                linear_vel = abs(float(odom['linear_velocity']))
                angular_vel = abs(float(odom['angular_velocity']))
            else:
                linear_vel = abs(float(observation[23]))
                angular_vel = abs(float(observation[40]))
        except Exception:
            linear_vel = 0.0
            angular_vel = 0.0
        
        if distance < DIST_THRESHOLD:
            # 增加停车奖励：速度越小，奖励越大
            stop_bonus = 100.0 * (1.0 - min(1.0, linear_vel + angular_vel))
            rewards['goal_reward'] = 1000.0 + stop_bonus*args.stop_bonus_k  # 完全到达 + 停车奖励
        elif distance < 0.5:  # 50cm内
            close_reward = (0.5 - distance)  # 线性递增奖励
            # 在接近目标时，鼓励减速
            if distance < 0.3:  # 30cm内开始鼓励减速
                slow_down_reward = 50.0*(1.0 - min(1.0, linear_vel))
                rewards['slow_down_reward'] = slow_down_reward*args.slow_down_reward_k
            rewards['approach_goal'] = close_reward*args.approach_reward_k
        elif distance < 1.0:  # 1m内  
            approach_reward = 50.0 * (1.0 - distance) # 接近奖励
            rewards['approach_goal'] = approach_reward*args.approach_reward_k

        # 4) 时间惩罚（随步数递增）
        time_penalty = 1 + (float(env_state['episode_steps']))
        rewards['time_penalty'] = -time_penalty*args.time_k

        # 5) 距离基奖励（距离越小越好）
        liner_distance_reward = args.liner_distance_reward
        if liner_distance_reward:
            dist_reward = min(125.0, ((-1)*distance+125) if distance > 1e-6 else 125)
        else:
            dist_reward = (-2)*(distance)*(distance)+125
        rewards['distance_reward'] = dist_reward*args.distance_k

        # 6) 距离变化（靠近奖励）
        prev_d = self.prev_distance_to_target if self.prev_distance_to_target is not None else distance
        dist_change = prev_d - distance
        rewards['distance_change_reward'] = dist_change*args.delta_distance_k

        # 7) 探索奖励（新位置）
        current_pos = env_state['get_sup_position']()
        robot_position = (float(current_pos[0]), float(current_pos[1]))
        # current_position = (round(robot_position[0], 2), round(robot_position[1], 2))
        # if current_position not in self.visited_locations:
        #     self.visited_locations.add(current_position)
        #     new_position_reward = 1
        #     if self.same_spot_steps > 2:
        #         new_position_reward *= 2.5
        #     rewards['exploration_reward'] = new_position_reward

        # 9) 移动奖励（位移）
        distance_moved = float(np.linalg.norm(np.array(robot_position) - np.array(self.reward_previous_position)))
        movement_reward = distance_moved*args.movement_reward_k
        rewards['movement_reward'] = movement_reward

        # 8) 靠近墙壁惩罚（LiDAR特征）
        # lidar_features = env_state['get_lidar_features']()
        # wall_prox_penalty = 0.0
        # for feature in lidar_features:
        #     if feature <= 0.25:
        #         wall_prox_penalty += (0.2 - float(feature))
        # rewards['wall_proximity_penalty'] = 0

        # 9) 精确的原地停留检测：参考成功代码的逻辑
        distance_moved = float(np.linalg.norm(np.array(robot_position) - np.array(self.reward_previous_position)))
        
        # 参考成功代码：使用0.005的更严格阈值
        # if distance_moved < 0.1:  # 单步移动小于0.2m，视为在原地
        #     self.same_spot_steps += 1
        #     self.consecutive_stuck_steps += 1
        #     # 检查是否连续卡住超过阈值
        #     if self.consecutive_stuck_steps >= self.stuck_termination_threshold:
        #         rewards['stuck_penalty'] = -1000.0
        #         env_state['terminate'] = True
        #         _debug(self, f"Terminating episode due to being stuck for {self.consecutive_stuck_steps} steps")
        # else:
        #     # 一旦有明显移动，重置所有计数器
        #     self.same_spot_steps = 0
        #     self.consecutive_stuck_steps = 0
        #     self.total_rotation_in_place = 0.0
            
        # 参考成功代码：额外的原地停留惩罚机制
        # same_spot_penalty = 0.0
        # if self.same_spot_steps > 3:  # 参考成功代码的阈值
        #     same_spot_penalty = self.same_spot_steps * 0.25
        #     # 靠近墙壁时惩罚减半，鼓励转向
        #     close_to_wall = any(feature <= 0.2 for feature in lidar_features)
        #     if close_to_wall:episode_reward
        #         same_spot_penalty *= 0.5
        #     # 距离目标近时惩罚加重
        #     if distance < 0.3:
        #         same_spot_penalty *= 1.5
        # rewards['same_spot_penalty'] = -same_spot_penalty

        # 如果在原地停留超过阈值，开始累积旋转
        # if self.same_spot_steps > self.stuck_steps_for_spin_check:
        current_yaw = env_state['get_sup_orientation']()[2]
        delta_yaw = current_yaw - self.previous_yaw_for_spin_check
        # 处理角度环绕问题
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
        self.total_rotation_in_place += delta_yaw

        # 实时更新上一步的朝向，用于计算下一步的角度变化
        self.previous_yaw_for_spin_check = env_state['get_sup_orientation']()[2]

        # 如果累计旋转超过阈值，给予一个惩罚
        if self.total_rotation_in_place > self.spin_termination_threshold:
            rewards['excessive_spin_penalty'] = -1000.0
        else:
            rewards['excessive_spin_penalty'] = 0.0
             
        # 早期原地打转检测：渐进性惩罚
        #if self.same_spot_steps > self.stuck_steps_for_spin_check:  # 原地停留超过3步开始惩罚
        #    early_spin_penalty = -10.0 * (self.same_spot_steps - 3)  # 渐进增加惩罚
        #    rewards['early_spin_penalty'] = early_spin_penalty
        rewards['early_spin_penalty'] = 0


        # 10) 角度奖励：仅在“前方有路径”且“与目标夹角小于90°”时给予奖励
        angle_to_target = self._calculate_angle_to_target(env_state)
        abs_angle = abs(angle_to_target)
        
        # # 安全获取 LiDAR 特征
        # try:
        #     lidar_features = env_state['get_lidar_features']()
        # except Exception:
        #     lidar_features = []

        # clear_path_to_target = False
        # front_clear = False

        # num_feats = len(lidar_features) if hasattr(lidar_features, '__len__') else 0
        # if num_feats > 0:
        #     # 将角度(-pi, pi)映射到索引[0, N-1]
        #     target_angle_index = int(((angle_to_target + math.pi) / (2.0 * math.pi)) * num_feats)
        #     target_angle_index = max(0, min(target_angle_index, num_feats - 1))
        #     clear_path_to_target = float(lidar_features[target_angle_index]) > 0.3

        #     # 判断“前方是否有路”：检查正前方附近的一小段扇区
        #     center_idx = int((0.5) * num_feats)  # angle=0 对应正前方
        #     half_width = max(1, num_feats // 36)  # 约±10°扇区（假设360等分）
        #     start_idx = max(0, center_idx - half_width)
        #     end_idx = min(num_feats - 1, center_idx + half_width)
        #     front_max = max(float(lidar_features[i]) for i in range(start_idx, end_idx + 1))
        #     front_clear = front_max > 0.3
        # else:
        #     # 无法获取LiDAR时，保守起见不发放角度奖励
        #     clear_path_to_target = False
        #     front_clear = False

        # # 仅在两个条件同时满足时给予角度奖励：
        # # 1) 前方有路径(front_clear)
        # # 2) 与目标角度差小于90度(abs_angle <= pi/2)
        # if front_clear and abs_angle <= (0.5 * math.pi) and clear_path_to_target:
        #     angle_reward = 10.0 if abs_angle <= 0.3 else (0.3 / abs_angle)
        # else:
        #     angle_reward = 0.0
        # rewards['angle_reward'] = angle_reward
        rewards['angle_reward'] = 0
        # 更新缓存
        self.prev_distance_to_target = distance
        self.reward_previous_position = robot_position
        total_reward = sum(rewards.values())

        _debug(self, f"""
        Total reward: {total_reward},\n
        collision_penalty: {rewards['collision_penalty']},\n
        time_penalty: {rewards['time_penalty']},\n
        distance_reward: {rewards['distance_reward']},\n
        distance_change_reward: {rewards['distance_change_reward']},\n
        movement_reward: {rewards['movement_reward']},\n
        excessive_spin_penalty: {rewards['excessive_spin_penalty']},\n
        early_spin_penalty: {rewards['early_spin_penalty']},\n
        angle_reward: {rewards['angle_reward']}
        """)
        # wall_proximity_penalty: {rewards['wall_proximity_penalty']},\n
        # same_spot_penalty: {rewards['same_spot_penalty']},\n
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
        
        # 稳定性奖励 - 惩罚大加速度（从env_state获取，加速度不再从obs索引）
        # 注意：此函数在calculate_reward内部调用时，需要从env_state读取；
        # 这里简化为0并在calculate_reward处处理主要奖励。
        linear_acc = 0.0
        angular_acc = 0.0
        stability_penalty = -abs(linear_acc) * 5.0 - abs(angular_acc) * 3.0
        reward += stability_penalty
        
        # 速度限制 - 惩罚高速
        try:
            if isinstance(observation, dict):
                odom = observation.get('_odometry_cache_', None)
                if odom is None:
                    odom = {'linear_velocity': 0.0}
                linear_vel = float(odom['linear_velocity'])
            else:
                linear_vel = float(observation[23])
        except Exception:
            linear_vel = 0.0
        velocity_penalty = -max(0, linear_vel - 1.0) * 3.0  # 超过1m/s惩罚
        reward += velocity_penalty
        
        # 角速度平滑性
        try:
            if isinstance(observation, dict):
                odom = observation.get('_odometry_cache_', None)
                if odom is None:
                    odom = {'angular_velocity': 0.0}
                angular_vel = abs(float(odom['angular_velocity']))
            else:
                angular_vel = abs(float(observation[40]))
        except Exception:
            angular_vel = 0.0
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
        try:
            if isinstance(observation, dict):
                # 通过 env_state 获取 LiDAR 特征（在calculate_reward中已有该函数）
                # 这里使用一个保守的默认，实际计算在主奖励中涵盖
                lidar_data = np.array([1.0]*20, dtype=np.float32)
            else:
                lidar_data = observation[0:20]
        except Exception:
            lidar_data = np.array([1.0]*20, dtype=np.float32)
        min_obstacle_dist = min(lidar_data) * 10.0  # 反归一化
        
        if min_obstacle_dist < 0.5:  # 距离障碍物小于0.5m
            safety_penalty = -20.0 * (0.5 - min_obstacle_dist)
            reward += safety_penalty
        
        # 保守速度奖励
        try:
            if isinstance(observation, dict):
                odom = observation.get('_odometry_cache_', None)
                if odom is None:
                    odom = {'linear_velocity': 0.0}
                linear_vel = float(odom['linear_velocity'])
            else:
                linear_vel = float(observation[23])
        except Exception:
            linear_vel = 0.0
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
