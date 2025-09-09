"""
姿态估计器 - 备用定位方案
提供GPS+IMU融合定位，用于AMCL失败时的备用方案
"""

import numpy as np
import math
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class PoseState:
    """位姿状态数据类"""
    position: np.ndarray      # [x, y, z]
    orientation: np.ndarray   # [roll, pitch, yaw]  
    velocity: np.ndarray      # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    linear_acceleration: np.ndarray  # [ax, ay, az]
    timestamp: float
    uncertainty: np.ndarray   # 各分量的不确定性


class LowPassFilter:
    """低通滤波器"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.filtered_value = None
    
    def update(self, new_value: np.ndarray) -> np.ndarray:
        """更新滤波值"""
        if self.filtered_value is None:
            self.filtered_value = new_value.copy()
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        
        return self.filtered_value.copy()
    
    def reset(self):
        """重置滤波器"""
        self.filtered_value = None


class KalmanFilter3D:
    """3D卡尔曼滤波器用于GPS+IMU融合"""
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.02):
        # 状态维度 [x, y, z, vx, vy, vz, ax, ay, az]
        self.dim_state = 9
        
        # 状态向量
        self.x = np.zeros(self.dim_state)
        
        # 协方差矩阵
        self.P = np.eye(self.dim_state) * 0.1
        
        # 状态转移矩阵
        self.dt = 0.05  # 20Hz默认
        self.update_transition_matrix()
        
        # 测量矩阵
        self.H_pos = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        
        self.H_vel = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0]])
        
        # 过程噪声和测量噪声
        self.Q = np.eye(self.dim_state) * process_noise
        self.R_pos = np.eye(3) * measurement_noise
        self.R_vel = np.eye(3) * measurement_noise * 2.0  # 速度测量噪声更大
        
        # 状态历史
        self.state_history = deque(maxlen=10)
    
    def update_transition_matrix(self):
        """更新状态转移矩阵"""
        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2, 0, 0],
            [0, 1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2, 0],
            [0, 0, 1, 0, 0, self.dt, 0, 0, 0.5*self.dt**2],
            [0, 0, 0, 1, 0, 0, self.dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, self.dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
    
    def predict(self):
        """预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 更新状态历史
        self.state_history.append(self.x.copy())
    
    def update_with_position(self, position: np.ndarray):
        """使用GPS位置更新"""
        y = position - self.H_pos @ self.x
        S = self.H_pos @ self.P @ self.H_pos.T + self.R_pos
        K = self.P @ self.H_pos.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_state) - K @ self.H_pos) @ self.P
    
    def update_with_velocity(self, velocity: np.ndarray):
        """使用速度测量更新"""
        y = velocity - self.H_vel @ self.x
        S = self.H_vel @ self.P @ self.H_vel.T + self.R_vel
        K = self.P @ self.H_vel.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_state) - K @ self.H_vel) @ self.P
    
    def get_state(self) -> Dict:
        """获取当前状态"""
        return {
            'position': self.x[0:3],
            'velocity': self.x[3:6],
            'acceleration': self.x[6:9],
            'covariance': np.diag(self.P)  # 主对角线作为不确定性估计
        }


class RobustPoseEstimator:
    """鲁棒姿态估计器 - GPS+IMU融合备用方案"""
    
    def __init__(self, supervisor, dt: float = 0.05):
        self.supervisor = supervisor
        self.dt = dt
        self.timestep = int(dt * 1000)
        
        # 传感器数据滤波器
        self.position_filter = LowPassFilter(alpha=0.1)
        self.velocity_filter = LowPassFilter(alpha=0.2)
        self.imu_filter = LowPassFilter(alpha=0.15)
        
        # 卡尔曼滤波器
        self.kalman_filter = KalmanFilter3D()
        
        # 状态缓存
        self.previous_state = None
        self.state_history = deque(maxlen=20)
        
        # 传感器接口
        self._initialize_sensors()
        
        # 不确定性估计
        self.position_uncertainty = np.array([0.05, 0.05, 0.01])  # 5cm, 5cm, 1cm
        self.velocity_uncertainty = np.array([0.01, 0.01, 0.005])  # 0.1m/s, 0.1m/s, 0.05m/s
        
    def _initialize_sensors(self):
        """初始化传感器接口"""
        # GPS传感器
        self.gps = self.supervisor.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
        
        # IMU/惯性单元
        self.imu = self.supervisor.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.timestep)
        
        # 陀螺仪
        self.gyro = self.supervisor.getDevice('gyro')
        if self.gyro:
            self.gyro.enable(self.timestep)
        
        # 罗盘
        self.compass = self.supervisor.getDevice('compass')
        if self.compass:
            self.compass.enable(self.timestep)
        
        # 机器人节点 - 用于获取绝对位置和姿态
        self.robot_node = self.supervisor.getFromDef('ROSbot')
        
    def update_state(self) -> PoseState:
        """更新并返回当前状态"""
        current_time = self.supervisor.getTime()
        
        # 1. 获取传感器原始数据
        sensor_data = self._get_sensor_data(current_time)
        
        # 2. 数据预处理和滤波
        filtered_data = self._filter_sensor_data(sensor_data)
        
        # 3. 卡尔曼滤波融合
        kalman_state = self._apply_kalman_filter(filtered_data)
        
        # 4. 计算派生状态
        derived_state = self._calculate_derived_states(kalman_state)
        
        # 5. 不确定性计算
        uncertainty = self._calculate_uncertainty(filtered_data, kalman_state)
        
        # 6. 构建完整状态
        current_state = PoseState(
            position=kalman_state['position'],
            orientation=derived_state['orientation'],
            velocity=kalman_state['velocity'],
            angular_velocity=derived_state['angular_velocity'],
            linear_acceleration=derived_state['linear_acceleration'],
            timestamp=current_time,
            uncertainty=uncertainty
        )
        
        # 7. 状态历史记录
        self._record_state_history(current_state)
        self.previous_state = current_state
        
        return current_state
    
    def _get_sensor_data(self, timestamp: float) -> Dict:
        """获取传感器原始数据"""
        sensor_data = {
            'timestamp': timestamp,
            'gps_available': False,
            'imu_available': False,
            'gyro_available': False,
            'compass_available': False
        }
        
        # GPS数据
        if self.gps:
            gps_values = self.gps.getValues()
            if gps_values and len(gps_values) >= 3:
                sensor_data.update({
                    'gps_position': np.array(gps_values[:3]),
                    'gps_available': True
                })
        
        # IMU数据
        if self.imu:
            imu_values = self.imu.getRollPitchYaw()
            if imu_values and len(imu_values) >= 3:
                sensor_data.update({
                    'imu_orientation': np.array(imu_values),
                    'imu_available': True
                })
        
        # 陀螺仪数据
        if self.gyro:
            gyro_values = self.gyro.getValues()
            if gyro_values and len(gyro_values) >= 3:
                sensor_data.update({
                    'gyro_angular_velocity': np.array(gyro_values),
                    'gyro_available': True
                })
        
        # 罗盘数据
        if self.compass:
            compass_values = self.compass.getValues()
            if compass_values and len(compass_values) >= 3:
                sensor_data.update({
                    'compass_direction': np.array(compass_values),
                    'compass_available': True
                })
        
        # 机器人节点数据（完美数据，仅用于对比和初始化）
        if self.robot_node:
            translation = self.robot_node.getField('translation').getSFVec3f()
            rotation = self.robot_node.getField('rotation').getSFRotation()
            
            sensor_data.update({
                'true_position': np.array(translation),
                'true_rotation': np.array(rotation),
                'true_available': True
            })
        
        return sensor_data
    
    def _filter_sensor_data(self, sensor_data: Dict) -> Dict:
        """滤波传感器数据"""
        filtered_data = sensor_data.copy()
        
        # GPS位置滤波
        if sensor_data.get('gps_position') is not None:
            filtered_pos = self.position_filter.update(sensor_data['gps_position'])
            filtered_data['filtered_position'] = filtered_pos
        
        # 速度估计（位置微分）
        current_time = sensor_data['timestamp']
        if self.previous_state:
            dt = current_time - self.previous_state.timestamp
            if dt > 1e-6:  # 避免除零
                velocity_estimate = (filtered_pos - self.previous_state.position) / dt
                filtered_velocity = self.velocity_filter.update(velocity_estimate)
                filtered_data['filtered_velocity'] = filtered_velocity
        
        # IMU方向滤波
        if sensor_data.get('imu_orientation') is not None:
            filtered_orient = self.imu_filter.update(sensor_data['imu_orientation'])
            filtered_data['filtered_orientation'] = filtered_orient
        
        return filtered_data
    
    def _apply_kalman_filter(self, filtered_data: Dict) -> Dict:
        """应用卡尔曼滤波融合"""
        # 预测步骤
        if 'filtered_velocity' in filtered_data:
            # 根据速度更新时间步长
            self.kalman_filter.dt = 0.05  # 固定时间步长
        else:
            self.kalman_filter.dt = 0.05
        
        self.kalman_filter.update_transition_matrix()
        self.kalman_filter.predict()
        
        # 更新步骤
        if 'filtered_position' in filtered_data:
            self.kalman_filter.update_with_position(filtered_data['filtered_position'])
        
        if 'filtered_velocity' in filtered_data:
            self.kalman_filter.update_with_velocity(filtered_data['filtered_velocity'])
        
        return self.kalman_filter.get_state()
    
    def _calculate_derived_states(self, kalman_state: Dict) -> Dict:
        """计算派生状态"""
        # 加速度计算
        acceleration = kalman_state['acceleration']
        
        # 角速度计算（多种来源融合）
        angular_velocity = np.zeros(3)
        
        # 来自IMU的直接测量
        if 'gyro_angular_velocity' in self._last_sensor_data:
            angular_velocity = self.imu_filter.update(self._last_sensor_data['gyro_angular_velocity'])
        else:
            # 如果没有陀螺仪，从姿态变化估算
            if self.previous_state:
                dt = kalman_state['timestamp'] - self.previous_state.timestamp if 'timestamp' in kalman_state else 0.05
                if dt > 1e-6:
                    # 简单的角速度估计
                    angle_change = (self._last_kalman_state.get('orientation', np.zeros(3)) - 
                                   self.previous_state.orientation) if 'orientation' in self._last_kalman_state else np.zeros(3)
                    angular_velocity = angle_change / dt
        
        return {
            'orientation': kalman_state.get('position', np.zeros(3)),  # 需要从其他来源获取
            'angular_velocity': angular_velocity,
            'linear_acceleration': acceleration
        }
    
    def _calculate_uncertainty(self, filtered_data: Dict, kalman_state: Dict) -> np.ndarray:
        """计算状态不确定性"""
        # 基础不确定性
        base_uncertainty = np.array([
            self.position_uncertainty[0],    # x位置
            self.position_uncertainty[1],    # y位置  
            self.position_uncertainty[2],    # z位置
            0.02,  # roll
            0.02,  # pitch
            0.03,  # yaw
            self.velocity_uncertainty[0],    # x速度
            self.velocity_uncertainty[1],    # y速度
            self.velocity_uncertainty[2],    # z速度
        ])
        
        # 根据传感器数据质量和KF协方差调整
        if 'covariance' in kalman_state:
            covariance_uncertainty = np.abs(kalman_state['covariance'][:6]) / 10.0
            adjusted_uncertainty = np.maximum(base_uncertainty, covariance_uncertainty[:6])
        else:
            adjusted_uncertainty = base_uncertainty
        
        return adjusted_uncertainty
    
    def _record_state_history(self, current_state: PoseState):
        """记录状态历史"""
        self.state_history.append(current_state)
        
        # 保持历史记录在合理范围
        if len(self.state_history) > 20:
            self.state_history.popleft()
    
    def get_velocity_from_history(self, window: int = 5) -> np.ndarray:
        """从历史记录获取平均速度"""
        if len(self.state_history) < window:
            return np.zeros(3)
        
        recent_velocities = [state.velocity for state in list(self.state_history)[-window:]]
        return np.mean(recent_velocities, axis=0)
    
    def get_position_trend(self, window: int = 10) -> np.ndarray:
        """获取位置趋势"""
        if len(self.state_history) < window:
            return np.zeros(3)
        
        recent_positions = [state.position for state in list(self.state_history)[-window:]]
        return np.mean(recent_positions, axis=0)
    
    def is_pose_stable(self, threshold: float = 0.1) -> bool:
        """检查位姿是否稳定"""
        if len(self.state_history) < 5:
            return False
        
        recent_positions = np.array([state.position for state in list(self.state_history)[-5:]])
        position_variance = np.var(recent_positions, axis=0)
        
        return np.all(position_variance < threshold)
    
    def estimate_uncertainty_from_amcl_comparison(self, amcl_position: np.ndarray) -> float:
        """通过与AMCL位置比较估计不确定性"""
        try:
            current_estimate = self.state_history[-1].position
            position_diff = np.linalg.norm(current_estimate - amcl_position)
            return position_diff
        except (IndexError, AttributeError):
            return 0.1  # 默认不确定性
    
    def reset(self):
        """重置姿态估计器"""
        # 重置滤波器
        self.position_filter.reset()
        self.velocity_filter.reset()
        self.imu_filter.reset()
        self.kalman_filter = KalmanFilter3D()
        
        # 重置历史
        self.state_history.clear()
        self.previous_state = None
        self._last_sensor_data = None
        self._last_kalman_state = None
    
    def get_pose_statistics(self) -> Dict:
        """获取位姿统计信息"""
        if len(self.state_history) < 3:
            return {}
        
        positions = np.array([state.position for state in self.state_history])
        orientations = np.array([state.orientation for state in self.state_history])
        
        return {
            'position_mean': np.mean(positions, axis=0),
            'position_std': np.std(positions, axis=0),
            'orientation_mean': np.mean(orientations, axis=0),
            'orientation_std': np.std(orientations, axis=0),
            'traveled_distance': self._calculate_traveled_distance(),
            'stability_score': self._calculate_stability_score()
        }
    
    def _calculate_traveled_distance(self) -> float:
        """计算总行程"""
        if len(self.state_history) < 2:
            return 0.0
        
        total_distance = 0.0
        positions = [state.position for state in self.state_history]
        
        for i in range(1, len(positions)):
            segment_distance = np.linalg.norm(positions[i] - positions[i-1])
            total_distance += segment_distance
        
        return total_distance
    
    def _calculate_stability_score(self) -> float:
        """计算稳定性分数"""
        if len(self.state_history) < 5:
            return 1.0
        
        positions = np.array([state.position for state in self.state_history])
        
        # 计算位置变化的变异系数
        position_changes = positions[1:] - positions[:-1]
        cv_scores = np.std(position_changes, axis=0) / (np.mean(np.abs(position_changes), axis=0) + 1e-10)
        
        stability_score = 1.0 - np.mean(cv_scores)  # 越稳定分数越高
        return np.clip(stability_score, 0.0, 1.0)