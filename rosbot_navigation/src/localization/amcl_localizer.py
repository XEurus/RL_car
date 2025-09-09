"""
自适应蒙特卡洛定位(AMCL)实现
提供粒子滤波定位，估计机器人位置和姿态
包含不确定性建模
"""

import numpy as np
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class Particle:
    """单个粒子"""
    x: float      # x位置
    y: float      # y位置  
    z: float      # z位置
    roll: float   # 横滚角
    pitch: float  # 俯仰角
    yaw: float    # 偏航角
    weight: float = 1.0  # 权重
    
    def __post_init__(self):
        # 归一化角度到[-π, π]
        self.roll = self._normalize_angle(self.roll)
        self.pitch = self._normalize_angle(self.pitch)
        self.yaw = self._normalize_angle(self.yaw)
    
    def _normalize_angle(self, angle):
        """归一化角度到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class ParticleFilter:
    """粒子滤波器核心"""
    
    def __init__(self, num_particles: int = 800):
        self.num_particles = num_particles
        self.particles = []
        self.weights = np.ones(num_particles) / num_particles
        
        # 自适应重采样参数
        self.alpha_slow = 0.001
        self.alpha_fast = 0.1
        self.w_slow = 0.0
        self.w_fast = 0.0
        
        # 运动模型噪声参数
        self.alpha1 = 0.1      # 旋转误差
        self.alpha2 = 0.01     # 平移误差
        self.alpha3 = 0.1      # 旋转误差
        self.alpha4 = 0.01     # 平移误差
        
        # 运动历史
        self.motion_history = deque(maxlen=10)
        
    def initialize_particles(self, mean_pose: np.ndarray, std_dev: List[float]):
        """用高斯分布初始化粒子"""
        self.particles = []
        
        for i in range(self.num_particles):
            x = np.random.normal(mean_pose[0], std_dev[0])
            y = np.random.normal(mean_pose[1], std_dev[1])
            z = np.random.normal(mean_pose[2], std_dev[2])
            roll = np.random.normal(mean_pose[3], std_dev[3])
            pitch = np.random.normal(mean_pose[4], std_dev[4])
            yaw = np.random.normal(mean_pose[5], std_dev[5])
            
            particle = Particle(x, y, z, roll, pitch, yaw, weight=1.0)
            self.particles.append(particle)
    
    def predict(self, control_input: Dict):
        """运动模型预测"""
        # 提取控制输入
        dx = control_input.get('dx', 0.0)
        dy = control_input.get('dy', 0.0)
        dyaw = control_input.get('dyaw', 0.0)
        
        delta_trans = math.sqrt(dx**2 + dy**2)
        
        for particle in self.particles:
            # 计算运动变换
            if delta_trans > 0.001:  # 忽略微小运动
                delta_rot1 = math.atan2(dy, dx)
                delta_rot2 = dyaw - delta_rot1
                
                # 添加运动噪声（真实误差模型）
                noisy_delta_rot1 = delta_rot1 + np.random.normal(0, self.alpha1 * abs(delta_rot1))
                noisy_delta_trans = delta_trans + np.random.normal(0, self.alpha2 * delta_trans)
                noisy_delta_rot2 = delta_rot2 + np.random.normal(0, self.alpha3 * abs(delta_rot2))
                
                # 更新粒子位置
                particle.x += noisy_delta_trans * math.cos(particle.yaw + noisy_delta_rot1)
                particle.y += noisy_delta_trans * math.sin(particle.yaw + noisy_delta_rot1)
                particle.z = 0.0  # 假设在2D平面运动
                
                # 更新粒子姿态
                particle.yaw += noisy_delta_rot1 + noisy_delta_rot2
                
                # 添加额外的平移噪声
                noisy_delta_trans2 = np.random.normal(0, self.alpha4 * noisy_delta_trans)
                particle.x += noisy_delta_trans2 * math.cos(particle.yaw)
                particle.y += noisy_delta_trans2 * math.sin(particle.yaw)
            else:
                # 纯旋转运动
                particle.yaw += dyaw + np.random.normal(0, self.alpha1 * abs(dyaw))
            
            # 确保角度在有效范围内
            particle.yaw = particle._normalize_angle(particle.yaw)
    
    def update_weights(self, measurements: np.ndarray):
        """更新粒子权重"""
        for i, particle in enumerate(self.particles):
            particle.weight = measurements[i] + 1e-10  # 防止零权重
        
        # 权重归一化
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        
        # 更新自适应重采样参数
        self._update_adaptive_params()
    
    def _update_adaptive_params(self):
        """更新自适应重采样参数"""
        w_avg = np.mean([p.weight for p in self.particles])
        
        if self.w_slow == 0:
            self.w_slow = w_avg
        else:
            self.w_slow += self.alpha_slow * (w_avg - self.w_slow)
        
        if self.w_fast == 0:
            self.w_fast = w_avg
        else:
            self.w_fast += self.alpha_fast * (w_avg - self.w_fast)
    
    def resample(self):
        """自适应重采样"""
        # 计算有效粒子数
        effective_n = self.effective_sample_size()
        
        # 自适应重采样决策
        if effective_n < 0.5 * self.num_particles:
            self._systematic_resample()
        elif self.w_fast > 0 and self.w_slow > 0:
            # 如果快速平均权重大于慢速平均权重，进行重采样
            if self.w_fast > 1.1 * self.w_slow:
                self._systematic_resample()
    
    def effective_sample_size(self) -> float:
        """计算有效粒子数"""
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights**2)
    
    def _systematic_resample(self):
        """系统重采样"""
        indices = np.arange(self.num_particles)
        cumulative_sum = np.cumsum([p.weight for p in self.particles])
        cumulative_sum[-1] = 1.0  # 避免数值误差
        
        # 系统采样点
        start = np.random.uniform(0, 1.0 / self.num_particles)
        step = 1.0 / self.num_particles
        samples = start + np.arange(self.num_particles) * step
        
        # 找到对应的粒子索引
        indices = np.searchsorted(cumulative_sum, samples)
        indices = np.clip(indices, 0, self.num_particles - 1)
        
        # 重采样粒子
        new_particles = []
        for idx in indices:
            old_particle = self.particles[idx]
            # 添加少量重采样噪声避免粒子枯竭
            new_particle = Particle(
                x=old_particle.x + np.random.normal(0, 0.01),
                y=old_particle.y + np.random.normal(0, 0.01),
                z=old_particle.z,
                roll=old_particle.roll + np.random.normal(0, 0.005),
                pitch=old_particle.pitch + np.random.normal(0, 0.005),
                yaw=old_particle.yaw + np.random.normal(0, 0.01),
                weight=1.0 / self.num_particles
            )
            new_particles.append(new_particle)
        
        self.particles = new_particles
    
    def estimate_pose(self) -> Dict:
        """估计机器人位姿"""
        if not self.particles:
            return None
        
        # 加权平均位置
        positions = np.array([[p.x, p.y, p.z] for p in self.particles])
        weights = np.array([p.weight for p in self.particles])
        
        estimated_position = np.average(positions, axis=0, weights=weights)
        
        # 角度平均（处理环形特性）
        orientations = np.array([[p.roll, p.pitch, p.yaw] for p in self.particles])
        estimated_orientation = self._circular_mean(orientations.T, weights)
        
        # 计算不确定性
        position_uncertainty = self._calculate_position_uncertainty(positions, weights)
        orientation_uncertainty = self._calculate_orientation_uncertainty(orientations, weights)
        
        # 计算动态参数（速度）
        velocity = self._estimate_velocity()
        
        return {
            'position_estimated': estimated_position,
            'orientation_estimated': estimated_orientation,
            'position_uncertainty': position_uncertainty,
            'orientation_uncertainty': orientation_uncertainty,
            'velocity_estimated': velocity,
            'weights_distribution': weights,
            'effective_sample_size': self.effective_sample_size()
        }
    
    def _circular_mean(self, angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """计算角度的环形平均值"""
        # 将角度转换为复数表示
        complex_angles = np.exp(1j * angles)
        
        # 计算加权平均复数
        weighted_complex = np.average(complex_angles, axis=1, weights=weights)
        
        # 转换回角度
        mean_angles = np.angle(weighted_complex)
        
        return mean_angles
    
    def _calculate_position_uncertainty(self, positions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """计算位置不确定性"""
        mean_position = np.average(positions, axis=0, weights=weights)
        
        # 计算加权方差
        variances = np.average((positions - mean_position)**2, axis=0, weights=weights)
        
        return np.sqrt(variances)
    
    def _calculate_orientation_uncertainty(self, orientations: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """计算姿态不确定性"""
        # 对于角度，计算环形方差
        mean_angles = self._circular_mean(orientations.T, weights)
        
        # 计算环形方差
        angle_diffs = orientations - mean_angles.reshape(1, -1)
        angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
        
        # 加权环形方差
        circular_variances = np.average(angle_diffs**2, axis=0, weights=weights)
        
        return np.sqrt(circular_variances)
    
    def _estimate_velocity(self) -> np.ndarray:
        """基于粒子运动估计速度"""
        if len(self.motion_history) < 2:
            return np.zeros(3)
        
        # 使用最近的运动历史
        recent_poses = list(self.motion_history)[-5:]  # 最近5个时刻
        
        if len(recent_poses) < 2:
            return np.zeros(3)
        
        # 计算平均速度
        velocities = []
        for i in range(1, len(recent_poses)):
            dt = recent_poses[i]['timestamp'] - recent_poses[i-1]['timestamp']
            if dt > 0:
                dx = recent_poses[i]['position'][0] - recent_poses[i-1]['position'][0]
                dy = recent_poses[i]['position'][1] - recent_poses[i-1]['position'][1]
                dz = recent_poses[i]['position'][2] - recent_poses[i-1]['position'][2]
                
                velocities.append([dx/dt, dy/dt, dz/dt])
        
        return np.mean(velocities, axis=0) if velocities else np.zeros(3)
    
    def has_converged(self, threshold: float = 0.1) -> bool:
        """检查是否收敛"""
        if not self.particles:
            return False
        
        weights = np.array([p.weight for p in self.particles])
        max_weight = np.max(weights)
        
        # 使用最大权重作为收敛指标
        return max_weight > threshold


class AMCLLocalizer:
    """AMCL定位器主类"""
    
    def __init__(self, num_particles: int = 800, initial_std: List[float] = None):
        self.particle_filter = ParticleFilter(num_particles)
        self.num_particles = num_particles
        
        # 初始不确定性参数 - 支持6D或3D
        if initial_std is None:
            # 默认6D不确定性 [x,y,z,roll,pitch,yaw]
            self.initial_std = [0.2, 0.2, 0.15, 0.05, 0.05, 0.1]
        elif len(initial_std) == 3:
            # 如果提供3D，自动扩展为6D
            self.initial_std = initial_std[:2] + [0.0] + [initial_std[2]-0.05, initial_std[2], initial_std[2]+0.05]
        else:
            self.initial_std = initial_std
        
        # 测量模型参数
        self.sigma_hit = 0.05      # 测量噪声
        self.z_max = 0.05         # 最大距离概率
        self.z_rand = 0.05        # 随机测量概率
        self.z_short = 0.01       # 短测量概率
        
        # 状态缓存
        self.previous_pose = None
        self.previous_time = None
        self.velocity_cache = deque(maxlen=5)  # 速度缓存
        
        # 不确定性跟踪
        self.position_uncertainty_history = deque(maxlen=100)
        self.convergence_history = deque(maxlen=50)
        
        # 训练模式标识
        self.is_training = False
        self.training_uncertainty_level = 0.0
    
    def initialize_with_pose(self, initial_pose: np.ndarray, initial_uncertainty: List[float] = None):
        """用初始位姿初始化粒子滤波器"""
        if initial_uncertainty is None:
            initial_uncertainty = self.initial_std
        
        # Debug information
        initial_pose = np.array(initial_pose)  # Convert to numpy array
        initial_uncertainty = np.array(initial_uncertainty)
        print(f"[DEBUG] initialize_with_pose: initial_pose shape={initial_pose.shape}, length={len(initial_pose)}")
        print(f"[DEBUG] initial_pose content: {initial_pose}")
        print(f"[DEBUG] initial_uncertainty length: {len(initial_uncertainty)}")
        
        # 确保初始姿势有6个维度 [x, y, z, roll, pitch, yaw]
        if len(initial_pose) == 3:
            # 如果只有位置和朝向，补全缺少的维度
            full_pose = np.concatenate([initial_pose[:2], [0.0], [0.0, 0.0], [initial_pose[2]]])
        elif len(initial_pose) == 6:
            full_pose = initial_pose
        else:
            raise ValueError(f"initial_pose must have 3 or 6 dimensions, got {len(initial_pose)}")
        
        self.particle_filter.initialize_particles(full_pose, initial_uncertainty)
        self.previous_pose = full_pose
        self.previous_time = None
        self.velocity_cache.clear()
    
    def reset(self):
        """重置定位器"""
        self.previous_pose = None
        self.previous_time = None
        self.velocity_cache.clear()
        self.position_uncertainty_history.clear()
        self.convergence_history.clear()
    
    def localize(self, lidar_scan: np.ndarray, wheel_odometry: Dict) -> Dict:
        """执行定位"""
        # 添加训练噪声（如果使用训练模式）
        if self.is_training:
            lidar_scan = self._add_training_noise(lidar_scan)
            wheel_odometry = self._augment_odometry_uncertainty(wheel_odometry)
        
        # 测量似然计算
        measurements = self._calculate_likelihoods(lidar_scan, wheel_odometry)
        
        # 粒子滤波更新
        self.particle_filter.predict(wheel_odometry)
        self.particle_filter.update_weights(measurements)
        self.particle_filter.resample()
        
        # 位姿估计
        pose_estimate = self.particle_filter.estimate_pose()
        
        # 添加噪声（训练模式）
        if self.is_training:
            pose_estimate = self._add_pose_noise(pose_estimate)
        
        # 状态历史记录
        self._record_pose_history(pose_estimate)
        
        # 不确定性计算
        pose_estimate['uncertainty_metrics'] = self._calculate_uncertainty_metrics()
        
        return pose_estimate
    
    def _calculate_likelihoods(self, lidar_scan: np.ndarray, wheel_odometry: Dict) -> np.ndarray:
        """计算每个粒子的测量似然度"""
        likelihoods = []
        
        for particle in self.particle_filter.particles:
            total_likelihood = 0.0
            valid_measurements = 0
            
            # 使用主要方向的LiDAR射线
            num_beams = min(len(lidar_scan), 20)
            angles = np.linspace(-math.pi/2, math.pi/2, num_beams)
            
            for i, (angle, range_measurement) in enumerate(zip(angles, lidar_scan[:num_beams])):
                # 世界坐标系中的射线
                global_angle = particle.yaw + angle
                beam_end_x = particle.x + range_measurement * math.cos(global_angle)
                beam_end_y = particle.y + range_measurement * math.sin(global_angle)
                
                # 计算到障碍物的距离（模拟似然场）
                # 这里需要实际的地图或障碍物信息
                dist_to_obstacle = self._estimate_obstacle_distance(beam_end_x, beam_end_y)
                
                if dist_to_obstacle is not None:
                    # 高斯似然函数
                    likelihood = math.exp(-(dist_to_obstacle**2) / (2 * self.sigma_hit**2))
                    total_likelihood += likelihood
                    valid_measurements += 1
            
            # 平均似然度 + 随机噪声项
            if valid_measurements > 0:
                avg_likelihood = total_likelihood / valid_measurements
                # 添加随机测量模型
                randomized_likelihood = (1 - self.z_rand) * avg_likelihood + self.z_rand * 0.5
                likelihoods.append(randomized_likelihood)
            else:
                likelihoods.append(self.z_rand * 0.5)
        
        return np.array(likelihoods)
    
    def _estimate_obstacle_distance(self, x: float, y: float) -> float:
        """估计到障碍物的最短距离（简单实现）"""
        # 简化的障碍物距离估计 - 实际应该使用地图
        # 这里返回一个合理的模拟值
        
        # 边界限制
        max_distance = 2.0
        
        # 简单的距离场模拟
        center_distance = math.sqrt(x**2 + y**2)
        
        if center_distance > 15.0:  # 超出工作区域
            return max_distance
        
        # 基于粒子位置分布的简单模型
        if self.particle_filter.particles:
            # 使用最近粒子的影响
            min_particle_dist = float('inf')
            for particle in self.particle_filter.particles[:50]:  # 使用部分粒子
                dist = math.sqrt((x - particle.x)**2 + (y - particle.y)**2)
                min_particle_dist = min(min_particle_dist, dist)
            
            if min_particle_dist < 0.5:
                return min_particle_dist
        
        return max_distance  # 默认返回最大距离
    
    def _add_training_noise(self, lidar_scan: np.ndarray) -> np.ndarray:
        """添加训练专用噪声"""
        noise_scale = self.uncertainty_level * self.training_uncertainty_level
        
        if noise_scale > 0:
            noisy_scan = lidar_scan + np.random.normal(0, noise_scale * 0.1, len(lidar_scan))
            return np.clip(noisy_scan, 0.01, 10.0)
        
        return lidar_scan
    
    def _augment_odometry_uncertainty(self, odometry: Dict) -> Dict:
        """增加里程计不确定性"""
        if self.uncertainty_level > 0:
            noise_scale = self.uncertainty_level * 0.1
            odometry = odometry.copy()
            odometry['dx'] += np.random.normal(0, noise_scale * abs(odometry.get('dx', 0)))
            odometry['dy'] += np.random.normal(0, noise_scale * abs(odometry.get('dy', 0)))
            odometry['dyaw'] += np.random.normal(0, noise_scale * abs(odometry.get('dyaw', 0)))
        
        return odometry
    
    def _add_pose_noise(self, pose_estimate: Dict) -> Dict:
        """给位姿估计添加噪声"""
        if self.uncertainty_level > 0:
            noise_scale = self.uncertainty_level * 0.05
            
            pose_estimate['position_estimated'] += np.random.normal(0, noise_scale, 3)
            pose_estimate['orientation_estimated'] += np.random.normal(0, noise_scale * 0.5, 3)
            pose_estimate['velocity_estimated'] += np.random.normal(0, noise_scale * 2.0, 3)
        
        return pose_estimate
    
    def _record_pose_history(self, pose_estimate: Dict):
        """记录位姿历史用于速度计算"""
        current_time = self._get_current_time()
        
        if self.previous_time is not None and current_time > self.previous_time:
            dt = current_time - self.previous_time
            
            # 计算速度（中心差分更稳定）
            current_pos = pose_estimate['position_estimated']
            prev_pos = self.previous_pose[:3] if self.previous_pose is not None else current_pos
            
            if dt > 0:
                velocity = (current_pos - prev_pos) / dt
                self.velocity_cache.append(velocity)
                
                # 添加到运动历史
                self.particle_filter.motion_history.append({
                    'timestamp': current_time,
                    'position': current_pos.copy(),
                    'orientation': pose_estimate['orientation_estimated'].copy()
                })
        
        self.previous_pose = np.concatenate([
            pose_estimate['position_estimated'],
            pose_estimate['orientation_estimated']
        ])
        self.previous_time = current_time
    
    def _calculate_uncertainty_metrics(self) -> Dict:
        """计算不确定性指标"""
        current_uncertainty = self.particle_filter.estimate_pose()
        
        # 位置不确定性
        pos_uncertainty = current_uncertainty.get('position_uncertainty', np.array([0.1, 0.1, 0.1]))
        
        # 姿态不确定性  
        orient_uncertainty = current_uncertainty.get('orientation_uncertainty', np.array([0.05, 0.05, 0.05]))
        
        # 收敛性指标
        convergence_score = self.particle_filter.has_converged()
        effective_n = self.effective_sample_size()
        
        # 记录历史
        self.position_uncertainty_history.append(np.mean(pos_uncertainty))
        self.convergence_history.append(convergence_score)
        
        return {
            'position_uncertainty': pos_uncertainty,
            'orientation_uncertainty': orient_uncertainty,
            'convergence_score': convergence_score,
            'effective_sample_size': effective_n,
            'avg_position_uncertainty': np.mean(list(self.position_uncertainty_history)) if self.position_uncertainty_history else 0.1
        }
    
    def get_current_uncertainty(self) -> float:
        """获取当前定位不确定性"""
        if hasattr(self.particle_filter, 'estimate_pose'):
            result = self.particle_filter.estimate_pose()
            if 'position_uncertainty' in result:
                return float(np.mean(result['position_uncertainty']))
        return 0.1  # 默认不确定性
    
    def get_particle_distribution(self) -> np.ndarray:
        """获取粒子分布用于可视化"""
        if self.particle_filter.particles:
            positions = np.array([[p.x, p.y, p.yaw, p.weight] for p in self.particle_filter.particles])
            return positions
        return np.empty((0, 4))
    
    def effective_sample_size(self) -> float:
        """获取有效样本大小"""
        return self.particle_filter.effective_sample_size()
    
    def has_converged(self) -> bool:
        """检查是否收敛"""
        return self.particle_filter.has_converged()
    
    def set_training_mode(self, training: bool):
        """设置训练模式"""
        self.is_training = training
    
    def set_uncertainty_level(self, level: float):
        """设置不确定性级别（训练用）"""
        self.uncertainty_level = np.clip(level, 0.0, 1.0)
    
    def _get_current_time(self) -> float:
        """获取当前时间"""
        # 这里应该使用Webots的时间或其他时间源
        import time
        return time.time()


class UncertaintyCurriculumTraining:
    """不确定性课程学习训练"""
    
    def __init__(self):
        self.uncertainty_levels = {
            'low': 0.1,      # 低不确定性训练
            'medium': 0.3,   # 中等不确定性
            'high': 0.6,     # 高不确定性
            'extreme': 0.9   # 极端不确定性（测试边界情况）
        }
        
        self.training_stages = [
            {'name': 'initial', 'uncertainty': 'low', 'duration': 50000},
            {'name': 'medium', 'uncertainty': 'medium', 'duration': 100000},
            {'name': 'advanced', 'uncertainty': 'high', 'duration': 150000},
            {'name': 'robustness_test', 'uncertainty': 'extreme', 'duration': 100000}
        ]
    
    def get_uncertainty_for_step(self, total_steps: int) -> float:
        """根据训练步数获取不确定性级别"""
        cumulative_steps = 0
        
        for stage in self.training_stages:
            cumulative_steps += stage['duration']
            if total_steps <= cumulative_steps:
                return self.uncertainty_levels[stage['uncertainty']]
        
        return self.uncertainty_levels['extreme']