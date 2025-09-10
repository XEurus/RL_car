#!/usr/bin/env python3
"""
直接训练ROSbot导航模型 - 绕过复杂环境配置
"""

import numpy as np
import torch
import gymnasium as gym
import math
from pathlib import Path
import time
import json
import argparse

# 添加路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入核心类
from src.localization.amcl_localizer import AMCLLocalizer, UncertaintyCurriculumTraining

# 使用SimpleEnvironment
class SimpleNavEnv(gym.Env):
    """简化的导航环境 - 直接集成AMCL"""
    
    def __init__(self, cargo_type='normal'):
        super().__init__()
        
        # 42维状态空间
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * 20 + [-20.0] * 6 + [-math.pi] * 3 + [-2.0] * 3 + [-20.0] * 6 + [-math.pi] * 4),
            high=np.array([10.0] * 20 + [20.0] * 6 + [math.pi] * 3 + [2.0] * 3 + [20.0] * 6 + [math.pi] * 4),
            dtype=np.float32
        )
        
        # 动作空间：线速度和角速度
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -2.0]),
            high=np.array([2.0, 2.0]),
            dtype=np.float32
        )
        
        self.cargo_type = cargo_type
        self.amcl_localizer = None
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([10.0, 8.0, 0.0])
        self.start_position = np.array([-8.0, -6.0, 0.0])
        
        # 轨迹记录
        self.trajectory = []
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 初始化AMCL定位器
        if self.amcl_localizer is None:
            self.amcl_localizer = AMCLLocalizer(num_particles=800)
        self.amcl_localizer.reset()
        
        # 重置位置
        self.current_position = self.start_position.copy()
        self.current_orientation = np.array([0.0, 0.0, 0.0])
        self.trajectory = []
        self.step_count = 0
        
        # 初始化AMCL（初始位置在起点附近）
        initial_pose = np.array([self.start_position[0], self.start_position[1], 0.0])
        self.amcl_localizer.initialize_with_pose(initial_pose)
        
        # 获取观察
        obs = self._get_observation()
        
        info = {
            'start_position': self.start_position.tolist(),
            'target_position': self.target_position.tolist(),
            'cargo_type': self.cargo_type
        }
        
        return obs, info
    
    def step(self, action):
        """执行动作"""
        # 执行动作（模拟运动）
        linear_vel = float(action[0])
        angular_vel = float(action[1])
        
        # 更新位置（简单的运动学模型）
        dt = 0.1  # 时间步长
        old_position = self.current_position.copy()
        
        self.current_position[0] += linear_vel * dt * math.cos(self.current_orientation[2])
        self.current_position[1] += linear_vel * dt * math.sin(self.current_orientation[2])
        self.current_position[2] = 0.0  # 保持在平面
        
        self.current_orientation[2] += angular_vel * dt
        
        # 确保角度在[-pi, pi]范围内
        while self.current_orientation[2] > math.pi:
            self.current_orientation[2] -= 2 * math.pi
        while self.current_orientation[2] < -math.pi:
            self.current_orientation[2] += 2 * math.pi
        
        # 更新轨迹
        self.trajectory.append(self.current_position.copy())
        self.step_count += 1
        
        # 模拟LiDAR数据
        lidar_data = self._simulate_lidar()
        
        # 模拟里程计数据
        odometry = {
            'dx': self.current_position[0] - old_position[0],
            'dy': self.current_position[1] - old_position[1],
            'dyaw': self.current_orientation[2] - 0.0,  # 简化yaw变化
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel
        }
        
        # AMCL定位更新
        if self.amcl_localizer:
            amcl_result = self.amcl_localizer.localize(lidar_data, odometry)
        else:
            amcl_result = self._create_mock_amcl_result()
        
        # 计算奖励
        reward = self._calculate_reward_amcl(amcl_result, action)
        
        # 检查终止条件
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_steps
        
        # 获取观察
        obs = self._get_observation_amcl(amcl_result)
        
        # 添加步信息
        info = {
            'position': self.current_position.tolist(),
            'target_position': self.target_position.tolist(),
            'distance_to_target': float(np.linalg.norm(self.current_position - self.target_position)),
            'amcl_uncertainty': float(np.mean(amcl_result.get('position_uncertainty', [0.1]))),
            'trajectory_length': len(self.trajectory),
            'cargo_type': self.cargo_type
        }
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_lidar(self):
        """模拟LiDAR数据"""
        num_beams = 20
        max_range = 10.0
        scan_ranges = []
        
        # 生成扇形扫描模式
        angles = np.linspace(-math.pi/2, math.pi/2, num_beams)
        
        for angle in angles:
            # 世界坐标系中的射线方向
            global_angle = self.current_orientation[2] + angle
            
            # 模拟到障碍物的距离（基于位置）
            x, y = self.current_position[0], self.current_position[1]
            
            # 基于地图边界和简单障碍物的距离模型
            boundary_distance = self._calculate_boundary_distance(angle)
            obstacle_distance = self._calculate_obstacle_distance(x, y, angle)
            
            # 取最小距离作为测量值
            actual_distance = min(boundary_distance, obstacle_distance)
            
            # 添加测量噪声
            noise = np.random.normal(0, 0.1)
            measured_distance = max(0.01, actual_distance + noise)
            
            # 归一化到[0, 1]范围
            normalized_distance = min(1.0, measured_distance / max_range)
            scan_ranges.append(normalized_distance)
        
        return np.array(scan_ranges, dtype=np.float32)
    
    def _calculate_boundary_distance(self, angle):
        """计算到边界的距离"""
        # 简单的矩形边界
        x, y = self.current_position[0], self.current_position[1]
        global_angle = self.current_orientation[2] + angle
        
        # 边界
        min_x, max_x = -15.0, 15.0
        min_y, max_y = -12.0, 12.0
        
        # 计算射线到各边界的交点距离
        cos_a = math.cos(global_angle)
        sin_a = math.sin(global_angle)
        
        distances = []
        
        # 到左右边界的距离
        if abs(cos_a) > 1e-6:
            if cos_a > 0:
                distances.append((max_x - x) / cos_a)
            else:
                distances.append((min_x - x) / cos_a)
        
        # 到上下边界的距离
        if abs(sin_a) > 1e-6:
            if sin_a > 0:
                distances.append((max_y - y) / sin_a)
            else:
                distances.append((min_y - y) / sin_a)
        
        return min(distances) if distances else 10.0
    
    def _calculate_obstacle_distance(self, x, y, angle):
        """计算到简单障碍物的距离"""
        # 创建几个简单的圆柱形障碍物
        obstacles = [
            {'x': 0.0, 'y': 0.0, 'radius': 1.5},  # 中心圆柱
            {'x': 5.0, 'y': 3.0, 'radius': 1.0},  # 右侧圆柱
            {'x': -4.0, 'y': 5.0, 'radius': 0.8}, # 左侧圆柱  
            {'x': 8.0, 'y': -2.0, 'radius': 1.2}, # 右下角圆柱
        ]
        
        global_angle = self.current_orientation[2] + angle
        ray_dir = np.array([math.cos(global_angle), math.sin(global_angle)])
        
        min_distance = 10.0
        
        for obstacle in obstacles:
            # 从当前位置到障碍物的向量
            to_obstacle = np.array([obstacle['x'] - x, obstacle['y'] - y])
            distance_to_obstacle = np.linalg.norm(to_obstacle)
            
            # 如果射线方向指向障碍物
            if distance_to_obstacle < min_distance:
                # 计算射线与圆的距离（简化版）
                min_distance = min(min_distance, distance_to_obstacle - obstacle['radius'])
        
        return max(min_distance, 0.01)
    
    def _create_mock_amcl_result(self):
        """创建模拟的AMCL结果"""
        return {
            'position_estimated': self.current_position.copy(),
            'orientation_estimated': self.current_orientation.copy(),
            'velocity_estimated': np.array([0.1, 0.0, 0.0]),
            'position_uncertainty': np.array([0.1, 0.1, 0.1]),
            'orientation_uncertainty': np.array([0.05, 0.05, 0.05])
        }
    
    def _get_observation_amcl(self, amcl_result):
        """基于AMCL结果创建观察"""
        obs = np.zeros(42, dtype=np.float32)
        
        # 1. LiDAR数据 (20维)
        obs[0:20] = self._simulate_lidar()
        
        # 2. AMCL定位结果 (12维)
        obs[20:23] = self.current_position  # 真实位置
        obs[23:26] = amcl_result['position_estimated'][:3]  # 估计位置 (只取前3维)
        obs[26:29] = self.current_orientation  # 真实姿态
        obs[29:32] = amcl_result['orientation_estimated'][:3]  # 估计姿态  
        obs[32:35] = amcl_result['velocity_estimated'][:3]  # 速度 (只取前3维)
        
        # 3. 导航信息 (6维)
        relative_target = self.target_position[:3] - self.current_position[:3]
        obs[35:38] = relative_target
        obs[38:41] = self.start_position[:3]  # 起点位置
        # 注意：下一个3维目标是终点，但我们只有42维总空间
        # 调整下标，只使用接下来的3维而不是6维
        
        # 4. 航向控制 (4维)
        target_vector = self.target_position[:3] - self.current_position[:3]
        target_heading = math.atan2(target_vector[1], target_vector[0])
        current_heading = self.current_orientation[2]
        heading_error = target_heading - current_heading
        
        # 归一化航向偏差
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        obs[38] = heading_error  # 重新使用导航信息的第4维开始
        obs[39] = target_heading
        obs[40] = 0.0  # 角速度
        obs[41] = 0.0  # 角加速度
        
        return obs
    
    def _calculate_reward_amcl(self, amcl_result, action):
        """基于AMCL结果计算奖励"""
        reward = 0.0
        
        # 1. 距离奖励
        estimated_pos = amcl_result['position_estimated']
        distance_to_target = np.linalg.norm(estimated_pos - self.target_position)
        
        prev_distance = np.linalg.norm(self.current_position - self.target_position)
        reward += (prev_distance - distance_to_target) * 10.0
        
        # 2. 航向奖励
        target_vector = self.target_position - estimated_pos
        target_heading = math.atan2(target_vector[1], target_vector[0])
        current_heading = amcl_result['orientation_estimated'][2]
        
        heading_error = target_heading - current_heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        reward += -abs(heading_error) * 2.0
        
        # 3. 货物类型特定奖励
        linear_vel = abs(action[0])
        angular_vel = abs(action[1])
        
        if self.cargo_type == 'normal':
            # 普通货物 - 平衡性能
            reward += -0.1  # 时间惩罚
            
        elif self.cargo_type == 'fragile':
            # 易碎货物 - 稳定性优先
            stability_penalty = -(linear_vel * 0.5 + angular_vel * 0.3)
            reward += stability_penalty
            if linear_vel > 1.0:  # 速度限制
                reward += -(linear_vel - 1.0) * 3.0
                
        elif self.cargo_type == 'dangerous':
            # 危险货物 - 安全优先
            reward += -(linear_vel * 2.0 + angular_vel * 1.5)
            if linear_vel > 0.8:
                reward += -(linear_vel - 0.8) * 5.0
        
        # 4. 不确定性惩罚
        uncertainty = np.mean(amcl_result.get('position_uncertainty', [0.1]))
        reward += -uncertainty * 1.0
        
        return reward
    
    def _get_observation(self):
        """获取观察（包含AMCL更新）"""
        # 模拟AMCL定位
        lidar_data = self._simulate_lidar()
        
        # 模拟里程计
        odometry = {
            'dx': 0.05,
            'dy': 0.02,
            'dyaw': 0.01,
            'linear_velocity': 0.1,
            'angular_velocity': 0.05
        }
        
        if self.amcl_localizer:
            amcl_result = self.amcl_localizer.localize(lidar_data, odometry)
            return self._get_observation_amcl(amcl_result)
        else:
            return self._get_observation_amcl(self._create_mock_amcl_result())
    
    def _check_termination(self):
        """检查终止条件"""
        # 到达目标
        distance = np.linalg.norm(self.current_position - self.target_position)
        if distance < 0.15:
            return True
        
        # 超出边界
        if abs(self.current_position[0]) > 20 or abs(self.current_position[1]) > 15:
            return True
        
        return False

def train_single_cargo(cargo_type='normal', steps=10000, args=None):
    """训练单一货物类型"""
    print(f"🚦 开始训练 {cargo_type} 类型...")
    
    # 创建环境
    env = SimpleNavEnv(cargo_type=cargo_type)
    
    # 使用标准TD3 - 使用getattr安全获取参数
    from stable_baselines3 import TD3
    model = TD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=getattr(args, 'learning_rate', 3e-4) if args else 3e-4,
        buffer_size=getattr(args, 'buffer_size', 100000) if args else 100000,
        learning_starts=getattr(args, 'learning_starts', 1000) if args else 1000,
        batch_size=getattr(args, 'batch_size', 256) if args else 256,
        gamma=getattr(args, 'gamma', 0.99) if args else 0.99,
        tau=getattr(args, 'tau', 0.005) if args else 0.005,
        policy_delay=getattr(args, 'policy_delay', 2) if args else 2,
        target_policy_noise=getattr(args, 'target_noise', 0.2) if args else 0.2,
        target_noise_clip=getattr(args, 'noise_clip', 0.5) if args else 0.5,
        verbose=1,
        tensorboard_log=f"./logs/simple_{cargo_type}"
    )
    
    # 打印关键参数
    print("\n📊 训练参数:")
    print(f"  - 学习率: {getattr(args, 'learning_rate', 3e-4) if args else 3e-4}")
    print(f"  - 缓冲区大小: {getattr(args, 'buffer_size', 100000) if args else 100000}")
    print(f"  - 批处理大小: {getattr(args, 'batch_size', 256) if args else 256}")
    print(f"  - 预热步数: {getattr(args, 'learning_starts', 1000) if args else 1000}")
    print(f"  - 折扣因子: {getattr(args, 'gamma', 0.99) if args else 0.99}")
    
    start_time = time.time()
    
    # 开始训练
    print(f"🎯 开始训练 - 步数: {steps}")
    model.learn(total_timesteps=steps, progress_bar=True)
    
    training_time = time.time() - start_time
    print(f"⏰ 训练完成，用时: {training_time:.1f}秒")
    
    # 保存模型
    Path("./models").mkdir(exist_ok=True)
    model_path = f"./models/simple_model_{cargo_type}.zip"
    model.save(model_path)
    
    return model, env, model_path

def test_model(model, env, cargo_type='normal', episodes=3):
    """测试模型性能"""
    print(f"\n🧪 测试 {cargo_type} 模型性能...")
    
    rewards = []
    lengths = []
    successes = 0
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0
        max_steps = 150
        
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated and info.get('distance_to_target', 999) < 0.2:
                successes += 1
                done = True
            elif truncated:
                done = True
        
        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {episode+1}: 奖励={total_reward:.2f}, 步数={steps}")
    
    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)
    success_rate = successes / episodes
    
    print(f"\n🎯 测试结果:")
    print(f"  📊 平均奖励: {avg_reward:.2f}")
    print(f"  📏 平均步数: {avg_length:.1f}")
    print(f"  🎯 成功率: {success_rate:.1%}")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'rewards': rewards,
        'lengths': lengths
    }

def main():
    """主函数 - 训练所有货物类型"""
    parser = argparse.ArgumentParser(description='直接训练 - 简化环境')
    parser.add_argument('--num_envs', type=int, default=1, help='并行环境数量（Dummy/Subproc）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # TD3算法相关参数
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--buffer_size', type=int, default=100000, help='经验回放缓冲区大小')
    parser.add_argument('--learning_starts', type=int, default=1000, help='预热步数，开始学习前收集的样本数量')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005, help='目标网络软更新系数')
    parser.add_argument('--policy_delay', type=int, default=2, help='策略延迟更新步数')
    parser.add_argument('--target_noise', type=float, default=0.2, help='目标策略噪声')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='噪声裁剪范围')
    
    args = parser.parse_args()

    print("="*70)
    print("🚀 ROSbot导航训练系统 - 直接版本")
    print("="*70)
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 训练配置
    training_configs = {
        'normal': {'steps': 5000, 'description': '普通货物 - 高性能导航'},
        'fragile': {'steps': 6000, 'description': '易碎货物 - 稳定性优先'},
        'dangerous': {'steps': 7000, 'description': '危险货物 - 安全优先'}
    }
    
    all_results = {}
    
    for cargo_type, config in training_configs.items():
        print(f"\n{'='*50}")
        print(f"📦 {config['description']}")
        print(f"🎯 训练步数: {config['steps']}")
        print('='*50)
        
        # 训练模型
        model, env, model_path = train_single_cargo(cargo_type, config['steps'], args=args)
        
        # 测试模型
        test_results = test_model(model, env, cargo_type)
        
        all_results[cargo_type] = {
            'config': config,
            'model_path': model_path,
            'test_results': test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n💾 模型已保存: {model_path}")
        
        # 关闭环境以释放资源
        try:
            env.close()
        except:
            pass
    
    # 保存完整的结果
    results_file = f"./results/training_summary_{int(time.time())}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 训练总结:")
    print('='*70)
    
    for cargo_type, result in all_results.items():
        test = result['test_results']
        print(f"\n📦 {cargo_type.upper()}:")
        print(f"  ✅ 成功率: {test['success_rate']:.1%}")
        print(f"  📊 平均奖励: {test['avg_reward']:.2f}")
        print(f"  📏 平均步数: {test['avg_length']:.1f}")
        print(f"  📁 模型: {result['model_path']}")
    
    print(f"\n💾 详细结果已保存: {results_file}")
    print('\n🎉 所有训练完成！')
    print("="*70)

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    main()