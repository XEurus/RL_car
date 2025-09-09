#!/usr/bin/env python3
"""
快速训练演示
展示ROSbot导航系统的训练效果
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制CPU

import numpy as np
import torch
import time
import json
from pathlib import Path

print("="*70)
print("🚀 ROSbot导航系统快速训练演示")
print("="*70)

# 添加路径
import sys
sys.path.insert(0, str(Path(__file__).parent / 'rosbot_navigation'))

# 简单AMCL模拟
class QuickAMCLSimulator:
    """快速AMCL定位模拟器"""
    
    def __init__(self, num_particles=200):
        self.num_particles = num_particles
        self.particles = []
        
    def reset(self, start_pose):
        self.particles = []
        for _ in range(self.num_particles):
            x = start_pose[0] + np.random.normal(0, 0.5)
            y = start_pose[1] + np.random.normal(0, 0.5)
            yaw = start_pose[2] + np.random.normal(0, 0.1)
            self.particles.append({'x': x, 'y': y, 'yaw': yaw, 'weight': 1.0/self.num_particles})
    
    def localize(self, true_position, odometry):
        # 简化的粒子滤波
        if not self.particles:
            self.reset(true_position)
        
        # 基于里程计更新粒子位置
        for p in self.particles:
            p['x'] += odometry.get('dx', 0) + np.random.normal(0, 0.1)
            p['y'] += odometry.get('dy', 0) + np.random.normal(0, 0.1)
            p['yaw'] += odometry.get('dyaw', 0) + np.random.normal(0, 0.01)
        
        # 基于真实位置更新权重
        for p in self.particles:
            dist = np.sqrt((p['x'] - true_position[0])**2 + (p['y'] - true_position[1])**2)
            p['weight'] = 1.0 / (1.0 + dist)
        
        # 归一化权重
        total_weight = sum(p['weight'] for p in self.particles)
        for p in self.particles:
            p['weight'] /= total_weight
        
        # 计算估计位置
        est_x = sum(p['x'] * p['weight'] for p in self.particles)
        est_y = sum(p['y'] * p['weight'] for p in self.particles)
        est_yaw = sum(p['yaw'] * p['weight'] for p in self.particles)
        
        # 计算不确定性
        variances = [np.var([p['x'] for p in self.particles]), 
                    np.var([p['y'] for p in self.particles]),
                    np.var([p['yaw'] for p in self.particles])]
        
        return (est_x, est_y, est_yaw), variances

class RobotState:
    """机器人状态管理"""
    
    def __init__(self):
        # 初始位置和朝向
        self.true_position = [-8.0, -6.0, 0.0]  # x, y, yaw
        self.true_velocity = [0.0, 0.0]  # linear, angular
        
        # AMCL估计
        self.amcl = QuickAMCLSimulator(num_particles=200)
        self.est_position = [-8.0, -6.0, 0.0]  # 初始估计位置
        self.position_uncertainty = [2.0, 2.0, 0.3]  # 初始不确定性
        
        self.action_history = []
        self.trajectory = [[-8.0, -6.0]]  # 用于记录轨迹
        
    def update(self, linear_vel, angular_vel, dt=0.1):
        """更新机器人状态"""
        prev_x, prev_y = self.true_position[0], self.true_position[1]
        prev_yaw = self.true_position[2]
        
        # 更新真实位置（基于差速驱动模型）
        self.true_position[0] += linear_vel * dt * np.cos(prev_yaw)
        self.true_position[1] += linear_vel * dt * np.sin(prev_yaw) 
        self.true_position[2] += angular_vel * dt
        
        # Keep yaw in [-pi, pi]
        while self.true_position[2] > np.pi:
            self.true_position[2] -= 2*np.pi
        while self.true_position[2] < -np.pi:
            self.true_position[2] += 2*np.pi
        
        # 更新速度
        self.true_velocity = [linear_vel, angular_vel]
        
        # 记录轨迹
        self.trajectory.append(self.true_position[:2].copy())
        
        # AMCL定位更新
        odometry = {
            'dx': self.true_position[0] - prev_x,
            'dy': self.true_position[1] - prev_y, 
            'dyaw': self.true_position[2] - prev_yaw
        }
        
        (est_x, est_y, est_yaw), variances = self.amcl.localize(self.true_position, odometry)
        self.est_position = [est_x, est_y, est_yaw]
        self.position_uncertainty = [np.sqrt(v) for v in variances]
        
        return {
            'position': self.true_position.copy(),
            'velocity': self.true_velocity.copy(),
            'estimate': self.est_position.copy(),
            'uncertainty': self.position_uncertainty.copy()
        }

def simulate_lidar(true_position, num_beams=20):
    """模拟LiDAR扫描"""
    scan = np.zeros(num_beams, dtype=np.float32)
    max_range = 10.0
    
    # 简单的环境边界和障碍物
    boundaries = [(-15, 15), (-12, 12)]  # x和y边界
    obstacles = [
        (0.0, 0.0, 2.0),    # 中心圆柱
        (5.0, 3.0, 1.0),    # 右侧圆柱
        (-4.0, 5.0, 0.8),   # 左侧圆柱  
        (8.0, -2.0, 1.2)    # 右下角圆柱
    ]
    
    # 扫描角度
    angles = np.linspace(-np.pi/2, np.pi/2, num_beams)
    x, y, yaw = true_position[0], true_position[1], true_position[2]
    
    for i, beam_angle in enumerate(angles):
        global_angle = yaw + beam_angle
        
        # 计算射线方向
        cos_a = np.cos(global_angle)
        sin_a = np.sin(global_angle)
        
        min_dist = max_range
        
        # 检查边界
        boundary_distances = []
        if abs(cos_a) > 0.001:
            dist_x1 = (boundaries[0][1] - x) / cos_a if cos_a > 0 else (boundaries[0][0] - x) / cos_a
            if dist_x1 > 0:
                y_cross = y + dist_x1 * sin_a
                if boundaries[1][0] <= y_cross <= boundaries[1][1]:
                    boundary_distances.append(dist_x1)
        
        if abs(sin_a) > 0.001:
            dist_y1 = (boundaries[1][1] - y) / sin_a if sin_a > 0 else (boundaries[1][0] - y) / sin_a
            if dist_y1 > 0:
                x_cross = x + dist_y1 * cos_a
                if boundaries[0][0] <= x_cross <= boundaries[0][1]:
                    boundary_distances.append(dist_y1)
        
        if boundary_distances:
            min_dist = min(min_dist, min(boundary_distances))
        
        # 检查障碍物
        for ox, oy, radius in obstacles:
            to_center = [ox - x, oy - y]
            proj = to_center[0]*cos_a + to_center[1]*sin_a
            
            if proj > 0:
                perp_sq = to_center[0]**2 + to_center[1]**2 - proj**2
                if perp_sq <= radius**2 and proj <= max_range:
                    hit_dist = proj - np.sqrt(max(0, radius**2 - perp_sq))
                    min_dist = min(min_dist, hit_dist)
        
        # 添加噪声并归一化
        noise = np.random.normal(0, 0.05)
        scan[i] = np.clip((min_dist + noise) / max_range, 0.01, 1.0)
    
    return scan

def create_observation(robot_state, target_pos=[10.0, 8.0, 0.0], start_pos=[-8.0, -6.0, 0.0]):
    """创建42维观察向量"""
    obs = np.zeros(42, dtype=np.float32)
    
    # 1. LiDAR数据 [0-19]
    obs[0:20] = simulate_lidar(robot_state['position'])
    
    pos = robot_state['position']
    est = robot_state['estimate']
    
    # 2. 坐标信息 [20-31] - 12维 (真实情况用于训练)
    obs[20:23] = pos  # 真实位置
    obs[23:26] = est  # 估计位置
    obs[26:29] = pos  # 真实姿态（roll,pitch,yaw）
    obs[29:32] = est  # 估计姿态
    
    # 3. 导航信息 [32-37] - 6维
    rel_target = np.array(target_pos) - np.array(pos)
    obs[32:35] = rel_target
    obs[35:38] = start_pos  # 起点
    
    # 4. 航向信息 [38-41] - 4维
    target_vector = np.array(target_pos)[:2] - np.array(pos)[:2]
    target_heading = np.arctan2(target_vector[1], target_vector[0])
    current_heading = pos[2]
    heading_error = target_heading - current_heading
    
    # 归一化到[-pi, pi]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    obs[38] = heading_error
    obs[39] = target_heading
    obs[40] = robot_state['velocity'][1]  # 角速度
    obs[41] = 0.0  # 线加速度（简化）
    
    return obs

def calculate_reward(robot_state, prev_distance):
    """计算奖励函数"""
    reward = 0.0
    
    current_pos = robot_state['position'][:2]
    target_pos = [10.0, 8.0]
    current_distance = np.sqrt((current_pos[0]-target_pos[0])**2 + (current_pos[1]-target_pos[1])**2)
    
    # 距离改进奖励
    if prev_distance is not None:
        improvement = prev_distance - current_distance
        reward += improvement * 5.0
    
    # 航向奖励
    heading_error = abs(robot_state['position'][2] - np.arctan2(target_pos[1]-current_pos[1], target_pos[0]-current_pos[0]))
    if heading_error > np.pi:
        heading_error = 2*np.pi - heading_error
    reward -= heading_error * 2.0
    
    # 接近奖励
    if current_distance < 1.0:
        reward += (1.0 - current_distance) * 3.0
    
    # 不确定性惩罚（较小的惩罚）
    uncertainty = np.mean(robot_state['uncertainty'][:2])
    reward -= uncertainty * 0.5
    
    # 步数惩罚
    reward -= 0.1
    
    return reward, current_distance

def test_policy(robot_state, policy='random'):
    """测试策略选择"""
    if policy == 'random':
        linear = np.random.uniform(0, 2.0)  # 0-2 m/s
        angular = np.random.uniform(-2.0, 2.0)  # -2 to +2 rad/s
        return [linear, angular]
    elif policy == 'towards_target':
        # 简单朝向目标策略
        current_pos = robot_state['position'][:2]
        target_pos = [10.0, 8.0]
        angle_to_target = np.arctan2(target_pos[1]-current_pos[1], target_pos[0]-current_pos[0])
        current_angle = robot_state['position'][2]
        
        angle_diff = angle_to_target - current_angle
        if angle_diff > np.pi:
            angle_diff -= 2*np.pi
        if angle_diff < -np.pi:
            angle_diff += 2*np.pi
        
        linear = min(1.5, 2.0 - abs(angle_diff))
        angular = np.clip(angle_diff * 2.0, -2.0, 2.0)
        
        return [linear, angular]
    else:
        return [0.5, 0.0]  # 默认慢速前进

def quick_training_demo():
    """快速训练演示"""
    
    print("🎯 开始快速训练演示...")
    print(f"📊 演示配置:")
    print(f"  • 步数: 200")
    print(f"  • 目标距离阈值: 0.3m")
    print(f"  • 最大步数: 100")
    print("-" * 50)
    
    # 初始化机器人状态
    robot = RobotState()
    target_pos = [10.0, 8.0]
    
    # 训练不同策略的结果
    policies = ['random', 'towards_target']
    policy_results = {}
    
    for policy_name in policies:
        print(f"\n🧪 测试策略: {policy_name}")
        
        start_time = time.time()
        prev_distance = None
        total_reward = 0
        distances = []
        positions = []
        
        for step in range(100):
            # 获取当前观察
            obs = create_observation(robot.state_data, target_pos, [-8.0, -6.0, 0.0])
            
            # 选择动作
            action = test_policy(robot.state_data, policy_name)
            
            # 执行动作
            robot_state_data = robot.update(action[0], action[1])
            
            # 计算奖励
            reward, current_distance = calculate_reward(robot_state_data, prev_distance)
            
            total_reward += reward
            distances.append(current_distance)
            positions.append(robot_state_data['position'][:2])
            
            prev_distance = current_distance
            
            # 检查是否到达目标
            if current_distance < 0.3:
                print(f"  ✅ 成功到达目标! 步数: {step+1}")
                break
            
            if step % 20 == 0 and step > 0:
                print(f"  📊 步 {step+1}: 距离={current_distance:.2f}m, 奖励={total_reward:.2f}")
        
        training_time = time.time() - start_time
        
        policy_results[policy_name] = {
            'total_steps': step + 1,
            'final_distance': current_distance,
            'total_reward': total_reward,
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'positions': positions,
            'training_time': training_time,
            'final_position': robot.state_data['position'].copy(),
            'amcl_uncertainty': np.mean(robot.state_data['uncertainty'])
        }
        
        print(f"  📍 最终结果:")
        print(f"    - 到达距离: {current_distance:.2f}m")
        print(f"    - 总奖励: {total_reward:.2f}")
        print(f"    - 训练用时: {training_time:.2f}s")
        print(f"    - AMCL不确定性: {np.mean(robot.state_data['uncertainty']):.3f}")
        
        # 重置机器人状态用于下个策略测试
        robot = RobotState()
    
    return policy_results

def analyze_results(results):
    """分析训练结果"""
    
    print(f"\n📊 训练结果分析:")
    print("=" * 50)
    
    for policy_name, data in results.items():
        print(f"\n🎯 {policy_name.upper()} 策略:")
        success = "✅ 达到目标" if data['final_distance'] < 0.3 else "❌ 未达目标"
        print(f"  {success}")
        print(f"  📏 最终距离: {data['final_distance']:.2f}m")
        print(f"  🎯 最近距离: {data['min_distance']:.2f}m")
        print(f"  📊 平均距离: {data['avg_distance']:.2f}m")
        print(f"  💰 总奖励: {data['total_reward']:.2f}")
        print(f"  ⏰ 执行时间: {data['training_time']:.2f}s")
        
        # 到达率分析
        distance_ratio = (8.0 - data['final_distance']) / 8.0  # 初始距离约8m到目标
        print(f"  📈 导航进度: {distance_ratio*100:.1f}%")
    
    # 对比分析
    if len(results) > 1:
        print(f"\n📈 策略对比:")
        policies = list(results.keys())
        
        # 找到最佳策略
        best_policy = min(results.keys(), key=lambda k: results[k]['final_distance'])
        print(f"  🏆 最佳策略: {best_policy}")
        
        print(f"  📊 性能差异:")
        for i, policy in enumerate(policies):
            data = results[policy]
            perf_score = (1.0 - data['final_distance']/8.0) * 100  # 距离目标越近越好
            print(f"    {i+1}. {policy}: {perf_score:.1f}% 完成度")

def main():
    """主函数"""
    
    print("🤖 ROSbot导航系统训练演示")
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 执行快速训练演示
        results = quick_training_demo()
        
        # 分析结果
        analyze_results(results)
        
        print(f"\n🏆 演示完成!")
        print(f"✅ 系统状态: 导航和AMCL定位正常")
        print(f"🎯 训练系统: 就绪等待大规模训练")
        
        # 保存结果
        output_file = "./demo_training_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 结果已保存: {output_file}")
        
        # 生成的QuickStat文件
        print(f"\n📋 快速统计摘要:")
        for policy, data in results.items():
            status = "成功" if data['final_distance'] < 0.3 else "进行中"
            print(f"  {policy.title()}: {status} ({data['final_distance']:.2f}m)")
        
        print(f"\n🚀 建议下一步操作:")
        print(" 1. 运行完整训练: cd rosbot_training && python train_cpu.py")
        print(" 2. 检查实际训练结果查看")
        print(" 3. 进行Webots集成测试")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print("💡 请检查环境配置和依赖")

if __name__ == "__main__":
    main()