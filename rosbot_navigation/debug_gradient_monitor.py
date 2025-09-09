"""
梯度监控和原地打转问题调试脚本
分析训练过程中的梯度变化、奖励分布和行为模式
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import argparse
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from src.environments.navigation_env import ROSbotNavigationEnv
from src.models.td3_robust import ImprovedTD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


class GradientMonitorCallback(BaseCallback):
    """梯度监控回调 - 记录和分析梯度变化"""
    
    def __init__(self, log_interval: int = 1000, save_path: str = "./debug_logs", verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 梯度历史记录
        self.gradient_history = {
            'actor': [],
            'critic': [],
            'timesteps': []
        }
        
        # 奖励和行为记录
        self.reward_history = []
        self.action_history = []
        self.position_history = []
        self.spinning_episodes = []
        
        # 统计信息
        self.gradient_stats = {}
        self.last_log_step = 0
        
    def _on_step(self) -> bool:
        """每步调用 - 收集梯度信息"""
        
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self._collect_gradient_data()
            self._analyze_behavior_patterns()
            self.last_log_step = self.num_timesteps
            
        return True
    
    def _collect_gradient_data(self):
        """收集梯度数据"""
        
        try:
            # 获取模型
            model = self.model
            
            # 收集Actor网络梯度
            actor_gradients = []
            if hasattr(model.policy, 'actor'):
                for param in model.policy.actor.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        actor_gradients.append(grad_norm)
            
            # 收集Critic网络梯度
            critic_gradients = []
            if hasattr(model.policy, 'critics'):
                for critic in model.policy.critics:
                    for param in critic.parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            critic_gradients.append(grad_norm)
            
            # 记录梯度统计
            actor_stats = {}
            critic_stats = {}
            
            if actor_gradients:
                actor_stats = {
                    'mean': np.mean(actor_gradients),
                    'std': np.std(actor_gradients),
                    'max': np.max(actor_gradients),
                    'min': np.min(actor_gradients)
                }
                self.gradient_history['actor'].append(actor_stats)
                
            if critic_gradients:
                critic_stats = {
                    'mean': np.mean(critic_gradients),
                    'std': np.std(critic_gradients),
                    'max': np.max(critic_gradients),
                    'min': np.min(critic_gradients)
                }
                self.gradient_history['critic'].append(critic_stats)
            
            self.gradient_history['timesteps'].append(self.num_timesteps)
            
            if self.verbose >= 1:
                actor_mean = actor_stats.get('mean', 0) if actor_stats else 0
                critic_mean = critic_stats.get('mean', 0) if critic_stats else 0
                print(f"Step {self.num_timesteps}: Actor grad mean: {actor_mean:.6f}, "
                      f"Critic grad mean: {critic_mean:.6f}")
                
        except Exception as e:
            print(f"梯度收集失败: {e}")
    
    def _analyze_behavior_patterns(self):
        """分析行为模式 - 检测原地打转"""
        
        try:
            # 获取环境信息
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
            else:
                env = self.training_env
            
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped
                
            # 记录位置和动作
            if hasattr(env, '_get_sup_position'):
                current_pos = env._get_sup_position()
                self.position_history.append({
                    'timestep': self.num_timesteps,
                    'x': current_pos[0],
                    'y': current_pos[1],
                    'z': current_pos[2] if len(current_pos) > 2 else 0
                })
            
            # 检测原地打转行为
            if hasattr(env, 'same_spot_steps') and hasattr(env, 'total_rotation_in_place'):
                if env.same_spot_steps > 10 and env.total_rotation_in_place > 1.0:
                    spinning_info = {
                        'timestep': self.num_timesteps,
                        'same_spot_steps': env.same_spot_steps,
                        'total_rotation': env.total_rotation_in_place,
                        'position': env._get_sup_position() if hasattr(env, '_get_sup_position') else [0, 0, 0]
                    }
                    self.spinning_episodes.append(spinning_info)
                    
                    if self.verbose >= 1:
                        print(f"🌀 检测到原地打转! 步数: {env.same_spot_steps}, 旋转: {env.total_rotation_in_place:.2f}rad")
                        
        except Exception as e:
            print(f"行为分析失败: {e}")
    
    def save_debug_data(self):
        """保存调试数据"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存梯度历史
        gradient_file = self.save_path / f"gradient_history_{timestamp}.json"
        with open(gradient_file, 'w') as f:
            json.dump(self.gradient_history, f, indent=2, default=str)
        
        # 保存原地打转记录
        spinning_file = self.save_path / f"spinning_episodes_{timestamp}.json"
        with open(spinning_file, 'w') as f:
            json.dump(self.spinning_episodes, f, indent=2, default=str)
        
        # 保存位置历史
        if self.position_history:
            position_file = self.save_path / f"position_history_{timestamp}.json"
            with open(position_file, 'w') as f:
                json.dump(self.position_history, f, indent=2, default=str)
        
        print(f"✅ 调试数据已保存到: {self.save_path}")


class RewardAnalyzer:
    """奖励分析器 - 分析奖励函数的问题"""
    
    def __init__(self):
        self.reward_components = []
        self.action_sequences = []
        self.environment_states = []
    
    def analyze_reward_function(self, env: ROSbotNavigationEnv, num_steps: int = 1000):
        """分析奖励函数在不同情况下的表现"""
        
        print("🔍 开始分析奖励函数...")
        
        # 重置环境
        obs, info = env.reset()
        
        reward_analysis = {
            'total_rewards': [],
            'reward_components': [],
            'actions': [],
            'spinning_detected': []
        }
        
        for step in range(num_steps):
            # 生成随机动作
            action = env.action_space.sample()
            
            # 记录动作前的状态 (处理Monitor包装)
            if hasattr(env, 'unwrapped'):
                actual_env = env.unwrapped
            else:
                actual_env = env
            
            if hasattr(actual_env, '_get_sup_position'):
                prev_pos = actual_env._get_sup_position()
            else:
                prev_pos = [0, 0, 0]  # 默认位置
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 分析奖励组成（需要修改环境以暴露奖励组成）
            if hasattr(env, '_last_reward_components'):
                reward_components = env._last_reward_components.copy()
            else:
                reward_components = {'total': reward}
            
            # 检测原地打转
            spinning = hasattr(env, 'same_spot_steps') and env.same_spot_steps > 10
            
            # 记录数据
            reward_analysis['total_rewards'].append(reward)
            reward_analysis['reward_components'].append(reward_components)
            reward_analysis['actions'].append(action.tolist())
            reward_analysis['spinning_detected'].append(spinning)
            
            if step % 100 == 0:
                print(f"Step {step}: Reward: {reward:.3f}, Spinning: {spinning}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        return reward_analysis
    
    def create_reward_visualization(self, analysis_data: Dict, save_path: str = "./debug_plots"):
        """创建奖励可视化图表"""
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('奖励函数分析报告', fontsize=16, fontweight='bold')
        
        # 1. 奖励分布直方图
        ax1 = axes[0, 0]
        rewards = analysis_data['total_rewards']
        ax1.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('奖励值')
        ax1.set_ylabel('频率')
        ax1.set_title('奖励分布')
        ax1.axvline(np.mean(rewards), color='red', linestyle='--', label=f'均值: {np.mean(rewards):.3f}')
        ax1.legend()
        
        # 2. 奖励时间序列
        ax2 = axes[0, 1]
        ax2.plot(rewards[:500], alpha=0.7)  # 显示前500步
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('奖励值')
        ax2.set_title('奖励时间序列 (前500步)')
        ax2.grid(True, alpha=0.3)
        
        # 3. 动作分布
        ax3 = axes[1, 0]
        actions = np.array(analysis_data['actions'])
        ax3.scatter(actions[:, 0], actions[:, 1], alpha=0.6, s=1)
        ax3.set_xlabel('线速度')
        ax3.set_ylabel('角速度')
        ax3.set_title('动作空间分布')
        ax3.grid(True, alpha=0.3)
        
        # 4. 原地打转检测
        ax4 = axes[1, 1]
        spinning_counts = np.cumsum(analysis_data['spinning_detected'])
        ax4.plot(spinning_counts)
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('累计原地打转次数')
        ax4.set_title('原地打转检测')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = save_path / f"reward_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 奖励分析图表已保存: {plot_file}")
        
        return str(plot_file)


def create_gradient_visualization(gradient_file: str, save_path: str = "./debug_plots"):
    """创建梯度可视化"""
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 读取梯度数据
    with open(gradient_file, 'r') as f:
        data = json.load(f)
    
    if not data['timesteps']:
        print("⚠️  没有梯度数据可供可视化")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('梯度分析报告', fontsize=16, fontweight='bold')
    
    timesteps = data['timesteps']
    
    # 1. Actor梯度变化
    ax1 = axes[0, 0]
    if data['actor']:
        actor_means = [stats['mean'] for stats in data['actor']]
        ax1.plot(timesteps, actor_means, label='Actor梯度均值', linewidth=2)
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('梯度幅度')
        ax1.set_title('Actor网络梯度变化')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Critic梯度变化
    ax2 = axes[0, 1]
    if data['critic']:
        critic_means = [stats['mean'] for stats in data['critic']]
        ax2.plot(timesteps, critic_means, label='Critic梯度均值', linewidth=2, color='orange')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('梯度幅度')
        ax2.set_title('Critic网络梯度变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 梯度稳定性
    ax3 = axes[1, 0]
    if data['actor']:
        actor_stds = [stats['std'] for stats in data['actor']]
        ax3.plot(timesteps, actor_stds, label='Actor梯度标准差', linewidth=2)
        ax3.set_xlabel('训练步数')
        ax3.set_ylabel('梯度标准差')
        ax3.set_title('梯度稳定性')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 梯度分布
    ax4 = axes[1, 1]
    if data['actor'] and data['critic']:
        all_actor_means = [stats['mean'] for stats in data['actor']]
        all_critic_means = [stats['mean'] for stats in data['critic']]
        
        ax4.hist(all_actor_means, bins=30, alpha=0.7, label='Actor', density=True)
        ax4.hist(all_critic_means, bins=30, alpha=0.7, label='Critic', density=True)
        ax4.set_xlabel('梯度幅度')
        ax4.set_ylabel('密度')
        ax4.set_title('梯度分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = save_path / f"gradient_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 梯度分析图表已保存: {plot_file}")
    return str(plot_file)


def diagnose_spinning_problem(cargo_type: str = 'normal', debug_steps: int = 5000):
    """诊断原地打转问题"""
    
    print(f"🔧 开始诊断 {cargo_type} 货物类型的原地打转问题...")
    
    # 创建环境
    env = ROSbotNavigationEnv(cargo_type=cargo_type)
    env = Monitor(env)
    
    # 创建梯度监控回调
    gradient_monitor = GradientMonitorCallback(
        log_interval=100,  # 每100步记录一次
        save_path="./debug_logs",
        verbose=1
    )
    
    # 创建模型
    model = ImprovedTD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./debug_tensorboard"
    )
    
    print(f"🚀 开始训练和监控 ({debug_steps} 步)...")
    
    # 开始训练和监控
    model.learn(
        total_timesteps=debug_steps,
        callback=gradient_monitor,
        progress_bar=True
    )
    
    # 保存调试数据
    gradient_monitor.save_debug_data()
    
    # 分析奖励函数
    print("🔍 分析奖励函数...")
    reward_analyzer = RewardAnalyzer()
    reward_analysis = reward_analyzer.analyze_reward_function(env, num_steps=1000)
    
    # 创建可视化
    reward_plot = reward_analyzer.create_reward_visualization(reward_analysis)
    
    # 查找最新的梯度文件
    debug_path = Path("./debug_logs")
    gradient_files = list(debug_path.glob("gradient_history_*.json"))
    if gradient_files:
        latest_gradient_file = max(gradient_files, key=os.path.getctime)
        gradient_plot = create_gradient_visualization(str(latest_gradient_file))
    else:
        gradient_plot = None
    
    env.close()
    
    # 生成诊断报告
    return generate_diagnosis_report(
        gradient_monitor, reward_analysis, 
        reward_plot, gradient_plot, cargo_type
    )


def generate_diagnosis_report(gradient_monitor, reward_analysis, 
                            reward_plot, gradient_plot, cargo_type):
    """生成诊断报告"""
    
    report = f"""
# 机器人原地打转问题诊断报告

**货物类型**: {cargo_type}  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔍 问题发现

### 1. 梯度分析
- **原地打转检测次数**: {len(gradient_monitor.spinning_episodes)}
- **最大连续原地打转步数**: {max([ep['same_spot_steps'] for ep in gradient_monitor.spinning_episodes]) if gradient_monitor.spinning_episodes else 0}

### 2. 奖励函数分析
- **平均奖励**: {np.mean(reward_analysis['total_rewards']):.3f}
- **奖励标准差**: {np.std(reward_analysis['total_rewards']):.3f}
- **原地打转比例**: {np.mean(reward_analysis['spinning_detected']) * 100:.1f}%

## 🚨 识别的问题

### 主要问题：
1. **奖励函数设计问题**: movement_reward被放大20倍可能导致不稳定学习
2. **梯度不稳定**: 需要检查梯度裁剪和学习率设置
3. **原地打转惩罚不足**: 当前惩罚机制可能不够敏感

### 次要问题：
- 奖励函数过于复杂，包含多个冲突的信号
- 探索策略可能不够均衡

## 💡 解决建议

### 立即修复：
1. **降低movement_reward的放大系数** (从20倍降到2-5倍)
2. **加强原地打转惩罚** (增加惩罚系数或降低检测阈值)
3. **添加梯度裁剪** (设置为0.5或1.0)

### 长期优化：
1. **简化奖励函数** - 减少冲突的奖励信号
2. **调整学习率调度** - 使用衰减学习率
3. **改进探索策略** - 使用更好的噪声策略

## 📊 可视化文件
- 奖励分析图: {reward_plot}
- 梯度分析图: {gradient_plot if gradient_plot else '未生成'}

## 🔧 推荐的配置修改

```python
# 修改奖励函数 (navigation_env.py 第879行)
rewards['movement_reward'] = movement_reward * 2  # 从 *20 改为 *2

# 加强原地打转检测
self.stuck_steps_for_spin_check = 5  # 从默认值降低
self.spin_termination_threshold = 3.14  # 降低阈值

# 添加梯度裁剪 (td3_robust.py)
torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
```

---
**注意**: 请在应用这些修改后重新训练模型，并监控训练过程中的梯度变化。
    """
    
    # 保存报告
    report_path = Path("./debug_logs") / f"diagnosis_report_{cargo_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 诊断报告已保存: {report_path}")
    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    return str(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="机器人原地打转问题诊断")
    parser.add_argument("--cargo_type", type=str, default="normal", 
                       choices=["normal", "fragile", "dangerous"],
                       help="货物类型")
    parser.add_argument("--debug_steps", type=int, default=5000,
                       help="调试训练步数")
    
    args = parser.parse_args()
    
    # 运行诊断
    report_path = diagnose_spinning_problem(args.cargo_type, args.debug_steps)
    print(f"\n✅ 诊断完成！报告路径: {report_path}")
