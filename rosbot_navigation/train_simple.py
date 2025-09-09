#!/usr/bin/env python3
"""
简化版ROSbot导航训练脚本
绕过复杂的导入问题
"""

import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import time
import sys
import json

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入组件，避免复杂的依赖链
try:
    # 尝试导入核心组件
    from src.localization.amcl_localizer import AMCLLocalizer
    from src.localization.pose_estimator import RobustPoseEstimator
    from src.models.td3_robust import ImprovedTD3, RobustTD3Policy, RosbotFeaturesExtractor
    from src.utils.navigation_utils import NavigationUtils
    print("✅ 核心组件导入成功")
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    IMPORT_SUCCESS = False

# 确保始终导入环境
from src.environments.navigation_env import ROSbotNavigationEnv
from stable_baselines3 import TD3

class SimpleTrainingManager:
    """简化训练管理器"""
    
    def __init__(self):
        self.device = "cpu"
        self.training_history = []
        
    def create_environment(self, cargo_type='normal'):
        """创建42维环境"""
        print(f"🚀 创建 {cargo_type} 环境...")
        env = ROSbotNavigationEnv(cargo_type=cargo_type)
        print(f"✅ 环境创建完成 - 状态空间: {env.observation_space.shape}, 动作空间: {env.action_space.shape}")
        return env
        
    def create_model(self, env, cargo_type='normal'):
        """创建TD3模型"""
        print(f"🤖 创建TD3模型...")
        
        # 使用标准的TD3模型
        model = TD3(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            verbose=1,
            tensorboard_log=f"./logs/td3_simple_{cargo_type}"
        )
        print("✅ TD3模型创建完成")
        return model
        
    def train_model(self, model, total_steps=10000, cargo_type='normal'):
        """训练模型"""
        print(f"🎯 开始训练 {cargo_type} 模型...")
        print(f"📊 总步数: {total_steps}")
        
        try:
            # 开始训练
            model.learn(
                total_timesteps=total_steps,
                log_interval=10,
                progress_bar=True
            )
            
            # 保存模型
            model_path = f"./models/td3_simple_{cargo_type}.zip"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
            print(f"✅ 训练完成，模型已保存: {model_path}")
            
            return {
                'status': 'success',
                'model_path': model_path,
                'total_steps': total_steps,
                'cargo_type': cargo_type
            }
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'cargo_type': cargo_type
            }
    
    def test_model(self, model, env, cargo_type='normal', num_episodes=3):
        """测试模型性能"""
        print(f"\n🧪 测试 {cargo_type} 模型性能...")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}...", end=" ")
            
            obs, info = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            max_steps = 100
            
            while not done and step_count < max_steps:
                # 模型预测动作
                action, _ = model.predict(obs, deterministic=True)
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # 检查是否成功到达目标
                if terminated and info.get('distance_to_target', 999) < 0.2:
                    success_count += 1
                    done = True
                elif truncated:
                    done = True
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            print(f"奖励: {episode_reward:.2f}, 步数: {step_count}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = success_count / num_episodes
        
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'success_rate': success_rate,
            'num_episodes': num_episodes
        }
        
        print(f"\n🎯 {cargo_type} 模型测试结果:")
        print(f"  📊 平均奖励: {avg_reward:.2f}")
        print(f"  🎯 成功率: {success_rate:.1%}")
        print(f"  📏 平均步数: {avg_length:.1f}")
        
        return results

def main():
    """主函数 - 简化训练流程"""
    
    print("="*70)
    print("🤖 简化版ROSbot导航训练系统")
    print("="*70)
    print(f"⏰ 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建训练管理器
    trainer = SimpleTrainingManager()
    
    # 训练normal类型
    cargo_type = 'normal'
    
    print(f"\n🚀 开始训练 {cargo_type} 类型...")
    
    # 1. 创建环境
    env = trainer.create_environment(cargo_type=cargo_type)
    
    # 2. 创建模型
    model = trainer.create_model(env, cargo_type=cargo_type)
    
    # 3. 训练模型
    train_result = trainer.train_model(
        model, 
        total_steps=5000,  # 开始用较少的步数
        cargo_type=cargo_type
    )
    
    if train_result['status'] == 'success':
        # 4. 测试模型
        test_results = trainer.test_model(model, env, cargo_type=cargo_type)
        
        # 保存完整结果
        final_result = {
            'train_result': train_result,
            'test_results': test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_type': 'simplified'
        }
        
        # 保存结果
        result_file = f"./results/simplified_training_{cargo_type}_{int(time.time())}.json"
        Path(result_file).parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(final_result, f, indent=2, default=str)
        
        print(f"\n💾 完整结果已保存: {result_file}")
        
        print(f"\n{'='*70}")
        print("🎉 训练总结:")
        print(f"  ✅ {cargo_type} 类型训练完成")
        print(f"  📊 测试成功率: {test_results['success_rate']:.1%}")
        print(f"  🎯 平均奖励: {test_results['avg_reward']:.2f}")
        print(f"  📁 模型文件: {train_result['model_path']}")
        print("="*70)
        
    else:
        print(f"❌ 训练失败: {train_result.get('error', '未知错误')}")
    
    # 关闭环境
    try:
        env.close()
        print("✅ 环境已关闭")
    except:
        pass

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()