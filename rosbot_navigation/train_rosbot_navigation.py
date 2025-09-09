#!/usr/bin/env python3
"""
ROSbot导航训练脚本
简化版本 - 直接训练Normal类型的货物导航模型
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

from src.environments.navigation_env import ROSbotNavigationEnv
from src.models.td3_robust import ImprovedTD3
from src.localization.amcl_localizer import UncertaintyCurriculumTraining

print("="*70)
print("🤖 ROSbot导航训练系统")
print("="*70)

def start_training(cargo_type: str = 'normal', total_steps: int = 50000, visualize: bool = True):
    """开始训练特定类型的货物导航"""
    
    print(f"🚀 开始训练: {cargo_type.upper()} 货物类型")
    print(f"📊 总训练步数: {total_steps}")
    print(f"🎯 状态空间: 42维 (LiDAR+AMCL+导航信息)")
    print(f"🎮 动作空间: 2维连续 [线速度, 角速度]")
    
    # 创建环境
    env = ROSbotNavigationEnv(cargo_type=cargo_type)
    print(f"✅ 环境创建完成: {cargo_type}")
    
    # 创建TD3模型
    model = ImprovedTD3(
        "MlpPolicy",
        env,
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
        tensorboard_log=f"./logs/td3_{cargo_type}",
        device="cpu"  # 使用CPU进行训练
    )
    print("✅ TD3模型创建完成")
    
    # 设置不确定性课程学习
    uncertainty_curriculum = UncertaintyCurriculumTraining()
    print("✅ 不确定性课程学习初始化完成")
    
    # 训练循环
    try:
        print(f"\n🎯 开始训练 {cargo_type} 模型...")
        
        # 开始训练
        model.learn(
            total_timesteps=total_steps,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # 保存模型
        model_path = f"./models/td3_rosbot_{cargo_type}.zip"
        model.save(model_path)
        print(f"✅ 模型已保存: {model_path}")
        
        # 测试模型
        test_results = test_model(model, env, cargo_type, num_episodes=5)
        
        return {
            'cargo_type': cargo_type,
            'total_steps': total_steps,
            'model_path': model_path,
            'test_results': test_results,
            'status': 'success'
        }
        
    except KeyboardInterrupt:
        print(f"⚠️  训练被用户中断")
        return {'status': 'interrupted', 'cargo_type': cargo_type}
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return {'status': 'failed', 'cargo_type': cargo_type, 'error': str(e)}

def test_model(model, env, cargo_type: str, num_episodes: int = 5):
    """测试训练好的模型"""
    print(f"\n🧪 测试 {cargo_type} 模型性能...")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        max_steps = 200
        
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
        
        # 打印进度
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = success_count / (episode + 1)
        
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Steps={step_count}, "
              f"Avg={avg_reward:.2f}, "
              f"Success={success_rate:.1%}")
    
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_reward': np.mean(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'success_rate': success_count / num_episodes
    }
    
    print(f"\n🎯 {cargo_type} 模型测试结果:")
    print(f"  📊 平均奖励: {results['avg_reward']:.2f}")
    print(f"  🎯 成功率: {results['success_rate']:.1%}")
    print(f"  📏 平均步数: {results['avg_length']:.1f}")
    
    return results

def train_all_cargo_types():
    """训练所有三种货物类型"""
    print("🎯 开始训练所有货物类型")
    
    training_results = {}
    
    # 定义训练配置
    training_config = {
        'normal': {'steps': 30000, 'description': '普通货物 - 标准导航'},
        'fragile': {'steps': 40000, 'description': '易碎货物 - 稳定性优先'},
        'dangerous': {'steps': 50000, 'description': '危险货物 - 安全优先'}
    }
    
    for cargo_type, config in training_config.items():
        print(f"\n{'='*50}")
        print(f"📦 {config['description']}")
        print(f"🎯 训练步数: {config['steps']}")
        print('='*50)
        
        # 开始训练
        result = start_training(
            cargo_type=cargo_type,
            total_steps=config['steps'],
            visualize=True
        )
        
        training_results[cargo_type] = result
        
        # 保存结果到文件
        result_file = f"./results/training_{cargo_type}_{int(time.time())}.json"
        Path(result_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n💾 训练结果已保存: {result_file}")
        
        # 短暂休息
        time.sleep(2)
    
    return training_results
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ROSbot导航训练')
    parser.add_argument('--cargo_type', type=str, default='all', 
                       choices=['normal', 'fragile', 'dangerous', 'all'],
                       help='货物类型（all表示训练所有类型）')
    parser.add_argument('--steps', type=int, default=30000,
                       help='训练步数（仅当指定单种货物类型时）')
    parser.add_argument('--no_visualize', action='store_true',
                       help='禁用可视化')
    
    args = parser.parse_args()
    
    # 开始训练
    if args.cargo_type == 'all':
        results = train_all_cargo_types()
        
        # 总结所有结果
        print(f"\n{'='*70}")
        print("🎯 所有训练完成！")
        print('='*70)
        
        for cargo_type, result in results.items():
            if result['status'] == 'success':
                test_results = result['test_results']
                print(f"\n📦 {cargo_type.upper()}:")
                print(f"  ✅ 训练成功 - 步数: {result['total_steps']}")
                print(f"  🎯 测试成功率: {test_results['success_rate']:.1%}")
                print(f"  📊 平均奖励: {test_results['avg_reward']:.2f}")
                print(f"  📁 模型文件: {result['model_path']}")
        
    else:
        result = start_training(
            cargo_type=args.cargo_type,
            total_steps=args.steps,
            visualize=not args.no_visualize
        )
        
        if result['status'] == 'success':
            print(f"\n🎯 训练完成！模型已保存到: {result['model_path']}")
            print(f"✅ 测试成功率: {result['test_results']['success_rate']:.1%}")
    
    print(f"\n{'='*70}")
    print("🎉 ROSbot导航训练系统 运行完成！")
    print("="*70)