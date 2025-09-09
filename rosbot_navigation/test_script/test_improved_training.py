"""
测试改进后的训练系统 - 解决5000步后原地打转和到达检测问题
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.navigation_env import ROSbotNavigationEnv  
from src.models.td3_robust import ImprovedTD3
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time


def test_long_term_training(cargo_type='normal', test_steps=8000):
    """测试长期训练中的原地打转问题和到达检测"""
    
    print(f"🧪 测试 {cargo_type} 长期训练稳定性 ({test_steps} 步)")
    print(f"✅ 新增修复项目:")
    print(f"   - 成功检测阈值: 15cm → 25cm (更容易成功)")
    print(f"   - 渐进式接近奖励: 1m内开始给予接近奖励")
    print(f"   - 学习率衰减: 5000步后开始线性衰减") 
    print(f"   - 梯度裁剪: max_norm=0.5")
    print("-" * 60)
    
    # 创建环境
    env = ROSbotNavigationEnv(cargo_type=cargo_type)
    env = Monitor(env)
    
    # 确保环境初始化
    env.reset()
    
    # 创建模型
    model = ImprovedTD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=f"./test_logs/improved_{cargo_type}"
    )
    
    print(f"🚀 开始长期训练测试...")
    start_time = time.time()
    
    # 记录训练过程中的关键指标
    phase_results = {}
    
    # 分阶段训练和评估
    phases = [
        ("early", 2000),    # 0-2k: 早期训练
        ("middle", 4000),   # 2-6k: 跨越5k步关键期
        ("late", 2000)      # 6-8k: 后期训练
    ]
    
    total_trained = 0
    
    for phase_name, phase_steps in phases:
        print(f"\n📈 {phase_name.title()} Phase: 训练 {phase_steps} 步 (总计: {total_trained}-{total_trained + phase_steps})")
        
        # 确保环境状态正确
        try:
            env.reset()
        except:
            pass
            
        # 训练 - 简化进度条
        print(f"    🚀 训练中... (目标: {phase_steps} 步)")
        model.learn(
            total_timesteps=phase_steps,
            progress_bar=False,  # 关闭进度条
            reset_num_timesteps=False,
            log_interval=10      # 每10个episode输出一次
        )
        
        total_trained += phase_steps
        
        # 评估当前阶段
        phase_results[phase_name] = evaluate_current_performance(model, env, phase_name, total_trained)
        
        # 特别关注5k步附近的表现
        if phase_name == "middle":
            print(f"⚠️  关键检查: 5000步后是否出现原地打转...")
            
    training_time = time.time() - start_time
    
    # 最终评估
    print(f"\n📊 最终评估...")
    final_results = detailed_evaluation(model, env, num_episodes=10)
    
    env.close()
    
    # 分析结果
    analyze_long_term_results(phase_results, final_results, training_time)
    
    return phase_results, final_results


def evaluate_current_performance(model, env, phase_name, total_steps):
    """评估当前阶段的性能"""
    
    print(f"   🔍 评估 {phase_name} 阶段性能...")
    
    rewards = []
    spinning_count = 0
    success_count = 0
    approach_count = 0  # 接近目标次数
    
    for episode in range(3):  # 每个阶段评估3个episodes
        # 确保环境重置
        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"     警告: 环境重置失败 {e}")
            continue
            
        episode_reward = 0
        episode_spinning = False
        episode_success = False
        episode_approach = False
        
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            
            # 检测问题
            if hasattr(env.unwrapped, 'same_spot_steps') and env.unwrapped.same_spot_steps > 8:
                episode_spinning = True
                
            if step_info.get('success', False):
                episode_success = True
                
            # 新增：检测是否接近目标
            if step_info.get('close_to_target', False):
                episode_approach = True
                
            if terminated or truncated:
                break
                
        rewards.append(episode_reward)
        if episode_spinning:
            spinning_count += 1
        if episode_success:
            success_count += 1
        if episode_approach:
            approach_count += 1
    
    result = {
        'phase': phase_name,
        'total_steps': total_steps,
        'avg_reward': np.mean(rewards),
        'spinning_rate': spinning_count / 3,
        'success_rate': success_count / 3,
        'approach_rate': approach_count / 3  # 接近目标的成功率
    }
    
    print(f"     - 平均奖励: {result['avg_reward']:.1f}")
    print(f"     - 原地打转率: {result['spinning_rate']*100:.1f}%")
    print(f"     - 成功率: {result['success_rate']*100:.1f}%")
    print(f"     - 接近目标率: {result['approach_rate']*100:.1f}%")
    
    return result


def detailed_evaluation(model, env, num_episodes=10):
    """详细评估"""
    
    total_rewards = []
    success_episodes = 0
    approach_episodes = 0
    collision_episodes = 0
    spinning_episodes = 0
    
    for episode in range(num_episodes):
        # 确保环境重置
        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"     警告: Episode {episode+1} 环境重置失败: {e}")
            continue
            
        episode_reward = 0
        episode_success = False
        episode_approach = False
        episode_collision = False
        episode_spinning = False
        
        for step in range(400):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            
            # 记录各种事件
            if step_info.get('success', False):
                episode_success = True
                
            if step_info.get('close_to_target', False):
                episode_approach = True
                
            if step_info.get('collision', False):
                episode_collision = True
                
            if hasattr(env.unwrapped, 'same_spot_steps') and env.unwrapped.same_spot_steps > 10:
                episode_spinning = True
                
            if terminated or truncated:
                break
                
        total_rewards.append(episode_reward)
        if episode_success:
            success_episodes += 1
        if episode_approach:
            approach_episodes += 1
        if episode_collision:
            collision_episodes += 1
        if episode_spinning:
            spinning_episodes += 1
            
        print(f"Episode {episode+1}: 奖励={episode_reward:.1f}, "
              f"成功={'✓' if episode_success else '✗'}, "
              f"接近={'✓' if episode_approach else '✗'}, "
              f"碰撞={'✓' if episode_collision else '✗'}, "
              f"打转={'✓' if episode_spinning else '✗'}")
    
    return {
        'avg_reward': np.mean(total_rewards),
        'reward_std': np.std(total_rewards),
        'success_rate': success_episodes / num_episodes,
        'approach_rate': approach_episodes / num_episodes,
        'collision_rate': collision_episodes / num_episodes,
        'spinning_rate': spinning_episodes / num_episodes
    }


def analyze_long_term_results(phase_results, final_results, training_time):
    """分析长期训练结果"""
    
    print(f"\n" + "="*60)
    print(f"📋 长期训练分析报告")
    print(f"="*60)
    
    print(f"⏱️  训练时间: {training_time:.1f}秒")
    print(f"📊 最终指标:")
    print(f"   - 平均奖励: {final_results['avg_reward']:.1f}")
    print(f"   - 成功率: {final_results['success_rate']*100:.1f}%")
    print(f"   - 接近目标率: {final_results['approach_rate']*100:.1f}%")
    print(f"   - 碰撞率: {final_results['collision_rate']*100:.1f}%")  
    print(f"   - 原地打转率: {final_results['spinning_rate']*100:.1f}%")
    
    print(f"\n📈 阶段性能变化:")
    for phase_name, results in phase_results.items():
        print(f"   {phase_name.title()}: 奖励={results['avg_reward']:.1f}, "
              f"打转率={results['spinning_rate']*100:.1f}%, "
              f"成功率={results['success_rate']*100:.1f}%")
    
    # 检查5k步问题是否解决
    middle_spinning = phase_results['middle']['spinning_rate']
    late_spinning = phase_results['late']['spinning_rate']
    
    if middle_spinning < 0.3 and late_spinning < 0.3:
        print(f"\n🎉 修复成功！5000步后原地打转问题已解决")
        success_status = "优秀"
    elif middle_spinning < 0.5 and late_spinning < 0.5:
        print(f"\n✅ 有显著改善！原地打转明显减少")
        success_status = "良好"
    else:
        print(f"\n⚠️  仍需改进：原地打转问题未完全解决")
        success_status = "一般"
    
    # 检查到达检测改善
    final_approach_rate = final_results['approach_rate']
    final_success_rate = final_results['success_rate']
    
    if final_approach_rate > 0.6 and final_success_rate > 0.3:
        print(f"🎯 到达检测机制改善显著！")
        arrival_status = "优秀"
    elif final_approach_rate > 0.4 and final_success_rate > 0.2:
        print(f"🎯 到达检测有所改善")
        arrival_status = "良好"
    else:
        print(f"🎯 到达检测仍需优化")
        arrival_status = "一般"
    
    print(f"\n📝 综合评估:")
    print(f"   - 原地打转解决: {success_status}")
    print(f"   - 到达检测改善: {arrival_status}")
    
    if success_status in ["优秀", "良好"] and arrival_status in ["优秀", "良好"]:
        print(f"\n✅ 推荐：可以开始长期训练")
        print(f"   建议命令: python train_stage1.py --cargo_type normal --steps 50000")
    else:
        print(f"\n🔧 建议：继续调优参数")


if __name__ == "__main__":
    print("🔧 长期训练稳定性测试")
    print("专门解决5000步后原地打转和到达检测问题")
    print("=" * 60)
    
    # 测试改进效果
    phase_results, final_results = test_long_term_training('normal', test_steps=8000)
