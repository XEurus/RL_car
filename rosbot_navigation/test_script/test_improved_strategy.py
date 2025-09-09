"""
测试改进后的策略 - 参考成功代码的设计
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.navigation_env import ROSbotNavigationEnv  
from src.models.td3_robust import ImprovedTD3
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time


def test_strategy_improvements(cargo_type='normal', test_steps=2000):
    """测试策略改进效果"""
    
    print(f"🧪 测试策略改进效果 ({test_steps} 步)")
    print(f"货物类型: {cargo_type}")
    print(f"✅ 参考成功代码的改进项目:")
    print(f"   - 预测性碰撞避免 (前方8.5cm内阻止前进)")
    print(f"   - 成功代码的动作缩放 (linear×0.12, angular×1.3)")
    print(f"   - 一致转向奖励机制 (帮助脱困)")
    print(f"   - 精确的原地停留检测 (0.5cm阈值)")
    print(f"   - 智能原地停留惩罚 (靠近墙壁时减半)")
    print("-" * 60)
    
    # 创建环境
    print("🏗️  创建环境...")
    env = ROSbotNavigationEnv(cargo_type=cargo_type)
    env = Monitor(env)
    
    # 手动重置确保环境状态正确
    print("🔄 初始化环境...")
    obs, info = env.reset()
    print(f"   观测维度: {obs.shape}")
    print(f"   目标位置: {info.get('target_position', 'N/A')}")
    
    # 创建模型
    print("🤖 创建TD3模型...")
    model = ImprovedTD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        buffer_size=50000,
        verbose=0,
        tensorboard_log=f"./test_logs/strategy_{cargo_type}"
    )
    
    print(f"✅ 模型创建成功")
    
    # 开始训练
    print(f"\n🚀 开始策略训练测试...")
    print(f"   目标步数: {test_steps}")
    print(f"   重点观察: 预测性避障、一致转向奖励、原地打转减少")
    
    start_time = time.time()
    
    try:
        # 分段训练，便于观察策略效果
        for segment in range(0, test_steps, 500):
            segment_steps = min(500, test_steps - segment)
            
            print(f"\n   📈 训练段 {segment//500 + 1}: {segment_steps} 步...")
            
            model.learn(
                total_timesteps=segment_steps,
                progress_bar=False,
                reset_num_timesteps=False,
                log_interval=50
            )
            
            current_time = time.time()
            elapsed = current_time - start_time
            total_steps = segment + segment_steps
            
            print(f"      ✓ {total_steps}/{test_steps} 步完成 (用时: {elapsed:.1f}s)")
            
            # 检查是否有策略改进的迹象
            if total_steps >= 1000:
                print(f"      📊 策略效果检查点...")
                
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        env.close()
        return None
    
    training_time = time.time() - start_time
    print(f"\n✅ 策略训练测试完成! 用时: {training_time:.1f}秒")
    
    # 策略效果评估
    print(f"\n📊 策略效果评估...")
    
    evaluation_results = []
    spinning_episodes = 0
    collision_episodes = 0 
    approach_episodes = 0  # 接近目标的episodes
    predictive_avoid_count = 0  # 预测性避障次数
    consistent_turn_count = 0   # 一致转向奖励次数
    
    for i in range(5):
        print(f"\n   🎮 评估Episode {i+1}/5:")
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_spinning = False
        episode_collision = False
        episode_approach = False
        episode_predictive_avoid = 0
        episode_consistent_turn = 0
        
        for step in range(300):
            action, _ = model.predict(obs, deterministic=True)
            
            # 检查是否会触发预测性避障 (仅用于统计)
            if action[0] > 0.3:
                lidar_features = obs[0:20] if len(obs) >= 20 else []
                if len(lidar_features) > 0:
                    center_idx = len(lidar_features) // 2
                    front_distances = lidar_features[max(0,center_idx-3):min(len(lidar_features),center_idx+4)]
                    min_front = min(front_distances) if front_distances else 1.0
                    if min_front * 10.0 <= 0.085:  # 反归一化检查
                        episode_predictive_avoid += 1
            
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # 检查各种状态
            if hasattr(env.unwrapped, 'same_spot_steps') and env.unwrapped.same_spot_steps > 8:
                episode_spinning = True
                
            if step_info.get('collision', False):
                episode_collision = True
                
            if step_info.get('close_to_target', False):
                episode_approach = True
                
            # 检查一致转向奖励 (通过奖励信息推断)
            # 这需要在环境中暴露更多信息，这里简化处理
            
            if terminated or truncated:
                break
                
        evaluation_results.append({
            'reward': episode_reward,
            'steps': episode_steps,
            'spinning': episode_spinning,
            'collision': episode_collision,
            'approach': episode_approach,
            'predictive_avoid': episode_predictive_avoid
        })
        
        if episode_spinning:
            spinning_episodes += 1
        if episode_collision:
            collision_episodes += 1
        if episode_approach:
            approach_episodes += 1
        predictive_avoid_count += episode_predictive_avoid
        
        print(f"      奖励: {episode_reward:.1f}, 步数: {episode_steps}")
        print(f"      原地打转: {'是' if episode_spinning else '否'}")
        print(f"      碰撞: {'是' if episode_collision else '否'}")
        print(f"      接近目标: {'是' if episode_approach else '否'}")
        print(f"      预测性避障: {episode_predictive_avoid}次")
    
    # 计算统计数据
    avg_reward = np.mean([r['reward'] for r in evaluation_results])
    avg_steps = np.mean([r['steps'] for r in evaluation_results])
    spinning_rate = spinning_episodes / 5
    collision_rate = collision_episodes / 5
    approach_rate = approach_episodes / 5
    
    env.close()
    
    # 结果分析
    print(f"\n" + "="*60)
    print(f"📋 策略改进效果分析:")
    print(f"="*60)
    print(f"⏱️  训练时间: {training_time:.1f}秒")
    print(f"📊 评估指标:")
    print(f"   - 平均奖励: {avg_reward:.1f}")
    print(f"   - 平均步数: {avg_steps:.1f}")
    print(f"   - 原地打转率: {spinning_rate*100:.1f}%")
    print(f"   - 碰撞率: {collision_rate*100:.1f}%")  
    print(f"   - 接近目标率: {approach_rate*100:.1f}%")
    print(f"   - 预测性避障总计: {predictive_avoid_count}次")
    
    print(f"\n🔍 策略改进分析:")
    
    # 预测性避障效果
    if predictive_avoid_count > 0:
        print(f"   ✅ 预测性避障生效 - 共阻止{predictive_avoid_count}次危险前进")
    else:
        print(f"   ⚠️  预测性避障未触发 - 可能阈值需要调整")
        
    # 原地打转改善
    if spinning_rate < 0.4:
        print(f"   ✅ 原地打转显著减少 - 策略改进有效")
    elif spinning_rate < 0.6:
        print(f"   🔄 原地打转有所改善 - 仍需进一步优化")
    else:
        print(f"   ❌ 原地打转问题仍然严重 - 需要调试策略参数")
    
    # 接近目标能力
    if approach_rate > 0.4:
        print(f"   ✅ 机器人能够接近目标 - 导航策略有效")
    else:
        print(f"   ⚠️  接近目标能力有限 - 可能需要调整奖励权重")
    
    # 综合评估
    strategy_score = 0
    if spinning_rate < 0.5:
        strategy_score += 30
    if collision_rate < 0.6:
        strategy_score += 20
    if approach_rate > 0.3:
        strategy_score += 25
    if predictive_avoid_count > 0:
        strategy_score += 15
    if avg_reward > -500:
        strategy_score += 10
    
    print(f"\n🎯 策略改进评分: {strategy_score}/100")
    
    if strategy_score >= 80:
        status = "优秀"
        print(f"🎉 策略改进效果优秀！可以进行长期训练")
        next_step = "python train_stage1.py --cargo_type normal --steps 20000"
    elif strategy_score >= 60:
        status = "良好" 
        print(f"✅ 策略改进效果良好，建议继续测试")
        next_step = "python test_improved_training.py"
    else:
        status = "需要改进"
        print(f"🔧 策略仍需改进，建议调整参数")
        next_step = "调整预测性避障阈值或奖励权重"
    
    print(f"\n📋 建议下一步:")
    print(f"   {next_step}")
    print("="*60)
    
    return {
        'status': status,
        'score': strategy_score,
        'avg_reward': avg_reward,
        'spinning_rate': spinning_rate,
        'collision_rate': collision_rate,
        'approach_rate': approach_rate,
        'predictive_avoid_count': predictive_avoid_count
    }


if __name__ == "__main__":
    print("🔧 策略改进测试 - 参考成功代码设计")
    print("=" * 60)
    
    # 运行策略改进测试
    result = test_strategy_improvements('normal', test_steps=2000)
