"""
基础训练测试 - 解决环境重置和进度条问题
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.navigation_env import ROSbotNavigationEnv  
from src.models.td3_robust import ImprovedTD3
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time


def test_basic_training(cargo_type='normal', test_steps=3000):
    """基础训练测试，确保环境和模型正常工作"""
    
    print(f"🧪 基础训练测试 ({test_steps} 步)")
    print(f"货物类型: {cargo_type}")
    print(f"✅ 改进项目:")
    print(f"   - 环境重置修复")
    print(f"   - 进度条简化")  
    print(f"   - 模型维度修复")
    print(f"   - 学习率衰减 (5k步后)")
    print("-" * 50)
    
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
    print("🤖 创建改进的TD3模型...")
    model = ImprovedTD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        buffer_size=50000,  # 降低缓冲区大小
        verbose=0,  # 降低详细度
        tensorboard_log=f"./test_logs/basic_{cargo_type}"
    )
    
    print(f"✅ 模型创建成功")
    print(f"   策略网络: MlpPolicy")
    print(f"   特征维度: 缩小规模 (16+16+16+16→128)")
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    print(f"   目标步数: {test_steps}")
    print(f"   预计时间: ~{test_steps//100:.0f}分钟")
    
    start_time = time.time()
    episode_count = 0
    last_episode_count = 0
    
    # 自定义训练循环，更好的控制输出
    try:
        print(f"   训练进行中... (每500步输出一次)")
        
        # 分段训练，每500步检查一次
        for segment in range(0, test_steps, 500):
            segment_steps = min(500, test_steps - segment)
            
            model.learn(
                total_timesteps=segment_steps,
                progress_bar=False,
                reset_num_timesteps=False,
                log_interval=100  # 减少内部日志
            )
            
            # 输出进度
            current_time = time.time()
            elapsed = current_time - start_time
            total_steps = segment + segment_steps
            
            print(f"   ✓ {total_steps:4d}/{test_steps} 步完成 "
                  f"(用时: {elapsed:.1f}s, 速度: {total_steps/elapsed:.1f} 步/秒)")
                  
            # 检查是否有学习率衰减
            if total_steps >= 5000:
                print(f"     📉 注意: 已达到5000步，学习率开始衰减")
                break
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        env.close()
        return None
    
    training_time = time.time() - start_time
    print(f"✅ 训练完成! 用时: {training_time:.1f}秒")
    
    # 简单评估
    print(f"\n📊 基础评估 (3 episodes)...")
    
    rewards = []
    success_count = 0
    
    for i in range(3):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        
        for step in range(200):  # 限制每个episode最多200步
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            
            if step_info.get('success', False):
                episode_success = True
                
            if terminated or truncated:
                break
                
        rewards.append(episode_reward)
        if episode_success:
            success_count += 1
            
        print(f"   Episode {i+1}: 奖励={episode_reward:.1f}, 成功={'✓' if episode_success else '✗'}")
    
    avg_reward = np.mean(rewards)
    success_rate = success_count / 3
    
    # 结果总结
    print(f"\n" + "="*50)
    print(f"📋 测试结果:")
    print(f"   训练步数: {test_steps}")
    print(f"   训练时间: {training_time:.1f}秒")
    print(f"   平均奖励: {avg_reward:.1f}")
    print(f"   成功率: {success_rate*100:.1f}%")
    
    if avg_reward > -800 and success_rate >= 0:
        print(f"✅ 基础训练成功! 环境和模型工作正常")
        status = "成功"
    else:
        print(f"⚠️  基础训练有问题，需要进一步调试")
        status = "需调试"
    
    env.close()
    
    print(f"="*50)
    
    return {
        'status': status,
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'training_time': training_time
    }


if __name__ == "__main__":
    print("🔧 基础训练系统测试")
    print("解决环境重置和进度条问题")
    print("=" * 50)
    
    # 运行测试
    result = test_basic_training('normal', test_steps=3000)
    
    if result and result['status'] == '成功':
        print(f"\n🎉 推荐下一步:")
        print(f"   python test_improved_training.py  # 长期训练测试")
        print(f"   python train_stage1.py --steps 10000  # 正式训练")
    else:
        print(f"\n🔧 建议先解决基础问题再进行复杂训练")
