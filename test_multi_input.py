#!/usr/bin/env python3
"""
测试多输入观测空间的集成
验证新的观测格式是否正确工作
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environments.navigation_env import ROSbotNavigationEnv
from stable_baselines3 import TD3
import gymnasium as gym

def test_observation_space():
    """测试观测空间定义"""
    print("=== 测试观测空间定义 ===")
    
    try:
        obs_space, act_space = ROSbotNavigationEnv.get_spaces()
        print(f"观测空间类型: {type(obs_space)}")
        print(f"观测空间: {obs_space}")
        print(f"动作空间: {act_space}")
        
        # 验证观测空间是Dict类型
        assert isinstance(obs_space, gym.spaces.Dict), "观测空间应该是Dict类型"
        
        # 验证包含所需的键
        required_keys = ['local_map', 'robot_state', 'navigation_info']
        for key in required_keys:
            assert key in obs_space.spaces, f"观测空间缺少键: {key}"
        
        # 验证各个空间的形状
        assert obs_space['local_map'].shape == (1, 200, 200), "local_map形状不正确"
        assert obs_space['robot_state'].shape == (12,), "robot_state形状不正确"
        assert obs_space['navigation_info'].shape == (10,), "navigation_info形状不正确"
        
        print("✅ 观测空间定义测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 观测空间定义测试失败: {e}")
        return False

def test_fake_env_with_multiinput():
    """测试假环境是否能正确处理多输入观测"""
    print("\n=== 测试假环境多输入处理 ===")
    
    try:
        obs_space, act_space = ROSbotNavigationEnv.get_spaces()
        
        # 创建假环境
        class _FakeEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample().astype(np.float32), {}
            def step(self, action):
                return self.observation_space.sample().astype(np.float32), 0.0, True, False, {}
        
        fake_env = _FakeEnv(obs_space, act_space)
        
        # 测试重置
        obs, info = fake_env.reset()
        print(f"重置观测类型: {type(obs)}")
        print(f"重置观测键: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")
        
        # 验证观测格式
        assert isinstance(obs, dict), "观测应该是字典格式"
        assert 'local_map' in obs, "观测缺少local_map"
        assert 'robot_state' in obs, "观测缺少robot_state"
        assert 'navigation_info' in obs, "观测缺少navigation_info"
        
        # 验证各部分形状
        assert obs['local_map'].shape == (1, 200, 200), f"local_map形状错误: {obs['local_map'].shape}"
        assert obs['robot_state'].shape == (12,), f"robot_state形状错误: {obs['robot_state'].shape}"
        assert obs['navigation_info'].shape == (10,), f"navigation_info形状错误: {obs['navigation_info'].shape}"
        
        print("✅ 假环境多输入处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 假环境多输入处理测试失败: {e}")
        return False

def test_td3_multiinput_policy():
    """测试TD3多输入策略创建"""
    print("\n=== 测试TD3多输入策略创建 ===")
    
    try:
        obs_space, act_space = ROSbotNavigationEnv.get_spaces()
        
        # 创建假环境
        class _FakeEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample().astype(np.float32), {}
            def step(self, action):
                return self.observation_space.sample().astype(np.float32), 0.0, True, False, {}
        
        fake_env = _FakeEnv(obs_space, act_space)
        
        # 创建TD3模型
        model = TD3(
            'MultiInputPolicy',
            fake_env,
            learning_rate=3e-4,
            buffer_size=1000,  # 小缓冲区用于测试
            learning_starts=100,
            batch_size=32,
            verbose=1,
            policy_kwargs={
                'features_extractor_kwargs': {
                    'normalized_image': True,
                    'cnn_output_dim': 256
                },
                'net_arch': {
                    'pi': [256, 256],
                    'qf': [256, 256]
                }
            }
        )
        
        print(f"模型创建成功: {type(model)}")
        print(f"策略类型: {type(model.policy)}")
        
        # 测试预测
        obs, _ = fake_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        print(f"预测动作形状: {action.shape}")
        print(f"预测动作: {action}")
        
        assert action.shape == (10,), f"动作形状错误: {action.shape}"
        
        print("✅ TD3多输入策略创建测试通过")
        return True
        
    except Exception as e:
        print(f"❌ TD3多输入策略创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试多输入观测空间集成...")
    
    tests = [
        test_observation_space,
        test_fake_env_with_multiinput,
        test_td3_multiinput_policy
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！多输入观测空间集成成功。")
        return True
    else:
        print("⚠️  部分测试失败，需要修复问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
