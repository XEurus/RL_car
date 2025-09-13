# -*- coding: utf-8 -*-
"""
SB3AsyncVectorEnvWrapper
- 使用 Gymnasium 的 AsyncVectorEnv 提供异步收集，减少最慢实例的拖慢效应。
- 对齐 Stable-Baselines3 期望的 VecEnv 接口（部分最小实现）。

注意：
- 仅实现了本项目需要的基础方法：reset、step
actions、close、render、seed、get_attr、set_attr、env_method。
- 观测、动作空间取自第一个子环境实例。
"""

from typing import Callable, List, Any
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym as gym  # 兼容


class SB3AsyncVectorEnvWrapper:
    def __init__(self, env_fns: List[Callable[[], Any]]):
        # 将旧版 Gym 风格环境适配为 Gymnasium v0.26 API，避免 reset/step 签名不匹配
        def _wrap_compat(fn: Callable[[], Any]):
            def _maker():
                env = fn()
                # 动态包装：提供 reset()->(obs, info) 和 step()->(obs, reward, terminated, truncated, info)
                class GymV26CompatWrapper(gym.Wrapper):
                    def reset(self, *, seed=None, options=None):
                        try:
                            if seed is not None and hasattr(self.env, 'seed'):
                                self.env.seed(seed)
                        except Exception:
                            pass
                        obs = self.env.reset()
                        info = {}
                        return obs, info

                    def step(self, action):
                        out = self.env.step(action)
                        if len(out) == 5:
                            # 已是新API
                            return out
                        obs, reward, done, info = out
                        terminated = bool(done)
                        truncated = bool(info.get('TimeLimit.truncated', False))
                        return obs, reward, terminated, truncated, info
                return GymV26CompatWrapper(env)
            return _maker

        wrapped_env_fns = [_wrap_compat(fn) for fn in env_fns]
        # 直接构建异步向量环境，由其内部推断 space（避免在主进程提前创建真实环境）
        self._vec = gym.vector.AsyncVectorEnv(wrapped_env_fns)
        self.num_envs = self._vec.num_envs
        # 从向量环境读取单环境的 space 定义
        # gymnasium 提供 single_observation_space / single_action_space
        self.observation_space = getattr(self._vec, 'single_observation_space', None)
        self.action_space = getattr(self._vec, 'single_action_space', None)

    def reset(self):
        obs, info = self._vec.reset()
        return obs

    def step_async(self, actions):
        # SB3 接口兼容：部分算法会调用，但 TD3 主流程使用 step 等待返回
        self._pending_actions = actions

    def step_wait(self):
        return self.step(self._pending_actions)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self._vec.step(actions)
        dones = np.logical_or(terminations, truncations)
        return obs, rewards, dones, infos

    def close(self):
        self._vec.close()

    def render(self, *args, **kwargs):
        return None

    def seed(self, seed=None):
        # 分发给子环境
        if seed is None:
            return
        for i in range(self.num_envs):
            try:
                self.env_method('seed', seed + i, indices=[i])
            except Exception:
                pass

    # 以下方法为 SB3 VecEnv 适配
    def get_attr(self, attr_name, indices=None):
        return self._vec.get_attr(attr_name, indices=indices)

    def set_attr(self, attr_name, value, indices=None):
        return self._vec.set_attr(attr_name, value, indices=indices)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return self._vec.env_method(method_name, *args, indices=indices, **kwargs)
