"""
基于 Webots GUI 的模型测试脚本
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# 确保可以从本目录下的 src/ 导入
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 项目内导入
from src.environments.navigation_env import ROSbotNavigationEnv
from src.environments.local_map_obs import LocalMapNavigationEnv
from src.models.td3_robust import ImprovedTD3
from src.utils.webots_launcher import start_webots_instance, attach_process_cleanup_to_env
from stable_baselines3 import TD3
import gymnasium as gym


class LocalMapGymWrapper(gym.Env):
    """将LocalMapNavigationEnv包装为标准的Gymnasium环境"""
    
    def __init__(self, local_map_env):
        super().__init__()
        self.env = local_map_env
        self.observation_space = local_map_env.observation_space
        self.action_space = local_map_env.action_space
        
        # 保存原始环境的方法
        self._get_sup_position = local_map_env._get_sup_position
        self._get_sup_orientation = local_map_env._get_sup_orientation
        self._get_lidar_data = local_map_env._get_lidar_data
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs, info, lidar_data = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def step(self, action):
        """执行动作"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def render(self, mode='human'):
        """渲染环境"""
        if hasattr(self.env, 'render'):
            return self.env.render(mode)
        return None
    
    def _calculate_distance_to_target(self):
        """计算到目标的距离"""
        return self.env.base_env._calculate_distance_to_target()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webots GUI 模型测试")
    parser.add_argument("--model", type=str,default="results/test_2025-09-16 18:58:03.800635/models/td3_normal_stage1_200000_2025-09-16 18:58:03.800635.zip", help="已训练模型的 .zip 路径 (SB3 格式)")
    parser.add_argument("--cargo_type", type=str, default="normal", choices=["normal", "fragile", "dangerous"], help="货物类型")
    parser.add_argument("--episodes", type=int, default=20, help="测试 episode 数")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性策略动作")
    parser.add_argument("--device", type=str, default="cuda", help="模型推理设备：auto/cpu/cuda")
    parser.add_argument("--world", type=str, default="/root/workspace/RL_car2/warehouse/worlds/warehouse2.wbt", help="可选，自定义 world 文件路径")
    parser.add_argument("--control_period_ms", type=int, default=200, help="控制周期 (ms)，与训练一致")
    parser.add_argument("--fast_mode", action="store_true", help="以 fast 模式运行 Webots (默认关闭以更贴近真实速度)")
    parser.add_argument("--debug", action="store_true", help="启用环境与脚本调试输出")
    return parser.parse_args()


def load_model(model_path: str, env, device: str = "auto"):
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_file}")
    # SB3 的 .load 可指定 device；加载后设置 env 用于 predict
    model = TD3.load(str(model_file), env=env, device=device)
    return model




def run_episode(env, model, deterministic: bool = True, episode_index: int = 0,
                debug: bool = False) -> dict:
    obs, info = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    # 距离信息（可选）
    try:
        dist0, _ = env._calculate_distance_to_target()
    except Exception:
        dist0 = np.nan

    print(f"\n==== 开始 Episode {episode_index+1} ====")
    print(f"初始距离目标: {dist0:.3f} m" if np.isfinite(dist0) else "初始距离目标: N/A")

    while not done:
        with torch.no_grad():
            # 在分布式训练中我们用 policy.predict() 得到已缩放动作，这里保持一致
            action, _ = model.policy.predict(obs, deterministic=deterministic)
        if debug:
            try:
                a_pairs = action.reshape(5, 2)
                print(f"[DEBUG][model_out] pairs(0..4): " + \
                      ", ".join([f"({a_pairs[i,0]:+.3f},{a_pairs[i,1]:+.3f})" for i in range(min(5, a_pairs.shape[0]))]))
                print(f"[DEBUG][obs] shape: {obs.shape}, dtype: {obs.dtype}, range: [{obs.min():.3f}, {obs.max():.3f}]")
            except Exception:
                print(f"[DEBUG][model_out] action shape: {getattr(action, 'shape', None)}")
                print(f"[DEBUG][obs] shape: {getattr(obs, 'shape', None)}")
        # 新模型直接输出正确的动作格式，无需转换
        if debug:
            try:
                a_pairs = action.reshape(5, 2)
                print(f"[DEBUG][model_output] pairs: " + \
                      ", ".join([f"({a_pairs[i,0]:+.3f},{a_pairs[i,1]:+.3f})" for i in range(5)]))
            except Exception:
                pass
        next_obs, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += float(reward)
        obs = next_obs
        done = bool(terminated or truncated)

        # 可选：打印部分关键信息
        try:
            d, _ = env._calculate_distance_to_target()
            print(f"step={steps:03d} reward={reward:+.4f} dist={d:.3f} terminated={terminated} truncated={truncated}")
        except Exception:
            print(f"step={steps:03d} reward={reward:+.4f} terminated={terminated} truncated={truncated}")

    # 结束统计
    try:
        dist_final, _ = env._calculate_distance_to_target()
    except Exception:
        dist_final = np.nan

    print(f"Episode {episode_index+1} 结束: 总奖励={total_reward:.4f}, 步数={steps}, 终止距离={dist_final:.3f} m" if np.isfinite(dist_final) else f"Episode {episode_index+1} 结束: 总奖励={total_reward:.4f}, 步数={steps}")

    return {
        "total_reward": total_reward,
        "steps": steps,
        "final_distance": float(dist_final) if np.isfinite(dist_final) else None,
    }


def main():
    args = parse_args()

    # 1) 启动 Webots（带 GUI）
    print("启动 Webots 实例 (GUI 模式)...")
    proc = None
    url: Optional[str] = None
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            proc, url = start_webots_instance(
                instance_id=0,
                world_path=args.world,
                headless=False,          # 强制 GUI
                fast_mode=bool(args.fast_mode),
                no_rendering=False,
                batch=False,
                minimize=False,
                stdout=True,
                stderr=True,
            )
            if proc != -2 and url:
                break
        except Exception as e:
            last_err = e
            print(f"启动 Webots 失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(1.0)
    if proc in (-2, None) or not url:
        raise RuntimeError(f"无法启动 Webots 或解析 URL: {last_err}")

    print(f"Webots extern controller URL: {url}")

    # 2) 构造环境（任务发布与训练一致：reset() 内部会调用 _set_navigation_task）
    base_env = ROSbotNavigationEnv(
        cargo_type=args.cargo_type,
        instance_id=0,
        controller_url=url,
        fast_mode=bool(args.fast_mode),
        control_period_ms=int(args.control_period_ms),
        debug=bool(args.debug),
    )
    
    # 使用LocalMapNavigationEnv包装基础环境，与训练保持一致
    local_map_env = LocalMapNavigationEnv(
        base_env=base_env,
        map_size=200,
        resolution=0.05,
        max_range=10.0
    )
    
    # 包装为标准的Gymnasium环境以兼容Stable Baselines3
    env = LocalMapGymWrapper(local_map_env)
    attach_process_cleanup_to_env(env, proc)

    # 3) 加载模型
    model = load_model(args.model, env, device=args.device)
    print("模型加载完成，开始测试...")

    # 4) 运行若干 episodes
    results = []
    try:
        for ep in range(int(args.episodes)):
            ep_res = run_episode(env, model, deterministic=bool(args.deterministic), episode_index=ep,
                                 debug=bool(args.debug))
            results.append(ep_res)
    except KeyboardInterrupt:
        print("\n收到中断信号，提前结束测试...")
    finally:
        try:
            env.close()
        except Exception:
            pass

    # 5) 汇总
    if results:
        avg_reward = float(np.mean([r["total_reward"] for r in results]))
        avg_steps = float(np.mean([r["steps"] for r in results]))
        final_dists = [r["final_distance"] for r in results if r["final_distance"] is not None]
        avg_final_dist = float(np.mean(final_dists)) if final_dists else None
        print("\n==== 测试汇总 ====")
        print(f"货物类型: {args.cargo_type}")
        print(f"Episodes: {len(results)}  平均奖励: {avg_reward:.4f}  平均步数: {avg_steps:.1f}")
        if avg_final_dist is not None:
            print(f"平均终止距离: {avg_final_dist:.3f} m")


if __name__ == "__main__":
    main()
