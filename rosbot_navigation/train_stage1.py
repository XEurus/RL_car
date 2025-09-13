"""
第一阶段训练脚本 - 基础模型训练
使用AMCL定位和课程学习训练三个独立的导航模型
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any

from src.environments.navigation_env import ROSbotNavigationEnv
from src.models.td3_robust import ImprovedTD3

import mlflow
import mlflow.pytorch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.utils.webots_launcher import start_webots_instance, attach_process_cleanup_to_env
from functools import partial
import os
import signal
import multiprocessing as mp
import time
from queue import Empty, Full
# ================= 分布式训练实现 ==================

def _actor_process(rank: int, cargo_type: str, args: argparse.Namespace, obs_q: mp.Queue, act_q: mp.Queue, exp_q: mp.Queue, stop_event: mp.Event):
    print(f"[ACTOR{rank}] 进程启动...")
    try:
        # 启动单实例 Webots 并连接
        proc, url = start_webots_instance(
            instance_id=rank,
            world_path=getattr(args, 'world', None) if args else None,
            headless=getattr(args, 'headless', True),
            fast_mode=getattr(args, 'fast_mode', True),
            no_rendering=getattr(args, 'no_rendering', True),
            batch=getattr(args, 'batch', True),
            minimize=getattr(args, 'minimize', True),
            stdout=True,
            stderr=True
        )
        print(f"[ACTOR{rank}] Webots 启动完成，URL: {url}")
        
        env = ROSbotNavigationEnv(
            cargo_type=cargo_type,
            instance_id=rank,
            controller_url=url,
            fast_mode=(args.fast_mode if args else True),
            control_period_ms=(args.control_period_ms if args else 200)
        )
        attach_process_cleanup_to_env(env, proc)
        print(f"[ACTOR{rank}] 环境初始化完成")

        obs, info = env.reset()
        print(f"[ACTOR{rank}] 开始采样循环")
        
        episode_steps = 0
        while not stop_event.is_set():
            # 1) 将观测放入队列，请求一个动作
            try:
                obs_q.put((rank, obs), timeout=0.1)
            except Full:
                # 观测队列满了，Learner 暂未处理，稍后再试，避免阻塞
                time.sleep(0.001)
                continue
            except Exception as e:
                print(f"[ACTOR{rank}] 发送观测异常: {e}")
                time.sleep(0.001)
                continue

            # 2) 等待 Learner 返回动作（只在成功发送观测后才等待）
            try:
                action = act_q.get(timeout=2.0)
            except Empty:
                # 未及时获得动作，重试
                # 不退出循环，避免 actor 过早终止
                continue
            except Exception as e:
                print(f"[ACTOR{rank}] 获取动作异常: {e}")
                continue

            # 3) 执行动作并写入经验
            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
            except Exception as e:
                print(f"[ACTOR{rank}] 环境 step 异常: {e}")
                continue

            try:
                exp_q.put((obs, action, reward, next_obs, done), timeout=0.1)
            except Full:
                # 经验队列满了，丢弃该条经验以保证流动性
                pass
            except Exception as e:
                print(f"[ACTOR{rank}] 写入经验异常: {e}")
                # 经验写入失败不应终止 actor
                pass

            obs = next_obs
            episode_steps += 1
            if done:
                try:
                    obs, info = env.reset()
                except Exception as e:
                    print(f"[ACTOR{rank}] reset 异常: {e}")
                    continue
                episode_steps = 0
                print(f"[ACTOR{rank}] Episode 完成")
    except KeyboardInterrupt:
        print(f"[ACTOR{rank}] 收到中断信号")
    except Exception as e:
        print(f"[ACTOR{rank}] 启动异常: {e}")
    finally:
        print(f"[ACTOR{rank}] 进程退出")
        try:
            env.close()
        except Exception:
            pass


def run_distributed_training(args: argparse.Namespace, model_path: str):
    """多Actor+单Learner 训练主控"""
    mp.set_start_method('spawn', force=True)
    num_actors = int(getattr(args, 'num_actors', max(1, args.num_envs if hasattr(args, 'num_envs') else 1)))

    # 队列：每个 actor 配一对 obs/act 队列；经验共享一个队列
    obs_queues = [mp.Queue(maxsize=8) for _ in range(num_actors)]
    act_queues = [mp.Queue(maxsize=8) for _ in range(num_actors)]
    exp_queue = mp.Queue(maxsize=4096)
    stop_event = mp.Event()
    init_event = mp.Event()  # 初始化完成信号

    # Learner 进程
    learner_proc = mp.Process(target=_learner_process, args=(args, model_path, obs_queues, act_queues, exp_queue, stop_event, init_event), daemon=True)
    learner_proc.start()
    
    # 等待 Learner 初始化完成
    print("[MAIN] 等待 Learner 初始化完成...")
    init_event.wait()  # 阻塞直到 learner 设置 init_event
    print("[MAIN] Learner 初始化完成，开始启动 Actors...")
    # 启动 Actors
    actors = []
    for i in range(num_actors):
        p = mp.Process(target=_actor_process, args=(i, args.cargo_type, args, obs_queues[i], act_queues[i], exp_queue, stop_event), daemon=True)
        p.start()
        actors.append(p)

    try:
        # 主进程只需等待 Learner 结束或 KeyboardInterrupt
        learner_proc.join()
    except KeyboardInterrupt:
        print("\n分布式训练被中断，正在清理...")
    finally:
        stop_event.set()
        for p in actors:
            try:
                p.join(timeout=5)
            except Exception:
                pass
        try:
            learner_proc.join(timeout=5)
        except Exception:
            pass


def _learner_process(args: argparse.Namespace, model_path: str, obs_queues: list, act_queues: list, exp_queue: mp.Queue, stop_event: mp.Event, init_event: mp.Event):
    """Learner：集中选择动作并训练TD3，异步消费经验。"""
    print("[LEARNER] 进程启动...")
    
    device = args.device if hasattr(args, 'device') else 'auto'

    # # 构建一个假的 env 以拿到空间（不做采样，仅供 SB3 初始化）
    # # 直接实例化一个 ROSbot 环境以获取 space 再关闭，避免多余 Webots：改为从类静态方法提供空间会更好
    # # 这里临时启动单个环境用于取 space，然后立即关闭 Webots 进程（成本较高，但一次性）
    # temp_proc, temp_url = start_webots_instance(
    #     instance_id=9999,
    #     world_path=getattr(args, 'world', None) if args else None,
    #     headless=True, fast_mode=True, no_rendering=True, batch=True, minimize=True,
    #     stdout=True, stderr=True
    # )
    # # 传入 controller_url 以连接到对应实例
    # temp_env = ROSbotNavigationEnv(cargo_type=args.cargo_type, instance_id=9999, controller_url=temp_url,
    #                                fast_mode=True, control_period_ms=args.control_period_ms)
    # obs_space = temp_env.observation_space
    # act_space = temp_env.action_space
    # try:
    #     temp_env.close()
    # except Exception:
    #     pass
    # try:
    #     os.kill(temp_proc.pid, signal.SIGTERM)
    # except Exception:
    #     pass
    
    # print("[LEARNER] 获取环境空间完成")

    # 直接通过静态方法获取空间定义，避免启动临时 Webots 实例
    try:
        obs_space, act_space = ROSbotNavigationEnv.get_spaces()
        print("[LEARNER] 获取环境空间完成(静态)")
    except Exception as e:
        print(f"[LEARNER] 获取环境空间失败: {e}")
        raise

    # 创建一个 Dummy 环境供 SB3 初始化（最小实现）
    import gymnasium as gym
    class _FakeEnv(gym.Env):
        def __init__(self, obs_space, act_space):
            self.observation_space = obs_space
            self.action_space = act_space
        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample().astype(np.float32), {}
        def step(self, action):
            return self.observation_space.sample().astype(np.float32), 0.0, True, False, {}

    fake_env = _FakeEnv(obs_space, act_space)

    model = ImprovedTD3(
        "MlpPolicy",
        fake_env,
        learning_rate=getattr(args, 'learning_rate', 3e-4),
        buffer_size=getattr(args, 'buffer_size', 100000),
        learning_starts=getattr(args, 'learning_starts', 10000),
        batch_size=getattr(args, 'batch_size', 256),
        gamma=getattr(args, 'gamma', 0.98),
        tau=getattr(args, 'tau', 0.005),
        gradient_steps=getattr(args, 'gradient_steps', 1),
        train_freq=getattr(args, 'train_freq', 1),
        policy_delay=getattr(args, 'policy_delay', 2),
        target_policy_noise=getattr(args, 'target_noise', 0.2),
        target_noise_clip=getattr(args, 'noise_clip', 0.5),
        verbose=1,
        tensorboard_log=None,
        device=device,
    )
    
    print("[LEARNER] TD3 模型初始化完成")
    # 向主进程发送初始化完成的信号
    init_event.set()
    # 分布式主循环
    total_steps = int(getattr(args, 'total_steps', 50000))
    collected = 0
    last_print = 0
    actions_served = 0
    
    # 奖励跟踪变量
    reward_history = []  # 存储所有奖励
    episode_count = 0
    last_reward_print = 0
    
    print(f"[LEARNER] 开始主循环，目标步数: {total_steps}")

    try:
        while not stop_event.is_set() and collected < total_steps:
            # 优先处理所有动作请求（避免 actor 超时）
            for i, (oq, aq) in enumerate(zip(obs_queues, act_queues)):
                # 避免使用不可靠的 empty()，直接使用 get_nowait 循环取尽
                while True:
                    try:
                        aid, obs = oq.get_nowait()
                    except Exception:
                        break  # 队列暂时无数据，处理下一个 actor
                    try:
                        # 将观测转为 numpy，并通过 policy.predict 获得缩放后的动作
                        obs_np = np.asarray(obs, dtype=np.float32)
                        with torch.no_grad():
                            action, _ = model.policy.predict(obs_np, deterministic=True)
                        aq.put(action, timeout=1.0)
                        actions_served += 1
                    except Exception as e:
                        print(f"[LEARNER] 生成/发送动作异常: {e}")
                        # 当前 actor 发送失败则跳出其处理，避免死循环
                        break
        
            experiences_added = 0
            
            for _ in range(64):
                try:
                    obs, action, reward, next_obs, done = exp_queue.get_nowait()
                    
                    # 记录奖励
                    reward_history.append(reward)  # 添加奖励到历史记录
                    
                    # 如果是结束状态，计算该episode的平均奖励
                    # if done:
                    #     episode_count += 1
                    #     # 计算最近100个奖励的平均值作为该episode的平均奖励
                    #     recent_rewards = reward_history[-100:] if len(reward_history) > 100 else reward_history
                    #     avg_episode_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                    #     print(f"[LEARNER] Episode {episode_count} 完成, 平均奖励: {avg_episode_reward:.4f}")
                    
                    model.replay_buffer.add(
                        obs=obs,
                        next_obs=next_obs,
                        action=action,
                        reward=reward,
                        done=done,
                        infos=[{}]
                    )
                    collected += 1
                    experiences_added += 1
                except Empty:
                    # 队列为空，退出循环
                    break
                except Exception as e:
                    # 其他异常，记录后跳过当前条目，继续尝试获取更多经验
                    print(f"[LEARNER] 读取经验异常: {e}")
                    raise e
            
            # if experiences_added > 0:
            #     print(f"[LEARNER] 本轮添加 {experiences_added} 个经验，总计: {collected}")
            # if collected < getattr(args, 'learning_starts') and collected % 100 == 0 and collected > 0:
            #     print(f"[LEARNER] 收集样本: {collected}, 缓冲区大小: {model.replay_buffer.size()}")
                
            # 训练
            if collected >= getattr(args, 'learning_starts') and model.replay_buffer.size() >= model.batch_size:
                # print(f"[LEARNER] 训练，收集样本: {collected}, 缓冲区大小: {model.replay_buffer.size()}")
                model.train(gradient_steps=getattr(args, 'gradient_steps', 1), batch_size=model.batch_size)

            # 降频打印
            if collected - last_print >= getattr(args, 'log_interval', 200):
                last_print = collected
                # 计算最近100个奖励的平均值
                recent_rewards = reward_history[-100:] if len(reward_history) > 100 else reward_history
                avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                print(f"[LEARNER] 收集样本: {collected}, 缓冲区大小: {model.replay_buffer.size()}, 最近100步平均奖励: {avg_reward:.4f}")
                
            # 奖励统计打印
            if collected - last_reward_print >= 1000:  # 每1000步打印一次奖励统计
                last_reward_print = collected
                if reward_history:
                    print(f"\n[LEARNER] === 奖励统计 (步数: {collected}) ===")
                    print(f"平均奖励: {sum(reward_history) / len(reward_history):.4f}")
                    print(f"最大奖励: {max(reward_history):.4f}")
                    print(f"最小奖励: {min(reward_history):.4f}")
                    print(f"最近100步平均: {sum(reward_history[-100:]) / min(100, len(reward_history)):.4f}")
                    print(f"已完成Episodes: {episode_count}")
                    print(f"==============================\n")
            
    except KeyboardInterrupt:
        print("[LEARNER] 收到中断信号")
    except Exception as e:
        print(f"[LEARNER] 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[LEARNER] 保存模型...")
        try:
            model.save(model_path)
            print(f"[LEARNER] 模型已保存到: {model_path}")
        except Exception as e:
            print(f"[LEARNER] 保存模型失败: {e}")


class MlflowCallback(BaseCallback):
    """自定义回调 - 将指标记录到MLflow (降频记录)"""
    def __init__(self, mlflow_log_interval: int = 500, verbose=0):
        super(MlflowCallback, self).__init__(verbose)
        self.mlflow_log_interval = int(max(1, mlflow_log_interval))
        self._last_log_step = 0

    def _on_step(self) -> bool:
        """按间隔记录日志到MLflow，避免每步写入造成IO瓶颈"""
        if (self.num_timesteps - self._last_log_step) < self.mlflow_log_interval:
            return True
        self._last_log_step = self.num_timesteps
        try:
            if self.logger and hasattr(self.logger, 'name_to_value'):
                for key, value in self.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value, step=self.num_timesteps)
        except Exception:
            pass
        return True


class AMCLMetricsCallback(BaseCallback):
    """AMCL指标记录回调"""
    
    def __init__(self, verbose=0):
        super(AMCLMetricsCallback, self).__init__(verbose)
        self.amcl_uncertainties = []
        self.convergence_scores = []
        
    def _on_step(self) -> bool:
        """每步记录AMCL指标"""
        try:
            # 获取环境的AMCL信息
            if hasattr(self.training_env, 'unwrapped'):
                env = self.training_env.unwrapped
                if hasattr(env, 'get_amcl_uncertainty'):
                    uncertainty = env.get_amcl_uncertainty()
                    self.amcl_uncertainties.append(uncertainty)
                    
                    convergence = env.amcl_localizer.has_converged()
                    self.convergence_scores.append(float(convergence))
                    
                    # 记录到MLflow
                    if len(self.amcl_uncertainties) % 100 == 0:
                        avg_uncertainty = np.mean(self.amcl_uncertainties[-100:])
                        avg_convergence = np.mean(self.convergence_scores[-100:])
                        
                        self.logger.record("amcl/position_uncertainty", avg_uncertainty)
                        self.logger.record("amcl/convergence_rate", avg_convergence)
                        
                        # 清空缓冲区
                        self.amcl_uncertainties = []
                        self.convergence_scores = []
        except Exception as e:
            # 静默处理AMCL指标获取错误
            pass
        
        return True


class CurriculumCallback(BaseCallback):
    """课程学习回调 - 根据进度调整不确定性"""
    
    def __init__(self, uncertainty_curriculum, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.uncertainty_curriculum = uncertainty_curriculum
        self.current_uncertainty = 0.1
        
    def _on_step(self) -> bool:
        """根据训练进度调整不确定性"""
        if hasattr(self.locals, 'total_timesteps'):
            total_steps = self.locals['total_timesteps']
            current_step = self.num_timesteps
            
            # 获取当前阶段的不确定性
            new_uncertainty = self.uncertainty_curriculum.get_uncertainty_for_step(current_step)
            
            if new_uncertainty != self.current_uncertainty:
                self.current_uncertainty = new_uncertainty
                
                # 更新环境的不确定性级别
                if hasattr(self.training_env, 'unwrapped'):
                    env = self.training_env.unwrapped
                    if hasattr(env, 'amcl_localizer'):
                        env.amcl_localizer.set_uncertainty_level(new_uncertainty)
                        
                        self.logger.record("curriculum/uncertainty_level", new_uncertainty)
                        self.logger.record("curriculum/training_progress", current_step / total_steps)
        
        return True


class ProgressLoggingCallback(BaseCallback):
    """进度日志记录回调（降频打印）"""
    
    def __init__(self, log_interval: int = 200, verbose=0):
        super(ProgressLoggingCallback, self).__init__(verbose)
        self.log_interval = int(max(1, log_interval))
        self.last_metrics_time = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        """定期记录训练进度"""
        # 检查是否完成了一个episode
        if self.locals.get('dones') is not None:
            for done in self.locals['dones']:
                if done:
                    self.episode_count += 1
        
        if self.num_timesteps - self.last_metrics_time >= self.log_interval:
            self.last_metrics_time = self.num_timesteps
            
            # 记录性能统计
            if hasattr(self.locals, 'infos') and self.locals['infos']:
                successes = [info.get('success', False) for info in self.locals['infos'] if info]
                if successes:
                    success_rate = sum(successes) / len(successes)
                    self.logger.record("stats/recent_success_rate", success_rate)
            
            # 记录训练信息
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.record("time/episodes", self.episode_count)
            
            print(f"训练进度: {self.num_timesteps} 步, {self.episode_count} episodes")
        
        return True


def _make_single_env(rank: int, cargo_type: str, args: argparse.Namespace):
    """顶层环境工厂函数（可被spawn方式pickle）。"""
    print(f"[ENV{rank}] 构造开始...")
    # 启动独立 Webots 实例（headless/fast）并获取 extern URL
    proc, url = start_webots_instance(
        instance_id=rank,
        world_path=getattr(args, 'world', None) if args else None,
        headless=getattr(args, 'headless', False) if args else False,
        fast_mode=getattr(args, 'fast_mode', True) if args else True,
        no_rendering=getattr(args, 'no_rendering', False) if args else False,
        batch=getattr(args, 'batch', False) if args else False,
        minimize=getattr(args, 'minimize', False) if args else False,
        stdout=getattr(args, 'stdout', False) if args else False,
        stderr=getattr(args, 'stderr', False) if args else False
    )
    # 传入 controller_url 以连接到对应实例
    env_i = ROSbotNavigationEnv(
        cargo_type=cargo_type,
        instance_id=rank,
        controller_url=url,
        fast_mode=(args.fast_mode if args else True),
        control_period_ms=(args.control_period_ms if args else 200)
    )
    attach_process_cleanup_to_env(env_i, proc)
    print(f"[ENV{rank}] 构造完成。")
    return Monitor(env_i)


def _make_single_env_connect(rank: int, cargo_type: str, args: argparse.Namespace, url: str):
    """仅连接到已有的 Webots 实例（由主进程预启动）。"""
    print(f"[ENV{rank}] 连接已有实例: {url}")
    env_i = ROSbotNavigationEnv(
        cargo_type=cargo_type,
        instance_id=rank,
        controller_url=url,
        fast_mode=(args.fast_mode if args else True),
        control_period_ms=(args.control_period_ms if args else 200)
    )
    return Monitor(env_i)


def setup_for_cargo_type(cargo_type: str, total_steps: int, device: str = 'auto', args: argparse.Namespace = None):
    """为特定货物类型设置训练"""
    
    print(f"\n=== 开始训练 {cargo_type} 货物导航模型 ===")
    print(f"总训练步数: {total_steps}")
    print(f"状态空间: 42维 (LiDAR+AMCL定位)")
    print(f"动作空间: 2维连续 [线速度, 角速度]")
    print(f"使用AMCL定位: 800粒子数")
    
    if args and getattr(args, 'num_envs', 1) > 1:
        # 使用异步向量环境以减少最慢实例的拖慢效应
        if getattr(args, 'async_vec', True):
            from src.utils.async_vec_wrapper import SB3AsyncVectorEnvWrapper
            # 可选预启动 Webots，避免在子进程中竞争启动造成卡顿
            if getattr(args, 'prelaunch_webots', True):
                pre_urls = []
                pre_pids = []
                for i in range(int(args.num_envs)):
                    print(f"[PRE] 串行预启动 Webots 实例 {i} ...")
                    proc, url = start_webots_instance(
                        instance_id=i,
                        world_path=getattr(args, 'world', None) if args else None,
                        headless=getattr(args, 'headless', False) if args else False,
                        fast_mode=getattr(args, 'fast_mode', True) if args else True,
                        no_rendering=getattr(args, 'no_rendering', False) if args else False,
                        batch=getattr(args, 'batch', False) if args else False,
                        minimize=getattr(args, 'minimize', False) if args else False,
                        stdout=getattr(args, 'stdout', False) if args else False,
                        stderr=getattr(args, 'stderr', False) if args else False
                    )
                    pre_urls.append(url)
                    pre_pids.append(proc.pid if hasattr(proc, 'pid') else None)
                # 保存到 args 以便主流程清理
                args._prelaunch_pids = pre_pids
                env_fns = [partial(_make_single_env_connect, i, cargo_type, args, pre_urls[i]) for i in range(int(args.num_envs))]
            else:
                env_fns = [partial(_make_single_env, i, cargo_type, args) for i in range(int(args.num_envs))]
            env = SB3AsyncVectorEnvWrapper(env_fns)
        else:
            # 回退：SubprocVecEnv 同步模式，使用 spawn 避免 fork 引发的死锁/资源继承问题
            print("使用 SubprocVecEnv(start_method='spawn') 创建并行环境...")
            if getattr(args, 'prelaunch_webots', True):
                pre_urls = []
                pre_pids = []
                for i in range(int(args.num_envs)):
                    print(f"[PRE] 串行预启动 Webots 实例 {i} ...")
                    proc, url = start_webots_instance(
                        instance_id=i,
                        world_path=getattr(args, 'world', None) if args else None,
                        headless=getattr(args, 'headless', False) if args else False,
                        fast_mode=getattr(args, 'fast_mode', True) if args else True,
                        no_rendering=getattr(args, 'no_rendering', False) if args else False,
                        batch=getattr(args, 'batch', False) if args else False,
                        minimize=getattr(args, 'minimize', False) if args else False,
                        stdout=getattr(args, 'stdout', False) if args else False,
                        stderr=getattr(args, 'stderr', False) if args else False
                    )
                    pre_urls.append(url)
                    pre_pids.append(proc.pid if hasattr(proc, 'pid') else None)
                args._prelaunch_pids = pre_pids
                env = SubprocVecEnv([partial(_make_single_env_connect, i, cargo_type, args, pre_urls[i]) for i in range(int(args.num_envs))], start_method='spawn')
            else:
                env = SubprocVecEnv([partial(_make_single_env, i, cargo_type, args) for i in range(int(args.num_envs))], start_method='spawn')
    else:
        # 单实例（不需要 SubprocVecEnv）
        env = _make_single_env(0, cargo_type, args)
    
    # 课程学习配置
    from src.localization.amcl_localizer import UncertaintyCurriculumTraining
    uncertainty_curriculum = UncertaintyCurriculumTraining()
    
    # 创建改进的TD3模型 - 使用getattr安全获取参数，确保命令行参数生效
    model = ImprovedTD3(
        "MlpPolicy",
        env,
        learning_rate=getattr(args, 'learning_rate', 3e-4) if args else 3e-4,
        buffer_size=getattr(args, 'buffer_size', 50000) if args else 50000,
        learning_starts=getattr(args, 'learning_starts', 1000) if args else 1000,
        batch_size=getattr(args, 'batch_size', 256) if args else 256,
        gamma=getattr(args, 'gamma', 0.98) if args else 0.98,
        tau=getattr(args, 'tau', 0.005) if args else 0.005,
        gradient_steps=getattr(args, 'gradient_steps', 1) if args else 1,
        train_freq=getattr(args, 'train_freq', 1) if args else 1,
        policy_delay=getattr(args, 'policy_delay', 2) if args else 2,
        target_policy_noise=getattr(args, 'target_noise', 0.2) if args else 0.2,
        target_noise_clip=getattr(args, 'noise_clip', 0.5) if args else 0.5,
        verbose=1,
        tensorboard_log=None,
        device=device,
    )
    
    return model, env, uncertainty_curriculum


def create_callbacks(env, uncertainty_curriculum, cargo_type: str, args: argparse.Namespace = None):
    """创建训练回调函数"""
    
    callbacks = []
    
    # AMCL指标记录 - AMCL已停用，暂时注释
    # amcl_callback = AMCLMetricsCallback(verbose=1)
    # callbacks.append(amcl_callback)
    
    # 课程学习回调 - AMCL已停用，暂时注释
    # curriculum_callback = CurriculumCallback(uncertainty_curriculum, verbose=1)
    # callbacks.append(curriculum_callback)
    
    # 进度记录回调
    progress_callback = ProgressLoggingCallback(
        log_interval=(getattr(args, 'log_interval', 200) if args else 200),
        verbose=1
    )
    callbacks.append(progress_callback)
    
    # MLflow记录回调
    mlflow_callback = MlflowCallback(
        mlflow_log_interval=(getattr(args, 'mlflow_log_interval', 500) if args else 500),
        verbose=1
    )
    callbacks.append(mlflow_callback)
    
    # 评估回调（可选，减少计算开销）
    if False:  # 可以开启评估
        eval_env = ROSbotNavigationEnv(cargo_type=cargo_type)
        eval_env = Monitor(eval_env)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/{cargo_type}_eval/",
            log_path=f"./logs/{cargo_type}_eval/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    return callbacks


def train_single_cargo_model(cargo_type: str, total_steps: int, model_save_path: str, device: str = 'auto', args: argparse.Namespace = None):
    """训练单个货物类型模型"""
    
    # 创建MLflow实验
    experiment_name = f"rosbot_navigation_{cargo_type}_stage1"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"training_{cargo_type}_{total_steps}"):
        
        # 记录实验参数 - 使用安全的getattr确保命令行参数有效
        params_dict = {
            "algorithm": "TD3",
            "cargo_type": cargo_type,
            "total_steps": total_steps,
            "state_dim": 42,
            "action_dim": 2,
            "amcl_particles": 800,
            "device": device,
        }
        
        # 安全获取所有算法参数并添加到记录中
        if args:
            # 使用命令行解析器的默认值
            algo_params = {
                "learning_rate": getattr(args, 'learning_rate', 1e-3),
                "buffer_size": getattr(args, 'buffer_size', 100000),
                "batch_size": getattr(args, 'batch_size', 256),
                "learning_starts": getattr(args, 'learning_starts', 10000),
                "gamma": getattr(args, 'gamma', 0.98),
                "tau": getattr(args, 'tau', 0.005),
                "gradient_steps": getattr(args, 'gradient_steps', 1),
                "train_freq": getattr(args, 'train_freq', 1),
                "policy_delay": getattr(args, 'policy_delay', 2),
                "target_noise": getattr(args, 'target_noise', 0.2),
                "noise_clip": getattr(args, 'noise_clip', 0.5),
                "num_envs": getattr(args, 'num_envs', 4)
            }
            params_dict.update(algo_params)
            
            # 记录Webots相关参数
            webots_params = {
                "headless": getattr(args, 'headless', False),
                "fast_mode": getattr(args, 'fast_mode', False),
                "control_period_ms": getattr(args, 'control_period_ms', 200),
                "seed": getattr(args, 'seed', 0)
            }
            params_dict.update(webots_params)
            
        # 记录所有参数
        mlflow.log_params(params_dict)
        
        # 打印关键训练参数
        print("\n📊 训练参数:")
        print(f"  - 学习率: {params_dict.get('learning_rate')}")
        print(f"  - 缓冲区大小: {params_dict.get('buffer_size')}")
        print(f"  - 批处理大小: {params_dict.get('batch_size')}")
        print(f"  - 预热步数: {params_dict.get('learning_starts')}")
        print(f"  - 折扣因子: {params_dict.get('gamma')}")
        print(f"  - 并行环境数: {params_dict.get('num_envs')}")
        
        # 设置训练环境
        model, env, uncertainty_curriculum = setup_for_cargo_type(cargo_type, total_steps, device=device, args=args)
        
        # 创建回调
        callbacks = create_callbacks(env, uncertainty_curriculum, cargo_type, args)
        
        print(f"\n开始训练 {cargo_type} 模型...")
        print(f"模型保存路径: {model_save_path}")
        
        # 开始训练
        try:
            model.learn(
                total_timesteps=total_steps,
                callback=callbacks,
                log_interval=10,
                progress_bar=True
            )
            
            # 保存模型
            model.save(model_save_path)
            
            # 记录训练完成信息
            # 从回调中获取episode数量
            episode_count = 0
            for callback in callbacks:
                if isinstance(callback, ProgressLoggingCallback):
                    episode_count = callback.episode_count
                    break
                    
            mlflow.log_metrics({
                "training_completed": 1.0,
                "final_timesteps": model.num_timesteps,
                "final_episodes": episode_count
            })
            
            print(f"\n{cargo_type} 模型训练完成！")
            print(f"模型已保存到: {model_save_path}")
            
        except KeyboardInterrupt:
            print(f"\n{cargo_type} 训练被用户中断")
            
            # 保存中断前的模型
            interrupted_path = model_save_path.replace('.zip', '_interrupted.zip')
            model.save(interrupted_path)
            print(f"中断模型已保存到: {interrupted_path}")
            
            # 记录中断信息
            mlflow.log_metrics({
                "training_interrupted": 1.0,
                "final_timesteps": model.num_timesteps
            })
            
            # 通知主进程初始化完成
            if init_event is not None:
                init_event.set()
                
            # 奖励跟踪
            reward_history = []  # 存储所有奖励
            episode_count = 0
            last_reward_print = 0
            
        except Exception as e:
            print(f"\n训练发生错误: {e}")
            
            # 记录错误信息
            mlflow.log_params({
                "training_error": str(e),
                "error_timesteps": model.num_timesteps if hasattr(model, 'num_timesteps') else 0
            })
            
            raise e
    
    return model, env


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ROSbot第一阶段训练脚本')
    parser.add_argument('--cargo_type', type=str, default='normal', 
                       choices=['normal', 'fragile', 'dangerous'],
                       help='货物类型')
    parser.add_argument('--total_steps', type=int, default=50000,
                       help='总训练步数')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型保存路径')
    parser.add_argument('--debug', action='store_true', 
                       help='调试模式')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='计算设备 (e.g., "cpu", "cuda", "auto")')
    # 日志与异步控制
    parser.add_argument('--log_interval', type=int, default=200, help='控制台打印间隔步数')
    parser.add_argument('--mlflow_log_interval', type=int, default=500, help='MLflow 记录间隔步数')
    parser.add_argument('--async_vec', type=bool, default=True, help='是否使用异步向量环境以减少最慢实例拖慢')
    parser.add_argument('--prelaunch_webots', type=bool, default=True, help='是否在主进程串行预启动 Webots 实例以避免并发启动卡顿')
    # 分布式 Actor-Learner
    parser.add_argument('--distributed', type=bool, default=True, help='启用多Actor+单Learner分布式训练')
    parser.add_argument('--num_actors', type=int, default=4, help='Actor 数量')
    # 并行/实例与Webots相关参数
    parser.add_argument('--num_envs', type=int, default=4, help='并行环境数量（>1启用多进程并行）')
    parser.add_argument('--world', type=str, default='/root/workspace/RL_car2/warehouse/worlds/warehouse4.wbt', help='Webots world 文件路径')
    parser.add_argument('--headless', type=bool,default=True, help='以无渲染/批处理模式启动 Webots')
    parser.add_argument('--fast_mode', type=bool,default=True, help='使用Webots FAST模式')
    parser.add_argument('--no-rendering',type=bool,default=True, help='渲染模式')
    parser.add_argument('--batch', type=bool,default=True, help='批处理模式')
    parser.add_argument('--minimize', type=bool,default=True, help='最小化模式')
    parser.add_argument('--control_period_ms', type=int, default=200, help='控制周期(ms)，用于减少控制往返')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    
    # TD3算法相关参数
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--buffer_size', type=int, default=100000, help='经验回放缓冲区大小')
    parser.add_argument('--learning_starts', type=int, default=10000, help='预热步数，开始学习前收集的样本数量')
    parser.add_argument('--batch_size', type=int, default=256, help='批处理大小')
    parser.add_argument('--gamma', type=float, default=0.98, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005, help='目标网络软更新系数')
    parser.add_argument('--gradient_steps', type=int, default=1, help='每步梯度更新次数')
    parser.add_argument('--train_freq', type=int, default=1, help='训练频率')
    parser.add_argument('--policy_delay', type=int, default=2, help='策略延迟更新步数')
    parser.add_argument('--target_noise', type=float, default=0.2, help='目标策略噪声')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='噪声裁剪范围')
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        print("启用调试模式")
        torch.autograd.set_detect_anomaly(True)
    
    # 创建模型保存目录
    models_dir = Path("./models/stage1")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型文件路径
    if args.model_path is None:
        model_path = str(models_dir / f"td3_{args.cargo_type}_stage1_{args.total_steps}.zip")
    else:
        model_path = args.model_path
    
    # 训练模型
    print(f"\n{'='*60}")
    print(f"开始 {args.cargo_type} 货物的第一阶段训练")
    print(f"总步数: {args.total_steps}")
    print(f"模型保存: {model_path}")
    print(f"状态空间: 42维")
    print(f"{'='*60}")
    
    # 显示关键训练参数 - 让用户确认参数已生效
    print(f"\n📋 已配置的训练参数:")
    print(f"  🧠 学习率: {args.learning_rate}")
    print(f"  📦 缓冲区大小: {args.buffer_size}")
    print(f"  🔥 预热步数: {args.learning_starts}")
    print(f"  📊 批处理大小: {args.batch_size}")
    print(f"  🔄 并行环境数: {args.num_envs}")
    print(f"  ⏱️ 控制周期: {args.control_period_ms} ms")
    print(f"{'='*60}\n")
    
    try:
        if args.distributed:
            run_distributed_training(args, model_path)
            print(f"\n分布式训练完成！模型保存到: {model_path}")
        else:
            model, env = train_single_cargo_model(
                cargo_type=args.cargo_type,
                total_steps=args.total_steps,
                model_save_path=model_path,
                device=args.device,
                args=args
            )
            
            print(f"\n训练完成！模型保存到: {model_path}")
            
            # 输出训练统计
            if hasattr(model, 'num_timesteps'):
                print(f"总训练步数: {model.num_timesteps}")
            
            # 创建训练完成标记
            completion_file = Path(model_path).parent / f"{args.cargo_type}_stage1_completed.txt"
            with open(completion_file, 'w') as f:
                f.write(f"Cargo type: {args.cargo_type}\n")
                f.write(f"Total steps: {args.total_steps}\n")
                f.write(f"Completed at: {datetime.now()}\n")
    except Exception as e:
        raise e
    finally:
        # 预启动实例清理（父进程兜底）
        if hasattr(args, '_prelaunch_pids') and args._prelaunch_pids:
            for pid in args._prelaunch_pids:
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
if __name__ == "__main__":
    # 将项目路径添加到Python路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    main()


