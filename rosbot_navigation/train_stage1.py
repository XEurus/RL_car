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


class MlflowCallback(BaseCallback):
    """自定义回调 - 将指标记录到MLflow (兼容旧版)"""
    def __init__(self, verbose=0):
        super(MlflowCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """在每一步记录所有可用的日志值到MLflow"""
        if self.logger and hasattr(self.logger, 'name_to_value'):
            for key, value in self.logger.name_to_value.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=self.num_timesteps)
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
    """进度日志记录回调"""
    
    def __init__(self, verbose=0):
        super(ProgressLoggingCallback, self).__init__(verbose)
        self.last_metrics_time = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        """定期记录训练进度"""
        # 检查是否完成了一个episode
        if self.locals.get('dones') is not None:
            for done in self.locals['dones']:
                if done:
                    self.episode_count += 1
        
        if self.num_timesteps - self.last_metrics_time >= 10000:  # 每10000步记录一次
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


def setup_for_cargo_type(cargo_type: str, total_steps: int, device: str = 'auto', args: argparse.Namespace = None):
    """为特定货物类型设置训练"""
    
    print(f"\n=== 开始训练 {cargo_type} 货物导航模型 ===")
    print(f"总训练步数: {total_steps}")
    print(f"状态空间: 42维 (LiDAR+AMCL定位)")
    print(f"动作空间: 2维连续 [线速度, 角速度]")
    print(f"使用AMCL定位: 800粒子数")
    
    # 创建环境（支持多实例并行）
    def make_env_fn(rank: int):
        def _init():
            # 启动独立 Webots 实例（headless/fast）并获取 extern URL
            # 启动Webots实例并传递所有相关命令行参数
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
            # 关闭时杀掉 Webots 进程
            attach_process_cleanup_to_env(env_i, proc)
            return Monitor(env_i)
        return _init

    if args and getattr(args, 'num_envs', 1) > 1:
        env = SubprocVecEnv([make_env_fn(i) for i in range(int(args.num_envs))])
    else:
        # 单实例（不需要 SubprocVecEnv）
        env = make_env_fn(0)()
    
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


def create_callbacks(env, uncertainty_curriculum, cargo_type: str):
    """创建训练回调函数"""
    
    callbacks = []
    
    # AMCL指标记录 - AMCL已停用，暂时注释
    # amcl_callback = AMCLMetricsCallback(verbose=1)
    # callbacks.append(amcl_callback)
    
    # 课程学习回调 - AMCL已停用，暂时注释
    # curriculum_callback = CurriculumCallback(uncertainty_curriculum, verbose=1)
    # callbacks.append(curriculum_callback)
    
    # 进度记录回调
    progress_callback = ProgressLoggingCallback(verbose=1)
    callbacks.append(progress_callback)
    
    # MLflow记录回调
    mlflow_callback = MlflowCallback(verbose=1)
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
                "learning_rate": getattr(args, 'learning_rate', 3e-4),
                "buffer_size": getattr(args, 'buffer_size', 50000),
                "batch_size": getattr(args, 'batch_size', 256),
                "learning_starts": getattr(args, 'learning_starts', 1000),
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
        callbacks = create_callbacks(env, uncertainty_curriculum, cargo_type)
        
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
    parser.add_argument('--buffer_size', type=int, default=50000, help='经验回放缓冲区大小')
    parser.add_argument('--learning_starts', type=int, default=100, help='预热步数，开始学习前收集的样本数量')
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


if __name__ == "__main__":
    # 将项目路径添加到Python路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    main()