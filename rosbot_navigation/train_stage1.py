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


def setup_for_cargo_type(cargo_type: str, total_steps: int, device: str = 'auto'):
    """为特定货物类型设置训练"""
    
    print(f"\n=== 开始训练 {cargo_type} 货物导航模型 ===")
    print(f"总训练步数: {total_steps}")
    print(f"状态空间: 42维 (LiDAR+AMCL定位)")
    print(f"动作空间: 2维连续 [线速度, 角速度]")
    print(f"使用AMCL定位: 800粒子数")
    
    # 创建环境
    env = ROSbotNavigationEnv(cargo_type=cargo_type)
    env = Monitor(env)  # 包装成监视环境
    
    # 课程学习配置
    from src.localization.amcl_localizer import UncertaintyCurriculumTraining
    uncertainty_curriculum = UncertaintyCurriculumTraining()
    
    # 创建改进的TD3模型
    model = ImprovedTD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=500000,
        learning_starts=10000,
        batch_size=2560,
        gamma=0.99,
        tau=0.005,
        gradient_steps=1,
        train_freq=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
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


def train_single_cargo_model(cargo_type: str, total_steps: int, model_save_path: str, device: str = 'auto'):
    """训练单个货物类型模型"""
    
    # 创建MLflow实验
    experiment_name = f"rosbot_navigation_{cargo_type}_stage1"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"training_{cargo_type}_{total_steps}"):
        
        # 记录实验参数
        mlflow.log_params({
            "algorithm": "TD3",
            "cargo_type": cargo_type,
            "total_steps": total_steps,
            "state_dim": 42,
            "action_dim": 2,
            "amcl_particles": 800,
            "learning_rate": 3e-4,
            "buffer_size": 500000,
            "batch_size": 256,
            "policy_delay": 2,
            "gamma": 0.99,
            "tau": 0.005,
            "device": device
        })
        
        # 设置训练环境
        model, env, uncertainty_curriculum = setup_for_cargo_type(cargo_type, total_steps, device=device)
        
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
    parser.add_argument('--total_steps', type=int, default=500000,
                       help='总训练步数')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型保存路径')
    parser.add_argument('--debug', action='store_true', 
                       help='调试模式')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='计算设备 (e.g., "cpu", "cuda", "auto")')
    
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
    print(f"{'='*60}\n")
    
    try:
        model, env = train_single_cargo_model(
            cargo_type=args.cargo_type,
            total_steps=args.total_steps,
            model_save_path=model_path,
            device=args.device
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