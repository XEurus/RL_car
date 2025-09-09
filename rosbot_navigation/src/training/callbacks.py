"""
训练回调函数模块 - 用于训练过程监控和数据收集
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy


class AMCLMetricsCallback(BaseCallback):
    """AMCL定位指标记录回调"""
    
    def __init__(self, verbose: int = 0):
        """
        初始化AMCL指标记录回调
        
        Args:
            verbose: 详细程度 (0: 无输出, 1: 每步输出, 2: 详细信息)
        """
        super().__init__(verbose)
        
        # 指标存储
        self.amcl_uncertainties = []
        self.convergence_scores = []
        self.position_errors = []
        self.episode_amcl_metrics = {}
        
        # 记录 interval
        self.record_interval = 100  # 每100步记录一次
        self.last_record_step = 0
        
        self.logger = logging.getLogger(__name__)
        
    def _on_step(self) -> bool:
        """每一步调用"""
        
        # 定期记录AMCL指标
        if self.num_timesteps - self.last_record_step >= self.record_interval:
            self._record_amcl_metrics()
            self.last_record_step = self.num_timesteps
            
        return True
        
    def _record_amcl_metrics(self):
        """记录AMCL指标"""
        
        try:
            # 获取训练环境
            if hasattr(self.training_env, 'unwrapped'):
                env = self.training_env.unwrapped
            else:
                return
                
            # 检查是否有AMCL定位器
            if not hasattr(env, 'get_amcl_uncertainty'):
                return
                
            # 获取AMCL不确定性
            uncertainty = env.get_amcl_uncertainty()
            self.amcl_uncertainties.append({
                'step': self.num_timesteps,
                'uncertainty': uncertainty
            })
            
            # 检查收敛性
            if hasattr(env, 'amcl_localizer'):
                convergence_rate = 1.0 if env.amcl_localizer.has_converged() else 0.0
                self.convergence_scores.append({
                    'step': self.num_timesteps,
                    'converged': convergence_rate
                })
                
            # 记录到logger
            if len(self.amcl_uncertainties) % 10 == 0:
                avg_uncertainty = np.mean([m['uncertainty'] for m in self.amcl_uncertainties[-10:]])
                convergence_rate = np.mean([m['converged'] for m in self.convergence_scores[-10:]]) if self.convergence_scores else 0.0
                
                self.logger.record("amcl/avg_position_uncertainty", avg_uncertainty)
                self.logger.record("amcl/convergence_rate", convergence_rate)
                
                if self.verbose >= 1:
                    self.logger.info(f"AMCL Metrics - Step: {self.num_timesteps}, "
                                   f"Uncertainty: {avg_uncertainty:.3f}, "
                                   f"Convergence Rate: {convergence_rate:.3f}")
                    
        except Exception as e:
            self.logger.error(f"记录AMCL指标失败: {e}")
            
    def _on_rollout_end(self) -> None:
        """剧集结束时调用"""
        
        # 记录剧集级别的AMCL统计
        if self.amcl_uncertainties:
            recent_uncertainties = [m['uncertainty'] for m in self.amcl_uncertainties[-10:]]
            avg_uncertainty = np.mean(recent_uncertainties)
            std_uncertainty = np.std(recent_uncertainties)
            
            # 记录到episode级别的统计
            self.episode_amcl_metrics[self.num_timesteps] = {
                'avg_uncertainty': avg_uncertainty,
                'std_uncertainty': std_uncertainty,
                'convergence_rate': np.mean([m['converged'] for m in self.convergence_scores[-10:]]) if self.convergence_scores else 0.0
            }
            
    def get_amcl_metrics(self) -> Dict[str, Any]:
        """获取收集的AMCL指标"""
        
        metrics = {
            'amcl_uncertainties': self.amcl_uncertainties,
            'convergence_scores': self.convergence_scores,
            'episode_metrics': self.episode_amcl_metrics,
            'summary': {
                'total_steps': self.num_timesteps,
                'avg_uncertainty': np.mean([m['uncertainty'] for m in self.amcl_uncertainties]) if self.amcl_uncertainties else 0.0,
                'avg_convergence_rate': np.mean([m['converged'] for m in self.convergence_scores]) if self.convergence_scores else 0.0
            }
        }
        
        return metrics
        

class CurriculumCallback(BaseCallback):
    """课程学习回调 - 根据训练进度调整不确定性级别"""
    
    def __init__(self, uncertainty_curriculum, verbose: int = 0):
        """
        初始化课程学习回调
        
        Args:
            uncertainty_curriculum: 不确定性课程学习管理器
            verbose: 详细程度
        """
        super().__init__(verbose)
        
        self.uncertainty_curriculum = uncertainty_curriculum
        self.current_uncertainty = 0.1
        self.uncertainty_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def _on_step(self) -> bool:
        """每一步调用"""
        
        # 检查是否需要调整不确定性级别
        if hasattr(self.locals, 'total_timesteps'):
            total_steps = self.locals['total_timesteps']
        else:
            total_steps = 500000  # 默认总步数
            
        current_step = self.num_timesteps
        
        # 获取当前不确定性级别
        new_uncertainty = self.uncertainty_curriculum.get_uncertainty_for_step(current_step)
        
        if new_uncertainty != self.current_uncertainty:
            self.current_uncertainty = new_uncertainty
            
            # 更新环境的不确定性级别
            if hasattr(self.training_env, 'unwrapped'):
                env = self.training_env.unwrapped
                if hasattr(env, 'amcl_localizer'):
                    env.amcl_localizer.set_uncertainty_level(new_uncertainty)
                    
            # 记录调整历史
            self.uncertainty_history.append({
                'step': current_step,
                'uncertainty': new_uncertainty,
                'progress': current_step / total_steps
            })
            
            # 记录到MLflow
            self.logger.record("curriculum/uncertainty_level", new_uncertainty)
            self.logger.record("curriculum/training_progress", current_step / total_steps)
            
            if self.verbose >= 1:
                self.logger.info(f"调整到不确定性级别: {new_uncertainty} (进度: {current_step/total_steps:.1%})")
                
        return True
        

class ProgressLoggingCallback(BaseCallback):
    """训练进度记录和可视化回调"""
    
    def __init__(self, verbose: int = 0, log_interval: int = 1000):
        """
        初始化进度记录回调
        
        Args:
            verbose: 详细程度
            log_interval: 记录间隔（步数）
        """
        super().__init__(verbose)
        
        self.log_interval = log_interval
        self.last_log_time = time.time()
        self.start_time = time.time()
        
        # 进度存储
        self.progress_history = []
        self.episode_rewards = []
        self.episode_lengths = []
        
        self.logger = logging.getLogger(__name__)
        
    def _on_step(self) -> bool:
        """每一步调用"""
        
        # 定期记录进度
        if self.num_timesteps % self.log_interval == 0:
            self._log_progress()
            
        return True
        
    def _log_progress(self):
        """记录训练进度"""
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 计算进度信息
        progress_info = {
            'timestamp': datetime.now().isoformat(),
            'timestep': self.num_timesteps,
            'episode': hasattr(self, '_episode_num') and getattr(self, '_episode_num', 0),
            'fps': self.log_interval / (current_time - self.last_log_time) if current_time > self.last_log_time else 0,
            'elapsed_time': elapsed_time,
            'estimated_time_remaining': self._estimate_time_remaining()
        }
        
        self.progress_history.append(progress_info)
        
        # 记录到训练日志
        self.logger.info(f"Progress: Step {self.num_timesteps}, "
                        f"Episodes: {progress_info['episode']}, "
                        f"FPS: {progress_info['fps']:.1f}, "
                        f"Elapsed: {self._format_time(elapsed_time)}")
                        
        # 更新最后记录时间
        self.last_log_time = current_time
        
    def _estimate_time_remaining(self) -> float:
        """估计剩余训练时间"""
        
        try:
            if hasattr(self.locals, 'total_timesteps'):
                total_steps = self.locals['total_timesteps']
            else:
                # 从常见配置中推断
                if hasattr(self, 'training_env') and hasattr(self.training_env, 'unwrapped'):
                    # 尝试从环境或模型配置中获取
                    total_steps = 500000  # 默认
                else:
                    return 0.0
                    
            remaining_steps = max(0, total_steps - self.num_timesteps)
            
            # 基于过去进度计算平均速度
            if len(self.progress_history) >= 2:
                recent_progress = self.progress_history[-5:]  # 最近5次记录
                avg_speed = np.mean([p['fps'] for p in recent_progress if p['fps'] > 0])
            else:
                avg_speed = 100  # 默认速度
                
            if avg_speed > 0:
                return remaining_steps / avg_speed
            else:
                return 0.0
                
        except Exception:
            return 0.0
            
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            return f"{seconds/3600:.1f}h"
            

class ModelCheckpointCallback(BaseCallback):
    """模型保存检查点回调"""
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, 
                 save_env_stats: bool = True, verbose: int = 0):
        """
        初始化保存检查点回调
        
        Args:
            save_freq: 保存频率（步数）
            save_path: 保存路径
            name_prefix: 文件名前缀
            save_env_stats: 是否保存环境统计信息
            verbose: 详细程度
        """
        super().__init__(verbose)
        
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_env_stats = save_env_stats
        self.name_prefix = name_prefix
        
        # 创建保存目录
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.last_save_step = 0
        self.checkpoint_count = 0
        
        self.logger = logging.getLogger(__name__)
        
    def _on_step(self) -> bool:
        """每一步调用"""
        
        # 定期保存检查点
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self._save_checkpoint()
            self.last_save_step = self.num_timesteps
            
        return True
        
    def _save_checkpoint(self):
        """保存检查点"""
        
        try:
            # 生成文件名
            filename = f"{self.name_prefix}_{self.checkpoint_count:04d}_step{self.num_timesteps}"
            model_path = self.save_path / f"{filename}.zip"
            
            # 保存模型
            self.model.save(str(model_path))
            
            # 保存环境统计信息（如果启用）
            if self.save_env_stats and hasattr(self.training_env, 'unwrapped'):
                env_stats = self._collect_env_stats()
                stats_path = self.save_path / f"{filename}_env_stats.json"
                with open(stats_path, 'w') as f:
                    json.dump(env_stats, f, indent=2, default=str)
                    
            self.checkpoint_count += 1
            
            if self.verbose >= 1:
                self.logger.info(f"保存检查点: {model_path}")
                
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            
    def _collect_env_stats(self) -> Dict[str, Any]:
        """收集环境统计信息"""
        
        env_stats = {
            'checkpoint_name': f"{self.name_prefix}_{self.checkpoint_count:04d}_step{self.num_timesteps}",
            'timestep': self.num_timesteps,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            env = self.training_env.unwrapped
            
            # 收集环境特定信息
            if hasattr(env, 'get_current_pose'):
                pose = env.get_current_pose()
                env_stats['current_pose'] = {
                    'position': pose.get('position', []),
                    'orientation': pose.get('orientation', [])
                }
                
            if hasattr(env, 'get_amcl_uncertainty'):
                env_stats['amcl_uncertainty'] = env.get_amcl_uncertainty()
                
            # 记录剧集统计
            if hasattr(self, 'locals') and 'infos' in self.locals:
                episode_stats = []
                for info in self.locals['infos']:
                    if isinstance(info, dict) and 'episode' in info:
                        episode_stats.append(info['episode'])
                        
                if episode_stats:
                    env_stats['recent_episodes'] = {
                        'count': len(episode_stats),
                        'avg_reward': np.mean([ep.get('r', 0) for ep in episode_stats]),
                        'avg_length': np.mean([ep.get('l', 0) for ep in episode_stats])
                    }
                    
        except Exception as e:
            self.logger.warning(f"收集环境统计信息失败: {e}")
            
        return env_stats
        

class EvaluationCallback(BaseCallback):
    """训练期间定期评估回调"""
    
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 10,
                 deterministic: bool = True, verbose: int = 0):
        """
        初始化评估回调
        
        Args:
            eval_env: 评估环境
            eval_freq: 评估频率
            n_eval_episodes: 每次评估的剧集数量
            deterministic: 评估时是否使用确定性策略
            verbose: 详细程度
        """
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        
        self.evaluation_results = []
        self.last_eval_step = 0
        
        self.logger = logging.getLogger(__name__)
        
    def _on_step(self) -> bool:
        """每一步调用"""
        
        # 定期进行评估
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate_model()
            self.last_eval_step = self.num_timesteps
            
        return True
        
    def _evaluate_model(self):
        """评估当前模型"""
        
        try:
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            collision_count = 0
            
            for episode in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                episode_reward = 0
                episode_length = 0
                
                while episode_length < self.eval_env.max_steps_per_episode:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, terminated, truncated, step_info = self.eval_env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # 检查成功和碰撞
                    if step_info.get('success', False):
                        success_count += 1
                    if step_info.get('collision', False):
                        collision_count += 1
                        
                    if terminated or truncated:
                        break
                        
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
            # 计算统计结果
            eval_result = {
                'timestep': self.num_timesteps,
                'timestamp': datetime.now().isoformat(),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'avg_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'avg_length': np.mean(episode_lengths),
                'success_rate': success_count / self.n_eval_episodes,
                'collision_rate': collision_count / self.n_eval_episodes
            }
            
            self.evaluation_results.append(eval_result)
            
            # 记录到训练日志
            self.logger.record("eval/mean_reward", eval_result['avg_reward'])
            self.logger.record("eval/std_reward", eval_result['std_reward'])
            self.logger.record("eval/mean_length", eval_result['avg_length'])
            self.logger.record("eval/success_rate", eval_result['success_rate'])
            self.logger.record("eval/collision_rate", eval_result['collision_rate'])
            
            if self.verbose >= 1:
                self.logger.info(f"模型评估 (Step {self.num_timesteps}): "
                               f"Avg Reward: {eval_result['avg_reward']:.2f}, "
                               f"Success Rate: {eval_result['success_rate']:.2%}, "
                               f"Collision Rate: {eval_result['collision_rate']:.1%}")
                               
        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            
    def get_evaluation_results(self) -> List[Dict[str, Any]]:
        """获取所有评估结果"""
        return self.evaluation_results.copy()
        
    def get_latest_evaluation(self) -> Optional[Dict[str, Any]]:
        """获取最新的评估结果"""
        return self.evaluation_results[-1] if self.evaluation_results else None