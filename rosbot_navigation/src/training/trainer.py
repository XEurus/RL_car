"""
多阶段训练器 - 支持TD3算法的三阶段训练流程
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from ..environments.navigation_env import ROSbotNavigationEnv
from .config_manager import TrainingConfigManager


@dataclass
class TrainingStageConfig:
    """训练阶段配置"""
    name: str
    total_timesteps: int
    learning_rate: float = 3e-4
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    use_curriculum: bool = True
    uncertainty_level: float = 0.1
    checkpoint_interval: int = 10000
    eval_interval: int = 5000
    eval_episodes: int = 10


@dataclass
class CargoTrainingConfig:
    """货物类型训练配置"""
    cargo_type: str
    stages: List[TrainingStageConfig]
    env_config: Dict = field(default_factory=dict)
    model_hyperparams: Dict = field(default_factory=dict)
    parallel_environments: int = 1
    use_gpu: bool = True
    random_seed: int = 42


class BaseTrainer(ABC):
    """基础训练器抽象类"""
    
    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """执行训练"""
        pass
        
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """评估模型"""
        pass


class MultiStageTrainer(BaseTrainer):
    """多阶段训练器 - 实现三阶段训练流程"""
    
    def __init__(self, config_manager: TrainingConfigManager):
        """
        初始化多阶段训练器
        
        Args:
            config_manager: 配置管理器
        """
        self.config_mgr = config_manager
        self.logger = logging.getLogger(__name__)
        self.training_history = {}
        self.current_stage = None
        
        # 训练状态跟踪
        self._setup_training_state_tracking()
        
    def _setup_training_state_tracking(self):
        """设置训练状态跟踪"""
        self.training_state = {
            'current_cargo_type': None,
            'current_stage': None,
            'total_timesteps': 0,
            'episode_count': 0,
            'best_reward': -np.inf,
            'training_start_time': None,
            'stage_start_times': {}
        }
        
    def create_stage_config(self, cargo_type: str, stage_name: str) -> TrainingStageConfig:
        """创建阶段配置"""
        
        stage_configs = {
            'stage1': TrainingStageConfig(
                name='基础能力阶段',
                total_timesteps=100000,
                learning_rate=3e-4,
                exploration_noise=0.1,
                policy_noise=0.2,
                uncertainty_level=0.1,
                use_curriculum=True
            ),
            'stage2': TrainingStageConfig(
                name='能力提升阶段', 
                total_timesteps=150000,
                learning_rate=2e-4,
                exploration_noise=0.15,
                policy_noise=0.25,
                uncertainty_level=0.3,
                use_curriculum=True
            ),
            'stage3': TrainingStageConfig(
                name='鲁棒性验证阶段',
                total_timesteps=200000,
                learning_rate=1e-4,
                exploration_noise=0.2,
                policy_noise=0.3,
                uncertainty_level=0.6,
                use_curriculum=True
            )
        }
        
        return stage_configs.get(stage_name, stage_configs['stage1'])
        
    def create_training_environment(self, cargo_type: str, num_envs: int = 1) -> ROSbotNavigationEnv:
        """创建训练环境"""
        
        def make_env():
            def _init():
                env = ROSbotNavigationEnv(cargo_type=cargo_type)
                return Monitor(env)
            return _init
            
        if num_envs == 1:
            return Monitor(ROSbotNavigationEnv(cargo_type=cargo_type))
        else:
            # 多环境并行训练
            return SubprocVecEnv([make_env() for _ in range(num_envs)])
            
    def create_model(self, env, stage_config: TrainingStageConfig, 
                     previous_model_path: str = None) -> ImprovedTD3:
        """创建训练模型"""
        
        model_config = self.config_mgr.get_model_config()
        
        model_params = {
            'policy': 'MlpPolicy',
            'env': env,
            'learning_rate': stage_config.learning_rate,
            'buffer_size': model_config.get('buffer_size', 500000),
            'learning_starts': model_config.get('learning_starts', 5000),
            'batch_size': model_config.get('batch_size', 256),
            'gamma': model_config.get('gamma', 0.99),
            'tau': model_config.get('tau', 0.005),
            'gradient_steps': model_config.get('gradient_steps', 1),
            'train_freq': model_config.get('train_freq', 1),
            'policy_delay': stage_config.policy_delay,
            'target_policy_noise': stage_config.policy_noise,
            'target_noise_clip': stage_config.noise_clip,
            'verbose': 1,
            'tensorboard_log': f'./logs/multi_stage/{stage_config.name}'
        }
        
        # 特殊货物类型参数调整
        if hasattr(env, 'cargo_type'):
            if env.cargo_type == 'fragile':
                model_params['buffer_size'] = 750000  # 更大数据缓冲区
                model_params['learning_starts'] = 10000  # 更长的预热期
                
            elif env.cargo_type == 'dangerous':
                model_params['learning_rate'] = stage_config.learning_rate * 0.8
                model_params['policy_noise'] = stage_config.policy_noise * 0.8
                
        # 创建模型
        model = ImprovedTD3(**model_params)
        
        # 从之前阶段恢复
        if previous_model_path and os.path.exists(previous_model_path):
            self.logger.info(f"从 {previous_model_path} 加载模型")
            model.load(previous_model_path)
            
        return model
        
    def train_stage(self, cargo_type: str, stage_name: str, 
                    callbacks: List[BaseCallback] = None,
                    resume_from: str = None) -> Dict[str, Any]:
        """
        训练单个阶段
        
        Args:
            cargo_type: 货物类型
            stage_name: 阶段名称
            callbacks: 自定义回调函数列表
            resume_from: 恢复训练的检查点路径
            
        Returns:
            训练结果字典
        """
        
        self.logger.info(f"开始训练 {cargo_type} - {stage_name}")
        
        # 获取阶段配置
        stage_config = self.create_stage_config(cargo_type, stage_name)
        
        # 更新训练状态
        self.training_state['current_cargo_type'] = cargo_type
        self.training_state['current_stage'] = stage_name
        self.training_state['stage_start_times'][stage_name] = datetime.now()
        
        # 创建环境
        env = self.create_training_environment(cargo_type)
        
        # 设置AMCL不确定性级别
        if stage_config.use_curriculum and hasattr(env, 'amcl_localizer'):
            env.amcl_localizer.set_uncertainty_level(stage_config.uncertainty_level)
            
        # 创建模型
        model = self.create_model(env, stage_config, resume_from)
        
        # 创建训练回调
        if callbacks is None:
            callbacks = self._create_default_callbacks(cargo_type, stage_name, stage_config)
            
        try:
            # 开始训练
            model.learn(
                total_timesteps=stage_config.total_timesteps,
                callback=callbacks,
                log_interval=10,
                progress_bar=True
            )
            
            # 训练完成后的处理
            result = self._process_training_results(model, env, cargo_type, stage_name, stage_config)
            
            self.logger.info(f"{cargo_type} - {stage_name} 训练完成！")
            
            return result
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            raise TrainingError(f"阶段训练失败: {e}")
            
    def train_pipeline(self, cargo_type: str, stages: List[str] = None) -> Dict[str, Any]:
        """
        训练完整的货物类型管道
        
        Args:
            cargo_type: 货物类型
            stages: 训练阶段列表
            
        Returns:
            训练结果字典
        """
        
        if stages is None:
            stages = ['stage1', 'stage2', 'stage3']
            
        self.logger.info(f"开始 {cargo_type} 完整训练流程")
        self.training_state['training_start_time'] = datetime.now()
        
        pipeline_results = {
            'cargo_type': cargo_type,
            'start_time': datetime.now(),
            'stages': {}
        }
        
        previous_model_path = None
        
        for stage in stages:
            try:
                # 训练当前阶段
                stage_result = self.train_stage(cargo_type, stage, resume_from=previous_model_path)
                
                # 更新结果
                pipeline_results['stages'][stage] = {
                    'status': 'completed',
                    'result': stage_result,
                    'model_path': stage_result['model_path']
                }
                
                # 准备下一阶段
                previous_model_path = stage_result['model_path']
                
                # 清理资源
                if hasattr(stage_result, 'env'):
                    stage_result['env'].close()
                    
            except Exception as e:
                self.logger.error(f"{stage} 阶段训练失败: {e}")
                
                pipeline_results['stages'][stage] = {
                    'status': 'failed', 
                    'error': str(e)
                }
                
                # 中断整个流程
                pipeline_results['status'] = 'failed'
                return pipeline_results
                
        # 计算总体结果
        pipeline_results['total_timesteps'] = sum(
            result['result']['total_timesteps'] 
            for result in pipeline_results['stages'].values() 
            if result['status'] == 'completed'
        )
        
        pipeline_results['end_time'] = datetime.now()
        pipeline_results['duration'] = pipeline_results['end_time'] - pipeline_results['start_time']
        pipeline_results['status'] = 'completed'
        
        self.logger.info(f"{cargo_type} 完整训练流程完成")
        
        return pipeline_results
        
    def evaluate_model(self, model_path: str, cargo_type: str, 
                       num_episodes: int = 50) -> Dict[str, float]:
        """
        评估训练好的模型
        
        Args:
            model_path: 模型文件路径
            cargo_type: 货物类型
            num_episodes: 评估剧集数量
            
        Returns:
            评估结果字典
        """
        
        self.logger.info(f"评估模型: {model_path}")
        
        # 创建评估环境
        env = self.create_training_environment(cargo_type)
        
        # 加载模型
        model = ImprovedTD3.load(model_path, env=env)
        
        # 评估指标
        total_rewards = []
        episode_lengths = []
        successes = 0
        collision_count = 0
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while episode_length < env.max_steps_per_episode:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
                    
                # 记录成功和碰撞
                if step_info.get('success', False):
                    successes += 1
                if step_info.get('collision', False):
                    collision_count += 1
                    
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        env.close()
        
        # 计算统计结果
        results = {
            'success_rate': successes / num_episodes,
            'collision_rate': collision_count / num_episodes,
            'average_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'average_length': np.mean(episode_lengths),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards)
        }
        
        return results
        
    def _create_default_callbacks(self, cargo_type: str, stage_name: str, 
                                  stage_config: TrainingStageConfig) -> List[BaseCallback]:
        """创建默认训练回调"""
        
        from .callbacks import AMCLMetricsCallback, CurriculumCallback, ProgressLoggingCallback
        
        callbacks = []
        
        # AMCL指标监控
        amcl_callback = AMCLMetricsCallback(verbose=0)
        callbacks.append(amcl_callback)
        
        # 课程学习 (如果启用)
        if stage_config.use_curriculum:
            from ..localization.amcl_localizer import UncertaintyCurriculumTraining
            curriculum = UncertaintyCurriculumTraining()
            curriculum_callback = CurriculumCallback(curriculum, verbose=0)
            callbacks.append(curriculum_callback)
            
        # 进度记录
        progress_callback = ProgressLoggingCallback(verbose=0)
        callbacks.append(progress_callback)
        
        return callbacks
        
    def _process_training_results(self, model, env, cargo_type: str, 
                                  stage_name: str, stage_config: TrainingStageConfig) -> Dict[str, Any]:
        """处理训练结果"""
        
        # 获取训练统计
        result = {
            'cargo_type': cargo_type,
            'stage': stage_name,
            'total_timesteps': model.num_timesteps,
            'stage_config': stage_config.__dict__,
            'training_duration': datetime.now() - self.training_state['stage_start_times'][stage_name],
            'model_parameters': sum(p.numel() for p in model.policy.parameters()),
        }
        
        # 模型保存
        model_path = f'./models/multi_stage/{cargo_type}_{stage_name}.zip'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        result['model_path'] = model_path
        
        # 基本评估
        eval_results = self._quick_evaluation(model, env, num_episodes=10)
        result.update(eval_results)
        
        return result
        
    def _quick_evaluation(self, model, env, num_episodes: int = 10) -> Dict[str, float]:
        """快速评估"""
        
        rewards = []
        success_rate = 0
        
        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            success = False
            
            for _ in range(env.max_steps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                
                episode_reward += reward
                success = success or step_info.get('success', False)
                
                if terminated or truncated:
                    break
                    
            rewards.append(episode_reward)
            success_rate += float(success)
            
        return {
            'eval_reward_mean': np.mean(rewards),
            'eval_reward_std': np.std(rewards),
            'eval_success_rate': success_rate / num_episodes
        }


class StageSpecificTrainer(MultiStageTrainer):
    """针对特定阶段的训练器 - 扩展基础功能"""
    
    def __init__(self, config_manager: TrainingConfigManager, stage_name: str):
        super().__init__(config_manager)
        self.stage_name = stage_name
        
    def train(self, cargo_type: str, **kwargs) -> Dict[str, Any]:
        """重写训练方法，专注于特定阶段"""
        return self.train_stage(cargo_type, self.stage_name, **kwargs)


class TrainingError(Exception):
    """训练异常类"""
    pass