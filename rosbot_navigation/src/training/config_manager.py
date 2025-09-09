"""
训练配置管理器 - 统一管理三阶段训练的不同配置
支持三种货物类型的差异化配置
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class CargoConfig:
    """货物类型配置"""
    type_name: str
    max_linear_velocity: float
    max_angular_velocity: float
    stability_penalty: float
    safety_distance: float = 0.3
    speed_constraint_level: int = 0  # 0: 普通, 1: 易碎品, 2: 危险品
    max_acceleration: float = 1.0
    min_linear_velocity: float = 0.1
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'distance': 10.0,
        'heading': -2.0,
        'stability': -5.0,
        'safety': -10.0,
        'time': -0.1
    })


@dataclass
class StageConfig:
    """训练阶段配置"""
    stage_name: str
    total_timesteps: int
    learning_rate: float
    exploration_noise: float
    policy_noise: float
    target_noise_clip: float
    policy_delay: int
    buffer_size: int
    batch_size: int
    uncertainty_level: float
    use_curriculum: bool
    checkpoint_interval: int
    description: str = ""


@dataclass
class LocalizationConfig:
    """定位配置"""
    amcl_particles: int = 800
    initial_std: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.15])
    motion_noise_alpha: List[float] = field(default_factory=lambda: [0.1, 0.01, 0.1, 0.01])
    measurement_noise_sigma: float = 0.05
    adaptive_resampling: bool = True
    resample_threshold: float = 0.5


class TrainingConfigManager:
    """训练配置管理器"""
    
    # 默认配置
    DEFAULT_CONFIGS = {
        'normal': CargoConfig(
            type_name='normal',
            max_linear_velocity=2.0,
            max_angular_velocity=2.0,
            stability_penalty=0.0,
            safety_distance=0.3,
            max_acceleration=1.0,
            reward_weights={
                'distance': 10.0,
                'heading': -2.0, 
                'stability': 0.0,
                'safety': -5.0,
                'time': -0.1
            }
        ),
        
        'fragile': CargoConfig(
            type_name='fragile',
            max_linear_velocity=1.0,
            max_angular_velocity=1.5,
            stability_penalty=-5.0,
            safety_distance=0.4,
            max_acceleration=0.5,
            speed_constraint_level=1,
            reward_weights={
                'distance': 8.0,
                'heading': -3.0,
                'stability': -10.0,  # 强调稳定性
                'safety': -8.0,
                'time': -0.15
            }
        ),
        
        'dangerous': CargoConfig(
            type_name='dangerous',
            max_linear_velocity=0.8,
            max_angular_velocity=1.0,
            stability_penalty=-2.0,
            safety_distance=0.6,
            max_acceleration=0.3,
            speed_constraint_level=2,
            reward_weights={
                'distance': 5.0,
                'heading': -4.0,
                'stability': -5.0,
                'safety': -20.0,  # 强调安全性
                'time': -0.2
            }
        )
    }
    
    # 三阶段训练配置
    STAGE_CONFIGS = {
        'stage1': StageConfig(
            stage_name='基础能力阶段',
            total_timesteps=100000,
            learning_rate=3e-4,
            exploration_noise=0.1,
            policy_noise=0.2,
            target_noise_clip=0.5,
            policy_delay=2,
            buffer_size=500000,
            batch_size=256,
            uncertainty_level=0.1,
            use_curriculum=True,
            checkpoint_interval=10000,
            description='低不确定性基础导航能力建立'
        ),
        
        'stage2': StageConfig(
            stage_name='能力提升阶段',
            total_timesteps=150000,
            learning_rate=2e-4,
            exploration_noise=0.15,
            policy_noise=0.25,
            target_noise_clip=0.5,
            policy_delay=2,
            buffer_size=750000,
            batch_size=256,
            uncertainty_level=0.3,
            use_curriculum=True,
            checkpoint_interval=15000,
            description='中等不确定性下的鲁棒性训练'
        ),
        
        'stage3': StageConfig(
            stage_name='鲁棒性验证阶段',
            total_timesteps=200000,
            learning_rate=1e-4,
            exploration_noise=0.2,
            policy_noise=0.3,
            target_noise_clip=0.5,
            policy_delay=2,
            buffer_size=1000000,
            batch_size=512,
            uncertainty_level=0.6,
            use_curriculum=True,
            checkpoint_interval=20000,
            description='高不确定性复杂场景验证'
        )
    }
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果不提供则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else None
        self.configs = {}
        
        # 加载配置
        self._load_configurations()
        
        # 验证配置完整性
        self._validate_configurations()
        
    def _load_configurations(self):
        """加载配置"""
        
        if self.config_path and self.config_path.exists():
            # 从配置文件加载
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_configs = yaml.safe_load(f)
                    
                # 解析货物类型配置
                if 'cargo_types' in loaded_configs:
                    for cargo_type, config_data in loaded_configs['cargo_types'].items():
                        self.configs[cargo_type] = self._parse_cargo_config(cargo_type, config_data)
                        
                # 解析阶段配置
                if 'training_stages' in loaded_configs:
                    for stage_name, stage_data in loaded_configs['training_stages'].items():
                        if stage_name not in self.STAGE_CONFIGS:
                            self.STAGE_CONFIGS[stage_name] = self._parse_stage_config(stage_name, stage_data)
                            
                # 解析AMCL配置
                if 'localization' in loaded_configs:
                    self.localization_config = self._parse_localization_config(
                        loaded_configs['localization']
                    )
                    
            except Exception as e:
                self.logger.warning(f"配置文件加载失败 {self.config_path}: {e}")
                self.logger.info("使用默认配置")
                
        # 如果配置文件加载失败或不存在，使用默认配置
        if not self.configs:
            self.configs = self.DEFAULT_CONFIGS.copy()
            self.localization_config = LocalizationConfig()
            
    def _parse_cargo_config(self, cargo_type: str, config_data: Dict) -> CargoConfig:
        """解析货物类型配置"""
        
        return CargoConfig(
            type_name=config_data.get('type_name', cargo_type),
            max_linear_velocity=config_data.get('max_linear_velocity', 1.0),
            max_angular_velocity=config_data.get('max_angular_velocity', 1.5),
            stability_penalty=config_data.get('stability_penalty', 0.0),
            safety_distance=config_data.get('safety_distance', 0.3),
            speed_constraint_level=config_data.get('speed_constraint_level', 0),
            max_acceleration=config_data.get('max_acceleration', 1.0),
            min_linear_velocity=config_data.get('min_linear_velocity', 0.1),
            reward_weights=config_data.get('reward_weights', {})
        )
        
    def _parse_stage_config(self, stage_name: str, config_data: Dict) -> StageConfig:
        """解析阶段配置"""
        
        return StageConfig(
            stage_name=stage_name,
            total_timesteps=config_data.get('total_timesteps', 100000),
            learning_rate=config_data.get('learning_rate', 3e-4),
            exploration_noise=config_data.get('exploration_noise', 0.1),
            policy_noise=config_data.get('policy_noise', 0.2),
            target_noise_clip=config_data.get('target_noise_clip', 0.5),
            policy_delay=config_data.get('policy_delay', 2),
            buffer_size=config_data.get('buffer_size', 500000),
            batch_size=config_data.get('batch_size', 256),
            uncertainty_level=config_data.get('uncertainty_level', 0.1),
            use_curriculum=config_data.get('use_curriculum', True),
            checkpoint_interval=config_data.get('checkpoint_interval', 10000),
            description=config_data.get('description', '')
        )
        
    def _parse_localization_config(self, config_data: Dict) -> LocalizationConfig:
        """解析定位配置"""
        
        return LocalizationConfig(
            amcl_particles=config_data.get('amcl', {}).get('num_particles', 800),
            initial_std=config_data.get('amcl', {}).get('initial_std', [0.2, 0.2, 0.15]),
            motion_noise_alpha=config_data.get('amcl', {}).get('motion_noise', {}).get('alpha', [0.1, 0.01, 0.1, 0.01]),
            measurement_noise_sigma=config_data.get('amcl', {}).get('measurement_noise', {}).get('sigma_hit', 0.05),
            adaptive_resampling=config_data.get('amcl', {}).get('adaptive_resampling', {}).get('enabled', True),
            resample_threshold=config_data.get('amcl', {}).get('adaptive_resampling', {}).get('threshold', 0.5)
        )
        
    def _validate_configurations(self):
        """验证配置完整性"""
        
        required_cargo_types = ['normal', 'fragile', 'dangerous']
        required_stages = ['stage1', 'stage2', 'stage3']
        
        # 验证货物类型配置
        for cargo_type in required_cargo_types:
            if cargo_type not in self.configs:
                self.logger.warning(f"缺少 {cargo_type} 配置，使用默认配置")
                self.configs[cargo_type] = self.DEFAULT_CONFIGS[cargo_type]
                
        # 验证阶段配置
        for stage_name in required_stages:
            if stage_name not in self.STAGE_CONFIGS:
                self.logger.warning(f"缺少 {stage_name} 配置，使用默认配置")
                default_stages = {
                    'stage1': self.STAGE_CONFIGS['stage1'],
                    'stage2': self.STAGE_CONFIGS['stage2'], 
                    'stage3': self.STAGE_CONFIGS['stage3']
                }
                self.STAGE_CONFIGS.update(default_stages)
                
        self.logger.info("配置验证完成")
        
    def get_cargo_config(self, cargo_type: str) -> CargoConfig:
        """获取货物类型配置"""
        
        if cargo_type not in self.configs:
            self.logger.warning(f"未知货物类型 {cargo_type}，使用默认配置")
            return self.configs['normal']
            
        return self.configs[cargo_type]
        
    def get_stage_config(self, stage_name: str, cargo_type: str = None) -> Dict[str, Any]:
        """获取阶段配置"""
        
        if stage_name not in self.STAGE_CONFIGS:
            self.logger.warning(f"未知训练阶段 {stage_name}")
            return {}
            
        stage_config = self.STAGE_CONFIGS[stage_name]
        
        # 如果指定了货物类型，合并货物类型特定配置
        if cargo_type:
            cargo_config = self.get_cargo_config(cargo_type)
            
            # 根据货物类型调整参数
            adjusted_config = stage_config.__dict__.copy()
            
            # 根据货物类型调整学习率和探索参数
            if cargo_type == 'fragile':
                adjusted_config['learning_rate'] *= 0.8
                adjusted_config['exploration_noise'] *= 0.7
            elif cargo_type == 'dangerous':
                adjusted_config['learning_rate'] *= 0.6
                adjusted_config['exploration_noise'] *= 0.5
                
            adjusted_config['cargo_type'] = cargo_type
            adjusted_config['stage'] = stage_name
            
            return adjusted_config
            
        return stage_config.__dict__
        
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        
        # 从stage1获取基本模型配置
        stage_config = self.get_stage_config('stage1')
        
        return {
            'learning_rate': stage_config.get('learning_rate', 3e-4),
            'buffer_size': stage_config.get('buffer_size', 500000),
            'learning_starts': stage_config.get('learning_starts', 5000),
            'batch_size': stage_config.get('batch_size', 256),
            'gamma': stage_config.get('gamma', 0.99),
            'tau': stage_config.get('tau', 0.005),
            'policy_delay': stage_config.get('policy_delay', 2),
            'gradient_steps': stage_config.get('gradient_steps', 1),
            'train_freq': stage_config.get('train_freq', 1),
            'target_policy_noise': stage_config.get('policy_noise', 0.2),
            'target_noise_clip': stage_config.get('target_noise_clip', 0.5)
        }
        
    def get_localization_config(self) -> LocalizationConfig:
        """获取定位配置"""
        return self.localization_config
        
    def get_environment_config(self, cargo_type: str) -> Dict[str, Any]:
        """获取环境配置"""
        
        cargo_config = self.get_cargo_config(cargo_type)
        
        return {
            'max_linear_velocity': cargo_config.max_linear_velocity,
            'max_angular_velocity': cargo_config.max_angular_velocity,
            'safety_distance': cargo_config.safety_distance,
            'reward_weights': cargo_config.reward_weights,
            'speed_constraint_level': cargo_config.speed_constraint_level,
            'amcl_config': self.localization_config.__dict__
        }
        
    def save_config(self, save_path: str):
        """保存当前配置到文件"""
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'cargo_types': {},
            'training_stages': {},
            'localization': self.localization_config.__dict__,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
        
        # 序列化货物类型配置
        for cargo_type, config in self.configs.items():
            config_data['cargo_types'][cargo_type] = {
                'type_name': config.type_name,
                'max_linear_velocity': config.max_linear_velocity,
                'max_angular_velocity': config.max_angular_velocity,
                'stability_penalty': config.stability_penalty,
                'safety_distance': config.safety_distance,
                'speed_constraint_level': config.speed_constraint_level,
                'max_acceleration': config.max_acceleration,
                'min_linear_velocity': config.min_linear_velocity,
                'reward_weights': config.reward_weights
            }
            
        # 序列化阶段配置
        for stage_name, config in self.STAGE_CONFIGS.items():
            config_data['training_stages'][stage_name] = {
                'stage_name': config.stage_name,
                'total_timesteps': config.total_timesteps,
                'learning_rate': config.learning_rate,
                'exploration_noise': config.exploration_noise,
                'policy_noise': config.policy_noise,
                'target_noise_clip': config.target_noise_clip,
                'policy_delay': config.policy_delay,
                'buffer_size': config.buffer_size,
                'batch_size': config.batch_size,
                'uncertainty_level': config.uncertainty_level,
                'use_curriculum': config.use_curriculum,
                'checkpoint_interval': config.checkpoint_interval,
                'description': config.description
            }
            
        # 保存配置文件
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
        self.logger.info(f"配置保存到: {save_path}")
        
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        
        return {
            'cargo_configs': {k: v.__dict__ for k, v in self.configs.items()},
            'stage_configs': {k: v.__dict__ for k, v in self.STAGE_CONFIGS.items()},
            'localization_config': self.localization_config.__dict__,
            'path': str(self.config_path) if self.config_path else 'default'
        }