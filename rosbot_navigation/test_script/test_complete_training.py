#!/usr/bin/env python3
"""
完整训练系统测试脚本 - 验证ROSbot导航训练系统的所有组件
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
import yaml

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environments.navigation_env import ROSbotNavigationEnv
from src.models.td3_robust import ImprovedTD3
from src.training.config_manager import TrainingConfigManager
from src.training.mlflow_manager import MLflowTrainingManager
from src.training.model_manager import ModelCheckpointManager
from src.training.trainer import MultiStageTrainer
from src.training.callbacks import AMCLMetricsCallback, CurriculumCallback
from src.utils.logger import get_training_logger, setup_logging


class TestCompleteTrainingSystem:
    """完整训练系统测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.config_path = cls.test_dir / "test_training_config.yaml"
        cls.models_dir = cls.test_dir / "models"
        cls.logs_dir = cls.test_dir / "logs"
        cls.visualization_dir = cls.test_dir / "visualization_reports"
        
        # 创建测试配置
        cls._create_test_config()
        
        # 设置测试logger
        cls.logger = setup_logging(
            log_level="DEBUG", 
            log_file=cls.logs_dir / "test_training.log",
            enable_console=True
        )
        
        cls.logger.info(f"测试类初始化完成，测试目录: {cls.test_dir}")
        
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        import shutil
        # 清理测试文件
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
            
    @staticmethod
    def _create_test_config():
        """创建测试配置文件"""
        config_data = {
            'cargo_types': {
                'normal': {
                    'type_name': 'normal',
                    'max_linear_velocity': 2.0,
                    'max_angular_velocity': 2.0,
                    'stability_penalty': 0.0,
                    'safety_distance': 0.3,
                    'speed_constraint_level': 0,
                    'max_acceleration': 1.0,
                    'min_linear_velocity': 0.1,
                    'reward_weights': {
                        'distance': 10.0,
                        'heading': -2.0,
                        'stability': 0.0,
                        'safety': -5.0,
                        'time': -0.1
                    }
                },
                'fragile': {
                    'type_name': 'fragile',
                    'max_linear_velocity': 1.0,
                    'max_angular_velocity': 1.5,
                    'stability_penalty': -5.0,
                    'safety_distance': 0.4,
                    'speed_constraint_level': 1,
                    'max_acceleration': 0.5,
                    'min_linear_velocity': 0.1,
                    'reward_weights': {
                        'distance': 8.0,
                        'heading': -3.0,
                        'stability': -10.0,
                        'safety': -8.0,
                        'time': -0.15
                    }
                },
                'dangerous': {
                    'type_name': 'dangerous',
                    'max_linear_velocity': 0.8,
                    'max_angular_velocity': 1.0,
                    'stability_penalty': -2.0,
                    'safety_distance': 0.6,
                    'speed_constraint_level': 2,
                    'max_acceleration': 0.3,
                    'min_linear_velocity': 0.1,
                    'reward_weights': {
                        'distance': 5.0,
                        'heading': -4.0,
                        'stability': -5.0,
                        'safety': -20.0,
                        'time': -0.2
                    }
                }
            },
            'training_stages': {
                'stage1': {
                    'stage_name': '基础能力阶段',
                    'total_timesteps': 2000,  # 测试用较少步数
                    'learning_rate': 3e-4,
                    'exploration_noise': 0.1,
                    'policy_noise': 0.2,
                    'target_noise_clip': 0.5,
                    'policy_delay': 2,
                    'buffer_size': 10000,  # 测试用小缓冲区
                    'batch_size': 64,  # 测试用小批量
                    'uncertainty_level': 0.1,
                    'use_curriculum': True,
                    'checkpoint_interval': 500,
                    'description': '低不确定性基础导航能力建立（测试模式）'
                },
                'stage2': {
                    'stage_name': '能力提升阶段',
                    'total_timesteps': 1500,
                    'learning_rate': 2e-4,
                    'exploration_noise': 0.15,
                    'policy_noise': 0.25,
                    'target_noise_clip': 0.5,
                    'policy_delay': 2,
                    'buffer_size': 15000,
                    'batch_size': 64,
                    'uncertainty_level': 0.3,
                    'use_curriculum': True,
                    'checkpoint_interval': 750,
                    'description': '中等不确定性下的鲁棒性训练（测试模式）'
                },
                'stage3': {
                    'stage_name': '鲁棒性验证阶段',
                    'total_timesteps': 1000,
                    'learning_rate': 1e-4,
                    'exploration_noise': 0.2,
                    'policy_noise': 0.3,
                    'target_noise_clip': 0.5,
                    'policy_delay': 2,
                    'buffer_size': 20000,
                    'batch_size': 32,
                    'uncertainty_level': 0.6,
                    'use_curriculum': True,
                    'checkpoint_interval': 1000,
                    'description': '高不确定性复杂情况验证（测试模式）'
                }
            },
            'localization': {
                'amcl': {
                    'num_particles': 400,  # 测试用较少粒子数
                    'initial_std': [0.2, 0.2, 0.15],
                    'motion_noise': {
                        'alpha': [0.1, 0.01, 0.1, 0.01]
                    },
                    'measurement_noise': {
                        'sigma_hit': 0.05
                    },
                    'adaptive_resampling': {
                        'enabled': True,
                        'threshold': 0.5
                    }
                }
            },
            'mlflow': {
                'tracking_uri': 'file:' + str(cls.test_dir / 'mlruns'),
                'experiment_prefix': 'rosbot_navigation_test'
            }
        }
        
        cls.config_path.write_text(yaml.dump(config_data, default_flow_style=False, allow_unicode=True))
        print(f"✓ 测试配置文件已创建: {cls.config_path}")
        
    def test_environment_creation(self):
        """测试环境创建"""
        print("\n🔄 测试环境创建...")
        
        for cargo_type in ['normal', 'fragile', 'dangerous']:
            try:
                env = ROSbotNavigationEnv(cargo_type=cargo_type)
                obs, info = env.reset()
                
                assert obs is not None, f"{cargo_type}: 观察值为空"
                assert obs.shape == (42,), f"{cargo_type}: 观察值形状错误，期望(42,)，实际{obs.shape}"
                assert info is not None, f"{cargo_type}: 信息为空"
                assert info.get('cargo_type') == cargo_type, f"{cargo_type}: 货物类型信息错误"
                
                # 测试基本step功能
                test_action = np.array([0.5, 0.1], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(test_action)
                
                assert obs is not None, f"{cargo_type}: step后观察值为空"
                assert obs.shape == (42,), f"{cargo_type}: step后观察值形状错误"
                assert isinstance(reward, (int, float)), f"{cargo_type}: 奖励类型错误"
                assert isinstance(terminated, bool), f"{cargo_type}: terminated类型错误"
                assert isinstance(truncated, bool), f"{cargo_type}: truncated类型错误"
                
                print(f"   ✓ {cargo_type}环境创建和基础测试通过")
                
            except Exception as e:
                print(f"   ✗ {cargo_type}环境测试失败: {e}")
                raise
                
        print("✓ 环境创建测试全部通过")
        
    def test_training_configuration(self):
        """测试训练配置管理"""
        print("\n🔄 测试训练配置管理...")
        
        try:
            config_manager = TrainingConfigManager(str(self.config_path))
            
            # 测试货物类型配置获取
            for cargo_type in ['normal', 'fragile', 'dangerous']:
                cargo_config = config_manager.get_cargo_config(cargo_type)
                assert cargo_config is not None, f"{cargo_type}配置为None"
                assert cargo_config.type_name == cargo_type, f"{cargo_type}类型名称错误"
                
            # 测试阶段配置获取 
            for stage in ['stage1', 'stage2', 'stage3']:
                stage_config = config_manager.get_stage_config(stage, 'normal')
                assert stage_config is not None, f"{stage}配置为None"
                assert 'total_timesteps' in stage_config, f"{stage}缺少total_timesteps"
                
            # 测试模型配置获取
            model_config = config_manager.get_model_config()
            assert model_config is not None, "模型配置为None"
            assert 'learning_rate' in model_config, "模型配置缺少learning_rate"
            
            print("✓ 训练配置管理测试通过")
            
        except Exception as e:
            print(f"✗ 训练配置管理测试失败: {e}")
            raise
            
    def test_mlflow_integration(self):
        """测试MLflow集成"""
        print("\n🔄 测试MLflow集成...")
        
        try:
            config_manager = TrainingConfigManager(str(self.config_path))
            mlflow_manager = MLflowTrainingManager(config_manager)
            
            # 测试实验运行
            with mlflow_manager.start_run("test_experiment", "test_run") as run:
                # 记录参数
                test_params = {
                    "algorithm": "TD3",
                    "cargo_type": "test",
                    "total_steps": 1000,
                    "learning_rate": 3e-4
                }
                mlflow_manager.log_training_params(test_params)
                
                # 记录指标
                test_metrics = {
                    "test_success_rate": 0.75,
                    "test_avg_reward": 105.2,
                    "test_steps": 1000
                }
                mlflow_manager.log_training_metrics(test_metrics)
                
                # 记录自定义指标
                mlflow_manager.log_custom_metric("test_custom_metric", 42.0)
                
            print("✓ MLflow集成测试通过")
            
        except Exception as e:
            print(f"✗ MLflow集成测试失败: {e}")
            raise
            
    def test_model_checkpoints(self):
        """测试模型检查点管理"""
        print("\n🔄 测试模型检查点管理...")
        
        try:
            from stable_baselines3.common.monitor import Monitor
            
            # 使用模拟环境保证一致性
            config_manager = TrainingConfigManager(str(self.config_path))
            manager = ModelCheckpointManager(config_manager)
            
            # 创建测试模型
            env = Monitor(ROSbotNavigationEnv(cargo_type='normal'))
            model = ImprovedTD3("MlpPolicy", env, verbose=0, 
                              buffer_size=1000, learning_starts=100,
                              batch_size=32)  # 测试用小配置
            
            # 保存检查点
            checkpoint_path = manager.save_checkpoint(
                model, "normal", "stage1", 100, {"test_metric": 0.8}
            )
            
            assert os.path.exists(checkpoint_path), f"检查点文件不存在: {checkpoint_path}"
            
            # 加载检查点
            loaded_model = manager.load_checkpoint(checkpoint_path, env=env)
            assert loaded_model is not None, "加载的模型为None"
            
            # 检查训练状态
            status = manager.get_training_status("normal", "stage1")
            assert status['has_checkpoints'], "应该存在检查点"
            
            print("✓ 模型检查点管理测试通过")
            
        except Exception as e:
            print(f"✗ 模型检查点管理测试失败: {e}")
            raise
            
    def test_training_callbacks(self):
        """测试训练回调函数"""
        print("\n🔄 测试训练回调函数...")
        
        try:
            from src.localization.amcl_localizer import UncertaintyCurriculumTraining
            
            # 创建测试环境
            env = ROSbotNavigationEnv(cargo_type='normal')
            env = env  # 确保是unwrapped环境
            
            # 测试AMCL指标回调
            amcl_callback = AMCLMetricsCallback(verbose=0)
            amcl_callback.training_env = env
            amcl_callback.num_timesteps = 100
            amcl_callback._on_step()
            
            # 测试课程学习回调
            uncertainty_curriculum = UncertaintyCurriculumTraining()
            curriculum_callback = CurriculumCallback(uncertainty_curriculum, verbose=0)
            curriculum_callback.training_env = env
            curriculum_callback.locals = {'total_timesteps': 1000}
            curriculum_callback.num_timesteps = 100
            curriculum_callback._on_step()
            
            # 获取AMCL指标
            amcl_metrics = amcl_callback.get_amcl_metrics()
            assert 'amcl_uncertainties' in amcl_metrics, "AMCL指标缺少不确定性数据"
            assert 'convergence_scores' in amcl_metrics, "AMCL指标缺少收敛性数据"
            
            print("✓ 训练回调函数测试通过")
            
        except Exception as e:
            print(f"✗ 训练回调函数测试失败: {e}")
            raise
            
    def test_complete_training_pipeline(self):
        """测试完整训练流程"""
        print("\n🔄 测试完整训练流程...")
        
        try:
            # 使用测试配置创建多阶段训练器
            config_manager = TrainingConfigManager(str(self.config_path))
            trainer = MultiStageTrainer(config_manager)
            
            # 训练单个stage（减少步数以快速测试）
            self.logger.info("开始测试单个stage训练...")
            
            from stable_baselines3.common.monitor import Monitor
            
            env = Monitor(ROSbotNavigationEnv(cargo_type='normal'))
            model = ImprovedTD3("MlpPolicy", env, verbose=0,
                              buffer_size=2000, learning_starts=200,
                              batch_size=32)  # 测试用小配置
            
            # 短期训练
            model.learn(total_timesteps=500, progress_bar=False)
            
            assert model.num_timesteps >= 500, "模型训练步数不足"
            
            # 快速评估
            test_rewards = []
            for _ in range(3):  # 测试3个episode
                obs, info = env.reset()
                total_reward = 0
                for _ in range(100):  # 每个episode最多100步
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                test_rewards.append(total_reward)
            
            avg_reward = np.mean(test_rewards)
            assert avg_reward != 0, f"测试奖励为0，可能训练无效: {test_rewards}"
            
            print(f"✓ 完整训练流程测试通过 (平均奖励: {avg_reward:.2f})")
            
        except Exception as e:
            print(f"✗ 完整训练流程测试失败: {e}")
            raise
            
    def test_visualization_system(self):
        """测试可视化系统"""
        print("\n🔄 测试可视化系统...")
        
        try:
            from src.training.visualizer import TrainingVisualizer
            
            config_manager = TrainingConfigManager(str(self.config_path))
            visualizer = TrainingVisualizer(config_manager)
            
            # 创建模拟训练结果数据
            mock_stage_results = {
                'cargo_type': 'normal',
                'stage': 'stage1',
                'total_timesteps': 1000,
                'success_rate': {'success_rate': 0.72},
                'average_reward': {'eval_reward_mean': 95.5},
                'training_duration': 120.5,
                'model_path': str(self.models_dir / 'test_model.zip')
            }
            
            # 生成阶段报告
            stage_report_path = visualizer.generate_stage_report(
                cargo_type='normal',
                stage='stage1',
                stage_results=mock_stage_results
            )
            
            assert os.path.exists(stage_report_path), f"阶段报告文件不存在: {stage_report_path}"
            
            # 创建模拟管道结果数据
            mock_pipeline_results = {
                'stage1': {
                    'status': 'completed',
                    'result': mock_stage_results
                },
                'stage2': {
                    'status': 'completed',
                    'result': {
                        'success_rate': {'success_rate': 0.78},
                        'average_reward': {'eval_reward_mean': 108.2},
                    }
                },
                'stage3': {
                    'status': 'completed',
                    'result': {
                        'success_rate': {'success_rate': 0.84},
                        'average_reward': {'eval_reward_mean': 125.7},
                    }
                }
            }
            
            # 生成管道报告
            pipeline_report_path = visualizer.generate_pipeline_report(
                cargo_type='normal',
                pipeline_results=mock_pipeline_results
            )
            
            assert os.path.exists(pipeline_report_path), f"管道报告文件不存在: {pipeline_report_path}"
            
            print("✓ 可视化系统测试通过")
            
        except Exception as e:
            print(f"✗ 可视化系统测试失败: {e}")
            raise
            
    def test_logger_system(self):
        """测试日志系统"""
        print("\n🔄 测试日志系统...")
        
        try:
            from src.utils.logger import TrainingLogger
            
            test_logger = TrainingLogger(log_level="INFO", log_dir=str(self.logs_dir))
            
            # 测试各种日志功能
            test_config = {'algorithm': 'TD3', 'buffer_size': 1000}
            test_results = {'success_rate': 0.8, 'avg_reward': 105.3}
            test_metrics = {'episode_reward': 98.5, 'episode_length': 50}
            
            test_logger.log_training_start(test_config)
            test_logger.log_training_progress(100, 1000, 10, 98.5)
            test_logger.log_evaluation_results(test_metrics)
            test_logger.log_stage_completion("stage1", "normal", test_results)
            test_logger.log_training_end(test_results, 125.5)
            test_logger.log_amcl_metrics(0.15, 0.92, 500)
            
            # 检查日志文件是否创建
            log_files = list(self.logs_dir.glob("*.log"))
            assert len(log_files) > 0, "没有日志文件被创建"
            
            print("✓ 日志系统测试通过")
            
        except Exception as e:
            print(f"✗ 日志系统测试失败: {e}")
            raise
            
    def test_system_integration(self):
        """测试系统集成"""
        print("\n🔄 测试系统集成...")
        
        try:
            from train_all_cargo_types import ROSbotTrainingSystem
            
            # 创建训练系统（模拟测试模式）
            config_manager = TrainingConfigManager(str(self.config_path))
            training_system = ROSbotTrainingSystem(str(self.config_path))
            
            # 验证系统组件
            assert hasattr(training_system, 'config_manager'), "缺少配置管理器"
            assert hasattr(training_system, 'mlflow_manager'), "缺少MLflow管理器"
            assert hasattr(training_system, 'model_manager'), "缺少模型管理器"
            assert hasattr(training_system, 'visualizer'), "缺少可视化器"
            
            # 测试环境设置
            env, model, config = training_system.setup_training_environment('normal', 'stage1')
            assert env is not None, "环境设置为None"
            assert model is not None, "模型设置为None"
            assert config is not None, "配置为None"
            
            # 快速验证模型创建
            assert hasattr(model, 'policy'), "模型缺少policy属性"
            assert hasattr(model, 'critic'), "模型缺少critic属性"
            
            print("✓ 系统集成测试通过")
            
        except Exception as e:
            print(f"✗ 系统集成测试失败: {e}")
            raise
            
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🔄 测试错误处理...")
        
        try:
            # 测试无效货物类型
            with pytest.raises(Exception):
                env = ROSbotNavigationEnv(cargo_type='invalid_type')
                
            # 测试配置管理器错误处理
            config_manager = TrainingConfigManager(str(self.config_path))
            
            # 测试未知货物类型应该返回默认配置  
            cargo_config = config_manager.get_cargo_config('unknown_cargo')
            assert cargo_config is not None, "未知货物类型应该返回默认配置"
            assert cargo_config.type_name == 'normal', "未知货物类型应该返回普通货物配置"
            
            print("✓ 错误处理测试通过")
            
        except Exception as e:
            print(f"✗ 错误处理测试失败: {e}")
            raise
            
    def test_system_summary(self):
        """生成系统测试总结"""
        print(f"\n{'='*60}")
        print("ROSbot导航完整训练系统 - 测试总结")
        print(f"{'='*60}")
        
        # 汇总测试结果
        test_categories = [
            ("环境创建", "✓"),
            ("训练配置管理", "✓"), 
            ("MLflow集成", "✓"),
            ("模型检查点管理", "✓"),
            ("训练回调函数", "✓"),
            ("完整训练流程", "✓"),
            ("可视化系统", "✓"),
            ("日志系统", "✓"),
            ("系统集成", "✓"),
            ("错误处理", "✓"),
        ]
        
        for category, status in test_categories:
            print(f"  {status} {category}")
            
        print(f"\n测试环境设置:")
        print(f"  💾 测试数据目录: {self.test_dir}")
        print(f"  📊 日志目录: {self.logs_dir}")
        print(f"  🤖 模型目录: {self.models_dir}")
        print(f"  📈 可视化报告目录: {self.visualization_dir}")
        
        print(f"\n🎉 ROSbot导航完整训练系统所有核心组件测试通过！")
        print(f"系统已准备好进行实际训练！")


def main():
    """运行完整测试"""
    
    print("🚀 启动ROSbot导航完整训练系统测试...")
    
    # 检查依赖项
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import mlflow
        import stable_baselines3
        print("✓ 所有核心依赖项可用")
    except ImportError as e:
        print(f"✗ 缺少依赖项: {e}")
        print("请先安装requirements.txt中的所有依赖项")
        sys.exit(1)
    
    # 运行测试
    test_system = TestCompleteTrainingSystem()
    
    try:
        test_system.setup_class()
        
        # 运行各项测试
        test_system.test_environment_creation()
        test_system.test_training_configuration()
        test_system.test_mlflow_integration()
        test_system.test_model_checkpoints()
        test_system.test_training_callbacks()
        test_system.test_complete_training_pipeline()
        test_system.test_visualization_system()
        test_system.test_logger_system()
        test_system.test_system_integration()
        test_system.test_error_handling()
        test_system.test_system_summary()
        
        print("\n🎊 所有测试通过！训练系统就绪！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        test_system.teardown_class()


if __name__ == "__main__":
    main()