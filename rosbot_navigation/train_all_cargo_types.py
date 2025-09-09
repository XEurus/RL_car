#!/usr/bin/env python3
"""
ROSbot导航完整训练系统 - 三阶段训练脚本
支持三种货物类型的综合训练，包含完整的多阶段训练流程
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass, field

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environments.navigation_env import ROSbotNavigationEnv
from src.models.td3_robust import ImprovedTD3
from src.training.trainer import MultiStageTrainer
from src.training.callbacks import AMCLMetricsCallback, CurriculumCallback, ProgressLoggingCallback
from src.training.config_manager import TrainingConfigManager
from src.training.mlflow_manager import MLflowTrainingManager
from src.training.model_manager import ModelCheckpointManager
from src.training.visualizer import TrainingVisualizer
from src.utils.logger import get_training_logger

# 货物类型定义
CARGO_TYPES = ['normal', 'fragile', 'dangerous']

class ROSbotTrainingSystem:
    """ROSbot导航训练系统主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化训练系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config_manager = TrainingConfigManager(config_path)
        self.mlflow_manager = MLflowTrainingManager(self.config_manager)
        self.model_manager = ModelCheckpointManager(self.config_manager)
        self.visualizer = TrainingVisualizer(self.config_manager)
        self.logger = get_training_logger()
        
        # 训练状态跟踪
        self.training_history = {}
        self.current_stage = "init"
        self.training_start_time = None
        
    def setup_training_environment(self, cargo_type: str, stage: str = "stage1"):
        """设置训练环境"""
        
        # 获取阶段配置
        stage_config = self.config_manager.get_stage_config(stage, cargo_type)
        
        # 创建环境
        env = ROSbotNavigationEnv(cargo_type=cargo_type)
        
        # 设置环境参数
        if stage_config.get("uncertainty_level"):
            env.amcl_localizer.set_uncertainty_level(stage_config["uncertainty_level"])
            
        # 创建模型
        model = self._create_td3_model(env, stage_config)
        
        return env, model, stage_config
        
    def _create_td3_model(self, env, stage_config: Dict):
        """创建TD3模型"""
        
        model_config = self.config_manager.get_model_config()
        
        model = ImprovedTD3(
            "MlpPolicy",
            env,
            learning_rate=stage_config.get("learning_rate", model_config["learning_rate"]),
            buffer_size=model_config["buffer_size"],
            learning_starts=model_config["learning_starts"],
            batch_size=model_config["batch_size"],
            gamma=model_config["gamma"],
            tau=model_config["tau"],
            gradient_steps=model_config["gradient_steps"],
            train_freq=model_config["train_freq"],
            policy_delay=model_config["policy_delay"],
            target_policy_noise=model_config["target_policy_noise"],
            target_noise_clip=model_config["target_noise_clip"],
            verbose=1,
            tensorboard_log=f"./logs/td3_{stage_config['cargo_type']}_{stage_config['stage']}"
        )
        
        return model
        
    def train_single_stage(self, cargo_type: str, stage: str, resume_from: str = None):
        """训练单个阶段"""
        
        stage_config = self.config_manager.get_stage_config(stage, cargo_type)
        total_steps = stage_config["training_steps"]
        
        self.logger.info(f"开始训练 {cargo_type} - {stage}")
        self.logger.info(f"总步数: {total_steps}")
        
        # 设置MLflow实验
        experiment_name = f"rosbot_{cargo_type}_{stage}"
        run_name = f"{cargo_type}_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with self.mlflow_manager.start_run(experiment_name, run_name) as run:
            
            # 设置训练环境
            env, model, config = self.setup_training_environment(cargo_type, stage)
            
            if resume_from:
                self.logger.info(f"从检查点恢复: {resume_from}")
                model = self.model_manager.load_checkpoint(model, resume_from)
                
            # 创建训练回调
            callbacks = self._create_training_callbacks(env, config, stage, cargo_type)
            
            try:
                # 开始训练
                model.learn(
                    total_timesteps=total_steps,
                    callback=callbacks,
                    log_interval=10,
                    progress_bar=True
                )
                
                # 保存阶段模型
                model_path = self.model_manager.save_stage_model(model, cargo_type, stage)
                
                # 记录训练结果
                training_result = {
                    "stage": stage,
                    "cargo_type": cargo_type,
                    "total_steps": model.num_timesteps,
                    "model_path": model_path,
                    "completed_at": datetime.now().isoformat(),
                    "success_rate": self._calculate_success_rate(model, env),
                    "average_reward": self._get_average_reward(model)
                }
                
                self.mlflow_manager.log_stage_results(training_result)
                
                self.logger.info(f"训练 {cargo_type}-{stage} 完成")
                self.logger.info(f"模型保存到: {model_path}")
                
                return training_result
                
            except Exception as e:
                self.logger.error(f"训练 {cargo_type}-{stage} 失败: {e}")
                
                # 保存异常的检查点
                error_checkpoint = self.model_manager.save_error_checkpoint(
                    model, cargo_type, stage, str(e)
                )
                
                raise TrainingError(f"阶段训练失败: {e}")
                
    def train_cargo_type_pipeline(self, cargo_type: str, stages: List[str] = None):
        """训练单个货物类型的完整流程"""
        
        if stages is None:
            stages = ["stage1", "stage2", "stage3"]
            
        self.logger.info(f"开始 {cargo_type} 货物类型完整训练流程")
        
        training_results = {}
        previous_model_path = None
        
        for stage in stages:
            try:
                result = self.train_single_stage(
                    cargo_type=cargo_type,
                    stage=stage,
                    resume_from=previous_model_path
                )
                
                training_results[stage] = result
                previous_model_path = result["model_path"]
                
                # 阶段间可视化
                self.visualizer.generate_stage_report(cargo_type, stage, result)
                
            except Exception as e:
                self.logger.error(f"{cargo_type} 在 {stage} 阶段失败: {e}")
                
                training_results[stage] = {
                    "status": "failed",
                    "error": str(e),
                    "stage": stage,
                    "cargo_type": cargo_type
                }
                
                break
                
        # 生成整体报告
        self._generate_cargo_pipeline_report(cargo_type, training_results)
        
        return training_results
        
    def train_all_cargo_types(self, parallel: bool = False, specific_types: List[str] = None):
        """训练所有货物类型"""
        
        if specific_types is None:
            cargo_types = CARGO_TYPES
        else:
            cargo_types = [ct for ct in specific_types if ct in CARGO_TYPES]
            
        self.logger.info(f"开始所有货物类型训练: {cargo_types}")
        self.training_start_time = datetime.now()
        
        overall_results = {}
        
        if parallel:
            # TODO: 实现并行训练
            self.logger.warning("并行训练暂未实现，使用顺序训练")
            
        for cargo_type in cargo_types:
            try:
                results = self.train_cargo_type_pipeline(cargo_type)
                overall_results[cargo_type] = results
                
            except Exception as e:
                self.logger.error(f"{cargo_type} 训练流程失败: {e}")
                overall_results[cargo_type] = {
                    "status": "failed",
                    "error": str(e),
                    "cargo_type": cargo_type
                }
                
        # 生成总体报告
        self._generate_overall_training_report(overall_results)
        
        return overall_results
        
    def _create_training_callbacks(self, env, config, stage, cargo_type):
        """创建训练回调"""
        
        callbacks = []
        
        # AMCL指标记录
        amcl_callback = AMCLMetricsCallback(verbose=1)
        callbacks.append(amcl_callback)
        
        # 课程学习回调
        if config.get("use_curriculum", True):
            from src.localization.amcl_localizer import UncertaintyCurriculumTraining
            curriculum = UncertaintyCurriculumTraining()
            curriculum_callback = CurriculumCallback(curriculum, verbose=1)
            callbacks.append(curriculum_callback)
            
        # 进度记录回调
        progress_callback = ProgressLoggingCallback(verbose=1)
        callbacks.append(progress_callback)
        
        # 模型检查点回调
        checkpoint_callback = self.model_manager.create_checkpoint_callback(
            cargo_type, stage, save_freq=config.get("checkpoint_freq", 10000)
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks
        
    def _calculate_success_rate(self, model, env, num_episodes: int = 50):
        """计算成功率"""
        
        successes = 0
        total_steps = 0
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_steps = 0
            
            while episode_steps < env.max_steps_per_episode:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                
                if terminated and info.get("success", False):
                    successes += 1
                    break
                    
                if terminated or truncated:
                    break
                    
            total_steps += episode_steps
            
        success_rate = successes / num_episodes
        avg_steps = total_steps / num_episodes
        
        return {
            "success_rate": success_rate,
            "average_steps": avg_steps
        }
        
    def _get_average_reward(self, model):
        """获取平均奖励"""
        
        if hasattr(model, "rollout_buffer") and model.rollout_buffer is not None:
            rewards = model.rollout_buffer.rewards
            if len(rewards) > 0:
                return float(np.mean(rewards))
        
        return 0.0
        
    def _generate_cargo_pipeline_report(self, cargo_type: str, results: Dict):
        """生成货物类型训练流程报告"""
        
        report_path = f"./reports/{cargo_type}_training_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        self.visualizer.generate_pipeline_report(cargo_type, results, report_path)
        
        self.logger.info(f"生成 {cargo_type} 训练流程报告: {report_path}")
        
    def _generate_overall_training_report(self, results: Dict):
        """生成总体训练报告"""
        
        report_path = f"./reports/overall_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        self.visualizer.generate_overall_report(results, report_path)
        
        self.logger.info(f"生成总体训练报告: {report_path}")


class TrainingError(Exception):
    """训练异常"""
    pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ROSbot导航完整训练系统')
    parser.add_argument('--config', type=str, default='./config/training_config.yaml',
                       help='训练配置文件路径')
    parser.add_argument('--cargo_types', nargs='+', choices=CARGO_TYPES,
                       default=None, help='指定货物类型')
    parser.add_argument('--stages', nargs='+', 
                       choices=['stage1', 'stage2', 'stage3'],
                       default=None, help='指定训练阶段')
    parser.add_argument('--parallel', action='store_true',
                       help='并行训练（暂未实现）')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--output_dir', type=str, default='./training_results',
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化报告')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    args = parser.parse_args()
    
    # 设置PyTorch配置
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("使用CPU训练")
    
    # 设置调试模式
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # 创建训练系统
    training_system = ROSbotTrainingSystem(args.config)
    
    try:
        # 开始训练
        print(f"\n{'='*60}")
        print("ROSbot导航训练系统启动")
        print(f"配置文件: {args.config}")
        print(f"货物类型: {args.cargo_types or '全部'}")
        print(f"训练阶段: {args.stages or '全部'}")
        print(f"使用设备: {device}")
        print(f"{'='*60}\n")
        
        results = training_system.train_all_cargo_types(
            parallel=args.parallel,
            specific_types=args.cargo_types
        )
        
        # 保存训练结果
        results_path = Path(args.output_dir) / "training_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n训练完成！结果保存到: {results_path}")
        
        # 生成可视化报告
        if args.visualize:
            print("生成可视化报告...")
            training_system.visualizer.generate_comprehensive_report(results)
            
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练系统错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()