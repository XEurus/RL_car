"""
MLflow训练实验管理器 - 统一跟踪三阶段训练
"""

import os
import mlflow
import mlflow.pytorch
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager
import logging
import numpy as np


class MLflowTrainingManager:
    """MLflow训练实验管理器"""
    
    def __init__(self, config_manager=None):
        """
        初始化MLflow管理器
        
        Args:
            config_manager: 配置管理器
        """
        self.config_mgr = config_manager
        self.logger = logging.getLogger(__name__)
        
        # 设置MLflow跟踪URI
        self.tracking_uri = self._get_tracking_uri()
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # 创建实验存储目录
        self.experiments_dir = Path("./mlruns")
        self.experiments_dir.mkdir(exist_ok=True)
        
        self.current_run = None
        self.current_experiment = None
        
    def _get_tracking_uri(self) -> str:
        """获取MLflow跟踪URI"""
        
        if self.config_mgr:
            config = self.config_mgr.get_all_configs()
            if 'mlflow' in config and 'tracking_uri' in config['mlflow']:
                return config['mlflow']['tracking_uri']
                
        # 默认本地文件跟踪
        return "file:./mlruns"
        
    @contextmanager
    def start_run(self, experiment_name: str, run_name: str = None, 
                  tags: Dict[str, Any] = None):
        """
        开始MLflow实验运行
        
        Args:
            experiment_name: 实验名称
            run_name: 运行名称
            tags: 附加标签
        """
        
        # 创建实验
        experiment = self.get_or_create_experiment(experiment_name)
        
        # 准备运行标签
        run_tags = {
            'training_system': 'rosbot_navigation',
            'start_time': datetime.now().isoformat()
        }
        
        if tags:
            run_tags.update(tags)
            
        try:
            # 开始运行
            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=run_name,
                tags=run_tags
            ) as run:
                self.current_run = run
                self.current_experiment = experiment
                
                self.logger.info(f"开始MLflow运行: {run_name} ({run.info.run_id})")
                
                yield run
                
                # 记录运行结束信息
                mlflow.set_tag('end_time', datetime.now().isoformat())
                mlflow.set_tag('status', 'completed')
                
                self.logger.info(f"MLflow运行完成: {run_name}")
                
        except Exception as e:
            self.logger.error(f"MLflow运行失败: {e}")
            if self.current_run:
                mlflow.set_tag('status', 'failed')
                mlflow.set_tag('error', str(e))
            raise
            
    def get_or_create_experiment(self, experiment_name: str) -> mlflow.entities.Experiment:
        """获取或创建实验"""
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags={
                        'created_at': datetime.now().isoformat(),
                        'type': 'navigation_training'
                    }
                )
                experiment = mlflow.get_experiment(experiment_id)
                self.logger.info(f"创建新实验: {experiment_name}")
            else:
                self.logger.info(f"使用现有实验: {experiment_name}")
                
            return experiment
            
        except Exception as e:
            self.logger.error(f"创建/获取实验失败: {e}")
            raise
            
    def log_training_params(self, params: Dict[str, Any]):
        """记录训练参数"""
        
        if not self.current_run:
            self.logger.warning("没有活动的MLflow运行，跳过参数记录")
            return
            
        try:
            # 记录基础参数
            mlflow.log_params({
                'algorithm': 'TD3',
                **params
            })
            
            # 记录货物类型特定参数
            if 'cargo_type' in params:
                self._log_cargo_specific_params(params['cargo_type'])
                
            self.logger.info(f"记录 {len(params)} 个训练参数")
            
        except Exception as e:
            self.logger.error(f"记录参数失败: {e}")
            
    def _log_cargo_specific_params(self, cargo_type: str):
        """记录货物类型特定参数"""
        
        if not self.config_mgr:
            return
            
        cargo_config = self.config_mgr.get_cargo_config(cargo_type)
        
        cargo_params = {
            'cargo_type_name': cargo_config.type_name,
            'max_linear_velocity': cargo_config.max_linear_velocity,
            'max_angular_velocity': cargo_config.max_angular_velocity,
            'stability_penalty': cargo_config.stability_penalty,
            'safety_distance': cargo_config.safety_distance,
            'speed_constraint_level': cargo_config.speed_constraint_level
        }
        
        mlflow.log_params(cargo_params)
        
    def log_training_metrics(self, metrics: Dict[str, float], step: int = None):
        """记录训练指标"""
        
        if not self.current_run:
            self.logger.warning("没有活动的MLflow运行，跳过指标记录")
            return
            
        try:
            # 过滤非数值指标
            numeric_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    numeric_metrics[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    numeric_metrics[key] = float(value)
                    
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)
                self.logger.debug(f"记录 {len(numeric_metrics)} 个训练指标")
                
        except Exception as e:
            self.logger.error(f"记录指标失败: {e}")
            
    def log_stage_results(self, stage_results: Dict[str, Any]):
        """记录阶段训练结果"""
        
        if not self.current_run:
            self.logger.warning("没有活动的MLflow运行，跳过结果记录")
            return
            
        try:
            # 记录阶段结果
            metrics = {
                'stage_success_rate': stage_results.get('success_rate', {}).get('success_rate', 0.0),
                'stage_avg_reward': stage_results.get('average_reward', {}).get('eval_reward_mean', 0.0),
                'stage_total_steps': stage_results.get('total_timesteps', 0),
                'stage_duration_seconds': self._parse_duration(stage_results.get('training_duration')),
            }
            
            mlflow.log_metrics(metrics)
            
            # 记录模型信息
            model_path = stage_results.get('model_path')
            if model_path and os.path.exists(model_path):
                mlflow.set_tag('model_path', model_path)
                
            # 记录阶段作为artifact
            stage_info = {
                'stage_results': stage_results,
                'recorded_at': datetime.now().isoformat()
            }
            
            with open('./temp_stage_results.json', 'w') as f:
                json.dump(stage_info, f, indent=2, default=str)
                
            mlflow.log_artifact('./temp_stage_results.json', 'stage_results')
            os.remove('./temp_stage_results.json')
            
            self.logger.info(f"记录阶段训练结果: {stage_results.get('stage', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"记录阶段结果失败: {e}")
            
    def log_pipeline_results(self, pipeline_results: Dict[str, Dict]):
        """记录整个训练流程的结果"""
        
        if not self.current_run:
            self.logger.warning("没有活动的MLflow运行，跳过流程结果记录")
            return
            
        try:
            # 计算总体指标
            total_success_rates = []
            total_rewards = []
            total_steps = 0
            
            for stage_name, stage_data in pipeline_results.items():
                if stage_data.get('status') == 'completed':
                    result = stage_data.get('result', {})
                    
                    # 成功率
                    success_info = result.get('success_rate', {})
                    if isinstance(success_info, dict) and 'success_rate' in success_info:
                        total_success_rates.append(success_info['success_rate'])
                        
                    # 平均奖励
                    reward_info = result.get('average_reward', {})
                    if isinstance(reward_info, dict) and 'eval_reward_mean' in reward_info:
                        total_rewards.append(reward_info['eval_reward_mean'])
                        
                    # 总步数
                    total_steps += result.get('total_timesteps', 0)
                    
            # 记录总体指标
            if total_success_rates:
                metrics = {
                    'pipeline_avg_success_rate': np.mean(total_success_rates),
                    'pipeline_max_success_rate': np.max(total_success_rates),
                    'pipeline_avg_reward': np.mean(total_rewards) if total_rewards else 0.0,
                    'pipeline_total_steps': total_steps,
                    'pipeline_num_stages': len([s for s in pipeline_results.values() if s.get('status') == 'completed'])
                }
                
                mlflow.log_metrics(metrics)
                
            # 记录流程结果
            pipeline_info = {
                'pipeline_results': pipeline_results,
                'summary_metrics': metrics if total_success_rates else {},
                'recorded_at': datetime.now().isoformat()
            }
            
            with open('./temp_pipeline_results.json', 'w') as f:
                json.dump(pipeline_info, f, indent=2, default=str)
                
            mlflow.log_artifact('./temp_pipeline_results.json', 'pipeline_results')
            os.remove('./temp_pipeline_results.json')
            
            self.logger.info(f"记录训练流程总体结果: {len(pipeline_results)} 个阶段")
            
        except Exception as e:
            self.logger.error(f"记录流程结果失败: {e}")
            
    def log_model_artifact(self, model_path: str, artifact_path: str = 'models'):
        """记录模型文件为artifact"""
        
        if not self.current_run:
            self.logger.warning("没有活动的MLflow运行，跳过模型artifact记录")
            return
            
        try:
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path)
                self.logger.info(f"记录模型artifact: {model_path}")
            else:
                self.logger.warning(f"模型文件不存在: {model_path}")
                
        except Exception as e:
            self.logger.error(f"记录模型artifact失败: {e}")
            
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """记录超参数"""
        
        if not self.current_run:
            self.logger.warning("没有活动的MLflow运行，跳过超参数记录")
            return
            
        try:
            mlflow.log_params(hyperparams)
            self.logger.info(f"记录 {len(hyperparams)} 个超参数")
            
        except Exception as e:
            self.logger.error(f"记录超参数失败: {e}")
            
    def log_training_progress(self, progress_stats: Dict[str, Any]):
        """记录训练进度"""
        
        if not self.current_run:
            self.logger.warning("没有活动的MLflow运行，跳过进度记录")
            return
            
        try:
            # 计算进度相关指标
            current_step = progress_stats.get('current_step', 0)
            total_steps = progress_stats.get('total_steps', 1)
            
            progress_metrics = {
                'training_progress_percent': (current_step / total_steps) * 100,
                'current_timestep': current_step,
                'total_timesteps': total_steps,
            }
            
            # 记录环境统计
            if 'env_stats' in progress_stats:
                env_stats = progress_stats['env_stats']
                progress_metrics.update({
                    'env_avg_reward': env_stats.get('avg_reward', 0.0),
                    'env_avg_length': env_stats.get('avg_length', 0.0),
                    'env_success_rate': env_stats.get('success_rate', 0.0),
                })
                
            mlflow.log_metrics(progress_metrics, step=current_step)
            
            self.logger.debug(f"记录训练进度: {current_step}/{total_steps}")
            
        except Exception as e:
            self.logger.error(f"记录训练进度失败: {e}")
            
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """获取当前运行信息"""
        
        if not self.current_run:
            return None
            
        return {
            'run_id': self.current_run.info.run_id,
            'experiment_id': self.current_run.info.experiment_id,
            'run_name': self.current_run.data.tags.get('mlflow.runName', 'unknown'),
            'status': self.current_run.info.status,
            'start_time': self.current_run.info.start_time,
            'end_time': getattr(self.current_run.info, 'end_time', None)
        }
        
    def log_custom_metric(self, name: str, value: float, step: int = None):
        """记录自定义指标"""
        
        try:
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(name, value, step=step)
        except Exception as e:
            self.logger.error(f"记录自定义指标失败: {name}={value}: {e}")
            
    def end_run(self, status: str = 'FINISHED'):
        """结束当前运行"""
        
        if self.current_run:
            try:
                mlflow.end_run(status=status)
                self.logger.info(f"结束MLflow运行: {status}")
            except Exception as e:
                self.logger.error(f"结束运行失败: {e}")
            
        self.current_run = None
        self.current_experiment = None
        
    def _parse_duration(self, duration) -> float:
        """解析持续时间"""
        
        if isinstance(duration, (int, float)):
            return float(duration)
        elif hasattr(duration, 'total_seconds'):
            return duration.total_seconds()
        elif isinstance(duration, str):
            try:
                # 尝试解析字符串格式的持续时间
                import datetime
                if 'day' in duration:
                    # 处理timedelta字符串
                    parts = duration.split()
                    days = int(parts[0])
                    time_part = datetime.datetime.strptime(parts[2], '%H:%M:%S.%f')
                    return days * 86400 + time_part.hour * 3600 + time_part.minute * 60 + time_part.second
                else:
                    # 尝试解析为秒数
                    return float(duration)
            except:
                return 0.0
        
        return 0.0
        
    def search_experiments(self, experiment_name_contains: str = None, 
                          tag_filters: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """搜索实验"""
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            # 构建搜索过滤器
            filter_string = ""
            if experiment_name_contains:
                filter_string += f"name LIKE '%{experiment_name_contains}%"  
                
            if tag_filters:
                for key, value in tag_filters.items():
                    if filter_string:
                        filter_string += " AND "
                    filter_string += f"tags.{key} = '{value}'"
                    
            experiments = client.search_experiments(filter_string=filter_string)
            
            experiment_info = []
            for exp in experiments:
                experiment_info.append({
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'artifact_location': exp.artifact_location,
                    'lifecycle_stage': exp.lifecycle_stage,
                    'tags': exp.tags
                })
                
            return experiment_info
            
        except Exception as e:
            self.logger.error(f"搜索实验失败: {e}")
            return []
            
    def get_run_history(self, experiment_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """获取运行历史"""
        
        try:
            experiment = self.get_or_create_experiment(experiment_name)
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=['attribute.start_time DESC'],
                max_results=max_results
            )
            
            run_history = []
            for _, run in runs.iterrows():
                run_info = {
                    'run_id': run['run_id'],
                    'run_name': run.get('tags.mlflow.runName', 'unknown'),
                    'status': run['status'],
                    'start_time': run['start_time'],
                    'end_time': run.get('end_time'),
                    'metrics': self._extract_metrics_from_run(run),
                    'params': self._extract_params_from_run(run),
                    'tags': self._extract_tags_from_run(run)
                }
                run_history.append(run_info)
                
            return run_history
            
        except Exception as e:
            self.logger.error(f"获取运行历史失败: {e}")
            return []
            
    def _extract_metrics_from_run(self, run_series) -> Dict[str, float]:
        """从运行序列中提取指标"""
        
        metrics = {}
        for key, value in run_series.items():
            if key.startswith('metrics.'):
                metric_name = key.replace('metrics.', '')
                if pd.notna(value):
                    metrics[metric_name] = value
                    
        return metrics
        
    def _extract_params_from_run(self, run_series) -> Dict[str, str]:
        """从运行序列中提取参数"""
        
        params = {}
        for key, value in run_series.items():
            if key.startswith('params.'):
                param_name = key.replace('params.', '')
                if pd.notna(value):
                    params[param_name] = str(value)
                    
        return params
        
    def _extract_tags_from_run(self, run_series) -> Dict[str, str]:
        """从运行序列中提取标签"""
        
        tags = {}
        for key, value in run_series.items():
            if key.startswith('tags.'):
                tag_name = key.replace('tags.', '')
                if pd.notna(value):
                    tags[tag_name] = str(value)
                    
        return tags


# 需要在文件顶部添加pandas导入
import pandas as pd