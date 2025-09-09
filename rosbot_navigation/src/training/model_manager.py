"""
模型检查点和版本管理器 - 支持三阶段训练的模型保存和恢复
"""

import os
import glob
import json
import shutil
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class ModelCheckpointManager:
    """模型检查点管理器"""
    
    def __init__(self, config_manager=None):
        """
        初始化模型管理器
        
        Args:
            config_manager: 配置管理器
        """
        self.config_mgr = config_manager
        self.logger = logging.getLogger(__name__)
        
        # 默认存储路径
        self.checkpoints_dir = Path("./models/checkpoints")
        self.stages_dir = Path("./models/stages")
        
        # 创建存储目录
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_info = {}
        
    def save_checkpoint(self, model: TD3, cargo_type: str, stage: str, 
                       step: int, metrics: Dict[str, Any] = None) -> str:
        """
        保存训练检查点
        
        Args:
            model: 要保存的模型
            cargo_type: 货物类型
            stage: 训练阶段
            step: 训练步数
            metrics: 当前训练的评估指标
            
        Returns:
            检查点文件路径
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{cargo_type}_{stage}_step{step}_{timestamp}"
        
        # 创建检查点目录
        checkpoint_dir = self.checkpoints_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存模型
            model_path = checkpoint_dir / f"{checkpoint_name}.zip"
            model.save(str(model_path))
            
            # 保存检查点元信息
            checkpoint_info = {
                'cargo_type': cargo_type,
                'stage': stage,
                'step': step,
                'timestamp': timestamp,
                'model_path': str(model_path),
                'metrics': metrics or {},
                'model_hash': self._calculate_model_hash(str(model_path))
            }
            
            # 保存信息文件
            info_path = checkpoint_dir / "checkpoint_info.json"
            with open(info_path, 'w') as f:
                json.dump(checkpoint_info, f, indent=2, default=str)
                
            # 保存训练状态
            if hasattr(model, 'training_env') and hasattr(model.training_env, 'envs'):
                self._save_env_state(checkpoint_dir, model.training_env.envs[0])
                
            self.checkpoint_info[checkpoint_name] = checkpoint_info
            
            self.logger.info(f"保存检查点: {checkpoint_name}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            # 清理部分保存的文件
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            raise
            
    def load_checkpoint(self, model_path: str, env=None, strict: bool = True) -> TD3:
        """
        加载检查点
        
        Args:
            model_path: 模型文件路径
            env: 环境实例（可选）
            strict: 是否严格模式（检查模型兼容性）
            
        Returns:
            加载的模型
        """
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"检查点文件不存在: {model_path}")
            
        try:
            # 加载模型
            if env:
                model = TD3.load(model_path, env=env)
            else:
                model = TD3.load(model_path)
                
            self.logger.info(f"加载检查点: {model_path}")
            
            # 检查模型兼容性（如果启用）
            if strict:
                self._validate_model_compatibility(model_path)
                
            return model
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            raise
            
    def save_stage_model(self, model: TD3, cargo_type: str, stage: str, 
                        final_metrics: Dict[str, Any] = None) -> str:
        """
        保存阶段完成模型
        
        Args:
            model: 要保存的模型
            cargo_type: 货物类型
            stage: 训练阶段
            final_metrics: 最终评估指标
            
        Returns:
            保存的模型文件路径
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{cargo_type}_{stage}_final_{timestamp}"
        
        model_path = self.stages_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存模型
            model_file = model_path / f"{model_name}.zip"
            model.save(str(model_file))
            
            # 保存最终评估结果
            if final_metrics:
                eval_file = model_path / "final_evaluation.json"
                with open(eval_file, 'w') as f:
                    json.dump(final_metrics, f, indent=2, default=str)
                    
            # 保存模型元信息
            model_info = {
                'cargo_type': cargo_type,
                'stage': stage,
                'timestamp': timestamp,
                'model_path': str(model_file),
                'final_metrics': final_metrics or {},
                'total_timesteps': getattr(model, 'n_timesteps', 0),
                'training_duration': getattr(model, 'training_time', 'unknown')
            }
            
            info_file = model_path / "model_info.json"
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
                
            self.logger.info(f"保存阶段模型: {model_name}")
            return str(model_file)
            
        except Exception as e:
            self.logger.error(f"保存阶段模型失败: {e}")
            raise
            
    def get_best_checkpoint(self, cargo_type: str, stage: str, 
                           metric: str = 'success_rate') -> Optional[str]:
        """
        获取指定类型的最佳检查点
        
        Args:
            cargo_type: 货物类型
            stage: 训练阶段
            metric: 评估指标
            
        Returns:
            最佳检查点路径，如果没有找到则返回None
        """
        
        candidates = self.list_checkpoints(cargo_type, stage)
        
        if not candidates:
            return None
            
        best_checkpoint = None
        best_score = -np.inf
        
        for checkpoint_path in candidates:
            try:
                info_path = Path(checkpoint_path).parent / "checkpoint_info.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        
                    score = info.get('metrics', {}).get(metric, 0.0)
                    if score > best_score:
                        best_score = score
                        best_checkpoint = checkpoint_path
                        
            except Exception as e:
                self.logger.warning(f"读取检查点信息失败 {checkpoint_path}: {e}")
                
        self.logger.info(f"找到最佳检查点: {best_checkpoint} (得分: {best_score})")
        return best_checkpoint
        
    def list_checkpoints(self, cargo_type: str = None, stage: str = None) -> List[str]:
        """
        列出检查点
        
        Args:
            cargo_type: 货物类型过滤（可选）
            stage: 阶段过滤（可选）
            
        Returns:
            检查点路径列表
        """
        
        pattern = f"*_{stage}_*" if stage else "*.zip"
        
        if cargo_type:
            pattern = f"{cargo_type}_{pattern}"
            
        checkpoint_files = list(self.checkpoints_dir.rglob(pattern))
        
        return [str(f) for f in checkpoint_files if f.suffix == '.zip']
        
    def list_stage_models(self, cargo_type: str = None, stage: str = None) -> List[str]:
        """
        列出阶段模型
        
        Args:
            cargo_type: 货物类型过滤（可选）
            stage: 阶段过滤（可选）
            
        Returns:
            模型路径列表
        """
        
        pattern = f"*_final_*"  # 阶段完成模型
        
        if cargo_type:
            pattern = f"{cargo_type}_{pattern}"
            
        if stage:
            pattern = f"*{stage}*_final_*"
            
        model_files = list(self.stages_dir.rglob(pattern))
        
        return [str(f) for f in model_files if f.suffix == '.zip']
        
    def create_checkpoint_callback(self, cargo_type: str, stage: str, 
                                 save_freq: int = 10000) -> BaseCallback:
        """
        创建检查点保存回调
        
        Args:
            cargo_type: 货物类型
            stage: 训练阶段
            save_freq: 保存频率（步数）
            
        Returns:
            检查点回调函数
        """
        
        class CheckpointCallback(BaseCallback):
            
            def __init__(self, save_freq, cargo_type, stage, manager):
                super().__init__()
                self.save_freq = save_freq
                self.cargo_type = cargo_type
                self.stage = stage
                self.manager = manager
                self.last_save_step = 0
                
            def _on_step(self) -> bool:
                # 定期保存检查点
                if self.num_timesteps - self.last_save_step >= self.save_freq:
                    try:
                        # 获取当前评估指标
                        metrics = self._collect_current_metrics()
                        
                        # 保存检查点
                        model_path = self.manager.save_checkpoint(
                            self.model, self.cargo_type, self.stage, 
                            self.num_timesteps, metrics
                        )
                        
                        self.last_save_step = self.num_timesteps
                        self.manager.logger.info(f"保存检查点 (step {self.num_timesteps}): {model_path}")
                        
                    except Exception as e:
                        self.manager.logger.error(f"检查点保存失败: {e}")
                        
                return True
                
            def _collect_current_metrics(self) -> Dict[str, Any]:
                """收集当前训练指标"""
                
                metrics = {
                    'current_step': self.num_timesteps,
                    'episode_count': self._episode_num if hasattr(self, '_episode_num') else 0
                }
                
                # 从训练日志中提取指标（如果有）
                if hasattr(self.locals, 'infos') and self.locals['infos']:
                    recent_reward = sum(info.get('episode', {}).get('r', 0) 
                                      for info in self.locals['infos'])
                    metrics['recent_avg_reward'] = recent_reward / len(self.locals['infos'])
                    
                return metrics
                
        return CheckpointCallback(save_freq, cargo_type, stage, self)
        
    def save_error_checkpoint(self, model: TD3, cargo_type: str, stage: str, 
                            error_message: str) -> str:
        """
        保存错误状态的检查点（用于调试）
        
        Args:
            model: 出错的模型
            cargo_type: 货物类型
            stage: 训练阶段
            error_message: 错误信息
            
        Returns:
            错误检查点路径
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{cargo_type}_{stage}_error_{timestamp}"
        
        checkpoint_dir = self.checkpoints_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存模型
            model_file = checkpoint_dir / f"{checkpoint_name}.zip"
            model.save(str(model_file))
            
            # 保存错误信息
            error_info = {
                'error_type': 'training_error',
                'error_message': error_message,
                'timestamp': timestamp,
                'cargo_type': cargo_type,
                'stage': stage,
                'current_step': getattr(model, 'num_timesteps', 0)
            }
            
            error_file = checkpoint_dir / "error_info.json"
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
                
            self.logger.warning(f"保存错误检查点: {checkpoint_name}")
            return str(model_file)
            
        except Exception as e:
            self.logger.error(f"保存错误检查点失败: {e}")
            raise
            
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """
        清理旧的检查点
        
        Args:
            keep_last: 保留最近几个检查点
        """
        
        all_checkpoints = self.list_checkpoints()
        
        # 按时间排序
        def get_checkpoint_time(checkpoint_path):
            try:
                info_path = Path(checkpoint_path).parent / "checkpoint_info.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                        return info.get('timestamp', '')
            except:
                return ''
                
        sorted_checkpoints = sorted(
            all_checkpoints, 
            key=lambda x: get_checkpoint_time(x), 
            reverse=True
        )
        
        # 删除旧的检查点
        for checkpoint_path in sorted_checkpoints[keep_last:]:
            try:
                checkpoint_dir = Path(checkpoint_path).parent
                shutil.rmtree(checkpoint_dir)
                self.logger.info(f"删除旧检查点: {checkpoint_dir}")
            except Exception as e:
                self.logger.error(f"删除检查点失败 {checkpoint_path}: {e}")
                
    def _calculate_model_hash(self, model_path: str) -> str:
        """计算模型文件哈希值"""
        
        import hashlib
        
        try:
            hash_md5 = hashlib.md5()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"
            
    def _validate_model_compatibility(self, model_path: str):
        """验证模型兼容性"""
        
        try:
            # 加载模型进行基本验证
            test_model = TD3.load(model_path)
            
            # 检查必要的模型组件
            required_attributes = ['policy', 'critic', 'target_policy', 'target_critic']
            for attr in required_attributes:
                if not hasattr(test_model, attr):
                    raise ValueError(f"模型缺少必要组件: {attr}")
                    
            self.logger.info(f"模型兼容性验证通过: {model_path}")
            
        except Exception as e:
            self.logger.error(f"模型兼容性验证失败: {e}")
            raise
            
    def _save_env_state(self, checkpoint_dir: Path, env):
        """保存环境状态（如需要）"""
        
        try:
            # 保存环境中的随机种子状态
            env_state = {
                'seed': getattr(env, 'seed', None),
                'reset_state': getattr(env, '_elapsed_steps', None)
            }
            
            if any(v is not None for v in env_state.values()):
                env_file = checkpoint_dir / "env_state.json"
                with open(env_file, 'w') as f:
                    json.dump(env_state, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.warning(f"保存环境状态失败: {e}")
            
    def get_training_status(self, cargo_type: str, stage: str) -> Dict[str, Any]:
        """
        获取训练状态
        
        Args:
            cargo_type: 货物类型
            stage: 训练阶段
            
        Returns:
            状态信息字典
        """
        
        status = {
            'cargo_type': cargo_type,
            'stage': stage,
            'has_checkpoints': False,
            'has_stage_model': False,
            'best_checkpoint': None,
            'latest_checkpoint': None,
            'stage_model': None
        }
        
        try:
            # 检查检查点
            checkpoints = self.list_checkpoints(cargo_type, stage)
            if checkpoints:
                status['has_checkpoints'] = True
                status['latest_checkpoint'] = max(checkpoints, key=os.path.getmtime)
                
                # 获取最佳检查点
                best_checkpoint = self.get_best_checkpoint(cargo_type, stage)
                if best_checkpoint:
                    status['best_checkpoint'] = best_checkpoint
                    
            # 检查阶段模型
            stage_models = self.list_stage_models(cargo_type, stage)
            if stage_models:
                status['has_stage_model'] = True
                status['stage_model'] = max(stage_models, key=os.path.getmtime)
                
        except Exception as e:
            self.logger.error(f"获取训练状态失败: {e}")
            
        return status
        
    def export_model_summary(self, output_path: str):
        """导出模型状态摘要"""
        
        summary = {
            'export_time': datetime.now().isoformat(),
            'checkpoints': {},
            'stage_models': {},
            'training_status': {}
        }
        
        # 汇总检查点信息
        for checkpoint in self.list_checkpoints():
            checkpoint_name = os.path.basename(checkpoint).replace('.zip', '')
            parts = checkpoint_name.split('_')
            
            if len(parts) >= 3:
                cargo_type = parts[0]
                stage = parts[1]
                
                if cargo_type not in summary['checkpoints']:
                    summary['checkpoints'][cargo_type] = {}
                if stage not in summary['checkpoints'][cargo_type]:
                    summary['checkpoints'][cargo_type][stage] = []
                    
                summary['checkpoints'][cargo_type][stage].append({
                    'path': checkpoint,
                    'modified_time': os.path.getmtime(checkpoint)
                })
                
        # 汇总阶段模型信息  
        for stage_model in self.list_stage_models():
            model_name = os.path.basename(stage_model).replace('.zip', '')
            parts = model_name.split('_')
            
            if len(parts) >= 3:
                cargo_type = parts[0]
                stage = parts[1]
                
                if cargo_type not in summary['stage_models']:
                    summary['stage_models'][cargo_type] = {}
                    
                summary['stage_models'][cargo_type][stage] = {
                    'path': stage_model,
                    'modified_time': os.path.getmtime(stage_model)
                }
                
        # 生成训练状态
        for cargo_type in ['normal', 'fragile', 'dangerous']:
            for stage in ['stage1', 'stage2', 'stage3']:
                status = self.get_training_status(cargo_type, stage)
                
                if cargo_type not in summary['training_status']:
                    summary['training_status'][cargo_type] = {}
                    
                summary['training_status'][cargo_type][stage] = status
                
        # 保存摘要
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        self.logger.info(f"模型状态摘要导出到: {output_path}")
        return str(output_path)