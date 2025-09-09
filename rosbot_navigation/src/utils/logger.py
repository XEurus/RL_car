"""
训练日志管理器 - 统一管理训练过程的日志记录
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
import coloredlogs


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    设置训练日志系统
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为None则使用默认路径
        max_bytes: 单个日志文件最大大小
        backup_count: 日志文件备份数量
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        
    Returns:
        配置好的logger实例
    """
    
    # 创建logger
    logger = logging.getLogger("rosbot_training")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 使用coloredlogs增加彩色输出
        try:
            coloredlogs.install(
                level=log_level,
                logger=logger,
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                field_styles={
                    'asctime': {'color': 'green'},
                    'name': {'color': 'blue'},
                    'levelname': {'color': 'yellow', 'bold': True},
                }
            )
        except ImportError:
            # coloredlogs不可用，使用标准输出
            pass
    
    # 文件输出
    if enable_file:
        if log_file is None:
            # 使用默认路径
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "rosbot_training.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler防止日志文件过大
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_training_logger(name: str = "rosbot_training") -> logging.Logger:
    """
    获取训练日志记录器
    
    Args:
        name: logger名称
        
    Returns:
        日志记录器实例
    """
    
    return logging.getLogger(name)


class TrainingLogger:
    """训练过程专用日志器"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "./logs"):
        """
        初始化训练日志器
        
        Args:
            log_level: 日志级别
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logging(
            log_level=log_level,
            log_file=self.log_dir / "training.log"
        )
        
        # 专门的训练指标日志器
        self.metrics_logger = setup_logging(
            log_level="INFO",
            log_file=self.log_dir / "metrics.log",
            enable_console=False  # 只写入文件
        )
        self.metrics_logger.name = "rosbot_metrics"
        
    def log_training_start(self, config: dict):
        """记录训练开始"""
        """
        记录训练开始信息
        
        Args:
            config: 训练配置
        """
        """记录训练开始"""
        self.logger.info("="*60)
        self.logger.info("ROSbot Navigation Training Started")
        self.logger.info("="*60)
        self.logger.info(f"Training Config: {config}")
        
    def log_training_end(self, results: dict, duration: float):
        """记录训练结束"""
        """
        记录训练结束信息
        
        Args:
            results: 训练结果
            duration: 训练持续时间（秒）
        """
        """记录训练结束"""
        self.logger.info("="*60)
        self.logger.info("ROSbot Navigation Training Completed")
        self.logger.info("="*60)
        self.logger.info(f"Training Results: {results}")
        self.logger.info(f"Total Duration: {self._format_duration(duration)}")
        
    def log_epoch_results(self, epoch: int, metrics: dict):
        """记录epoch结果"""
        """
        记录每个epoch的训练结果
        
        Args:
            epoch: epoch编号
            metrics: 训练指标
        """
        """记录epoch结果"""
        self.logger.info(f"Epoch {epoch}: {metrics}")
        self.metrics_logger.info(f"epoch,{epoch},{self._dict_to_csv(metrics)}")
        
    def log_stage_completion(self, stage: str, cargo_type: str, results: dict):
        """记录阶段完成"""
        """
        记录训练阶段完成信息
        
        Args:
            stage: 训练阶段
            cargo_type: 货物类型
            results: 训练结果
        """
        """记录阶段完成"""
        self.logger.info(f"Stage {stage} completed for {cargo_type}")
        self.logger.info(f"Results: {results}")
        
    def log_model_save(self, model_path: str, step: int):
        """记录模型保存"""
        """
        记录模型保存信息
        
        Args:
            model_path: 模型保存路径
            step: 训练步数
        """
        """记录模型保存"""
        self.logger.info(f"Model saved at step {step}: {model_path}")
        
        
    def log_evaluation_results(self, metrics: dict):
        """记录评估结果"""
        """
        记录模型评估结果
        
        Args:
            metrics: 评估指标
        """
        """记录评估结果"""
        self.logger.info(f"Evaluation: {metrics}")
        self.metrics_logger.info(f"eval,{datetime.now().isoformat()},{self._dict_to_csv(metrics)}")
        
    def log_error(self, error: Exception, context: str = None):
        """记录错误信息"""
        """
        记录训练过程中的错误
        
        Args:
            error: 异常信息
            context: 错误上下文
        """
        """记录错误信息"""
        error_msg = f"Training Error: {str(error)}"
        if context:
            error_msg = f"{error_msg} (Context: {context})"
        self.logger.error(error_msg)
        
    def log_amcl_metrics(self, uncertainty: float, convergence_rate: float, step: int):
        """记录AMCL指标"""
        """
        记录AMCL定位指标
        
        Args:
            uncertainty: 定位不确定性
            convergence_rate: 收敛率
            step: 训练步数
        """
        """记录AMCL指标"""
        self.metrics_logger.info(f"amcl,{step},{uncertainty},{convergence_rate}")
        
    def log_training_progress(self, step: int, total_steps: int, episode: int, avg_reward: float):
        """记录训练进度"""
        """
        记录训练进度
        
        Args:
            step: 当前步数
            total_steps: 总步数
            episode: 当前剧集
            avg_reward: 平均奖励
        """
        """记录训练进度"""
        progress = step / total_steps * 100
        self.logger.info(f"Progress: {progress:.1f}% (Step {step}/{total_steps}, Episode {episode}, Avg Reward: {avg_reward:.2f})")
        
        if step % 10000 == 0:  # 每万步记录一次详细指标
            self.metrics_logger.info(f"progress,{step},{progress},{episode},{avg_reward}")
            
    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
        """
        将秒数格式化为易读的时间字符串
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        """格式化持续时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
            
    def _dict_to_csv(self, data: dict) -> str:
        """将字典转换为CSV格式字符串"""
        """
        将字典数据转换为CSV格式，便于后续分析
        
        Args:
            data: 字典数据
            
        Returns:
            CSV格式字符串
        """
        """将字典转换为CSV格式字符串"""
        csv_parts = []
        for key, value in data.items():
            if isinstance(value, float):
                csv_parts.append(f"{key}:{value:.4f}")
            else:
                csv_parts.append(f"{key}:{value}")
        return ",".join(csv_parts)