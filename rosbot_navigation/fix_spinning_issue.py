"""
修复原地打转问题的自动化脚本
根据诊断结果自动应用修复方案
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import re


def backup_files():
    """备份原始文件"""
    
    backup_dir = Path("./backups") / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_backup = [
        "src/environments/navigation_env.py",
        "src/models/td3_robust.py",
        "config/training_config.yaml"
    ]
    
    for file_path in files_to_backup:
        if Path(file_path).exists():
            shutil.copy2(file_path, backup_dir / Path(file_path).name)
            print(f"✅ 备份文件: {file_path} -> {backup_dir / Path(file_path).name}")
    
    return str(backup_dir)


def fix_reward_function():
    """修复奖励函数中的问题"""
    
    env_file = Path("src/environments/navigation_env.py")
    if not env_file.exists():
        print(f"❌ 文件不存在: {env_file}")
        return False
    
    # 读取文件
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: 降低movement_reward的放大系数
    content = re.sub(
        r'rewards\[\'movement_reward\'\]\s*=\s*movement_reward\s*\*\s*20',
        'rewards[\'movement_reward\'] = movement_reward * 2',
        content
    )
    
    # 修复2: 加强原地打转检测
    content = re.sub(
        r'self\.stuck_steps_for_spin_check\s*=\s*\d+',
        'self.stuck_steps_for_spin_check = 5',
        content
    )
    
    content = re.sub(
        r'self\.spin_termination_threshold\s*=\s*[\d.]+',
        'self.spin_termination_threshold = 3.14',
        content
    )
    
    # 修复3: 增加原地打转惩罚力度
    content = re.sub(
        r'rewards\[\'excessive_spin_penalty\'\]\s*=\s*-500\.0',
        'rewards[\'excessive_spin_penalty\'] = -1000.0',
        content
    )
    
    # 修复4: 添加更敏感的原地打转检测
    insertion_point = content.find("# 如果累计旋转超过阈值，给予一个惩罚")
    if insertion_point != -1:
        additional_check = """
        # 额外的原地打转检测 - 更敏感
        if self.same_spot_steps > 3:  # 降低检测阈值
            position_penalty = -10.0 * self.same_spot_steps  # 累进惩罚
            rewards['position_stagnation_penalty'] = position_penalty
        """
        content = content[:insertion_point] + additional_check + "\n        " + content[insertion_point:]
    
    # 写回文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已修复奖励函数中的问题")
    return True


def fix_gradient_issues():
    """修复梯度相关问题"""
    
    td3_file = Path("src/models/td3_robust.py")
    if not td3_file.exists():
        print(f"❌ 文件不存在: {td3_file}")
        return False
    
    # 读取文件
    with open(td3_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加梯度裁剪
    if "torch.nn.utils.clip_grad_norm_" not in content:
        # 在ImprovedTD3类的learn方法中添加梯度裁剪
        learn_method_start = content.find("def learn(")
        if learn_method_start != -1:
            # 找到方法的结束位置，在return语句前添加
            return_pos = content.find("return super().learn(", learn_method_start)
            if return_pos != -1:
                gradient_clip_code = """
        
        # 添加梯度裁剪以提高训练稳定性
        def _clip_gradients(self):
            if hasattr(self.policy, 'actor'):
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
            if hasattr(self.policy, 'critics'):
                for critic in self.policy.critics:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        
        # 重写训练步骤以包含梯度裁剪
        original_train = super()._update_policy
        def _update_policy_with_clipping(gradient_steps, batch_size):
            result = original_train(gradient_steps, batch_size)
            self._clip_gradients()
            return result
        super()._update_policy = _update_policy_with_clipping
        
        """
                content = content[:return_pos] + gradient_clip_code + content[return_pos:]
    
    # 写回文件
    with open(td3_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已添加梯度裁剪机制")
    return True


def fix_training_config():
    """修复训练配置"""
    
    config_file = Path("config/training_config.yaml")
    if not config_file.exists():
        print(f"❌ 文件不存在: {config_file}")
        return False
    
    # 读取文件
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修改学习率配置
    content = re.sub(
        r'learning_rate:\s*3e-4',
        'learning_rate: 1e-4  # 降低学习率提高稳定性',
        content
    )
    
    # 修改噪声参数
    content = re.sub(
        r'exploration_noise:\s*0\.\d+',
        'exploration_noise: 0.05  # 降低探索噪声',
        content
    )
    
    # 写回文件
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已优化训练配置")
    return True


def create_enhanced_callback():
    """创建增强的训练回调"""
    
    callback_file = Path("src/training/spinning_fix_callback.py")
    
    callback_code = '''"""
增强的训练回调 - 专门处理原地打转问题
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any


class AntiSpinningCallback(BaseCallback):
    """防原地打转回调"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.spinning_threshold = 10  # 原地打转检测阈值
        self.spinning_penalty = 2.0   # 原地打转时的学习率惩罚
        self.position_history = []
        self.spinning_episodes = 0
        
    def _on_step(self) -> bool:
        """每一步调用"""
        
        # 检测原地打转
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
        else:
            env = self.training_env
            
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped
            
        # 记录位置
        if hasattr(env, '_get_sup_position'):
            current_pos = env._get_sup_position()
            self.position_history.append(current_pos[:2])  # 只记录x,y
            
            # 保持历史长度
            if len(self.position_history) > 50:
                self.position_history.pop(0)
            
            # 检测原地打转
            if len(self.position_history) >= self.spinning_threshold:
                recent_positions = np.array(self.position_history[-self.spinning_threshold:])
                position_std = np.std(recent_positions, axis=0)
                
                # 如果位置标准差很小，说明在原地打转
                if np.mean(position_std) < 0.05:  # 5cm范围内
                    self._handle_spinning_detection()
        
        return True
    
    def _handle_spinning_detection(self):
        """处理原地打转检测"""
        
        self.spinning_episodes += 1
        
        if self.verbose >= 1:
            print(f"🌀 检测到原地打转 (第{self.spinning_episodes}次)")
        
        # 动态调整学习率
        if hasattr(self.model, 'lr_schedule'):
            current_lr = self.model.lr_schedule(1)
            new_lr = current_lr / self.spinning_penalty
            
            # 更新优化器学习率
            if hasattr(self.model.policy, 'actor_optimizer'):
                for param_group in self.model.policy.actor_optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            if hasattr(self.model.policy, 'critic_optimizer'):
                for param_group in self.model.policy.critic_optimizer.param_groups:
                    param_group['lr'] = new_lr
        
        # 添加随机扰动以打破循环
        if hasattr(self.model.policy, 'actor'):
            with torch.no_grad():
                for param in self.model.policy.actor.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * 0.001
                        param.add_(noise)
        
        # 记录到tensorboard
        self.logger.record("anti_spinning/detections", self.spinning_episodes)
        self.logger.record("anti_spinning/current_lr", new_lr if 'new_lr' in locals() else 0)


class GradientStabilizationCallback(BaseCallback):
    """梯度稳定化回调"""
    
    def __init__(self, max_grad_norm: float = 0.5, verbose: int = 0):
        super().__init__(verbose)
        self.max_grad_norm = max_grad_norm
        self.grad_norms = []
        
    def _on_step(self) -> bool:
        """每一步调用"""
        
        # 收集梯度范数
        if hasattr(self.model.policy, 'actor'):
            actor_grad_norm = 0.0
            for param in self.model.policy.actor.parameters():
                if param.grad is not None:
                    actor_grad_norm += param.grad.data.norm(2).item() ** 2
            actor_grad_norm = actor_grad_norm ** 0.5
            self.grad_norms.append(actor_grad_norm)
            
            # 记录梯度统计
            if len(self.grad_norms) >= 100:
                avg_grad_norm = np.mean(self.grad_norms[-100:])
                self.logger.record("gradients/actor_norm_avg", avg_grad_norm)
                
                # 如果梯度过大，输出警告
                if avg_grad_norm > 10.0:
                    print(f"⚠️  检测到大梯度: {avg_grad_norm:.3f}")
        
        return True
'''
    
    # 写入文件
    with open(callback_file, 'w', encoding='utf-8') as f:
        f.write(callback_code)
    
    print(f"✅ 已创建增强回调: {callback_file}")
    return True


def create_test_script():
    """创建测试脚本验证修复效果"""
    
    test_file = Path("test_spinning_fix.py")
    
    test_code = '''"""
测试原地打转修复效果
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.environments.navigation_env import ROSbotNavigationEnv  
from src.models.td3_robust import ImprovedTD3
from src.training.spinning_fix_callback import AntiSpinningCallback, GradientStabilizationCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np


def test_spinning_fix(cargo_type='normal', test_steps=2000):
    """测试原地打转修复效果"""
    
    print(f"🧪 测试 {cargo_type} 货物类型的原地打转修复...")
    
    # 创建环境
    env = ROSbotNavigationEnv(cargo_type=cargo_type)
    env = Monitor(env)
    
    # 创建回调
    anti_spinning_cb = AntiSpinningCallback(verbose=1)
    gradient_cb = GradientStabilizationCallback(verbose=1)
    
    # 创建模型
    model = ImprovedTD3(
        policy='MlpPolicy',
        env=env,
        learning_rate=1e-4,  # 使用修复后的较低学习率
        verbose=1,
        tensorboard_log=f"./test_logs/{cargo_type}"
    )
    
    print(f"🚀 开始测试训练 ({test_steps} 步)...")
    
    # 开始训练
    model.learn(
        total_timesteps=test_steps,
        callback=[anti_spinning_cb, gradient_cb],
        progress_bar=True
    )
    
    # 评估结果
    print(f"✅ 测试完成!")
    print(f"📊 原地打转检测次数: {anti_spinning_cb.spinning_episodes}")
    
    if anti_spinning_cb.spinning_episodes < test_steps // 500:  # 期望少于0.2%的步数
        print("🎉 修复效果良好 - 原地打转明显减少!")
    else:
        print("⚠️  仍需进一步调优")
    
    env.close()
    return anti_spinning_cb.spinning_episodes


if __name__ == "__main__":
    # 测试三种货物类型
    cargo_types = ['normal', 'fragile', 'dangerous']
    
    for cargo_type in cargo_types:
        spinning_count = test_spinning_fix(cargo_type, test_steps=1000)
        print(f"{cargo_type}: {spinning_count} 次原地打转\\n")
'''
    
    # 写入文件
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print(f"✅ 已创建测试脚本: {test_file}")
    return True


def apply_all_fixes():
    """应用所有修复"""
    
    print("🔧 开始应用原地打转问题修复...")
    print("="*60)
    
    # 1. 备份原始文件
    print("1. 备份原始文件...")
    backup_dir = backup_files()
    
    # 2. 修复奖励函数
    print("\\n2. 修复奖励函数...")
    fix_reward_function()
    
    # 3. 修复梯度问题
    print("\\n3. 添加梯度裁剪...")
    fix_gradient_issues()
    
    # 4. 修复训练配置
    print("\\n4. 优化训练配置...")
    fix_training_config()
    
    # 5. 创建增强回调
    print("\\n5. 创建增强训练回调...")
    create_enhanced_callback()
    
    # 6. 创建测试脚本
    print("\\n6. 创建测试脚本...")
    create_test_script()
    
    print("\\n" + "="*60)
    print("✅ 所有修复已应用完成!")
    print(f"📁 原始文件备份位置: {backup_dir}")
    print("\\n📋 下一步:")
    print("1. 运行 'python test_spinning_fix.py' 测试修复效果")
    print("2. 如果满意，可以开始正式训练")
    print("3. 如果需要回滚，从备份目录恢复原始文件")
    print("="*60)


if __name__ == "__main__":
    apply_all_fixes()
