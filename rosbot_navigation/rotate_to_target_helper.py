"""
旋转到目标辅助函数 - 参考成功代码实现
"""

import math
import numpy as np


def rotate_to_target_method(self):
    """
    旋转机器人朝向目标 - 参考成功代码的实现
    在episode开始时使用
    """
    print("🎯 正在旋转机器人朝向目标...")
    
    # 获取当前观察
    obs = self._get_observation()
    
    # 计算到目标的角度 (观察的最后一个元素是角度)
    if len(obs) >= 2:
        angle_to_target = float(obs[-1])  # 最后一个元素是角度
        
        rotation_steps = 0
        max_rotation_steps = 50  # 防止无限循环
        
        # 当角度偏差大于0.1弧度时继续旋转
        while abs(angle_to_target) >= 0.1 and rotation_steps < max_rotation_steps:
            # 决定旋转方向
            rotation_speed = 2.0 if angle_to_target > 0 else -2.0
            
            # 执行纯旋转 (线速度=0, 角速度=rotation_speed)
            if hasattr(self, 'fl_motor') and self.fl_motor:
                # 差分驱动：左轮和右轮反向旋转
                wheel_base = 0.053  # 轮距
                angular_to_wheel = wheel_base / 2.0 * rotation_speed
                
                # 左轮
                self.fl_motor.setVelocity(-angular_to_wheel)
                if hasattr(self, 'rl_motor') and self.rl_motor:
                    self.rl_motor.setVelocity(-angular_to_wheel)
                
                # 右轮
                self.fr_motor.setVelocity(angular_to_wheel)
                if hasattr(self, 'rr_motor') and self.rr_motor:
                    self.rr_motor.setVelocity(angular_to_wheel)
            
            # 执行一步仿真
            self.supervisor.step(self.timestep)
            
            # 重新获取观察和角度
            obs = self._get_observation()
            if len(obs) >= 2:
                angle_to_target = float(obs[-1])
            
            rotation_steps += 1
            
            # 调试输出
            if rotation_steps % 10 == 0:
                print(f"   旋转中... 当前角度偏差: {angle_to_target:.3f}rad ({rotation_steps}步)")
        
        # 停止旋转
        if hasattr(self, 'fl_motor') and self.fl_motor:
            self.fl_motor.setVelocity(0.0)
            if hasattr(self, 'fr_motor') and self.fr_motor:
                self.fr_motor.setVelocity(0.0)
            if hasattr(self, 'rl_motor') and self.rl_motor:
                self.rl_motor.setVelocity(0.0)
            if hasattr(self, 'rr_motor') and self.rr_motor:
                self.rr_motor.setVelocity(0.0)
        
        # 执行一步以应用停止
        self.supervisor.step(self.timestep)
        
        final_obs = self._get_observation()
        final_angle = float(final_obs[-1]) if len(final_obs) >= 2 else angle_to_target
        
        print(f"✅ 旋转完成！最终角度偏差: {final_angle:.3f}rad (用时{rotation_steps}步)")
        
        return True
    
    print("⚠️  无法获取角度信息，跳过旋转")
    return False


# 将方法添加到环境类的辅助函数
def add_rotate_to_target_method(env_class):
    """
    将旋转到目标的方法添加到环境类
    """
    env_class._rotate_to_target = rotate_to_target_method
    return env_class
