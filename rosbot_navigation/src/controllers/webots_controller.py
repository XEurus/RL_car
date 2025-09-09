"""
Webots控制器基类
专用的ROSbot控制器，支持与AMCL定位的集成
"""

from controller import Robot, Supervisor
import numpy as np
from typing import Dict, List, Optional
import time
import math


class ROSbotWebotsController:
    """ROSbot专用的Webots控制器基类"""
    
    def __init__(self, robot_node_name: str = 'ROSbot'):
        """初始化控制器"""
        self.robot = Robot()
        self.supervisor = Supervisor()
        
        # 基本时间步
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # 初始化传感器
        self._initialize_sensors()
        
        # 初始化执行器
        self._initialize_actuators()
        
        # 状态缓存
        self.state_buffer = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0],
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'left_wheel_speed': 0.0,
            'right_wheel_speed': 0.0
        }
        
        # 时间相关
        self.previous_time = self.robot.getTime()
        
        # 机器人参数
        self.robot_params = {
            'wheel_base': 0.235,     # 轴距 (m)
            'wheel_radius': 0.08,    # 轮径 (m) 
            'max_linear_velocity': 2.0,      # 最大线速度 (m/s)
            'max_angular_velocity': 2.0,     # 最大角速度 (rad/s)
            'max_wheel_speed': 6.28   # 电机最大角速度 (rad/s)
        }
    
    def _initialize_sensors(self):
        """初始化所有传感器设备"""
        self.sensors = {}
        
        # LiDAR传感器
        self.sensors['lidar'] = self.robot.getDevice('lidar')
        if self.sensors['lidar']:
            self.sensors['lidar'].enablePointCloud()
            self.sensors['lidar'].enable(self.timestep)
        
        # GPS位置传感器
        self.sensors['gps'] = self.robot.getDevice('gps')
        if self.sensors['gps']:
            self.sensors['gps'].enable(self.timestep)
        
        # IMU惯性单元
        self.sensors['imu'] = self.robot.getDevice('inertial unit')
        if self.sensors['imu']:
            self.sensors['imu'].enable(self.timestep)
        
        # 陀螺仪
        self.sensors['gyro'] = self.robot.getDevice('gyro')
        if self.sensors['gyro']:
            self.sensors['gyro'].enable(self.timestep)
        
        # 罗盘
        self.sensors['compass'] = self.robot.getDevice('compass')
        if self.sensors['compass']:
            self.sensors['compass'].enable(self.timestep)
        
        # 碰撞检测
        self.sensors['touch'] = self.robot.getDevice('touch sensor')
        if self.sensors['touch']:
            self.sensors['touch'].enable(self.timestep)
        
        # 获取机器人节点（用于获取真实状态）
        self.robot_node = self.supervisor.getFromDef('ROSbot')
    
    def _initialize_actuators(self):
        """初始化执行器"""
        self.actuators = {}
        
        # 左右轮电机
        self.actuators['left_motor'] = self.robot.getDevice('left wheel motor')
        self.actuators['right_motor'] = self.robot.getDevice('right wheel motor')
        
        # 设置为速度控制模式
        if self.actuators['left_motor']:
            self.actuators['left_motor'].setPosition(float('inf'))
            self.actuators['left_motor'].setVelocity(0.0)
        
        if self.actuators['right_motor']:
            self.actuators['right_motor'].setPosition(float('inf'))
            self.actuators['right_motor'].setVelocity(0.0)
    
    def get_lidar_data(self) -> np.ndarray:
        """获取LiDAR数据"""
        if not self.sensors.get('lidar'):
            return np.zeros(20, dtype=np.float32)
        
        ranges = self.sensors['lidar'].getRangeImage()
        if ranges and len(ranges) >= 20:
            # 均匀采样20个点
            indices = np.linspace(0, len(ranges)-1, 20, dtype=int)
            data = np.array([ranges[i] for i in indices])
            # 限制范围和归一化
            data = np.clip(data, 0.01, 10.0) / 10.0
            return data.astype(np.float32)
        return np.zeros(20, dtype=np.float32)
    
    def get_gps_position(self) -> np.ndarray:
        """获取GPS位置"""
        if not self.sensors.get('gps'):
            return np.array([0.0, 0.0, 0.0])
        
        gps_values = self.sensors['gps'].getValues()
        if gps_values and len(gps_values) >= 3:
            return np.array(gps_values[:3])
        return np.array([0.0, 0.0, 0.0])
    
    def get_imu_orientation(self) -> np.ndarray:
        """获取IMU姿态"""
        if not self.sensors.get('imu'):
            return np.array([0.0, 0.0, 0.0])
        
        imu_values = self.sensors['imu'].getRollPitchYaw()
        if imu_values and len(imu_values) >= 3:
            return np.array(imu_values)
        return np.array([0.0, 0.0, 0.0])
    
    def get_gyro_data(self) -> np.ndarray:
        """获取陀螺仪数据"""
        if not self.sensors.get('gyro'):
            return np.array([0.0, 0.0, 0.0])
        
        gyro_values = self.sensors['gyro'].getValues()
        if gyro_values and len(gyro_values) >= 3:
            return np.array(gyro_values)
        return np.array([0.0, 0.0, 0.0])
    
    def get_compass_data(self) -> np.ndarray:
        """获取罗盘数据"""
        if not self.sensors.get('compass'):
            return np.array([0.0, 0.0, 0.0])
        
        compass_values = self.sensors['compass'].getValues()
        if compass_values and len(compass_values) >= 3:
            return np.array(compass_values)
        return np.array([0.0, 0.0, 0.0])
    
    def get_touch_sensor_value(self) -> float:
        """获取碰撞传感器值"""
        if not self.sensors.get('touch'):
            return 0.0
        
        return self.sensors['touch'].getValue()
    
    def get_robot_true_state(self) -> Dict:
        """获取机器人真实状态（来自Supervisor）"""
        if not self.robot_node:
            return {}
        
        # 获取真实位置和旋转
        translation = self.robot_node.getField('translation').getSFVec3f()
        rotation = self.robot_node.getField('rotation').getSFRotation()
        
        return {
            'position': translation,
            'rotation': rotation,
            'timestamp': self.robot.getTime()
        }
    
    def get_wheel_speeds(self) -> Tuple[float, float]:
        """获取当前轮速"""
        left_speed = 0.0
        right_speed = 0.0
        
        if self.actuators.get('left_motor'):
            left_speed = self.actuators['left_motor'].getVelocity()
        
        if self.actuators.get('right_motor'):
            right_speed = self.actuators['right_motor'].getVelocity()
        
        return left_speed, right_speed
    
    def calculate_wheel_velocity(self, left_speed: float, right_speed: float) -> Tuple[float, float]:
        """计算车体速度（正运动学）"""
        # 逆运动学计算
        linear_vel = (left_speed + right_speed) * self.robot_params['wheel_radius'] / 2.0
        angular_vel = (right_speed - left_speed) * self.robot_params['wheel_radius'] / self.robot_params['wheel_base']
        
        return linear_vel, angular_vel
    
    def calculate_wheel_command(self, linear_vel: float, angular_vel: float) -> Tuple[float, float]:
        """计算轮速命令（逆运动学）"""
        # 约束速度
        linear_vel = np.clip(linear_vel, -self.robot_params['max_linear_velocity'], 
                           self.robot_params['max_linear_velocity'])
        angular_vel = np.clip(angular_vel, -self.robot_params['max_angular_velocity'], 
                             self.robot_params['max_angular_velocity'])
        
        # 逆运动学
        left_wheel_speed = (linear_vel - angular_vel * self.robot_params['wheel_base'] / 2) / self.robot_params['wheel_radius']
        right_wheel_speed = (linear_vel + angular_vel * self.robot_params['wheel_base'] / 2) / self.robot_params['wheel_radius']
        
        # 约束轮速
        left_wheel_speed = np.clip(left_wheel_speed, -self.robot_params['max_wheel_speed'], 
                                  self.robot_params['max_wheel_speed'])
        right_wheel_speed = np.clip(right_wheel_speed, -self.robot_params['max_wheel_speed'], 
                                   self.robot_params['max_wheel_speed'])
        
        return left_wheel_speed, right_wheel_speed
    
    def set_wheel_velocities(self, left_speed: float, right_speed: float):
        """设置轮速"""
        # 速度限制
        left_speed = np.clip(left_speed, -self.robot_params['max_wheel_speed'], self.robot_params['max_wheel_speed'])
        right_speed = np.clip(right_speed, -self.robot_params['max_wheel_speed'], self.robot_params['max_wheel_speed'])
        
        # 设置速度
        if self.actuators.get('left_motor'):
            self.actuators['left_motor'].setVelocity(left_speed)
        
        if self.actuators.get('right_motor'):
            self.actuators['right_motor'].setVelocity(right_speed)
    
    def step_sensors(self):
        """执行一次传感器步进"""
        for name, sensor in self.sensors.items():
            if sensor:
                sensor.step(self.timestep)
    
    def get_all_sensor_data(self) -> Dict:
        """获取所有传感器数据"""
        current_time = self.robot.getTime()
        
        return {
            'timestamp': current_time,
            'lidar': self.get_lidar_data(),
            'gps_position': self.get_gps_position(),
            'imu_orientation': self.get_imu_orientation(),
            'gyro_data': self.get_gyro_data(),
            'compass_data': self.get_compass_data(),
            'touch_sensor': self.get_touch_sensor_value(),
            'true_state': self.get_robot_true_state(),
            'wheel_speeds': self.get_wheel_speeds()
        }
    
    def reset_actuators(self):
        """重置所有执行器"""
        self.set_wheel_velocities(0.0, 0.0)
    
    def get_controller_info(self) -> Dict:
        """获取控制器信息"""
        return {
            'timestep': self.timestep,
            'current_time': self.robot.getTime(),
            'sensors_enabled': [name for name, sensor in self.sensors.items() if sensor],
            'actuators_enabled': [name for name, actuator in self.actuators.items() if actuator],
            'robot_params': self.robot_params.copy()
        }


class AMCLWebotsInterface:
    """AMCL与Webots的接口层"""
    
    def __init__(self, webots_controller: ROSbotWebotsController):
        self.controller = webots_controller
        
        # 状态缓存
        self.previous_odometry = {'dx': 0.0, 'dy': 0.0, 'dyaw': 0.0}
        self.previous_position = np.zeros(3)
        self.previous_time = 0.0
        
    def get_observation_for_amcl(self) -> Dict:
        """获取用于AMCL的观测数据"""
        sensor_data = self.controller.get_all_sensor_data()
        current_time = sensor_data['timestamp']
        
        # 计算里程计
        current_pos = sensor_data['true_state'].get('position', [0.0, 0.0, 0.0])
        current_pos = np.array(current_pos[:2])  # 只取x,y
        
        if self.previous_time > 0 and current_time > self.previous_time:
            dt = current_time - self.previous_time
            if dt > 0:
                dx = (current_pos[0] - self.previous_position[0]) / dt
                dy = (current_pos[1] - self.previous_position[1]) / dt
            else:
                dx, dy = 0.0, 0.0
        else:
            dx, dy = 0.0, 0.0
        
        # 计算角度变化
        orientation = sensor_data['imu_orientation']
        current_yaw = orientation[2] if len(orientation) > 2 else 0.0
        dyaw = current_yaw - self.previous_odometry.get('current_yaw', 0.0)
        
        # 更新缓存
        self.previous_odometry = {
            'dx': dx,
            'dy': dy,
            'dyaw': dyaw,
            'current_yaw': current_yaw,
            'left_speed': sensor_data['wheel_speeds'][0],
            'right_speed': sensor_data['wheel_speeds'][1]
        }
        
        self.previous_position = current_pos.copy()
        self.previous_time = current_time
        
        return {
            'lidar_scan': sensor_data['lidar'],
            'odometry': self.previous_odometry,
            'current_pose': sensor_data['true_state'].get('position', [0.0, 0.0, 0.0])
        }
    
    def apply_amcl_correction(self, amcl_result: Dict) -> Dict:
        """应用AMCL校正结果"""
        if not amcl_result:
            return {'status': 'no_correction', 'reason': 'AMCL_result_none'}
        
        corrected_pose = {
            'position': amcl_result.get('position_estimated', [0.0, 0.0, 0.0]),
            'orientation': amcl_result.get('orientation_estimated', [0.0, 0.0, 0.0]),
            'uncertainty': amcl_result.get('position_uncertainty', [0.1, 0.1, 0.1]),
            'convergence': amcl_result.get('convergence_score', False)
        }
        
        return {
            'status': 'correction_applied',
            'corrected_pose': corrected_pose,
            'original_pose': amcl_result.get('original_pose', None)
        }


if __name__ == "__main__":
    """控制器基类测试"""
    print("ROSbot Webots控制器基类")
    print("这是一个基类模块，请在具体环境中使用")