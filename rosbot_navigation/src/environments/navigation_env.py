"""
ROSbot导航环境 - 支持AMCL定位的42维状态空间
基于Webots仿真环境和真实定位算法
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import os
from typing import Dict, Tuple, List, Optional
from collections import deque

from ..localization.amcl_localizer import AMCLLocalizer
from ..localization.pose_estimator import RobustPoseEstimator
from ..utils.navigation_utils import NavigationUtils
from ..utils.reward_functions import RewardFunctions
from controller import Supervisor, GPS, InertialUnit, Gyro, Compass
WEBOTS_AVAILABLE = True


class ROSbotNavigationEnv(gym.Env):
    """
    ROSbot导航环境类
    
    状态空间：42维
    - LiDAR数据 (20维)
    - 机器人位姿状态 (12维) 
    - 目标导航信息 (6维)
    - 航向控制信息 (4维)
    
    动作空间：连续2维 [线速度, 角速度]
    """
    @staticmethod
    def get_spaces():
        """无需连接 Webots，返回与环境一致的观测/动作空间定义"""
        pi = math.pi
        obs_low = np.array(
            [0.0] * 20 +
            [-20.0, -20.0, -2.0] +
            [-pi, -pi, -pi] +
            [-2.0, -2.0, -2.0] +
            [-2.0, -2.0] +
            [-1.0] +
            [-20.0, -20.0, -2.0] +
            [-20.0, -20.0, -2.0] +
            [-pi] +
            [-pi] +
            [-2.0] +
            [-1.0],
            dtype=np.float32
        )
        obs_high = np.array(
            [1.0] * 20 +
            [20.0, 20.0, 2.0] +
            [pi, pi, pi] +
            [2.0, 2.0, 2.0] +
            [2.0, 2.0] +
            [1.0] +
            [20.0, 20.0, 2.0] +
            [20.0, 20.0, 2.0] +
            [pi] +
            [pi] +
            [2.0] +
            [1.0],
            dtype=np.float32
        )
        obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        return obs_space, act_space
    
    def __init__(self, 
                 cargo_type: str = 'normal', 
                 instance_id: Optional[int] = None,
                 controller_url: Optional[str] = None,
                 fast_mode: bool = True,
                 control_period_ms: int = 200):
        super(ROSbotNavigationEnv, self).__init__()
        
        # 货物类型
        self.cargo_type = cargo_type
        self.is_training = True
        self.instance_id = instance_id if instance_id is not None else 0
        self.control_period_ms = int(control_period_ms) if control_period_ms and control_period_ms > 0 else 200
        
        # 启用初始朝向目标
        self._rotate_to_target_on_reset = True
        
        # 42维状态空间定义
        pi = math.pi
        self.observation_space = spaces.Box(
            low=np.array(
                # LiDAR数据 (20维) - 归一化后
                [0.0] * 20 +
                # 机器人位姿状态 (12维)
                [-20.0, -20.0, -2.0] +  # 位置(3)
                [-pi, -pi, -pi] +       # 姿态(3)
                [-2.0, -2.0, -2.0] +    # 速度(3)
                [-2.0, -2.0] +          # 上一时刻速度(2)
                [-1.0] +                # 加速度(1)
                # 目标导航信息 (6维)
                [-20.0, -20.0, -2.0] +  # 目标相对位置(3)
                [-20.0, -20.0, -2.0] +  # 起点位置(3)
                # 航向控制 (4维)
                [-pi] +                 # 航向偏差(1)
                [-pi] +                 # 目标航向角(1)
                [-2.0] +                # 角速度(1)
                [-1.0]                  # 角加速度(1)
            ),
            high=np.array(
                # LiDAR数据 (20维) - 归一化后
                [1.0] * 20 +
                # 机器人位姿状态 (12维)
                [20.0, 20.0, 2.0] +     # 位置(3)
                [pi, pi, pi] +          # 姿态(3)
                [2.0, 2.0, 2.0] +       # 速度(3)
                [2.0, 2.0] +            # 上一时刻速度(2)
                [1.0] +                 # 加速度(1)
                # 目标导航信息 (6维)
                [20.0, 20.0, 2.0] +     # 目标相对位置(3)
                [20.0, 20.0, 2.0] +     # 起点位置(3)
                # 航向控制 (4维)
                [pi] +                  # 航向偏差(1)
                [pi] +                  # 目标航向角(1)
                [2.0] +                 # 角速度(1)
                [1.0]                   # 角加速度(1)
            ),
            dtype=np.float32
        )
        
        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),    # 线速度[0-1], 角速度[-1,1] (归一化值，实际会乘以系数)
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Webots控制器 - 使用兼容性层
        # 允许通过 controller_url 连接到特定的 Webots 实例（外部控制器）
        
        # 设置最大连接重试次数（在并行实例较多时适当增加重试和间隔）
        max_tries = 5
        retry_delay = 3  # 秒
        connected = False
        import time
        
        for try_num in range(max_tries):
            try:
                if controller_url:
                    print(f"🔌 实例 {self.instance_id} 连接到 Webots ({try_num+1}/{max_tries}): {controller_url}")
                    # 解析URL并设置环境变量
                    if controller_url.startswith('tcp://'):
                        # TCP连接格式：tcp://localhost:1234
                        import urllib.parse
                        parsed = urllib.parse.urlparse(controller_url)
                        host = parsed.hostname or 'localhost'
                        port = parsed.port or 10000 + (self.instance_id * 100)
                        
                        # 设置Webots的连接参数
                        os.environ['WEBOTS_SERVER'] = host
                        os.environ['WEBOTS_PORT'] = str(port)
                        # 同时设置标准的控制器URL，确保选择到正确的机器人（包含?name=...时生效）
                        os.environ['WEBOTS_CONTROLLER_URL'] = str(controller_url)
                        print(f"   设置连接: {host}:{port}")
                    else:
                        # 其他格式直接设置
                        os.environ['WEBOTS_CONTROLLER_URL'] = str(controller_url)
                else:
                    print(f"🔌 实例 {self.instance_id} 使用默认 Webots 连接 ({try_num+1}/{max_tries})")
                
                print(f"🤖 实例 {self.instance_id} 初始化 Supervisor...")
                self.robot = Supervisor()
                self.supervisor = self.robot
                print(f"✅ 实例 {self.instance_id} Supervisor 初始化成功")
                connected = True
                break
                
            except Exception as e:
                print(f"⚠️ 实例 {self.instance_id} 连接失败 ({try_num+1}/{max_tries}): {e}")
                if try_num < max_tries - 1:
                    print(f"⏳ 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    # 尝试更改URL格式
                    if controller_url and '?name=' not in controller_url and 'tcp://' in controller_url:
                        controller_url = f"{controller_url}?name=rosbot"
                        print(f"🔄 修改URL格式: {controller_url}")
        
        if not connected:
            print(f"❌ 实例 {self.instance_id} 多次尝试后仍无法连接")
            raise ConnectionError(f"无法连接到 Webots 实例 {self.instance_id}")
            
        self.timestep = int(self.supervisor.getBasicTimeStep())
        # 切换到 FAST 模式（若可用）
        try:
            if fast_mode and hasattr(self.supervisor, 'simulationSetMode') and hasattr(Supervisor, 'SIMULATION_MODE_FAST'):
                self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)
        except Exception:
            pass
        
        # 传感器设备
        self._setup_sensors()
        
        # 执行器设备  
        self._setup_actuators()
        
        # AMCL定位器
        # self.amcl_localizer = AMCLLocalizer(
        #     num_particles=800,
        #     initial_std=[0.2, 0.2, 0.15]
        # )
        
        # 辅助定位器（备用/融合）
        
        self.pose_estimator = RobustPoseEstimator(self.supervisor)
        
        # 导航工具
        self.nav_utils = NavigationUtils()
        
        # 状态变量
        self.state_buffer = {
            'previous_position': np.zeros(3),
            'previous_velocity': np.zeros(3),
            'previous_orientation': np.zeros(3),
            'previous_time': 0.0
        }
        
        # 初始化AMCL结果
        self.amcl_result = {
            'position_estimated': np.zeros(3),
            'orientation_estimated': np.zeros(3),
            'velocity_estimated': np.zeros(3),
            'position_uncertainty': 0.1
        }
        
        # 任务信息
        self.task_info = {
            'start_pos': np.zeros(3),
            'target_pos': np.zeros(3),
            'cargo_type': cargo_type
        }
        
        # 训练参数
        self.max_steps_per_episode = 300
        self.collision_threshold = 0.05
        self.success_threshold = 0.25  # 放宽到25cm，更容易成功
        
        # 轨迹跟踪
        self.trajectory = []
        self.min_obstacle_distance = float('inf')

        # 反打转与卡滞检测状态
        self.spin_steps = 0
        self.no_progress_steps = 0
        self.last_distance_to_target = None
        self._stuck = False
        # 旋转/前进阈值（单位：m/s 与 rad/s）
        self.spin_linear_speed_threshold = 0.05
        self.spin_angular_speed_threshold = 1.0

        # 探索/移动奖励与角度相关状态
        self.episode_steps = 0
        
        # 奖励函数实例
        self.reward_functions = RewardFunctions()
    
    def _setup_sensors(self):
        """初始化传感器设备"""
    
        # LiDAR传感器 - 在proto文件中名称为"laser"
        self.lidar = self.supervisor.getDevice('laser')
        if self.lidar:
            self.lidar.enablePointCloud()
            self.lidar.enable(self.timestep)
        
        # 直接通过supervisor获取机器人位置
        self.robot_node = self.supervisor.getSelf()
        
        # # IMU/姿态传感器 - 在proto文件中名称为"imu"
        # self.inertial_unit = self.supervisor.getDevice('imu')
        # if self.inertial_unit:
        #     self.inertial_unit.enable(self.timestep)
        # IMU设备无法获取，使用supervisor功能代替
        self.inertial_unit = None
        
        # 使用IMU内置的陀螺仪功能
        # self.gyro = self.inertial_unit  # IMU已经包含陀螺仪功能        
        # 使用supervisor功能代替陀螺仪
        self.gyro = None
        
        # 使用距离传感器进行碰撞检测
        # 获取前方距离传感器
        self.collision_sensors = []
        sensor_names = ['fl_range', 'fr_range', 'rl_range', 'rr_range']
        for name in sensor_names:
            sensor = self.supervisor.getDevice(name)
            if sensor:
                sensor.enable(self.timestep)
                self.collision_sensors.append(sensor)
        
        # 设置碰撞检测阈值（单位：米）
        self.collision_distance_threshold = 0.05

        # 用于基于节点识别的碰撞检测
        # 从 rosbot.proto 和 warehouse2.wbt 文件中提取的名称
        self.wheel_defs = {'front left wheel', 'front right wheel', 'rear left wheel', 'rear right wheel'}
        self.ground_defs = {'floor'}
    
    def _setup_actuators(self):
        """初始化执行器设备"""
        # 获取机器人节点
        self.robot_node = self.supervisor.getFromDef('rosbot')
        
        # 根据proto文件中的电机名称获取电机
        self.fl_motor = self.supervisor.getDevice('fl_wheel_joint')
        self.fr_motor = self.supervisor.getDevice('fr_wheel_joint')
        self.rl_motor = self.supervisor.getDevice('rl_wheel_joint')
        self.rr_motor = self.supervisor.getDevice('rr_wheel_joint')
        
        # 为了与原代码兼容，定义左右电机（使用前轮）
        self.left_motor = self.fl_motor
        self.right_motor = self.fr_motor
        
        # 设置电机模式
        for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
        
        # 读取电机最大速度以进行限幅（若获取失败则回退到26.0rad/s）
        try:
            speeds = []
            for motor in [self.fl_motor, self.fr_motor, self.rl_motor, self.rr_motor]:
                if motor:
                    speeds.append(motor.getMaxVelocity())
            self.max_motor_speed = float(min(speeds)) if speeds else 26.0
        except Exception:
            self.max_motor_speed = 26.0

        # 保存底盘参数，避免硬编码分散
        self.wheel_base = 0.22
        self.wheel_radius = 0.043

        # 轮速平滑缓存，避免瞬时大跃迁导致不稳定
        self._prev_left_speed = 0.0
        self._prev_right_speed = 0.0
    
    def reset(self, seed=None, options=None):
        """重置环境和状态"""
        super().reset(seed=seed)
        
        # 设置新的导航任务
        self._set_navigation_task()
        
        # 重置AMCL定位器
        # self.amcl_localizer.reset()
        
        # 重置状态缓存
        self._reset_state_buffer()
        
        # 重置轨迹记录
        self.trajectory = []
        self.min_obstacle_distance = float('inf')
        
        # 初始化速度变量，用于平滑控制
        self.last_linear_vel = 0.0
        self.last_angular_vel = 0.0
        self._last_cmd_linear_vel = 0.0
        self._last_cmd_angular_vel = 0.0
        self.spin_steps = 0
        self.no_progress_steps = 0
        self.last_distance_to_target = None
        self._stuck = False
        self.episode_steps = 0

        # 重置奖励函数状态
        _pos = self._get_sup_position()
        _orient = self._get_sup_orientation()
        self.reward_functions.reset(_pos, _orient)
        
        # 如果电机存在，重置电机速度为0
        if hasattr(self, 'fl_motor') and self.fl_motor:
            self.fl_motor.setVelocity(0.0)
        if hasattr(self, 'fr_motor') and self.fr_motor:
            self.fr_motor.setVelocity(0.0)
        if hasattr(self, 'rl_motor') and self.rl_motor:
            self.rl_motor.setVelocity(0.0)
        if hasattr(self, 'rr_motor') and self.rr_motor:
            self.rr_motor.setVelocity(0.0)
        
        # 等待一小段时间，确保机器人完全停止
        # 增加等待时间，确保初始化完全
        for _ in range(5):
            self.supervisor.step(self.timestep)
        
        # 机器人初始朝向目标 (可选)
        if hasattr(self, '_rotate_to_target_on_reset') and self._rotate_to_target_on_reset:
            self._rotate_to_target()
        
        # 获取初始观察
        observation = self._get_observation()
        
        # 设置初始AMCL状态
        # self._initialize_amcl_state()
        
        info = {
            'start_position': self.task_info['start_pos'].copy(),
            'target_position': self.task_info['target_pos'].copy(),
            'cargo_type': self.cargo_type
        }
        
        return observation, info
    
    def step(self, action):
        """执行动作并返回新状态"""
        # 执行动作
        self._execute_action(action)
        # 更新奖励函数中的动作历史
        self.reward_functions.update_action_history(action)
        # 更新当前旋转（使用命令角速度作为近似）
        self.reward_functions.update_current_rotation(float(getattr(self, '_last_cmd_angular_vel', 0.0)))
        # 累计步数
        self.episode_steps = int(self.episode_steps) + 1
        
        # Step仿真 - 为了适应控制频率，确保每次step足够的时间
        # control_period_ms（默认200ms），timestep可能是32ms，所以需要多次step
        steps_per_control = max(1, int(self.control_period_ms / self.timestep))
        for _ in range(steps_per_control):
            self.supervisor.step(self.timestep)
        
        # 获取新观察
        observation = self._get_observation()
        
        # 更新AMCL状态
        # self._update_amcl_state()
        
        # 计算奖励
        reward = self._calculate_reward(action, observation)
        
        # 检查终止条件
        terminated = self._check_termination()
        
        # 检查截断条件
        truncated = self._check_truncation()
        
        # 更新状态信息
        info = self._get_step_info()
        
        # 记录机器人位置到轨迹
        if self.robot_node:
            try:
                current_pos = self._get_sup_position()
                self.trajectory.append(current_pos)
                # 打印当前位置，用于调试
                # print(f"机器人位置: {current_pos}")
            except Exception as e:
                print(f"记录轨迹错误: {e}")
        
        return observation, reward, terminated, truncated, info

    def _calculate_distance_to_target(self):
        """计算到当前目标的距离及目标位置"""
        current_pos = self._get_sup_position()
        target_pos = self.task_info['target_pos']
        distance = float(np.linalg.norm(current_pos - target_pos))
        return distance, target_pos

    def _calculate_angle_to_target(self, target_pos):
        """计算当前航向与目标方向的角度偏差（-π, π）"""
        current_pos = self._get_sup_position()
        current_orient = self._get_sup_orientation()
        vec_to_target = target_pos - current_pos
        target_heading = math.atan2(float(vec_to_target[1]), float(vec_to_target[0]))
        current_heading = float(current_orient[2])
        angle = target_heading - current_heading
        angle = math.atan2(math.sin(angle), math.cos(angle))
        return angle

    def _get_lidar_features(self):
        """获取用于奖励的LiDAR特征（0-1，约对应0-10m归一化）"""
        data = self._get_lidar_data()
        return data if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
    
    def _set_navigation_task(self):
        """设置导航任务"""
        # 根据货物类型获取起点和目标位置
        start_pos, target_pos = self.nav_utils.get_navigation_task(self.cargo_type)
        
        self.task_info['start_pos'] = np.array(start_pos, dtype=np.float32)
        self.task_info['target_pos'] = np.array(target_pos, dtype=np.float32)
        
        # 重置机器人位置
        self._reset_robot_position(start_pos)
    
    def _reset_robot_position(self, position):
        """重置机器人位置"""
        if self.robot_node:
            self.robot_node.getField('translation').setSFVec3f(list(position))
            self.robot_node.getField('rotation').setSFRotation([0, 1, 0, 0])
            # 清零机器人及其子节点的物理状态，避免残余速度导致起跳/漂移
            try:
                self.robot_node.resetPhysics()
            except Exception:
                pass
            # 额外执行几个周期以稳定着地
            for _ in range(2):
                self.supervisor.step(self.timestep)
                
    def _rotate_to_target(self):
        """将机器人朝向目标位置 - 使用电机控制自然旋转"""
        if self.robot_node and 'target_pos' in self.task_info:
            # 获取当前位置和目标位置
            current_pos = self._get_sup_position()
            target_pos = self.task_info['target_pos']
            
            # 计算朝向目标的方向向量
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            
            # 计算目标朝向角度（偏航角）
            target_yaw = math.atan2(dy, dx)
            
            # 获取当前朝向
            current_orientation = self._get_sup_orientation()
            current_yaw = current_orientation[2]
            
            # 计算需要旋转的角度（最短路径）
            delta_yaw = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
            

            # 2. 重置物理状态，确保没有残余动量
            try:
                self.robot_node.resetPhysics()
            except Exception:
                pass
            
            # 3. 直接使用平移和旋转重置机器人，确保正确的姿态
            # 获取当前位置
            current_translation = self.robot_node.getField('translation').getSFVec3f()
            
            # 设置新的旋转 - 只改变Y轴旋转（偏航角），保持其他轴为0
            # Webots中，机器人应该是平放在地面上的，所以我们需要保持X和Z轴的旋转为0
            # 标准姿态是[0, 1, 0, angle]，表示绕Y轴旋转angle角度
            new_rotation = [0, 0, 1, target_yaw]
            
            # 重新设置机器人的位置和姿态
            self.robot_node.getField('translation').setSFVec3f(current_translation)
            self.robot_node.getField('rotation').setSFRotation(new_rotation)
            
            
            # 确保电机已初始化
            # if not (hasattr(self, 'fl_motor') and hasattr(self, 'fr_motor') and 
            #         hasattr(self, 'rl_motor') and hasattr(self, 'rr_motor')):
            #     print("警告：电机未初始化，无法进行电机控制旋转")
            #     return
                
            # # 设置旋转速度（根据角度差异调整）
            # rotation_speed = 1.0  # 基础旋转速度
            
            # # 根据旋转方向设置电机速度
            # if delta_yaw > 0:  # 需要逆时针旋转
            #     left_speed = -rotation_speed
            #     right_speed = rotation_speed
            # else:  # 需要顺时针旋转
            #     left_speed = rotation_speed
            #     right_speed = -rotation_speed
                
            # # 旋转精度阈值（弧度）
            # precision_threshold = 0.05
            
            # # 最大旋转时间（防止无限循环）
            # max_rotation_time = 10.0  # 秒
            # start_time = self.supervisor.getTime()
            
            # # 开始旋转
            # while abs(delta_yaw) > precision_threshold:
            #     # 设置电机速度
            #     self.fl_motor.setVelocity(left_speed)
            #     self.rl_motor.setVelocity(left_speed)
            #     self.fr_motor.setVelocity(right_speed)
            #     self.rr_motor.setVelocity(right_speed)
                
            #     # 执行仿真步骤
            #     self.supervisor.step(self.timestep)
                
            #     # 获取当前朝向并计算剩余角度
            #     current_orientation = self._get_sup_orientation()
            #     current_yaw = current_orientation[2]
            #     delta_yaw = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
                
            #     # 检查是否超时
            #     if self.supervisor.getTime() - start_time > max_rotation_time:
            #         print("旋转超时，停止旋转")
            #         break
                    
            #     # 根据剩余角度调整速度（接近目标时减速）
            #     if abs(delta_yaw) < 0.3:
            #         speed_factor = abs(delta_yaw) / 0.3  # 线性减速
            #         speed_factor = max(0.3, speed_factor)  # 保持最小速度
                    
            #         if delta_yaw > 0:  # 需要逆时针旋转
            #             left_speed = -rotation_speed * speed_factor
            #             right_speed = rotation_speed * speed_factor
            #         else:  # 需要顺时针旋转
            #             left_speed = rotation_speed * speed_factor
            #             right_speed = -rotation_speed * speed_factor
            
            # # 旋转完成，停止电机
            # self.fl_motor.setVelocity(0.0)
            # self.rl_motor.setVelocity(0.0)
            # self.fr_motor.setVelocity(0.0)
            # self.rr_motor.setVelocity(0.0)
            
            # 等待物理引擎稳定
            for _ in range(5):
                self.supervisor.step(self.timestep)
                
            # # 最终角度差
            # final_orientation = self._get_sup_orientation()
            # final_delta = math.atan2(math.sin(target_yaw - final_orientation[2]), 
            #                         math.cos(target_yaw - final_orientation[2]))
            
            # print(f"机器人已旋转到目标方向，最终角度差: {math.degrees(final_delta):.2f}°")
    
    def _get_sup_position(self):
        """获取机器人位置（使用supervisor功能获取位置）"""
        if self.robot_node:
            # 使用supervisor API获取机器人位置
            position = self.robot_node.getPosition()
            return np.array(position, dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    
    def _get_sup_orientation(self):
        """获取机器人朝向（使用supervisor API）"""
        # 直接使用supervisor API获取朝向
        if self.robot_node:
            # 从机器人节点获取旋转矩阵
            rotation = self.robot_node.getOrientation()
            # 转换为欧拉角 (roll, pitch, yaw)
            # 从旋转矩阵提取欧拉角
            # 矩阵格式为[r11 r21 r31 r12 r22 r32 r13 r23 r33]
            if len(rotation) >= 9:
                # 简化计算，只关注yaw角（绕z轴旋转）
                # yaw = atan2(r21, r11)
                yaw = math.atan2(rotation[1], rotation[0])
                # 简化roll和pitch计算
                roll = 0.0
                pitch = 0.0
                return np.array([roll, pitch, yaw], dtype=np.float32)
        
        # 如果无法获取，返回零向量
        return np.zeros(3, dtype=np.float32)
    
    def _reset_state_buffer(self):
        """重置状态缓存"""
        # 获取当前位置作为初始状态
        current_pos = self._get_sup_position()
        current_orient = self._get_sup_orientation()
        current_vel = np.zeros(3, dtype=np.float32)
        current_time = self.supervisor.getTime()
        
        self.state_buffer = {
            'previous_position': current_pos.copy(),
            'previous_velocity': current_vel.copy(),
            'previous_orientation': current_orient.copy(),
            'previous_time': current_time
        }
    
    def _initialize_amcl_state(self):
        """初始化AMCL状态"""
        # 获取初始LiDAR数据
        lidar_data = self._get_lidar_data()
        
        # 获取初始里程计数据
        odometry_data = self._get_odometry_data()
        
        # 初始化AMCL
        try:
            self.amcl_localizer.initialize_with_pose(
                self.task_info['start_pos'],
                [0.0, 0.0, 0.0]  # 初始朝向0
            )
            
            # 执行初始定位
            result = self.amcl_localizer.localize(lidar_data, odometry_data)
            if result is not None and isinstance(result, dict):
                # 确保所有必要的键都存在
                required_keys = ['position_estimated', 'orientation_estimated', 'velocity_estimated']
                if all(key in result for key in required_keys):
                    self.amcl_result = result
        except Exception as e:
            print(f"AMCL初始化失败: {e}")
            # 使用默认值
            self.amcl_result = {
                'position_estimated': self.task_info['start_pos'].copy(),
                'orientation_estimated': np.array([0.0, 0.0, 0.0]),
                'velocity_estimated': np.zeros(3),
                'position_uncertainty': 0.1
            }
    
    def _execute_action(self, action):
        """执行动作"""
        
        # 预测性碰撞避免
        # if action[0] > 0.3:  # 如果试图显著前进
        #     lidar_features = self._get_lidar_features()
        #     if len(lidar_features) > 0:
        #         # 查看前方激光雷达读数
        #         center_index = len(lidar_features) // 2
        #         ray_span = 3  # 中心两侧各3束光线
        #         front_indices = range(max(0, center_index - ray_span), 
        #                             min(len(lidar_features), center_index + ray_span + 1))
        #         front_distances = [lidar_features[i] for i in front_indices]
        #         min_front_distance = min(front_distances) if front_distances else 1.0
                
        #         # # 如果前方太靠近障碍物，阻止前进动作
        #         # if min_front_distance <= 0.085:  # 8.5cm阈值
        #         #     action = [0.0, action[1]]  # 只允许旋转
        #         #     print(f"⚠️  预测性避障: 前方{min_front_distance:.3f}m，阻止前进")

        #         # 不阻止前进，但如果前方距离过近，给予惩罚信号
        #         if min_front_distance <= 0.085:  # 8.5cm阈值
        #             # 设置惩罚标志，供奖励函数读取
        #             self._predictive_obstacle_penalty = True
        #             # 可选：打印调试信息
        #             # print(f"⚠️  预测性避障惩罚: 前方{min_front_distance:.3f}m，给予惩罚")
        #         else:
        #             self._predictive_obstacle_penalty = False
        # else:
        #     self._predictive_obstacle_penalty = False
        
        # 动作缩放 - 参考成功代码的缩放系数
        linear_vel = float(action[0])
        angular_vel = float(action[1])
        
        # 差分驱动计算
        left_speed, right_speed = self._diff_drive_kinematics(linear_vel, angular_vel)

        # 平滑限速：限制单步变化，避免瞬时大扭矩引发不稳定
        max_motor_speed = getattr(self, 'max_motor_speed', 26.0)
        # 每步允许的最大变化（与设备能力成比例）
        max_delta = max_motor_speed * 0.2  # 例如 20%/step
        left_speed = float(np.clip(left_speed,
                                   getattr(self, '_prev_left_speed', 0.0) - max_delta,
                                   getattr(self, '_prev_left_speed', 0.0) + max_delta))
        right_speed = float(np.clip(right_speed,
                                    getattr(self, '_prev_right_speed', 0.0) - max_delta,
                                    getattr(self, '_prev_right_speed', 0.0) + max_delta))
        
        # 设置四个电机速度 - 左侧两个轮子相同速度，右侧两个轮子相同速度
        if self.fl_motor and self.fr_motor and self.rl_motor and self.rr_motor:
            # 左侧电机
            self.fl_motor.setVelocity(left_speed)
            self.rl_motor.setVelocity(left_speed)
            
            # 右侧电机
            self.fr_motor.setVelocity(right_speed)
            self.rr_motor.setVelocity(right_speed)
            
            # 打印调试信息
            # print(f"设置电机速度 - 左: {left_speed:.4f}, 右: {right_speed:.4f} (线速度: {linear_vel:.4f}, 角速度: {angular_vel:.4f})")

        # 记录本次轮速用于下次平滑
        self._prev_left_speed = left_speed
        self._prev_right_speed = right_speed
        # 记录命令速度用于反打转检测
        self._last_cmd_linear_vel = linear_vel
        self._last_cmd_angular_vel = angular_vel
    
    def _diff_drive_kinematics(self, linear_vel, angular_vel):
        """差速驱动运动学计算"""
        # 使用与模型一致的底盘与设备参数
        wheel_base = getattr(self, 'wheel_base', 0.22)
        wheel_radius = getattr(self, 'wheel_radius', 0.043)
        max_motor_speed = getattr(self, 'max_motor_speed', 26.0)
        
        # 逆运动学计算
        left_wheel_speed = (linear_vel - angular_vel * wheel_base / 2) / wheel_radius
        right_wheel_speed = (linear_vel + angular_vel * wheel_base / 2) / wheel_radius
        
        # 速度限制
        left_wheel_speed = np.clip(left_wheel_speed, -max_motor_speed, max_motor_speed)
        right_wheel_speed = np.clip(right_wheel_speed, -max_motor_speed, max_motor_speed)
        
        return left_wheel_speed, right_wheel_speed
    
    def _get_observation(self):
        """获取42维观察值"""
        observation = np.zeros(42, dtype=np.float32)
        
        # 1. LiDAR数据 (0-19)
        observation[0:20] = self._get_lidar_data()
        
        # 2. AMCL定位结果 (20-31) - 核心改进
        # amcl_state = self._get_amcl_state()
        # observation[20:32] = amcl_state
        # 使用supervisor数据替代AMCL
        supervisor_pose_state = self._get_supervisor_pose_state()
        observation[20:32] = supervisor_pose_state
        
        # 3. 目标导航信息 (32-37)
        nav_info = self._get_navigation_info()
        observation[32:38] = nav_info
        
        # 4. 航向控制信息 (38-41)
        heading_info = self._get_heading_info()
        observation[38:42] = heading_info
        
        return observation
    
    def _get_supervisor_pose_state(self):
        """获取基于supervisor的位姿状态 (12维), 模拟amcl_state的输出"""
        position = self._get_sup_position()
        orientation = self._get_sup_orientation()

        # 计算速度
        current_time = self.supervisor.getTime()
        dt = current_time - self.state_buffer['previous_time']
        if dt > 0:
            velocity = (position - self.state_buffer['previous_position']) / dt
        else:
            velocity = self.state_buffer['previous_velocity']

        # 更新状态缓存，用于下次计算速度
        self.state_buffer['previous_position'] = position.copy()
        self.state_buffer['previous_orientation'] = orientation.copy()
        self.state_buffer['previous_velocity'] = velocity.copy()
        self.state_buffer['previous_time'] = current_time

        # 使用odometry估计的加速度
        acceleration = self._estimate_acceleration() # 1维

        # 模拟amcl_state的12维输出
        # 格式: est_pos(3), est_orient(3), vel(3), prev_vel(2), accel(1)
        amcl_state_replacement = np.concatenate([
            position,       # 3维 - 估计位置 (使用真实位置)
            orientation,    # 3维 - 估计姿态 (使用真实姿态)
            velocity,       # 3维 - 估计速度
            self.state_buffer['previous_velocity'][:2],  # 2维 - 历史速度（只取xy）
            acceleration   # 1维 - 加速度（只取线性加速度）
        ])

        return amcl_state_replacement


    def _get_amcl_state(self):
        """获取AMCL定位状态 (12维)"""
        # 获取LiDAR和里程计数据
        return None
        # lidar_data = self._get_lidar_data()
        # odometry_data = self._get_odometry_data()
        
        # # 执行AMCL定位
        # try:
        #     result = self.amcl_localizer.localize(lidar_data, odometry_data)
        #     if result is not None and isinstance(result, dict):
        #         # 确保所有必要的键都存在
        #         required_keys = ['position_estimated', 'orientation_estimated', 'velocity_estimated']
        #         if all(key in result for key in required_keys):
        #             self.amcl_result = result
        # except Exception as e:
        #     print(f"AMCL定位获取失败: {e}")
        
        # # 组合状态向量
        # # 注意：总共应该是12维，但当前是14维，需要调整
        # position = self.amcl_result['position_estimated'][:3]  # 确保是3维
        # orientation = self.amcl_result['orientation_estimated'][:3]  # 确保是3维
        # velocity = self.amcl_result['velocity_estimated'][:3]  # 确保是3维
        # prev_velocity = self.state_buffer['previous_velocity'][:2]  # 只取前2维
        # acceleration = self._estimate_acceleration()  # 2维
        
        # amcl_state = np.concatenate([
        #     position,       # 3维 - 估计位置
        #     orientation,    # 3维 - 估计姿态
        #     velocity,       # 3维 - 估计速度
        #     prev_velocity,  # 2维 - 历史速度（只取xy）
        #     acceleration    # 1维 - 加速度（只取线性加速度）
        # ])
        
        # return amcl_state
    
    def _get_lidar_data(self):
        """获取LiDAR数据"""
        if self.lidar is None:
            return np.zeros(20, dtype=np.float32)
            
        try:
            # 真实Webots模式
            ranges = self.lidar.getRangeImage()
            if ranges and len(ranges) >= 20:
                # 均匀采样20个数据点
                indices = np.linspace(0, len(ranges)-1, 20, dtype=int)
                data = np.array([ranges[i] for i in indices])
                # 限制范围和归一化
                data = np.clip(data, 0.01, 10.0) / 10.0
                return data.astype(np.float32)
        except AttributeError:
            raise Exception(AttributeError)
    
    def _get_odometry_data(self):
        """获取里程计数据"""
        # 基于轮速计算里程计
        if self.fl_motor and self.fr_motor and self.rl_motor and self.rr_motor:
            # 获取四个轮子的速度
            fl_speed = self.fl_motor.getVelocity()
            fr_speed = self.fr_motor.getVelocity()
            rl_speed = self.rl_motor.getVelocity()
            rr_speed = self.rr_motor.getVelocity()
            
            # 计算左右侧平均速度
            left_speed = (fl_speed + rl_speed) / 2.0
            right_speed = (fr_speed + rr_speed) / 2.0
            
            # 逆运动学计算车体速度
            linear_vel, angular_vel = self._calculate_body_velocity(left_speed, right_speed)
            
            # 计算位姿变化（基于上一时刻）
            dt = self.timestep / 1000.0  # 转换为秒
            
            if self.state_buffer['previous_time'] > 0:
                dx = linear_vel * dt * math.cos(self.state_buffer['previous_orientation'][2])
                dy = linear_vel * dt * math.sin(self.state_buffer['previous_orientation'][2])
                dyaw = angular_vel * dt
            else:
                dx, dy, dyaw = 0.0, 0.0, 0.0
            
            return {
                'dx': dx, 'dy': dy, 'dyaw': dyaw,
                'linear_velocity': linear_vel,
                'angular_velocity': angular_vel,
                'left_speed': left_speed,
                'right_speed': right_speed
            }
        
        return {'dx': 0.0, 'dy': 0.0, 'dyaw': 0.0, 'linear_velocity': 0.0, 'angular_velocity': 0.0}
    
    def _calculate_body_velocity(self, left_speed, right_speed):
        """计算车体速度（正运动学）"""
        wheel_radius = 0.043  # 轮径 - 与_diff_drive_kinematics中保持一致
        wheel_base = 0.22   # 轴距 - 与_diff_drive_kinematics中保持一致
        
        linear_vel = (left_speed + right_speed) * wheel_radius / 2.0
        angular_vel = (right_speed - left_speed) * wheel_radius / wheel_base
        
        return linear_vel, angular_vel
    
    def _estimate_acceleration(self):
        """估计加速度（只返回线性加速度）"""
        dt = self.timestep / 1000.0
        
        if dt > 0 and self.state_buffer['previous_time'] > 0:
            # 线加速度
            current_vel = np.linalg.norm(self.state_buffer['previous_velocity'][:2])
            prev_vel = np.linalg.norm(self.state_buffer['previous_velocity'][:2])
            linear_acc = (current_vel - prev_vel) / dt
        else:
            linear_acc = 0.0
        
        # 只返回线性加速度（1维）
        return np.array([linear_acc], dtype=np.float32)
    
    def _get_navigation_info(self):
        """获取导航目标信息 (6维)"""
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']  # 使用AMCL位置
        target_pos = self.task_info['target_pos']
        start_pos = self.task_info['start_pos']
        
        # 目标相对位置
        relative_target = target_pos - current_pos
        
        return np.concatenate([
            relative_target,  # 3维
            start_pos,        # 3维
        ]).astype(np.float32)
    
    def _calculate_angular_acceleration(self):
        """计算角加速度"""
        dt = self.timestep / 1000.0  # 转换为秒
        
        if dt > 0 and self.state_buffer['previous_time'] > 0:
            # 当前角速度
            # current_angular_vel = self.amcl_result['velocity_estimated'][2]
            odometry_data = self._get_odometry_data()
            current_angular_vel = odometry_data['angular_velocity']
            # 上一时刻角速度
            prev_angular_vel = self.state_buffer['previous_velocity'][2]
            # 角加速度
            angular_acc = (current_angular_vel - prev_angular_vel) / dt
            return angular_acc
        
        return 0.0
    
    def _get_heading_info(self):
        """获取航向控制信息 (4维)"""
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']
        current_orient = self._get_sup_orientation() # self.amcl_result['orientation_estimated']
        target_pos = self.task_info['target_pos']
        
        # 计算当前到目标的向量
        target_vector = target_pos - current_pos
        target_heading = math.atan2(target_vector[1], target_vector[0])
        
        # 当前航向角（使用估计姿态）
        current_heading = current_orient[2]  # yaw角
        
        # 航向偏差
        heading_error = target_heading - current_heading
        # 归一化到[-π, π]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # 角速度和角加速度
        # angular_velocity = self.amcl_result['velocity_estimated'][2]
        odometry_data = self._get_odometry_data()
        angular_velocity = odometry_data['angular_velocity']
        angular_acceleration = self._calculate_angular_acceleration()
        
        return np.array([
            heading_error,
            target_heading, 
            angular_velocity,
            angular_acceleration
        ], dtype=np.float32)
    
    def _update_amcl_state(self):
        """更新AMCL状态"""
        # AMCL 停用
        pass
        # # 获取最新传感器数据
        # lidar_data = self._get_lidar_data()
        # odometry_data = self._get_odometry_data()
        
        # # AMCL定位更新
        # try:
        #     result = self.amcl_localizer.localize(lidar_data, odometry_data)
        #     if result is not None and isinstance(result, dict):
        #         # 确保所有必要的键都存在
        #         required_keys = ['position_estimated', 'orientation_estimated', 'velocity_estimated']
        #         if all(key in result for key in required_keys):
        #             self.amcl_result = result
        # except Exception as e:
        #     print(f"AMCL定位更新失败: {e}")
        
        # # 更新历史状态
        # current_time = self.supervisor.getTime()
        
        # self.state_buffer['previous_position'] = self.amcl_result['position_estimated'].copy()
        # self.state_buffer['previous_velocity'] = self.amcl_result['velocity_estimated'].copy()
        # self.state_buffer['previous_orientation'] = self.amcl_result['orientation_estimated'].copy()
        
        # if self.state_buffer['previous_time'] == 0:
        #     self.state_buffer['previous_time'] = current_time
    
    def _calculate_reward(self, action, observation):
        """使用RewardFunctions类计算奖励"""
        # 准备环境状态字典，传递给奖励函数
        env_state = {
            'get_sup_position': self._get_sup_position,
            'get_sup_orientation': self._get_sup_orientation,
            'calculate_distance_to_target': self._calculate_distance_to_target,
            'get_lidar_features': self._get_lidar_features,
            'detect_collision_simple': self._detect_collision_simple,
            'episode_steps': self.episode_steps,
            'cargo_type': self.cargo_type,
            'success_threshold': self.success_threshold,
            'task_info': self.task_info
        }
        
        # 调用奖励函数计算奖励
        return self.reward_functions.calculate_reward(action, observation, env_state)
    
    # 货物专用奖励函数已移至reward_functions.py
    
    def _check_termination(self):
        """检查终止条件"""
        collision_detected = False

        # 全局步计数与朝向历史初始化
        try:
            self._global_step = getattr(self, '_global_step', 0) + 1
            setattr(self, '_global_step', self._global_step)
            if not hasattr(self, '_heading_hist'):
                self._heading_hist = []  # 简单列表作为滑窗
            if not hasattr(self, '_last_pos'):
                self._last_pos = self._get_sup_position().copy()
            if not hasattr(self, '_last_contact_check_step'):
                self._last_contact_check_step = -999999
        except Exception:
            # 若异常，继续执行但不启用条件触发
            self._heading_hist = []
            self._last_pos = self._get_sup_position().copy()
            self._last_contact_check_step = -999999

        # 基于平面运动估计朝向，并维护滑动窗口
        try:
            curr_pos = self._get_sup_position()
            dx = float(curr_pos[0] - self._last_pos[0])
            dy = float(curr_pos[1] - self._last_pos[1])
            self._last_pos = curr_pos.copy()
            import math
            # 仅在移动幅度超过极小阈值时更新朝向估计
            if (dx * dx + dy * dy) > 1e-6:
                heading = math.atan2(dy, dx)
                self._heading_hist.append(heading)
                if len(self._heading_hist) > 10:
                    self._heading_hist = self._heading_hist[-10:]
        except Exception:
            pass

        # 只有在“朝向在滑窗内基本不变”且“距离上次查询超过间隔”时，才调用昂贵的接触点查询
        should_check_contacts = False
        try:
            if len(self._heading_hist) >= 6:
                hmax = max(self._heading_hist)
                hmin = min(self._heading_hist)
                if (hmax - hmin) < 0.02:  # 约 1.1 度
                    if (self._global_step - self._last_contact_check_step) >= 5:
                        should_check_contacts = True
        except Exception:
            pass

        # 1. selfCollision检测 (Webots) - 仅在触发条件满足时进行昂贵查询
        if should_check_contacts and self.robot_node:
            try:
                contact_points = self.robot_node.getContactPoints(includeDescendants=True)
                self._last_contact_check_step = self._global_step
                for cp in contact_points:
                    other_node_id = cp.getNodeId()
                    contact_z_height = cp.getPoint()[2]
                    other_node = self.supervisor.getFromId(other_node_id)
                    if other_node is None:
                        continue
                    other_node_name = ""
                    name_field = other_node.getField("name")
                    if name_field:
                        other_node_name = name_field.getSFString()
                    # 正常车轮-地面接触判定
                    is_ground_contact = other_node_name in self.ground_defs
                    is_at_floor_level = abs(contact_z_height) < 0.01
                    if is_ground_contact and is_at_floor_level:
                        continue
                    collision_detected = True
                    break
            except Exception as e:
                # 静默或降频打印
                # print(f"通过getContactPoints检测碰撞时发生错误: {e}")
                pass

        if collision_detected:
            return True
        
        # 2. 到达目标
        current_pos = self._get_sup_position()
        current_distance = np.linalg.norm(current_pos - self.task_info['target_pos'])
        if current_distance < self.success_threshold:
            # print(f"成功到达目标点！距离: {current_distance:.4f}m")
            return True
        
        # 3. 新增：原地打转终止条件
        if self.reward_functions.is_excessive_spin():
            # print(f"检测到原地打转超过阈值，任务终止。")
            return True
        
        # 4. 超时/步数限制
        if len(self.trajectory) > self.max_steps_per_episode:
            # print("超过最大步数限制，任务终止。")
            return True
            
        return False
    
    def _check_truncation(self):
        """检查截断条件"""
        # 位置边界检查
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']
        if abs(current_pos[0]) > 20 or abs(current_pos[1]) > 20:
            return True
            
        return False
    
    def _get_step_info(self):
        """获取步信息"""
        current_pos = self._get_sup_position() # self.amcl_result['position_estimated']
        target_pos = self.task_info['target_pos']
        
        # 计算距离信息
        distance_to_target = float(np.linalg.norm(current_pos - target_pos))
        
        info = {
            'position': current_pos.tolist(),
            'target_position': target_pos.tolist(),
            'distance_to_target': distance_to_target,
            'amcl_uncertainty': 0.0, # float(self.amcl_result.get('position_uncertainty', 0.1)),
            'min_obstacle_distance': float(self.min_obstacle_distance),
            'trajectory_length': len(self.trajectory),
            'close_to_target': distance_to_target < 0.5,  # 接近目标标志
            'very_close_to_target': distance_to_target < 0.25,  # 非常接近目标
            'cargo_type': self.cargo_type
        }
        
        return info

    def _detect_collision_simple(self) -> bool:
        """轻量碰撞检测：
        1) 使用接触点排除正常的轮-地面接触
        2) 回退使用红外距离传感器阈值
        """
        # 方法1：接触点检测
        try:
            if self.robot_node:
                contact_points = self.robot_node.getContactPoints(includeDescendants=True)
                for cp in contact_points:
                    other_node_id = cp.getNodeId()
                    contact_z_height = cp.getPoint()[2]
                    other_node = self.supervisor.getFromId(other_node_id)
                    if other_node is None:
                        continue
                    other_node_name = ""
                    name_field = other_node.getField("name")
                    if name_field:
                        other_node_name = name_field.getSFString()
                    is_ground_contact = other_node_name in getattr(self, 'ground_defs', {'floor'})
                    is_at_floor_level = abs(contact_z_height) < 0.01
                    # 过滤正常的轮-地面接触
                    if is_ground_contact and is_at_floor_level:
                        continue
                    return True
        except Exception:
            pass

        # 方法2：距离传感器阈值
        try:
            threshold = getattr(self, 'collision_distance_threshold', 0.05)
            for sensor in getattr(self, 'collision_sensors', []) or []:
                value = sensor.getValue()
                if value is not None and value < threshold:
                    return True
        except Exception:
            pass

        return False
    
    def get_amcl_uncertainty(self):
        """获取AMCL定位不确定性"""
        return 0.0 # self.amcl_localizer.get_current_uncertainty()
    
    def get_current_pose(self):
        """获取当前位姿"""
        # 使用supervisor数据
        pos = self._get_sup_position()
        orient = self._get_sup_orientation()
        # 从里程计获取速度
        odometry_data = self._get_odometry_data()
        vel = np.array([odometry_data['linear_velocity'], 0, odometry_data['angular_velocity']])

        return {
            'position': pos,
            'orientation': orient,
            'velocity': vel
        }
        # return {
        #     'position': self.amcl_result['position_estimated'].copy(),
        #     'orientation': self.amcl_result['orientation_estimated'].copy(),
        #     'velocity': self.amcl_result['velocity_estimated'].copy()
        # }
    
    def set_training_mode(self, mode: bool):
        """设置训练模式"""
        self.is_training = mode
        # self.amcl_localizer.set_training_mode(mode)

    def close(self):
        """释放资源并尽量优雅地停止控制器"""
        try:
            if hasattr(self, 'supervisor') and self.supervisor and hasattr(self.supervisor, 'simulationSetMode') and hasattr(Supervisor, 'SIMULATION_MODE_PAUSE'):
                self.supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        except Exception:
            pass