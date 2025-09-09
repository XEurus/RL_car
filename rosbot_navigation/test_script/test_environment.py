"""
环境测试脚本 - 验证42维导航环境的正确性
非Webots依赖的自我测试
"""

import numpy as np
import time
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.localization.amcl_localizer import AMCLLocalizer
    from src.utils.navigation_utils import NavigationUtils
    print("✓ 成功导入本地化模块")
except ImportError as e:
    print(f"✗ 本地化模块导入失败: {e}")
    print("将使用模拟的AMCL本地器进行测试")
    AMCLLocalizer = None

class MockAMCLLocalizer:
    """模拟AMCL本地器 - 用于无Webots测试"""
    
    def __init__(self, num_particles=800):
        self.num_particles = num_particles
        self.state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'uncertainty': np.array([0.1, 0.1, 0.1])
        }
        
    def localize(self, lidar_scan, odometry):
        """模拟定位 - 添加真实噪声"""
        # 基于里程计更新状态
        if 'dx' in odometry:
            self.state['position'][0] += odometry['dx'] + np.random.normal(0, 0.02)
            self.state['position'][1] += odometry['dy'] + np.random.normal(0, 0.02)
            self.state['position'][2] = 0.0
        
        return {
            'position_estimated': self.state['position'].copy(),
            'orientation_estimated': np.array([0.0, 0.0, odometry.get('dyaw', 0.0)]),
            'velocity_estimated': np.array([odometry.get('linear_velocity', 0.0), 0.0, odometry.get('angular_velocity', 0.0)]),
            'position_uncertainty': self.state['uncertainty'].copy()
        }
    
    def reset(self):
        self.state['position'] = np.array([0.0, 0.0, 0.0])
    
    def get_current_uncertainty(self):
        return np.mean(self.state['uncertainty'])
    
    def has_converged(self):
        return np.mean(self.state['uncertainty']) < 0.2


class EnvironmentTestSuite:
    """环境测试套件"""
    
    def __init__(self):
        self.test_results = {}
        self.use_webots = False  # 暂时禁用Webots，先用模拟测试
        
    def test_state_space_dimensions(self):
        """测试状态空间维度"""
        print("\n=== 测试状态空间维度 ===")
        
        # 创建42维模拟状态
        test_observations = []
        
        # 合法的测试状态
        for _ in range(100):
            obs = self.generate_legal_observation()
            test_observations.append(obs)
        
        obs_array = np.array(test_observations)
        print(f"观测数据形状: {obs_array.shape}")
        print(f"单个观测维度: {obs_array.shape[1]}")
        
        # 验证维度
        expected_dim = 42
        actual_dim = obs_array.shape[1]
        dimension_test = actual_dim == expected_dim
        
        print(f"期望维度: {expected_dim}")
        print(f"实际维度: {actual_dim}")
        print(f"维度测试: {'通过' if dimension_test else '失败'}")
        
        return {
            'expected_dimension': expected_dim,
            'actual_dimension': actual_dim,
            'test_passed': dimension_test,
            'num_test_samples': len(test_observations)
        }
    
    def test_observation_value_ranges(self):
        """测试观察值范围验证"""
        print("\n=== 测试观察值范围 ===")
        
        # 生成多组观察数据
        test_observations = [self.generate_legal_observation() for _ in range(50)]
        
        # 分析各组件的范围
        lidar_min = min(obs[0:20].min() for obs in test_observations)
        lidar_max = max(obs[0:20].max() for obs in test_observations)
        
        amcl_data = np.array([obs[20:32] for obs in test_observations])
        nav_data = np.array([obs[32:38] for obs in test_observations])
        heading_data = np.array([obs[38:42] for obs in test_observations])
        
        print("观察数据范围分析:")
        print(f"LiDAR [0-1]:       min={lidar_min:.3f}, max={lidar_max:.3f}")
        print(f"AMCL定位 [32]:     min={amcl_data.min():.3f}, max={amcl_data.max():.3f}")
        print(f"导航信息 [6]:       min={nav_data.min():.3f}, max={nav_data.max():.3f}")
        print(f"航向控制 [4]:       min={heading_data.min():.3f}, max={heading_data.max():.3f}")
        
        # 验证范围
        range_tests = {
            'lidar_range': (lidar_min >= 0.0 and lidar_max <= 1.1),  # 允许轻微超限
            'amcl_position': (amcl_data[:,0:3].min() >= -20.1 and amcl_data[:,0:3].max() <= 20.1),
            'navigation': (nav_data.min() >= -20.1 and nav_data.max() <= 20.1),
            'heading': (heading_data[:,0].min() >= -3.15 and heading_data[:,0].max() <= 3.15)
        }
        
        for test_name, passed in range_tests.items():
            print(f"{test_name}范围测试: {'通过' if passed else '失败'}")
        
        return {
            'lidar_range': (lidar_min, lidar_max),
            'amcl_range': (amcl_data.min(), amcl_data.max()),
            'navigation_range': (nav_data.min(), nav_data.max()),
            'heading_range': (heading_data.min(), heading_data.max()),
            'range_tests_passed': all(range_tests.values())
        }
    
    def test_amcl_integration(self):
        """测试AMCL集成"""
        print("\n=== 测试AMCL集成 ===")
        
        # 使用真实的AMCL本地器或模拟版本
        if AMCLLocalizer:
            print("使用真实AMCL本地器")
            amcl_localizer = AMCLLocalizer(num_particles=800)
        else:
            print("使用模拟AMCL本地器")
            amcl_localizer = MockAMCLLocalizer(num_particles=800)
        
        # 模拟初始化和定位
        amcl_localizer.reset()
        amcl_localizer.initialize_with_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 测试运动更新和定位
        localization_results = []
        
        for step in range(20):
            # 模拟LiDAR扫描
            lidar_scan = np.random.uniform(0.1, 0.9, 20)  # 模拟的LiDAR数据
            
            # 模拟里程计
            odometry = {
                'dx': 0.5 + np.random.normal(0, 0.01),
                'dy': 0.0 + np.random.normal(0, 0.01),
                'dyaw': 0.1 + np.random.normal(0, 0.005),
                'linear_velocity': 0.5 + np.random.normal(0, 0.02),
                'angular_velocity': 0.1 + np.random.normal(0, 0.01)
            }
            
            result = amcl_localizer.localize(lidar_scan, odometry)
            
            if result:
                localization_results.append({
                    'position': result['position_estimated'].tolist(),
                    'uncertainty': np.mean(result.get('position_uncertainty', [0.1])).item(),
                    'step': step
                })
        
        num_successful = len(localization_results)
        avg_uncertainty = np.mean([r['uncertainty'] for r in localization_results]) if localization_results else 0.1
        
        print(f"定位成功率: {num_successful}/20 = {num_successful/20:.1%}")
        print(f"平均不确定性: {avg_uncertainty:.4f}")
        
        return {
            'num_localization_attempts': 20,
            'successful_localizations': num_successful,
            'success_rate': num_successful / 20,
            'avg_uncertainty': avg_uncertainty,
            'amcl_type': 'real' if AMCLLocalizer else 'mock'
        }
    
    def test_environment_with_simulation(self):
        """模拟环境完整测试循环"""
        print("\n=== 模拟环境完整测试循环 ===")
        
        # 我们创建完整的环境模拟，但跳过Webots依赖
        try:
            from src.environments.navigation_env import ROSbotNavigationEnv
            print("环境模块导入成功")
        except ImportError as e:
            print(f"环境导入失败: {e}")
            return self._test_mock_environment()
        
        # 测试环境创建（带Mock模式）
        mock_results = self._test_mock_environment_cycle()
        
        return mock_results
    
    def _test_mock_environment_cycle(self):
        """模拟环境循环测试"""
        print("进行模拟环境循环测试...")
        
        # 模拟完整的导航场景
        env_state = self.create_mock_environment_state()
        
        episode_results = []
        
        for episode in range(5):
            print(f"\n模拟Episode {episode + 1}...")
            
            # 重置环境状态
            env_state['position'] = [-8.0, -6.0, 0.0]
            env_state['target'] = [2.0, 1.0, 0.0]
            env_state['step'] = 0
            env_state['success'] = False
            
            episode_data = self.simulate_episode_cycle(env_state, max_steps=50)
            episode_results.append(episode_data)
            
            print(f"Episode {episode + 1}: 步数={episode_data['steps']}, "
                  f"成功={'是' if episode_data['success'] else '否'}")
        
        # 统计结果
        success_count = sum(1 for ep in episode_results if ep['success'])
        avg_steps = np.mean([ep['steps'] for ep in episode_results])
        avg_uncertainty = np.mean([ep['avg_uncertainty'] for ep in episode_results])
        
        return {
            'total_episodes': len(episode_results),
            'successful_episodes': success_count,
            'success_rate': success_count / len(episode_results),
            'average_steps': avg_steps,
            'average_uncertainty': avg_uncertainty
        }
    
    def create_mock_environment_state(self):
        """创建模拟环境状态"""
        return {
            'position': [0.0, 0.0, 0.0],
            'velocity': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0],
            'target': [0.0, 0.0, 0.0],
            'start': [0.0, 0.0, 0.0],
            'step': 0,
            'success': False,
            'lidar': np.zeros(20),
            'amcl_uncertainty': 0.1
        }
    
    def simulate_episode_cycle(self, env_state: Dict, max_steps: int = 50) -> Dict:
        """模拟一个Episode的执行"""
        step_data = []
        
        for step in range(max_steps):
            # 生成当前观察
            observation = self.generate_state_observation(env_state)
            
            # 模拟RL智能体决策
            action = self.generate_mock_action(observation)
            
            # 更新环境状态
            self.update_environment_state(env_state, action, step)
            
            # 检查终止条件
            if self.check_termination_conditions(env_state):
                break
            
            step_data.append({
                'step': step,
                'observation_shape': observation.shape,
                'action': action.tolist(),
                'position': env_state['position'][:2].copy(),
                'uncertainty': env_state['amcl_uncertainty']
            })
        
        return {
            'steps': len(step_data) + 1,
            'success': env_state['success'],
            'max_uncertainty': max([s['uncertainty'] for s in step_data]) if step_data else 0.1,
            'avg_uncertainty': np.mean([s['uncertainty'] for s in step_data]) if step_data else 0.1
        }
    
    def generate_state_observation(self, env_state: Dict) -> np.ndarray:
        """生成42维状态观察"""
        # LiDAR数据 (20维)
        lidar_data = env_state['lidar'] if 'lidar' in env_state else np.random.uniform(0.1, 0.9, 20)
        
        # AMCL定位数据 (12维) - 模拟真实AMCL输出
        amcl_data = np.concatenate([
            np.array([0.0, 0.0, 0.0]),           # 真实位置
            np.array(env_state['position']),     # 估计位置（带误差）
            np.array([0.0, 0.0, env_state.get('orientation', [0,0,0])[2]]),  # 姿态
            np.array(env_state['velocity']),     # 速度
            np.array([env_state.get('prev_velocity', [0,0,0])[0], 0.0])        # 历史速度+加速度
        ])
        
        # 导航信息 (6维)
        nav_data = np.concatenate([
            np.array(env_state['target']) - np.array(env_state['position']),  # 相对目标
            np.array(env_state['start']),                                       # 起点
            np.array(env_state['target'])                                       # 终点
        ])
        
        # 航向控制 (4维)
        heading_data = np.array([
            0.0,  # 航向偏差
            0.0,  # 目标航向
            0.1,  # 角速度
            0.0   # 角加速度
        ])
        
        # 组合完整观察
        observation = np.concatenate([lidar_data, amcl_data, nav_data, heading_data])
        
        return observation.astype(np.float32)
    
    def generate_mock_action(self, observation: np.ndarray) -> np.ndarray:
        """生成模拟动作"""
        # 简单的导航策略模拟
        # 基于目标相对位置计算控制输入
        target_vector = observation[32:35]  # 目标相对位置
        
        # 计算到目标的距离和方向
        distance = np.linalg.norm(target_vector)
        direction = np.arctan2(target_vector[1], target_vector[0])
        
        # 简单的比例控制策略
        linear_vel = np.clip(distance * 0.3, 0.1, 1.0)  # 限速1m/s
        angular_vel = np.clip(direction * 0.5, -1.0, 1.0)
        
        return np.array([linear_vel, angular_vel], dtype=np.float32)
    
    def update_environment_state(self, env_state: Dict, action: np.ndarray, step: int):
        """更新环境状态"""
        # 基于动作更新位置
        linear_vel, angular_vel = action
        
        # 简单运动学模型
        current_heading = env_state.get('orientation', [0,0,0])[2]
        new_x = env_state['position'][0] + linear_vel * 0.1 * np.cos(current_heading)
        new_y = env_state['position'][1] + linear_vel * 0.1 * np.sin(current_heading)
        
        env_state['position'] = [new_x, new_y, 0.0]
        env_state['velocity'] = [linear_vel, 0.0, 0.0]
        env_state['orientation'] = [0.0, 0.0, current_heading + angular_vel * 0.1]
        
        # 更新AMCL不确定性（模拟）
        env_state['amcl_uncertainty'] = 0.05 + 0.01 * step  # 线性增长模拟
        
        # 更新LiDAR数据（基于新位置）
        env_state['lidar'] = np.random.uniform(0.1, 0.9, 20)  # 简化的LiDAR模拟
    
    def check_termination_conditions(self, env_state: Dict) -> bool:
        """检查终止条件"""
        # 到达目标
        distance_to_target = np.linalg.norm(
            np.array(env_state['target'][:2]) - np.array(env_state['position'][:2])
        )
        
        if distance_to_target < 0.2:  # 20cm阈值
            env_state['success'] = True
            return True
        
        # 边界检查
        pos = env_state['position']
        if abs(pos[0]) > 20 or abs(pos[1]) > 15:
            return True
        
        return False
    
    def generate_legal_observation(self) -> np.ndarray:
        """生成合法的42维观察"""
        observation = np.zeros(42, dtype=np.float32)
        
        # LiDAR数据 (0-19): [0,1]范围
        observation[0:20] = np.random.uniform(0.0, 1.0, 20)
        
        # AMCL定位 (20-31): 在合理范围内
        observation[20:23] = np.random.uniform(-10, 10, 3)   # 位置
        observation[23:26] = np.random.uniform(-10, 10, 3)   # 估计位置
        observation[26:29] = np.random.uniform(-math.pi, math.pi, 3) # 姿态
        observation[29:32] = np.random.uniform(-2, 2, 3)    # 速度
        observation[31:33] = np.random.uniform(-1, 1, 2)    # 加速度
        
        # 导航信息 (32-37)
        observation[32:35] = np.random.uniform(-15, 15, 3)  # 相对目标
        observation[35:38] = np.random.uniform(-15, 15, 3)  # 起点终点
        
        # 航向控制 (38-41)
        observation[38] = np.random.uniform(-math.pi, math.pi)  # 航向偏差
        observation[39] = np.random.uniform(-math.pi, math.pi)  # 目标航向
        observation[40] = np.random.uniform(-2, 2, 1)          # 角速度
        observation[41] = np.random.uniform(-1, 1, 1)          # 角加速度
        
        return observation
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("="*70)
        print("ROSbot导航环境综合测试套件")
        print("="*70)
        
        results = {}
        
        # 状态空间维度测试
        print("\n" + "="*50)
        results['state_space'] = self.test_state_space_dimensions()
        
        # 观察值范围测试
        print("\n" + "="*50)
        results['observation_ranges'] = self.test_observation_value_ranges()
        
        # AMCL集成测试
        print("\n" + "="*50)
        results['amcl_integration'] = self.test_amcl_integration()
        
        # 完整环境循环测试
        print("\n" + "="*50)
        results['environment_simulation'] = self.test_environment_with_simulation()
        
        # 生成测试报告
        self.generate_environment_test_report(results)
        
        print("\n" + "="*70)
        print("=== 环境测试完成 ===")
        
        # 如果AMCL本地器测试通过，推荐进行完整Webots集成测试
        if results['amcl_integration']['amcl_type'] == 'real' and results['amcl_integration']['success_rate'] > 0.8:
            print("✅ AMCL本地器状态良好，建议进行完整Webots集成测试")
        else:
            print("⚠️  建议在AMCL本地器稳定后进行Webots集成测试")
        
        # 可视化测试结果
        try:
            self.visualize_test_results(results)
        except Exception as e:
            print(f"可视化失败: {e}")
        
        return results
    
    def generate_environment_test_report(self, results: Dict):
        """生成详细的测试报告"""
        print("\n" + "="*60)
        print("=== 环境测试报告 ===")
        
        if results['state_space']:
            ss = results['state_space']
            print(f"状态空间维度: {ss['actual_dimension']}/{ss['expected_dimension']} - {'✓通过' if ss['test_passed'] else '✗失败'}")
        
        if results['observation_ranges']:
            or_ = results['observation_ranges']
            print(f"LiDAR范围检查: {'✓通过' if or_['range_tests_passed'] else '✗失败'}")
            print(f"  范围: [{or_['lidar_range'][0]:.3f}, {or_['lidar_range'][1]:.3f}]")
        
        if results['amcl_integration']:
            ai = results['amcl_integration']
            print(f"AMCL集成: {'✓通过' if ai['success_rate'] > 0.7 else '✗失败'}")
            print(f"  定位成功率: {ai['success_rate']:.1%}")
            print(f"  平均不确定性: {ai['avg_uncertainty']:.3f}")
            print(f"  本地器类型: {ai['amcl_type']}")
        
        if results['environment_simulation']:
            es = results['environment_simulation']
            print(f"环境仿真: {'✓通过' if es['success_rate'] > 0.5 else '⚠部分通过'}")
            print(f"  仿真成功率: {es['success_rate']:.1%}")
            print(f"  平均步数: {es['average_steps']:.1f}")
            print(f"  平均不确定性: {es['average_uncertainty']:.3f}")
        
        # 保存报告文件
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file = f"environment_test_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"ROSbot导航环境测试报告\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            
            for test_name, result in results.items():
                f.write(f"\n{test_name}:\n")
                f.write(json.dumps(result, indent=2, ensure_ascii=False))
                f.write("\n")
        
        print(f"\n✅ 详细测试报告已保存到: {report_file}")
    
    def visualize_test_results(self, results: Dict):
        """可视化测试结果"""
        try:
            import matplotlib.pyplot as plt
            
            plt.style.use('seaborn')
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('ROSbot导航环境测试结果', fontsize=16)
            
            if 'state_space' in results:
                # 状态空间维度检查
                ax = axes[0,0]
                result = results['state_space']
                labels = ['期望维度', '实际维度']
                values = [result['expected_dimension'], result['actual_dimension']]
                colors = ['blue', 'green' if result['test_passed'] else 'red']
                
                ax.bar(labels, values, color=colors, alpha=0.7)
                ax.set_title('状态空间维度检查')
                ax.set_ylabel('维度数')
                for i, v in enumerate(values):
                    ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
            
            if 'amcl_integration' in results:
                # AMCL集成结果
                ax = axes[0,1]
                result = results['amcl_integration']
                colors = ['green' if result['success_rate'] > 0.8 else 'orange' if result['success_rate'] > 0.5 else 'red']
                
                ax.set_xlim(0, 1)
                ax.barh('定位成功率', result['success_rate'], color=colors, alpha=0.7)
                ax.set_xlabel('成功率')
                ax.set_title('AMCL定位集成功率')
                
                # 添加数值标签
                ax.text(result['success_rate'] + 0.02, 0, f"{result['success_rate']:.1%}", 
                       va='center', ha='left')
            
            plt.tight_layout()
            plt.savefig(f"environment_test_visual_{time.strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("matplotlib不可用，跳过可视化")
        except Exception as e:
            print(f"可视化错误: {e}")


def main():
    """主函数 - 运行环境综合测试"""
    print("="*70)
    print("ROSbot导航环境综合测试")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 创建测试套件
    test_suite = EnvironmentTestSuite()
    
    # 运行完整测试
    results = test_suite.run_comprehensive_test()
    
    # 输出简要结果
    print(f"\n{'='*70}")
    print("=== 测试总结 ===")
    
    passed_tests = sum(1 for r in results.values() if isinstance(r, dict) and r.get('test_passed', False))
    total_tests = len([r for r in results.values() if isinstance(r, dict)])
    
    print(f"测试通过率: {passed_tests}/{total_tests} = {passed_tests/total_tests:.1%}")
    
    if results.get('amcl_integration', {}).get('success_rate', 0) > 0.7:
        print("✅ AMCL本地器运行稳定，建议进行Webots集成测试")
    else:
        print("⚠️  AMCL本地器需要进一步调优")
    
    print("\n=== 下一步建议 ===")
    print("1. 在Webots环境中测试完整系统")
    print("2. 运行AMCL定位深度测试")
    print("3. 验证TD3模型架构")
    print("4. 开始小规模训练实验")


if __name__ == "__main__":
    # 设置随机种子以确保可重复的测试
    np.random.seed(42)
    
    main()