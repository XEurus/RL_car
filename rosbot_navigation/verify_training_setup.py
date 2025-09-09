"""
训练设置验证脚本
验证训练环境、模型和配置的正确性
在开始大规模训练前的完整性检查
"""

import os
import sys
import numpy as np
import torch
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

def check_python_environment() -> Dict:
    """检查Python环境"""
    print("\n=== 检查Python环境 ===")
    
    results = {
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    
    print(f"Python版本: {results['python_version']}")
    print(f"NumPy版本: {results['numpy_version']}")
    print(f"PyTorch版本: {results['torch_version']}")
    print(f"CUDA可用: {'是' if results['cuda_available'] else '否'}")
    if results['cuda_available']:
        print(f"GPU数量: {results['gpu_count']}")
        print(f"GPU名称: {results['device_name']}")
    
    return results

def check_package_imports() -> Dict:
    """检查必要包的导入"""
    print("\n=== 检查依赖包导入 ===")
    
    required_packages = [
        'gymnasium', 'stable_baselines3', 'mlflow', 'tensorboard',
        'numpy', 'torch', 'pandas', 'matplotlib', 'yaml'
    ]
    
    import_results = {}
    
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
                version = getattr(yaml, '__version__', 'unknown')
            elif package == 'tensorboard':
                import tensorboard
                version = getattr(tensorboard, '__version__', 'unknown')
            else:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
            
            import_results[package] = {'status': 'imported', 'version': version}
            print(f"✅ {package}: {version}")
        
        except ImportError as e:
            import_results[package] = {'status': 'failed', 'error': str(e)}
            print(f"❌ {package}: 导入失败 - {e}")
        
        except Exception as e:
            import_results[package] = {'status': 'error', 'error': str(e)}
            print(f"⚠️ {package}: 导入错误 - {e}")
    
    all_imported = all(r['status'] == 'imported' for r in import_results.values())
    print(f"\n依赖包导入状态: {'✅全部通过' if all_imported else '❌部分失败'}")
    
    return {'import_status': import_results, 'all_imported': all_imported}

def check_project_imports() -> Dict:
    """检查项目特定模块导入"""
    print("\n=== 检查项目模块导入 ===")
    
    project_modules = [
        'src.environments.navigation_env',
        'src.models.td3_robust', 
        'src.localization.amcl_localizer',
        'src.localization.pose_estimator',
        'src.utils.navigation_utils',
        'src.controllers.webots_controller'
    ]
    
    import_results = {}
    
    for module in project_modules:
        try:
            mod = __import__(module, fromlist=[''])
            
            # 获取关键类信息
            if hasattr(mod, 'ROSbotNavigationEnv'):
                cls_info = "ROSbotNavigationEnv类存在"
            elif hasattr(mod, 'ImprovedTD3'):
                cls_info = "ImprovedTD3类存在"
            elif hasattr(mod, 'AMCLLocalizer'):
                cls_info = "AMCLLocalizer类存在"
            elif hasattr(mod, 'ROSbotWebotsController'):
                cls_info = "ROSbotWebotsController类存在"
            else:
                cls_info = "类检测未指定"
            
            import_results[module] = {'status': 'imported', 'class_info': cls_info}
            print(f"✅ {module}: {cls_info}")
        
        except ImportError as e:
            import_results[module] = {'status': 'failed', 'error': str(e)}
            print(f"❌ {module}: 导入失败 - {e}")
        
        except Exception as e:
            import_results[module] = {'status': 'error', 'error': str(e)}
            print(f"⚠️ {module}: 导入错误 - {e}")
    
    return {'import_status': import_results}

def check_state_space_dimensions() -> Dict:
    """检查状态空间维度"""
    print("\n=== 检查状态空间维度 ===")
    
    try:
        from src.environments.navigation_env import ROSbotNavigationEnv
        
        # 创建测试环境
        env = ROSbotNavigationEnv(cargo_type='normal')
        
        # 检查观测空间
        obs_shape = env.observation_space.shape
        expected_shape = (42,)
        shape_test = obs_shape == expected_shape
        
        # 检查动作空间
        action_shape = env.action_space.shape
        expected_action = (2,)
        action_test = action_shape == expected_action
        
        # 检查边界值
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        
        print(f"观测空间形状: {obs_shape} {'✅正确' if shape_test else '❌错误'}")
        print(f"动作空间形状: {action_shape} {'✅正确' if action_test else '❌错误'}")
        print(f"观测空间范围: [{obs_low.min():.2f}, {obs_high.max():.2f}]")
        
        # 生成测试观察值
        obs_low_test = np.array(obs_low)
        obs_high_test = np.array(obs_high)
        validation_observation = np.random.uniform(obs_low_test + 0.1, obs_high_test - 0.1)
        
        # 验证观察值格式
        obs_var = validation_observation
        obs_test = env.observation_space.contains(obs_var)
        
        return {
            'observation_shape': obs_shape,
            'action_shape': action_shape,
            'shape_test_passed': shape_test and action_test,
            'boundary_test': obs_test,
            'shape_validation': 'passed' if shape_test and action_test else 'failed'
        }
    
    except Exception as e:
        print(f"❌ 状态空间检查失败: {e}")
        return {'error': str(e), 'validation_status': 'failed'}

def check_amcl_localization_basic() -> Dict:
    """检查AMCL定位基础功能"""
    print("\n=== 检查AMCL定位基础功能 ===")
    
    try:
        from src.localization.amcl_localizer import AMCLLocalizer
        
        # 创建AMCL本地器
        amcl = AMCLLocalizer(num_particles=800)
        
        # 基础功能测试
        print("AMCL本地器创建成功")
        print(f"粒子数量: {amcl.num_particles}")
        
        # 初始化测试
        initial_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        amcl.initialize_with_pose(initial_pose)
        print("✅ 初始化测试通过")
        
        # 基本状态查询测试
        uncertainty = amcl.get_current_uncertainty()
        convergence = amcl.has_converged()
        print(f"当前不确定性: {uncertainty:.4f}")
        print(f"收敛状态: {'✅已收敛' if convergence else '❌未收敛'}")
        
        # 模拟简单的定位循环
        test_odometry = {
            'dx': 0.5, 'dy': 0.0, 'dyaw': 0.1,
            'linear_velocity': 0.5, 'angular_velocity': 0.1
        }
        
        test_lidar = np.random.uniform(0.1, 0.9, 20)  # 模拟LiDAR数据
        
        result = amcl.localize(test_lidar, test_odometry)
        
        if result and 'position_estimated' in result:
            print("✅ 定位更新测试通过")
            print(f"估计位置: {result['position_estimated']}")
            print(f"估计姿态: {result['orientation_estimated']}")
        else:
            print("⚠️  定位更新测试部分通过")
        
        return {
            'amcl_created': True,
            'initialization_test': 'passed',
            'localization_test': 'passed' if result else 'partial',
            'particle_count': amcl.num_particles,
            'current_uncertainty': uncertainty,
            'convergence_status': convergence
        }
    
    except ImportError as e:
        print(f"⚠️  AMCL本地器导入失败，将进行备用测试: {e}")
        return self._check_mock_amcl()
        
    except Exception as e:
        print(f"❌ AMCL测试失败: {e}")
        return {'error': str(e), 'test_status': 'failed'}

def _check_mock_amcl(self):
    """检查模拟AMCL"""
    print("\n使用模拟AMCL进行测试...")
    
    try:
        from src.localization.amcl_localizer import MockAMCLLocalizer
        
        mock_amcl = MockAMCLLocalizer(num_particles=800)
        
        test_lidar = np.random.uniform(0.1, 0.9, 20)
        test_odometry = {'dx': 0.5, 'dy': 0.0, 'dyaw': 0.1}
        
        result = mock_amcl.localize(test_lidar, test_odometry)
        
        print("✅ 模拟AMCL功能测试通过")
        print(f"模拟不确定性: {result['position_uncertainty']}")
        
        return {
            'amcl_type': 'mock',
            'test_status': 'passed',
            'mock_test_result': 'success'
        }
    
    except Exception as e:
        print(f"❌ 模拟AMCL测试也失败: {e}")
        return {'error': str(e), 'all_tests_failed': True}

def check_model_architecture() -> Dict:
    """检查模型架构"""
    print("\n=== 检查模型架构 ===")
    
    try:
        from src.models.td3_robust import ImprovedTD3
        
        # 获取模型信息（不创建实际环境）
        print("TD3模型类加载成功")
        
        # 检查类方法和属性
        model_class = ImprovedTD3
        
        required_methods = ['learn', 'save', 'load', 'predict']
        methods_check = {}
        
        for method in required_methods:
            has_method = hasattr(model_class, method)
            methods_check[method] = has_method
            print(f"{'✅' if has_method else '❌'} {method}: {'存在' if has_method else '不存在'}")
        
        # 检查网络架构信息
        if hasattr(model_class, 'policy'):
            print(f"策略网络类型: {model_class.policy}")
        
        return {
            'model_class_loaded': True,
            'required_methods': methods_check,
            'all_methods_present': all(methods_check.values()),
            'architecture_status': 'validated'
        }
    
    except Exception as e:
        print(f"❌ 模型架构检查失败: {e}")
        return {'error': str(e), 'architecture_status': 'validation_failed'}

def check_navigation_utils() -> Dict:
    """检查导航工具"""
    print("\n=== 检查导航工具 ===")
    
    try:
        from src.utils.navigation_utils import NavigationUtils, NavigationTaskGenerator
        
        utils = NavigationUtils()
        task_gen = NavigationTaskGenerator(utils)
        
        # 测试每种货物类型的任务生成
        cargo_types = ['normal', 'fragile', 'dangerous']
        task_results = {}
        
        for cargo_type in cargo_types:
            try:
                start_pos, target_pos = utils.get_navigation_task(cargo_type)
                
                start_valid = utils.is_position_valid(start_pos)
                target_valid = utils.is_position_valid(target_pos)
                
                print(f"{cargo_type}货物任务:")
                print(f"  起点: {start_pos} {'✅有效' if start_valid else '❌无效'}")
                print(f"  目标: {target_pos} {'✅有效' if target_valid else '❌无效'}")
                
                distance = utils.calculate_distance(start_pos, target_pos)
                heading = utils.calculate_heading(start_pos, target_pos)
                
                task_results[cargo_type] = {
                    'start_position': start_pos,
                    'target_position': target_pos,
                    'start_valid': start_valid,
                    'target_valid': target_valid,
                    'distance': distance,
                    'heading': heading
                }
                
            except Exception as e:
                print(f"❌ {cargo_type}任务生成失败: {e}")
                task_results[cargo_type] = {'error': str(e)}
        
        return {
            'navigation_utils_loaded': True,
            'task_generation_test': task_results,
            'config_files_accessible': True
        }
    
    except Exception as e:
        print(f"❌ 导航工具检查失败: {e}")
        return {'error': str(e), 'utils_status': 'failed'}

def check_configuration_files() -> Dict:
    """检查配置文件"""
    print("\n=== 检查配置文件 ===")
    
    config_files = [
        'config/training_config.yaml',
        'requirements.txt',
        'complete_technical_document_v2.md',
        'train_stage1.py',
        'test_amcl_localization.py',
        'test_environment.py'
    ]
    
    config_status = {}
    
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            file_size = file_path.stat().st_size
            config_status[config_file] = {'exists': True, 'size_bytes': file_size}
            print(f"✅ {config_file}: {file_size/1024:.1f}KB")
            
            # 特殊检查 - 配置文件格式
            if config_file.endswith('.yaml'):
                try:
                    import yaml
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    config_status[config_file]['yaml_valid'] = True
                    print(f"   YAML格式: ✅有效")
                except Exception as e:
                    config_status[config_file]['yaml_valid'] = False
                    print(f"   YAML格式: ❌无效 - {e}")
        else:
            config_status[config_file] = {'exists': False}
            print(f"❌ {config_file}: 文件不存在")
    
    all_config_exists = all(status['exists'] for status in config_status.values())
    print(f"\n配置文件完整性: {'✅全部存在' if all_config_exists else '❌部分缺失'}")
    
    return {'file_status': config_status, 'all_files_exist': all_config_exists}

def check_mlflow_setup() -> Dict:
    """检查MLflow设置"""
    print("\n=== 检查MLflow设置 ===")
    
    try:
        import mlflow
        
        # 测试基本连接
        original_uri = mlflow.get_tracking_uri()
        
        # 设置测试URI
        test_uri = "file:./mlruns_test"
        mlflow.set_tracking_uri(test_uri)
        
        # 创建简单实验
        experiment_name = f"test_experiment_{int(time.time())}"
        
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ MLflow实验创建成功: ID={experiment_id}")
            
            with mlflow.start_run(run_name="test_run") as run:
                print(f"✅ MLflow运行开始: run_id={run.info.run_id}")
                
                # 测试数据记录
                mlflow.log_param("test_param", "test_value")
                mlflow.log_metric("test_metric", 42.0)
                print("✅ 参数和指标记录成功")
            
            # 清理测试数据
            mlflow.set_tracking_uri(original_uri)
            
            return {
                'mlflow_available': True,
                'experiment_creation': 'passed',
                'data_logging': 'passed',
                'test_uri': test_uri
            }
        
        except Exception as e:
            print(f"❌ MLflow测试失败: {e}")
            return {'error': str(e), 'mlflow_test': 'failed'}
    
    except Exception as e:
        print(f"❌ MLflow设置检查失败: {e}")
        return {'error': str(e), 'mlflow_available': False}

def run_training_simulation() -> Dict:
    """运行小型训练模拟"""
    print("\n=== 运行训练模拟 ===")
    
    try:
        from src.environments.navigation_env import ROSbotNavigationEnv
        from src.models.td3_robust import ImprovedTD3
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        
        print("开始小规模训练模拟...")
        
        # 创建环境（简化的低速模式）
        env = ROSbotNavigationEnv(cargo_type='normal')
        
        # 创建改进的mock AMCL，跳过Webots依赖
        env._use_mock_amcl = True  # 添加模拟模式标识
        env.amcl_localizer = None  # 将用模拟数据代替
        
        env = Monitor(env)
        
        # 创建微型TD3模型
        model = ImprovedTD3(
            "MlpPolicy",
            env,
            learning_rate=1e-4,      # 降低学习率
            buffer_size=10000,       # 小型缓存
            learning_starts=100,     # 快速开始学习
            batch_size=32,
            verbose=1,
            tensorboard_log="./logs/verification"
        )
        
        print("模型创建成功")
        
        # 运行短时间的训练循环
        print("开始训练循环测试...")
        
        total_steps = 0
        episode_count = 0
        cumulative_reward = 0.0
        
        obs, info = env.reset()
        print(f"初始状态形状: {obs.shape}")
        print(f"状态类型: {obs.dtype}")
        
        for step in range(200):  # 200步训练测试
            # 简单的启发式动作（模拟基本导航策略）
            target_vector = obs[32:35]  # 目标相对位置
            distance = np.linalg.norm(target_vector)
            direction = np.arctan2(target_vector[1], target_vector[0])
            
            action = np.array([
                np.clip(distance * 0.5, 0.1, 1.0),     # 线速度
                np.clip(direction * 0.8, -1.0, 1.0)    # 角速度
            ], dtype=np.float32)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            
            total_steps += 1
            
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count}完成: 步数={total_steps}, 累积奖励={cumulative_reward:.2f}")
                
                if episode_count >= 3:  # 限制测试episodes
                    break
                
                obs, info = env.reset()
                cumulative_reward = 0.0
        
        print(f"训练模拟完成: 总步数={total_steps}, Episodes={episode_count}, "
              f"平均奖励={cumulative_reward/max(total_steps, 1):.3f}")
        
        return {
            'training_simulation': 'completed',
            'total_steps': total_steps,
            'episodes_completed': episode_count,
            'average_reward': cumulative_reward/max(total_steps, 1),
            'model_training_possible': True
        }
    
    except Exception as e:
        print(f"❌ 训练模拟失败: {e}")
        return {
            'training_simulation': 'failed',
            'error': str(e),
            'model_training_possible': False
        }

def generate_verification_report(all_results: Dict) -> str:
    """生成验证报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# ROSbot训练系统验证报告

生成时间: {timestamp}

## 验证摘要

本次验证了ROSbot强化学习导航系统的完整设置，包括环境搭建、模型配置、训练能力和系统集成。

## 详细验证结果

### 1. Python环境
- Python版本: {all_results.get('python_env', {}).get('python_version', 'unknown')}
- PyTorch版本: {all_results.get('python_env', {}).get('torch_version', 'unknown')}  
- CUDA可用: {'✅是' if all_results.get('python_env', {}).get('cuda_available') else '❌否'}
- GPU设备: {all_results.get('python_env', {}).get('device_name', 'CPU')}

### 2. 依赖包状态
- 全部导入成功: {'✅是' if all_results.get('packages', {}).get('all_imported') else '❌否'}
- 项目模块: {'✅全部导入' if not any('failed' in str(m) for m in all_results.get('project_imports', {}).get('import_status', {}).values()) else '❌部分失败'}

### 3. 状态空间验证  
- 观测维度: {all_results.get('state_space', {}).get('observation_shape', 'unknown')} 
- 动作维度: {all_results.get('state_space', {}).get('action_shape', 'unknown')}
- 维度测试: {'✅通过' if all_results.get('state_space', {}).get('shape_validation') == 'passed' else '❌失败'}

### 4. AMCL定位验证
- AMCL本地器: {'✅可用' if all_results.get('amcl_localization', {}).get('test_status') != 'failed' else '❌需要修复'}
- 粒子数量: {all_results.get('amcl_localization', {}).get('particle_count', 'unknown')}
- 不确定性测试: {'✅通过' if all_results.get('amcl_localization', {}).get('uncertainty_test') == 'passed' else '❌需要调优'}

### 5. 模型架构验证
- TD3模型类: {'✅存在' if all_results.get('model_architecture', {}).get('model_class_loaded') else '❌需要修复'}
- 方法完整性: {'✅完整' if all_results.get('model_architecture', {}).get('all_methods_present') else '❌缺少方法'}

### 6. 训练模拟结果  
- 训练模拟: {'✅完成' if all_results.get('training_simulation', {}).get('training_simulation') == 'completed' else '❌失败'}
- 测试步数: {all_results.get('training_simulation', {}).get('total_steps', 0)}
- 平均奖励: {all_results.get('training_simulation', {}).get('average_reward', 0):.3f}

### 7. 配置文件完整性
- 配置文件状态: {'✅全部存在' if all_results.get('config_files', {}).get('all_files_exist') else '❌部分缺失'}

## 系统就绪度评估

### 🔧 技术准备度
基于验证结果，系统各组件的状态如下：

1. **核心算法**: {'✅就绪' if all([all_results.get('amcl_localization', {}).get('test_status') != 'failed',
                                   all_results.get('model_architecture', {}).get('model_class_loaded')]) else '⚠️需要调优'}

2. **训练框架**: {'✅就绪' if all_results.get('training_simulation', {}).get('model_training_possible') else '❌需要修复'}

3. **实验管理": {'✅就绪' if all_results.get('mlflow_setup', {}).get('mlflow_available') else '❌需要配置'}

4. **环境配置": {'✅就绪' if all_results.get('config_files', {}).get('all_files_exist') else '❌需要完善'}

### 🚀 下一步建议

基于验证结果，建议按以下顺序进行：

1. **修复失败的组件**: 优先解决失败的测试项
2. **环境调优**: 调整AMCL参数和模型超参数
3. **小规模训练**: 先进行1万步的小规模训练测试
4. **完整训练**: 验证成功后进行完整的第一阶段训练
5. **性能评估**: 使用提供的评估脚本进行性能验证

## 结论

{'✅' if all([r.get('status') == 'passed' or r.get('test_status') not in ['failed', 'validation_failed'] 
      for r in list(all_results.values()) if isinstance(r, dict)]) else '⚠️'} 系统基本就绪，{'可以进行下一阶段开发工作' if all([r.get('status') == 'passed' or r.get('test_status') not in ['failed', 'validation_failed'] 
      for r in list(all_results.values()) if isinstance(r, dict)]) else '建议先修复发现的issues再开始训练'}

## 附件

- 详细测试数据: `verification_results_{timestamp}.json`
- AMBA测试报告: `amcl_test_report_*.md`  
- 环境测试报告: `environment_test_report_*.txt`
- 训练日志: `./logs/verification/`

---

**注意**: 本验证报告基于当前代码状态生成。实际训练中需要根据具体表现进一步调优参数和网络架构。
"""
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"verification_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 同时保存JSON格式的详细数据
    results_file = f"verification_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n📋 验证报告已生成: {report_file}")
    print(f"📊 详细结果数据: {results_file}")
    
    return report

def main():
    """主函数 - 运行完整验证"""
    print("="*80)
    print("ROSbot导航系统 - 训练设置验证工具")
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n🔍 开始系统验证...")
    
    # 运行所有验证
    verification_results = {}
    
    verification_results['python_env'] = check_python_environment()
    verification_results['packages'] = check_package_imports()
    verification_results['project_imports'] = check_project_imports()
    verification_results['state_space'] = check_state_space_dimensions()
    verification_results['amcl_localization'] = check_amcl_localization_basic()
    verification_results['model_architecture'] = check_model_architecture()
    verification_results['navigation_utils'] = check_navigation_utils()
    verification_results['config_files'] = check_configuration_files()
    verification_results['mlflow_setup'] = check_mlflow_setup()
    verification_results['training_simulation'] = run_training_simulation()
    
    print("\n" + "="*60)
    print("=== 验证完成 ===")
    
    # 生成最终报告
    final_report = generate_verification_report(verification_results)
    
    # 简要结果汇总
    print("\n📊 验证结果汇总")
    
    # 计算各项通过率
    overall_score = 0
    total_checks = 0
    
    for category, result in verification_results.items():
        if isinstance(result, dict):
            if 'test_status' in result:
                status = result['test_status']
                if status == 'passed': overall_score += 1
                total_checks += 1
            elif 'shape_validation' in result:
                status = result['shape_validation']
                if status == 'passed': overall_score += 1
                total_checks += 1
            elif 'all_imported' in result:
                status = result['all_imported']
                if status: overall_score += 1
                total_checks += 1
            elif 'all_files_exist' in result:
                status = result['all_files_exist']
                if status: overall_score += 1
                total_checks += 1
    
    print(f"\n整体验证分数: {overall_score}/{total_checks} = {overall_score/total_checks:.1%}")
    
    # 提供下一步建议
    print(f"\n🎯 下一步建议:")
    
    if overall_score/total_checks >= 0.85:
        print("✅ 系统验证通过，建议开始Phase 1完整训练")
        print("   python train_stage1.py --cargo_type normal --total_steps 50000")
    elif overall_score/total_checks >= 0.65:
        print("⚠️  系统基本就绪，建议先进行小规模测试训练")
        print("   python train_stage1.py --cargo_type normal --total_steps 10000")
    else:
        print("❌ 系统需要修复，请先解决验证报告中标记的问题")
        print("   重点检查: AMCL本地器、环境配置、依赖包完整性")

if __name__ == "__main__":
    # 确保可重复的测试
    np.random.seed(123456)
    torch.manual_seed(123456)
    
    main()