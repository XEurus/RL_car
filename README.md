# ROSbot 导航系统：基于强化学习的端到端解决方案

本项目旨在通过深度强化学习，赋予 ROSbot 在复杂动态环境中自主导航的能力。我们基于 Stable-Baselines3 框架，实现了 TD3 算法，并在 Webots 仿真环境中进行了大量的训练与验证。

## 技术选型

*   **强化学习框架:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
*   **仿真平台:** Webots
*   **实验追踪:** MLflow
*   **核心算法:** TD3 (Twin Delayed Deep Deterministic Policy Gradient)

## 核心设计理念与特性

为了实现高效且鲁棒的导航策略，我们设计并实现了一系列创新特性，这些是本项目的核心亮点。

### 1. 训练策略：双维度课程学习 (Dual-Dimension Curriculum Learning)

我们摒弃了单一难度的训练方式，设计了一套双维度、可配置的课程学习体系，旨在引导智能体循序渐进地掌握复杂技能。

*   **垂直课程：动态障碍物递增**
    我们定义了一个基于全局训练步数的“难度时间表” (`obstacle_curriculum_steps` & `obstacle_curriculum_counts`)。随着训练的深入，环境中的障碍物数量会自动增加，迫使智能体从简单的避障学起，逐步适应更拥挤、更复杂的场景。

*   **阶段锁定机制 (`--lock_obstacles_per_stage`)**
    这是我们课程学习设计的关键。当启用此模式时，进入下一个难度阶段后，系统并不会生成一组全新的随机障碍物。相反，它会**在前一阶段已有的障碍物集合基础上，新增一个或多个障碍物**。这种“累加式”的难度增长，确保了模型在探索新技能的同时，不会“遗忘”已经掌握的简单场景，极大地提升了学习的稳定性和连续性。

*   **水平课程：任务参数自适应**
    与障碍物课程并行，我们还设计了基于任务阶段 (`easy`, `medium`, `hard`) 的参数自适应机制。在训练初期，环境会设置更长的回合时间 (`max_episode_steps`) 和更宽松的成功判定条件，鼓励智能体进行更广泛的探索。随着训练的进行，这些参数会变得愈发严苛，促使智能体优化其导航效率。

### 2. 奖励工程：面向行为的精细化塑造

我们认为，精细的奖励函数是塑造理想行为的关键。因此，我们将奖励函数设计为多个子项的加权和，每一项都针对一个特定的行为目标，并且其权重均可通过命令行参数 (`--<reward_name>_k`) 进行精细调整。

*   **目标导向奖励**
    *   `--distance_k`: 核心驱动力，鼓励智能体不断接近目标点。
    *   `--angle_reward_k`: 鼓励智能体使其朝向与目标方向保持一致，减少无效的侧向移动。
    *   `--directional_movement_k`: 直接奖励朝向目标方向的移动分量，确保每一步都在“做有用功”。

*   **安全与效率奖励**
    *   `--wall_proximity_penalty_k`: 对靠近墙壁或障碍物的行为施加惩罚，是保证安全的核心。
    *   `--time_k`: 对每一步施加微小的负奖励，以鼓励智能体寻找更短的路径，提高效率。

*   **行为平滑性与探索奖励**
    *   `--angle_change_k`: 惩罚剧烈的转向动作，使轨迹更平滑、更符合物理规律。
    *   `--early_spin_penalty_k`: 抑制在起点附近的原地打转行为，鼓励智能体快速开始有效探索。
    *   `--front_clear_k`: 奖励前方开阔无障碍的区域，引导智能体向更安全的方向探索。

### 3. 实验的可复现性与深度分析

我们高度重视实验的可复现性和结果分析的便捷性。

*   **MLflow 全方位追踪:** 训练脚本与 MLflow 深度集成。每次运行，不仅会自动记录所有超参数和性能指标，**甚至会将当前版本的训练脚本和整个 `src` 目录作为“物料” (Artifacts) 上传**。这保证了任何一次实验结果都具备完全的可追溯性和可复现性。

*   **轨迹可视化与数据导出:** 在每个回合结束后，系统会自动生成并保存一张包含机器人路径、奖励分布、起点、终点和障碍物布局的详细轨迹图。这些可视化结果对于调试和直观理解智能体行为至关重要。同时，原始轨迹数据也会被导出为 `.csv` 和 `.jsonl` 文件，便于进行更深入的离线数据分析。

## 使用指南

1.  **环境配置:**
    ```bash
    git clone <your-repo-url>
    cd RL_car
    pip install -r requirements.txt
    ```

2.  **开始训练:**
    我们提供了高度可配置的训练脚本 `train_single.py`。您可以直接运行以使用默认参数，或通过命令行进行深度定制。

    *   **启动一次标准训练:**
        ```bash
        python rosbot_navigation/train_single.py
        ```

    *   **高级训练示例 (微调与课程学习):**
        以下示例展示了如何从一个预训练模型开始，针对 `dangerous` 货物类型进行微调，同时启用了我们设计的“阶段锁定”课程学习，并为本次运行指定了实验名称。
        ```bash
        python rosbot_navigation/train_single.py \
            --experiment_name "FineTune-Dangerous-Cargo-V2" \
            --leaner_name "TD3-FineTune-Run1" \
            --cargo_type "dangerous" \
            --total_steps 200000 \
            --pretrained_model_path "/path/to/your/pretrained_model.zip" \
            --learning_rate 1e-4 \
            --enable_obstacle_curriculum true \
            --lock_obstacles_per_stage true \
            --draw_trajectory true
        ```

3.  **评估模型:**
    使用 `test_model_webots.py` 脚本来加载并评估您训练好的模型。
    ```bash
    python rosbot_navigation/test_model_webots.py --model_path "/path/to/your/trained_model.zip"
    ```


现在有以下几个工作：
1.测试统计。要统计平均加速度和平均线速度，统计成功率，统计碰撞率。我想在测试时，碰撞后不结束，向后退一定距离再重新触发模型。
2. 取货点手动操作车辆。在取货点为目的地时，将取货点目的地x-2作为目标点，在模型完成后，先将车辆转向对准目标点，移动到目标点，再转到正对的角度上，之后向前开两米。这个仅限于到达取货点。取货点作为出发点时，也先向前开两米，再使用模型控制。
3. 接入手动控制。允许手动选择任务类型，即手动控制任务，用于图形化展示，与自动进行测试分开。
4. 接入controller中的功能。 @simple_controller.py 根据该代码，@simple_controller.py#L279-280 把识别图像的功能和想后端发送数据的功能都添加到手动控制任务中。把键盘控制运动作为一个单独的模式也保存下来。自动测试时不需要识别和向后端发送数据。

完成上述工作，并进行测试



while not done:
        with torch.no_grad():
            action, _ = model.policy.predict(env._get_observation(), deterministic=deterministic)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # # 碰撞处理（依据 info['_last_collision_info']）
        # collision_info = info.get('_last_collision_info')
        # if collision_info:
        #     collision_count += 1
        #     print(f"💥 检测到碰撞! 第 {collision_count} 次. 开始执行后退操作...")
        #     # 后退操作：发送负向速度指令
        #     backup_speed = -5.0  # 后退速度
        #     backup_duration_ms = 1000  # 后退持续时间 (ms)
        #     backup_steps = int(backup_duration_ms / env.control_period_ms)

        #     for _ in range(backup_steps):
        #         _set_raw_wheel_velocities(env, backup_speed, backup_speed)
        #         env.robot.step(env.timestep)
            
        #     # 后退后停止
        #     _set_raw_wheel_velocities(env, 0, 0)
        #     env.robot.step(env.timestep)
        #     print("后退完成，恢复模型控制.")

        #     # 由于我们不希望碰撞终止episode，重置终止状态
        #     # 注意：这会覆盖掉环境因碰撞返回的 terminated=True
        #     terminated = False

        steps += 1
        total_reward += float(reward)
        done = bool(terminated or truncated)

        # 若为取货点目的地：当到达接近点（x-2）时，执行手动停靠序列
        if done:
            dist_final, _ = env._calculate_distance_to_target()
            success = dist_final < 0.3
            if test_id in [1,2,3] and success:
                try:
                    env.task_info['target_pos']=target_pos
                    done_2=False
                    while not done_2:
                        with torch.no_grad():
                            action, _ = model.policy.predict(env._get_observation(), deterministic=deterministic)

                        next_obs, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        total_reward += float(reward)
                        done_2 = bool(terminated or truncated)
                        if done_2:
                            turn_to_angle(env, target_pos[2])
                            env._send_wheel_velocities(0.0, 0.0)
                            env.robot.step(env.timestep)
                    # cur = np.array(env._get_sup_position(), dtype=float)
                    # if np.linalg.norm(cur[:2] - approach_target[:2]) < 0.5:
                    #     print("模型已到达接近点，切换到手动停靠序列...")
                    #     # 1) 转向最终目标点
                    #     angle_to_target = env._calculate_angle_to_target(target_pos)
                    #     turn_to_angle(env, angle_to_target)
                    #     # 2) 直行到最终目标
                    #     cur_after_turn = np.array(env._get_sup_position(), dtype=float)
                    #     dist_to_final = float(np.linalg.norm(target_pos[:2] - cur_after_turn[:2]))
                    #     move_forward_distance(env, dist_to_final, speed=0.6)
                    #     # 3) 转到正对的角度（这里采用 180°）
                    #     turn_to_angle(env, target_pos[2])
                    #     # # 4) 向前开两米
                    #     # move_forward_distance(env, 2.0, speed=0.6)
                    #     # 停止
                    #     env._send_wheel_velocities(0.0, 0.0)
                    #     env.robot.step(env.timestep)
                    #     # 结束本回合并标记手动对接成功
                    #     manual_docked = True
                except Exception:
                    raise