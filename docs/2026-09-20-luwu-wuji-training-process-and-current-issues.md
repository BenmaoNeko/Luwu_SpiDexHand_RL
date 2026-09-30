# Luwu 灵巧手 Wuji/MJLab 训练流程与问题记录

最后更新：2026-09-20  
项目：`Luwu_Hand_Wuji` / `LuwuHand_Reorient`  
状态：训练已暂停；最新可用 checkpoint 为 `model_1600.pt`；当前模型尚未完成有效的方块重定向。

本文记录从真实手标定资料、URDF/MJCF 建模、Wuji 注册、训练环境排障、服务器训练，到最新 checkpoint 拉回本地预览的完整过程。文中“当前”指 2026-09-20 的代码和训练状态。

## 1. 项目目标与边界

目标是将当前 Arm_Hand_002 的四指灵巧手接入 Wuji/MJLab 的方块在手重定向任务，用强化学习训练 16 个灵巧手关节。

确定的边界如下：

- 机械臂不参与强化学习控制。
- 机械臂在仿真中作为固定物体存在，只保留视觉和碰撞几何，不保留机械臂关节执行器。
- 机械臂 J4 固定为 `+1.5 rad`，使掌心轴线垂直地面，方块可以从上方放在掌心。
- 只有灵巧手 16 个关节参与动作输出。
- 真实实验没有条件观测方块在腕部 Tag 坐标系中的位置，因此该位置不进入 Actor policy observation；只保留在 Critic 的 privileged observation 中。
- 方块 reset 时固定放在掌心对应位置，只改变六个面方向，不从高处掉落，也不加入 reset 时位置噪声和外力扰动。

## 2. 作为依据的真实硬件资料

本次仿真和训练使用的真实手映射来自：

- `/home/benmao/桌面/Open_Arm-Dexterous_Hand/Arm_Hand_002/docs/rebotarm_hand_calibration_record_2026-09-17.md`
- `/home/benmao/桌面/Open_Arm-Dexterous_Hand/Arm_Hand_002/src/rebotarm_hand/config/spidex_hand_mapping.yaml`
- `/home/benmao/桌面/Open_Arm-Dexterous_Hand/Arm_Hand_002/docs/rebotarm_hand_ros_integration.md`

真实手使用 16 个舵机：`11–14`、`21–24`、`31–34`、`41–44`。机械臂 J1–J6 使用独立的 `can0` 链路，与灵巧手串口映射分开。

### 2.1 实物手指到 URDF 手指的最终映射

| URDF 手指 | URDF 关节 | 实物手指 | 实物电机组 |
| --- | --- | --- | --- |
| Finger 1 | `joint7`–`joint10` | 实物 1 | `11–14` |
| Finger 2 | `joint11`–`joint14` | 实物 4 | `41–44` |
| Finger 3 | `joint15`–`joint18` | 实物 3 | `31–34` |
| Finger 4 | `joint19`–`joint22` | 实物 2 | `21–24` |

即：

~~~text
URDF 1 <- 实物 1（11–14）
URDF 2 <- 实物 4（41–44）
URDF 3 <- 实物 3（31–34）
URDF 4 <- 实物 2（21–24）
~~~

每根手指的方向模式为：

~~~text
joint1: -1
joint2: -1
joint3: +1
joint4: -1
~~~

真实手的原始位置换算为：

~~~text
joint_rad = direction × (raw_position - zero_raw) × (2π / 1023)
~~~

URDF 限位以真实映射文件和 URDF 为准，不在训练配置中重新发明一套关节范围。

### 2.2 真实 Home 与仿真 Home

真实标定中，十字构型使用每根实物手指第一个关节 `-35°` 的命令；对应到 URDF 后方向相反。

当前训练使用的仿真 Home 是已经确认过的 URDF 反向姿态：

~~~text
right_finger1_joint1 = +0.610865238
right_finger2_joint1 = +0.610865238
right_finger3_joint1 = +0.610865238
right_finger4_joint1 = +0.610865238

所有 joint2、joint3、joint4 = 0
~~~

因此当前训练 Home 不是“所有关节角为零”。它对应真实灵巧手的已确认 Home 映射。

## 3. 从 URDF 到 MuJoCo/MJLab 的建模过程

### 3.1 姿态调整

早期曾误以为需要移动 J5，之后根据用户确认改为 J4：

~~~text
J4 = +1.5 rad
~~~

该姿态被烘焙进固定 MJCF。训练时不再控制 J1–J6，也不会为机械臂创建执行器。

### 3.2 碰撞几何的处理过程

最初的计划是将所有碰撞体生成凸包，以降低碰撞计算量；之后为了保证灵巧手指尖、掌心和方块接触的真实性，最终采用了分区策略：

| 部分 | 最终碰撞几何 |
| --- | --- |
| 机械臂底座和 link1–link5 | 凸包 STL |
| 掌心 link6 | 原始 STL 网格 |
| 四根手指全部 link1–link4 | 原始 STL 网格 |
| 掌心辅助支撑区域 | 一块薄的静态 box collider |

最终 MJCF：

~~~text
/home/benmao/桌面/Luwu_Hand_Wuji/reference/wuji-mjlab-main/src/wuji_mjlab/assets/robots/luwu_hand/mjcf/luwu_mjlab.xml
~~~

掌心辅助支撑碰撞体为无质量薄盒体，主要用于避免方块在无手指接触时立即从掌心滑落。它不参与质量计算：

~~~xml
<geom name="right_palm_support_collision"
      type="box"
      size="0.03 0.03 0.0005"
      density="0" />
~~~

### 3.3 执行器与限制

当前模型只为 `joint7`–`joint22` 创建 16 个位置型 PD 执行器：

- 动作维度：16
- 最大扭矩：`±0.5 N·m`
- `kp = 1.0`（启动随机化范围 `0.8–1.2`）
- `kv = 0.15`（名义值；启动 `kd_range` 随机化范围为 `0.8–1.2`）
- `ctrlrange`：按 URDF 每个关节的上下限
- 机械臂没有执行器

这样训练输出的是灵巧手关节位置偏移动作，最终力矩由 MuJoCo 执行器限制在 `0.5 N·m` 内。

## 4. Wuji 任务注册

任务注册位置：

~~~text
/home/benmao/桌面/Luwu_Hand_Wuji/reference/wuji-mjlab-main/src/wuji_mjlab/tasks/reorient/config/luwu_hand/
~~~

主要文件：

- `env_cfgs.py`：Luwu 环境绑定和 reset 覆盖
- `luwu_constants.py`：Luwu Home、方块初始位置和六面方向
- `__init__.py`：注册 `LuwuHand_Reorient` 和 `LuwuHand_Reorient_Light`
- `luwu_hand_cfg.py`：固定机械臂和 16 DOF 灵巧手实体配置

当前训练任务使用：

~~~text
LuwuHand_Reorient
~~~

## 5. Observation、Action 和网络结构

### 5.1 Actor observation

Actor policy 的输入维度为 `162`，包含三帧历史：

| 观测项 | 单帧维度 | 三帧维度 | 说明 |
| --- | ---: | ---: | --- |
| `noisy_joint_angles` | 16 | 48 | 带噪关节角 |
| `qpos_error` | 16 | 48 | 目标关节位置误差 |
| `cube_ori_error` | 6 | 18 | 方块方向误差的 6D 表示 |
| `action_history` | 16 | 48 | 历史动作 |
| 合计 | 54 | 162 | Actor 输入 |

Actor 不读取方块在腕部 Tag 坐标系中的位置。

### 5.2 Critic observation

Critic 输入维度为 `344`，保留仿真中可以获得但真实实验难以直接测量的 privileged state，包括：

- `cube_pos_in_tag`
- `cube_pos_in_tag_clean`
- 方块线速度和角速度
- 指尖位置
- 真实关节位置和速度
- 扰动方向/速度
- cage 状态
- domain randomization 参数
- 掌心姿态

这是 asymmetric Actor-Critic 设计：Actor 面向真实可部署观测，Critic 可以使用仿真特权信息辅助价值估计。

### 5.3 Action

动作维度为 `16`，对应四根手指的 16 个关节。使用关节位置 offset action，并经过 EMA 平滑和 warmup：

~~~text
action term: joint_pos
action_scale: 0.5
ema_alpha: 0.5
warmup_time_s: 0.4
~~~

动作不是直接输出舵机原始位置，也不是直接输出 torque；它通过位置 PD 执行器作用到灵巧手关节。

### 5.4 PPO 网络

训练日志中加载的 Actor：

~~~text
162 -> 512 -> 256 -> 128 -> 32 -> (2, 16)
activation: ELU
distribution: SoftplusGaussianDistribution
min_std: 0.2
~~~

Critic：

~~~text
344 -> 512 -> 512 -> 256 -> 128 -> 1
activation: ELU
~~~

## 6. 方块 reset 和环境一致性问题

### 6.1 目标 reset 行为

Luwu reset 被设计为：

- 方块位置固定在校准掌心位置；
- `pos_noise = 0`；
- 只从六个离散方块面方向中选择一个；
- 移除 reset 时外力扰动；
- 移除方块质量、尺寸、质心的 startup randomization；
- 不因 reset 把方块从高处释放。

### 6.2 训练初期的主要 bug

最初为了让机械臂“完全固定”，直接删除了 `reset_robot_pose` 事件。这个做法造成了固定机械臂和方块的环境原点不一致：

1. 固定 base 机械臂通过 MuJoCo mocap 放置。
2. 每个并行环境的方块位置会加上该环境的 `env_origins`。
3. 删除 `reset_robot_pose` 后，所有机械臂 mocap 都停在世界原点。
4. 方块却被放到各自环境原点。
5. 在 4096 环境、`env_spacing=0.75` 时，部分方块离手掌可以达到数十米。

这解释了早期训练长期奖励约为 `-661`、方块掉落和训练完全失败的问题。

### 6.3 最终修复

没有删除 `reset_robot_pose`，而是保留该 reset 事件并将 pose noise 设为空范围：

~~~python
reset_robot_pose = cfg.events["reset_robot_pose"]
reset_robot_pose.params["pose_range"] = {}
~~~

这不是让机械臂在 episode 中运动，而是在 reset 时把固定 base 的 mocap replica 放到各自环境原点；episode 内机械臂仍完全静止。

### 6.4 4 环境真实验证

本机使用真实 `ManagerBasedRlEnv`、4 个环境和训练配置 reset 后验证：

- 机械臂 mocap 位置等于各自 `env_origins`；
- 四个环境的方块—掌心相对位置完全一致；
- 结果：

~~~text
RESET_ALIGNMENT=PASS
~~~

因此，当前训练日志中方块可以稳定处于掌心，说明这一类环境原点错位问题已经解决。

## 7. 奖励函数（接触引导修复后）

当前 Luwu active reward terms：

| 奖励项 | 权重 | 作用与预算 |
| --- | ---: | --- |
| `hand_pose` | `-0.02` | 手部姿态软惩罚 |
| `action_rate` | `-0.1` | 动作变化软惩罚 |
| `torque` | `-2.4` | 归一化扭矩软惩罚，硬限矩仍为 `0.5 N·m` |
| `tip_slide` | `-0.3` | 指尖滑动惩罚 |
| `cage_escape` | `-500.0` | 方块离开有效空间惩罚 |
| `finger_collision` | `-1.0` | 手指自碰撞惩罚 |
| `approach_progress` | `+1.0` | 指尖接近方块表面，episode 总预算 ≤`0.25` |
| `contact_milestones` | `+1.0` | 一指稳定接触 `+0.125`，二指稳定接触再 `+0.125` |
| `goal_progress` | `+1.0` | 仅在指尖接触时支付，每目标总预算 ≤`2.0` |
| `rotation_success` | `+5.0` | 旋转目标满足 4°、接触、连续保持条件后一次发放 |
| `flip_success` | `+5.0` | 翻面成功一次发放，随后 `flip_completed` terminal |

Luwu 已移除 `orientation_alignment`、`hold_escalation` 和 `palm_detach`。因此静止托举不能获得旧的高姿态满额奖励；没有指尖接触时，目标进展和成功奖励均为零。接触靠近奖励是有限的，不能靠反复接触或靠近-远离循环刷分。

## 8. 服务器训练流程

### 8.1 服务器环境

服务器：远端 GPU 服务器（连接地址不纳入仓库）  
远端项目：

~~~text
/root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main
~~~

训练 Python：

~~~text
/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/bin/python
~~~

服务器 GPU：NVIDIA GeForce RTX 4090。

### 8.2 训练命令

本次 3000 轮训练使用：

~~~bash
cd /root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/lib:${LD_LIBRARY_PATH:-}"

/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/bin/python \
  scripts/train/train_rsl_rl.py \
  --task LuwuHand_Reorient \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.logger tensorboard \
  --gpu-ids '[0]'
~~~

注意：`tyro` 对 `gpu_ids` 的 union/list 参数要求使用 `'[0]'`，写成 `--gpu-ids 0` 会解析失败。

### 8.3 代码同步

训练前同步了本机：

~~~text
/home/benmao/桌面/Luwu_Hand_Wuji/reference/wuji-mjlab-main/src
/home/benmao/桌面/Luwu_Hand_Wuji/reference/wuji-mjlab-main/scripts
~~~

到服务器对应目录。同步后验证了以下文件的本机/远端 SHA256 一致：

- `src/wuji_mjlab/tasks/reorient/config/luwu_hand/env_cfgs.py`
- `src/wuji_mjlab/tasks/reorient/scripts/view_task.py`
- `src/wuji_mjlab/tasks/reorient/tests/test_pipeline_invariants.py`

远端语法编译检查通过。

## 9. 训练结果与当前诊断

### 9.1 808 轮

~~~text
Mean reward: -59.68
Mean episode length: 1000
Mean action std: 0.20
orientation error: 2.1714 rad
goal_reach_count: 0
fingertip_contact_count: 0.0002
cage_drop: 0
cube_height_above_palm: 0.1258
joint_acceleration_rms: 178.9612
joint_vel_rms: 0.5513
~~~

此时已经没有明显方块掉落，但策略还没有建立指尖操作。

### 9.2 1585 轮

~~~text
Mean reward: -1.67
Mean episode length: 1000
Mean action std: 0.20
orientation error: 2.1464 rad
goal_reach_count: 0
fingertip_contact_count: 0.0002
cage_drop: 0
cube_height_above_palm: 0.1258
joint_acceleration_rms: 69.4107
joint_vel_rms: 0.2439
~~~

相比 808 轮：

- 平均奖励从 `-59.68` 降到 `-1.67`，数值上明显改善；
- 关节速度和加速度明显降低；
- 扭矩惩罚从 `-1.7236` 降到 `-0.5992`；
- 方块稳定性保持不变；
- 但姿态误差只从 `2.1714` 降到 `2.1464 rad`，改善约 1.1%；
- 指尖接触、成功次数和 hold counter 没有任何实质进展。

### 9.3 当前结论

当前策略已经学会：

~~~text
保持方块在掌心 + 减少动作 + 减少扭矩和速度
~~~

当前策略尚未学会：

~~~text
用指尖建立接触 + 将方块从掌心托举/夹持 + 旋转到目标面
~~~

因此，这次训练不能判定为任务成功。主要问题已经从“环境错位/方块掉落”转移为“策略陷入静态托举局部最优”。

### 9.4 `cube_survival_steps = 500.5` 的解释

该指标是 episode 内计数的平均值，不是最终计数。对完整 1000 步 episode，计数平均值约为：

~~~text
(1 + 2 + ... + 1000) / 1000 = 500.5
~~~

所以 `500.5` 与 `Mean episode length = 1000` 一致，不表示方块只存活了半个 episode。

## 10. 暂停训练后的 checkpoint handoff

训练暂停后，远端最新 checkpoint 是：

~~~text
/root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main/logs/rsl_rl/wuji_reorient/2026-09-20_11-53-14_Luwu_Reorient/model_1600.pt
~~~

用户日志显示到 `1585/3000`，但保存策略每 50 轮保存一次，所以最新保存文件是 `model_1600.pt`。

同时拉取了：

- `model_1600.pt`
- `events.out.tfevents.1789876402.gpufree-container.3145723.0`
- `params/env.yaml`
- `params/agent.yaml`

本地 handoff 目录：

~~~text
/home/benmao/桌面/Luwu_Hand_Wuji/reference/wuji-mjlab-main/logs/rsl_rl/wuji_reorient/remote-2026-09-20_11-53-14_Luwu_Reorient/
~~~

`model_1600.pt` SHA256：

~~~text
162cb82b5418db26089ebe4f43d3610c741815e5470785811e2f2cbc5a50dd70
~~~

## 11. 最新模型本地预览

### 11.1 checkpoint 加载验证

使用项目现有 trained-policy 入口进行了短时 headless smoke test：

- checkpoint 严格加载成功；
- Actor observation：`162`；
- action：`16`；
- Luwu play 环境 reset 事件正常存在；
- 视频 smoke test 输出到：

~~~text
/home/benmao/桌面/Luwu_Hand_Wuji/.handoff/2026-09-20_luwu_model1600_smoke/
~~~

### 11.2 本地交互式预览命令

~~~bash
cd /home/benmao/桌面/Luwu_Hand_Wuji/reference/wuji-mjlab-main

export PYTHONNOUSERSITE=1
export DISPLAY=:1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/home/benmao/mambaforge/envs/wheel_leg/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"

/home/benmao/mambaforge/envs/wheel_leg/bin/python \
  scripts/play/play_rsl_rl.py \
  --task LuwuHand_Reorient \
  --agent trained \
  --checkpoint-file "$PWD/logs/rsl_rl/wuji_reorient/remote-2026-09-20_11-53-14_Luwu_Reorient/model_1600.pt" \
  --num-envs 1 \
  --device cpu \
  --viewer native
~~~

原生 MuJoCo viewer 操作：

- `Enter`：执行完整 `env.reset()`；
- `Space`：暂停/继续仿真；
- `Esc`：关闭窗口。

## 12. 已完成的验证

当前已完成的证据：

1. 本机 Luwu/reorient 全量测试：`81 passed`。
2. 固定机械臂多环境 reset 原点一致性测试：`RESET_ALIGNMENT=PASS`。
3. 本机相关 Python 文件 `compileall` 通过。
4. 服务器同步后关键文件 SHA256 一致。
5. 服务器相关 Python 文件语法编译通过。
6. `model_1600.pt` 从服务器拉取后 SHA256 一致。
7. checkpoint strict load 通过，Actor/Action 维度匹配。
8. 本地 trained-policy smoke test 通过。
9. 确定性 Contact Test 通过：不加载训练模型，手动驱动单根指尖从 Home 到方块；验证了 `raw MuJoCo contact -> live MJWarp contact -> tip_object_contact.data.found=1 -> fingertip_contact_count=1` 的完整链路。

Contact Test 的最新结果（finger 1，CPU，20 步）：

~~~text
raw pair: `object/cube ↔ robot/right_finger1_link4_col`
sensor force norm: `2.062876`
metric: `1.0` at step `9`
~~~

本次真实验证使用 1 个环境、CPU、固定 seed `20260920`，没有构造 PPO runner，也没有加载 checkpoint。

以下结论目前不能声称已经成立：

- 灵巧手已经学会方块重定向；
- 已经出现有效指尖夹持；
- 已经有非零任务成功率；
- 当前 checkpoint 可以直接用于真实手部署。

## 13. 当前未解决问题

### 13.1 历史问题：静态托举局部最优（已针对性修复）

旧训练确实没有明显鼓励“建立指尖接触”：指尖接触只是 Metrics，策略可以选择不动手指而保持方块在掌心，并通过降低 action/torque cost 获得局部最优。本轮已加入接近、接触里程碑和接触门控进展，并将成功判定绑定到真实指尖接触；是否足以让 PPO 收敛仍需新一轮训练验证。

### 13.2 需要区分策略问题和接触检测问题

训练日志中的 `fingertip_contact_count ≈ 0` 仍然说明策略没有建立指尖接触；但接触链本身已经通过独立测试，不再把它作为当前首要怀疑点。

确定性 Contact Test 已确认：

1. MuJoCo raw contact 能识别 `object/cube` 与目标 `right_fingerN_link4_col`；
2. live MJWarp contact 列表包含同一 geom pair；
3. `tip_object_contact.data.found` 对应槽位变为 1；
4. `fingertip_contact_count` 同步变为 1。

因此，下一步重点应转向策略是否主动把指尖移动到方块，而不是继续修改 contact sensor 的几何匹配。

### 13.3 旧 checkpoint 的动作抖动记录

1585 轮时：

~~~text
action_delta_rms = 0.2718
action_jerk_rms = 0.4550
joint_acceleration_rms = 69.4107
~~~

虽然相比 808 轮已经降低，但仍应在预览中观察手指是否有高频抖动。当前不是扭矩饱和问题，因为：

~~~text
torque_saturation_ratio = 0
~~~

### 13.4 旧 checkpoint 不能直接用于真实手

当前 checkpoint 的 `goal_reach_count = 0`，且姿态误差仍约 `2.15 rad`。它最多证明仿真环境能够稳定运行，不能证明完成了灵巧手重定向任务，也不能作为真实舵机部署策略。

## 14. 新奖励实现后的下一步

建议按以下顺序推进：

1. 先用本地原生预览检查新任务的 home、掌心支撑和目标切换；`R`/`Enter` 使用训练同一 reset 路径。
2. 用短周期 ablation 对比零动作、随机动作和接触脚本，确认静止总回报低于实际接触/操作轨迹。
3. 再启动短训练，重点观察首次接触时间、接触率、旋转目标完成数、翻面完成率、掉落率和最小姿态误差。
4. 确认 CPU/GPU MJWarp 的接触传感器时序一致后，再决定是否恢复完整训练。
5. 新奖励实现已完成代码验证，但尚未宣称 PPO 已经学会旋转或翻面。

## 15. 相关代码与资料索引

核心代码：

- `reference/wuji-mjlab-main/src/wuji_mjlab/assets/robots/luwu_hand/mjcf/luwu_mjlab.xml`
- `reference/wuji-mjlab-main/src/wuji_mjlab/assets/robots/luwu_hand/luwu_hand_cfg.py`
- `reference/wuji-mjlab-main/src/wuji_mjlab/tasks/reorient/config/luwu_hand/env_cfgs.py`
- `reference/wuji-mjlab-main/src/wuji_mjlab/tasks/reorient/config/luwu_hand/luwu_constants.py`
- `reference/wuji-mjlab-main/src/wuji_mjlab/tasks/reorient/reorient_terms.py`
- `reference/wuji-mjlab-main/src/wuji_mjlab/tasks/reorient/mdp/metrics.py`
- `reference/wuji-mjlab-main/src/wuji_mjlab/tasks/reorient/scripts/view_task.py`
- `reference/wuji-mjlab-main/scripts/play/play_rsl_rl.py`
- `reference/wuji-mjlab-main/scripts/train/train_rsl_rl.py`
- `tools/contact_test.py`

测试：

- `reference/wuji-mjlab-main/src/wuji_mjlab/tasks/reorient/tests/test_pipeline_invariants.py`
- `tests/test_contact_test.py`

真实手资料：

- `/home/benmao/桌面/Open_Arm-Dexterous_Hand/Arm_Hand_002/docs/rebotarm_hand_calibration_record_2026-09-17.md`
- `/home/benmao/桌面/Open_Arm-Dexterous_Hand/Arm_Hand_002/src/rebotarm_hand/config/spidex_hand_mapping.yaml`
- `/home/benmao/桌面/Open_Arm-Dexterous_Hand/Arm_Hand_002/docs/rebotarm_hand_ros_integration.md`
