# Luwu Hand 第一成功模型基线

记录日期：2026-09-21（Asia/Shanghai）  
任务：`LuwuHand_Reorient`  
运行名：`Luwu_Reorient`  
状态：训练完成，作为后续实验的第一成功基线

## 1. 固化模型与训练产物

远端训练目录：

```text
/root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main/logs/rsl_rl/wuji_reorient/2026-09-20_18-01-59_Luwu_Reorient
```

训练命令目标为 3000 个 iteration，项目最终保存的最后 checkpoint 是 `model_2999.pt`，没有生成 `model_3000.pt`。

本地归档目录：

[`baseline-2026-09-21-first-success/`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success)

归档内容：

- [`model_2999.pt`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/model_2999.pt)
- [`events.out.tfevents...`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/events.out.tfevents.1789898527.gpufree-container.3287352.0)
- [`env.yaml`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/env.yaml)
- [`agent.yaml`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/agent.yaml)
- [`metadata.json`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/metadata.json)
- [`tensorboard_bundle.json`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/tensorboard_bundle.json)

文件校验：

| 文件 | SHA-256 |
|---|---|
| `model_2999.pt` | `17ca59396093228b87b9488b1b6d4ac3576b095df4187d8c4387211de6dc4d3d` |
| TensorBoard event | `910f0ae6751965b262b9eca3c9dce02d0faa532d2086a1b69d5def3e29ccdfc7` |

模型接口尺寸：actor observation `162`，critic observation `344`，action `16`。

## 2. 训练环境与 reset 基线

- 并行环境：4096
- 环境间距：`0.75 m`
- MuJoCo timestep：`0.01 s`
- decimation：5
- 控制周期：`0.05 s`
- episode 上限：`50 s`
- 机械臂：完全固定，MJCF 中使用 J4=`+1.5 rad` 的固定姿态，不配置机械臂关节执行器
- 灵巧手：16 个关节执行器
- 碰撞：Luwu 手使用原始 STL mesh collision；当前训练使用项目 MJCF 中的碰撞网格配置
- 方块初始位置：`(-0.0264848718, 8.59749199e-8, 0.40649396) m`
- 方块初始姿态：`wxyz=(1, 0, 0, 0)`，A 面朝上
- 方块初始线速度、角速度：均为 0
- reset 方块位置噪声：0
- reset 时不使用方块外力、质量、尺寸扰动事件

方块每个 episode 从 A 面朝上的静止状态开始；任务目标由 command term 决定。

## 3. 任务目标与成功判定

训练任务采样为：

- 50%：增量旋转任务，每次目标为当前姿态继续旋转 `+15°`
- 50%：翻面任务，目标为相对当前初始面翻转 `90°`，A→B
- 成功姿态误差阈值：`4°`
- 成功保持：5 个控制步
- 成功必须满足至少 1 个指尖接触方块
- 接触任务同步评估：开启
- 旋转成功后进入下一个 `+15°` 目标
- 翻面成功后触发 `flip_completed`，结束当前 episode

注意：本条训练基线与本地预览的 4+4 分组逻辑不同。4+4（0–3 旋转、4–7 翻面）只用于预览观察；本次 4096 环境训练仍使用 command 内部的 50/50 任务采样。

## 4. Actor / Critic 观测

### Actor：162 维

历史长度为 3，包含：

- `noisy_joint_angles`：16 × 3 = 48
- `qpos_error`：16 × 3 = 48
- `cube_ori_error`：6 × 3 = 18
- `action_history`：16 × 3 = 48

actor 不包含方块在腕部 Tag 坐标系的位置，也不直接包含接触状态；这是为了匹配实际实验中无法观测方块 Tag 位置的条件。

### Critic：344 维

critic 保留 privileged information，包括：

- 方块在 Tag 坐标系的位置
- 方块线速度、角速度
- 指尖位置
- 干净的方块位置和姿态误差
- 真实关节位置、关节速度
- 接触相关状态、cage 状态、扰动/域随机化参数
- 手掌旋转 6D 表示

接触传感器 `tip_object_contact` 用于成功判定和奖励门控，但没有加入 actor 观测。

## 5. 动作、关节限位与 PD

- 动作类型：16 维 joint position target
- `action_scale=1.0`
- `use_joint_limits=True`
- EMA 平滑：`alpha=0.5`
- warmup：`0.4 s`
- 所有关节限位以 URDF/MJCF 为准
- 舵机力矩硬限制：`±0.5 N·m`
- XML 名义位置控制：`kp=1.0`，`kv=0.15`
- startup PD 随机化：`kp∈[0.8,1.2]`，`kd∈[0.8,1.2]`

当前限位分组：

| 关节 | 限位（rad） |
|---|---:|
| finger1/finger4 J1 | `[-0.8, 1.3]` |
| finger2/finger3 J1 | `[-0.8, 1.0]` |
| 所有 J2 | `[-1.5, 0.5]` |
| 所有 J3 | `[-1.5, 1.5]` |
| 所有 J4 | `[-1.0, 1.0]` |

home 状态：每根手指 J1 为 `+0.610865238 rad`，其余关节为 XML 默认零位。

## 6. 奖励函数基线

当前奖励函数已经移除旧的、会让不接触也获得高分的项：

- `orientation_alignment`
- `hold_escalation`
- `palm_detach`
- `hand_pose`
- `tip_slide`

当前启用的奖励项：

| 奖励项 | 权重 | 作用与关键参数 |
|---|---:|---|
| `approach_progress` | `+1.0` | 指尖到方块表面的有界单调接近进度；`step_budget=2.0` |
| `contact_milestones` | `+1.0` | 稳定一指接触 `1.0`，稳定二指接触 `1.0`，稳定时间 `0.1 s` |
| `goal_progress` | `+1.0` | 只有指尖接触有效时才奖励姿态进度；`step_budget=10.0` |
| `rotation_success` | `+20.0` | 增量旋转目标完成的一次性奖励 |
| `flip_success` | `+30.0` | 翻面目标完成的一次性奖励 |
| `action_rate` | `-0.02` | 对实际 EMA joint target 的平滑性惩罚 |
| `torque` | `-0.02` | 以 `0.5 N·m` 为上限归一化的平方力矩惩罚，取 mean |
| `finger_collision` | `-0.05` | 归一化手指自碰撞惩罚 |
| `cage_escape` | `-0.2` | 有界方块 cage 逃逸惩罚 |
| `cage_drop` | `-10.0` | 方块掉落稀疏惩罚 |

奖励设计的关键约束是：姿态进度必须经过指尖接触门控；接近和接触奖励只提供有限的探索引导；旋转和翻面成功奖励是有界的一次性事件奖励。

## 7. PPO 与网络参数

- 随机种子：42
- 最大迭代：3000
- 最后保存迭代：2999
- 每次 rollout：40 steps/env
- 保存间隔：50 iterations
- actor MLP：`512 → 256 → 128`
- critic MLP：`512 → 512 → 256 → 128`
- 激活：ELU
- actor observation normalization：开启
- actor distribution：`SoftplusGaussianDistribution`
- 初始 std：`0.5`
- 最小 std：`0.2`
- PPO learning rate：`1e-4`
- learning epochs：4
- mini-batches：32
- `gamma=0.99`
- `lambda=0.95`
- entropy coefficient：`0.001`
- desired KL：`0.01`
- max grad norm：`1.0`
- value loss coefficient：`0.5`
- clip parameter：`0.2`
- clipped value loss：关闭
- logger：TensorBoard

## 8. Iteration 2999 最终指标

TensorBoard event 文件在 step=`2999` 的最终值：

| 指标 | 数值 |
|---|---:|
| Mean reward | `3271.04883` |
| Mean episode length | `729.710022` |
| Mean action std | `0.267668` |
| orientation error | `0.102115 rad` |
| success valid | `0.840863` |
| goal reach count | `52.7926` |
| rotation goal reach count | `52.7863` |
| command goal reach count | `106.8000` |
| command rotation goal reach count | `106.5171` |
| command flip success count | `0.282897` |
| fingertip contact count | `2.65209` |
| finger collision frequency | `0.874294` |
| cube survival steps | `361.833` |
| success interval | `10.0423` |
| cage drop termination | `0.125` |
| flip completed termination | `1.575` |

这条基线已经显示出有效旋转和接触学习信号；同时，`finger_collision_frequency=0.874294` 仍是后续优化的主要风险指标。`rotation_goal_reach_count` 高于翻面计数是预期现象：旋转成功后会继续串联下一个目标，而翻面成功会结束 episode。

## 9. 代码基线校验

本次只归档模型、训练事件和参数，没有修改奖励函数或训练参数。用于复现实验的关键源文件及 SHA-256 已记录在 [`metadata.json`](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/metadata.json) 的 `source_hashes` 字段中。

后续任何奖励、任务比例、动作范围、PD、reset 或 termination 修改，都应以本文件和该归档目录为对照组，并在实验记录中明确写出相对本基线的差异。
