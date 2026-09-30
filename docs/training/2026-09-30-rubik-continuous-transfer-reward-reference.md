# Rubik 连续迁移训练奖励函数基准

记录日期：2026-09-30  
训练 run：`Luwu_Rubik_Continuous_Transfer`  
任务：`LuwuHand_Rubik_UFace_ContinuousTransfer`  
训练轮次：3000（最终 checkpoint：iteration 2999）

本文记录该次已完成训练实际使用的奖励配置，不代表代码里其它 Rubik 或方块任务的默认值。权重以本次 run 的 [env.yaml](../../logs/rsl_rl/wuji_reorient/remote-2026-09-29_18-17-51_Luwu_Rubik_Continuous_Transfer/env.yaml) 为准；奖励实现见 [rewards.py](../../src/wuji_mjlab/tasks/reorient/mdp/rewards.py)，任务成功条件见 [rubik_command.py](../../src/wuji_mjlab/tasks/reorient/mdp/rubik_command.py)。

## 奖励项

| 奖励项 | 权重 | 作用与发放条件 |
| --- | ---: | --- |
| `action_rate` | `-0.04` | 惩罚归一化关节目标的一阶、二阶变化；使用 EMA 处理后的关节目标，不直接惩罚 PPO 的原始动作噪声。 |
| `torque` | `-0.02` | 力矩平方惩罚，以 `0.5 N·m` 为归一化上限，并对执行器取均值。 |
| `cage_escape` | `-0.2` | 有界惩罚魔方离开掌心支撑区域，抑制滑出/逃逸。 |
| `finger_collision` | `-0.07` | 归一化的手指间自碰撞惩罚；它不是指尖与魔方接触惩罚。 |
| `approach_progress` | `+1.0` | 指尖接近魔方表面的奖励。依各指尖表面距离维护 episode 内最佳进展，已获得的进展不会因退开再靠近重复支付。`step_budget=2.0`，魔方半边长 `0.02855 m`。 |
| `contact_milestones` | `+1.0` | 接触里程碑：至少 1 个指尖连续接触 `0.1 s`，一次性 `+1`；至少 2 个指尖连续接触 `0.1 s`，再一次性 `+1`。两项分别计时、每个 episode 各发一次；训练任务无效或发生掉落时不发。 |
| `cage_drop` | `-10.0` | 魔方掉落/越出掌心 cage 时的稀疏惩罚。 |
| `rubik_task_progress` | `+10.0` | 当前任务阶段的姿态势能进展；需要至少一个指尖接触，且只能比较连续的同一目标/阶段。每步势能差截断在 `[-0.1, +0.1]`，因此反向偏离也会产生负进展。 |
| `rubik_flip_to_top` | `+10.0` | 完成翻面目标的一次性奖励：至少 1 指接触；目标面朝向掌心法线的误差在 `0.20 rad` 内，U 层角误差在 `10°` 内；魔方根角速度不大于 `0.8 rad/s`；保持 8 步。 |
| `rubik_u_turn_success` | `+20.0` | 完成 U 层转动目标的一次性奖励：至少 1 指接触；魔方正立误差在 `0.20 rad` 内；U 层角误差不大于 `5°`；U 层角速度不大于 `0.45 rad/s`；保持 8 步。 |
| `rubik_cube_yaw_success` | `+3.0` | 完成整颗魔方绕掌心法线的目标转动，目标为随机正负 `15°`：至少 1 指接触，魔方保持正立、偏航误差不大于 `3°`、根角速度不大于 `0.45 rad/s`，并保持 8 步。 |
| `rubik_u_turn_progress` | `+5.0` | U 层转动过程奖励。每个目标只按历史最佳角度误差改善支付；每个目标最多累计 `+5`，须处于转面阶段、有指尖接触且魔方正立。来回抖动或退回后重新前进不会重复获得已支付的进展分。 |

奖励管理器启用 `scale_rewards_by_dt`。因此表中是奖励项权重，不应直接等同于每个仿真步最终入账的分数；速率和稀疏事件函数会结合 `step_dt` 处理，使奖励积分不随仿真步长简单线性改变。

## 任务比例与课程

- 起始任务分布为 `45%` 翻面到 U 面、`45%` U 层转动、`10%` 整颗魔方约 `±15°` 偏航。
- 翻面成功后继续当前 episode；全局步数达到 `40000` 后，翻面目标衔接到 U 层转动目标，不因单次成功直接结束 episode。
- 本次最终翻面课程为 stage 0（侧面目标）；U 层转动课程为 stage 2（含 `90°` 与 `180°` 目标）。晋级依据是实际成功次数和成功率门槛。
- 魔方仅 U 层可转，其余层锁定；标称质量 `80 g`，质量 DR 范围 `63–80 g`；边长 `57.1 mm`。

## 读 TensorBoard 时的注意事项

- `Episode_Reward/<term>` 是该奖励项加权后的 episode 累计值，应和任务采样比例、episode 长度一起看。
- `rubik_turn90_success_rate`、`rubik_turn180_success_rate`、`rubik_side_success_rate`、`rubik_bottom_success_rate`、`rubik_yaw_success_rate` 是比总 reward 更直接的任务成功指标。
- `fingertip_contact_count` 衡量指尖接触；`finger_collision_frequency` 衡量指间自碰撞，二者不是同一件事。
- 过程奖励上升不等于离散目标已完成；判断成功时还要看成功率/次数与角度误差。

## 本次 run 产物

- Run 目录：[remote-2026-09-29_18-17-51_Luwu_Rubik_Continuous_Transfer](../../logs/rsl_rl/wuji_reorient/remote-2026-09-29_18-17-51_Luwu_Rubik_Continuous_Transfer/)
- 实际奖励配置：目录内 `env.yaml`
- 最终模型：目录内 `model_2999.pt`
- 完整 TensorBoard 历史：目录内 `events.out.tfevents.1790677083.gpufree-container.3737415.0`
- 训练和 checkpoint 元数据：目录内 `handoff-metadata.json`

当前训练代码的工作目录是项目根目录 `/home/benmao/桌面/Luwu_Hand_Wuji`；后续训练和修改直接在该目录进行，不再从 `reference/wuji-mjlab-main` 启动。
