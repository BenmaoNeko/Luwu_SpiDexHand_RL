# Luwu FromScratch Curriculum：model_2999 训练交接

日期：2026-09-24  
任务：`LuwuHand_Reorient_FromScratch_Curriculum`  
Run：`2026-09-23_17-52-30_Luwu_FromScratch_Curriculum`  
状态：已到 iteration 2999/3000；服务器检查时训练进程已结束。

## 最终模型与原始训练数据

| 项目 | 位置 / 值 |
|---|---|
| 本地 checkpoint | [model_2999.pt](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/model_2999.pt) |
| 远端 checkpoint | `/root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main/logs/rsl_rl/wuji_reorient/2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/model_2999.pt` |
| 文件大小 | 10,296,373 bytes |
| SHA-256 | `399b3876a714fc9ed19e9305742942f6c7595a10672d11b9ce46bb955d888313` |
| TensorBoard event（本地） | [events.out.tfevents.1790157158.gpufree-container.654817.0](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/events.out.tfevents.1790157158.gpufree-container.654817.0) |
| Event SHA-256 | `3859f3c3566e4a5c6fcdc67ab64336fbfe6e1c6b8b6d169a42f15f730f1a3c94` |
| Reward 时间序列 CSV | [reward_timeseries.csv](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/reward_timeseries.csv) |
| 保存的运行配置 | [env.yaml](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/params/env.yaml)、[agent.yaml](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/params/agent.yaml) |

Event 文件包含 72 个 scalar tags；本次续训的时间序列覆盖 iteration 350–2999，共 2,650 个点。训练从上一 run 的 `model_350.pt` 恢复；`agent.yaml` 记录 `resume=true`、`max_iterations=2650`、`num_steps_per_env=40`、`seed=42`、4096 个环境及 `save_interval=50`。checkpoint 内部迭代号已确认是 2999。

## 模型接口与环境

- Checkpoint actor 输入：162 维；critic 输入：344 维；动作：16 维。checkpoint 内 `obs_normalizer._mean` 形状为 `(1, 162)`、actor 首层权重形状为 `(512, 162)`。
- Actor 使用 3 帧关节角、关节目标误差、方块姿态误差和动作历史；训练配置中的方块姿态观测启用视觉 pose DR，随机姿态注入概率为 2%。
- 动作是映射到 URDF 全关节限位的绝对关节位置目标，`action_scale=1.0`；EMA `alpha=0.5`，warmup `0.4 s`；舵机力矩硬限制为 `0.5 N·m`。
- 固定机械臂，训练对象为 16 关节 Luwu 灵巧手；4096 个并行环境，物理步长 `0.01 s`、控制步长 `0.05 s`。
- 课程结束在 Stage 3：全其他面连续翻面，目标分布为 70% 翻面、30% 每次约 15° 的旋转；翻面不会终止 episode。

### Checkpoint 与当前代码的观测差异

此 checkpoint 是 162 维 actor。较新的本地配置在 policy 末尾增加了 `vision_valid`、`vision_age`、`vision_confidence`、`vision_innovation` 四个 metadata，actor 因此变成 166 维。预览 model_900 时，为严格加载旧接口，只从预览 policy config 中移除了这四项；没有放宽 `strict=True`。以同一配置预览 model_2999 时也要保留这个处理。

## 最终训练表现

| 指标（iteration 2999） | 值 |
|---|---:|
| Mean reward | 1661.08 |
| Mean episode length | 967.40 |
| Mean value loss | 54.3043 |
| Mean action std | 0.31 |
| Goal reach / episode | 23.4448 |
| Rotation successes / episode | 9.7585 |
| Flip successes / episode | 13.6862 |
| Consecutive flip count | 13.6862 |
| Face coverage / episode | 4.0316 |
| Cube survival steps | 487.2298 |
| Fingertip contact count | 3.1398 |
| Finger collision frequency | 0.4482 |
| Action delta RMS / jerk RMS | 0.8745 / 1.3350 |
| Joint velocity RMS / acceleration RMS | 1.8348 / 341.5792 |
| Episode timeout / cage drop metrics | 3.8000 / 0.2750 |

最终 curriculum 阶段为 3。滚动统计为：翻面 episode 成功率 `70.20%`、平均连续翻面 `27.30`、面覆盖 `4.48`、cage-drop 率 `8.75%`。因此策略已学会连续翻面并包含对面翻转，但平均覆盖仍不到六面；指间碰撞频率约 44.8%，是 sim-to-real 前必须重点检查的风险。Mean reward 受本次奖励权重影响，不应单独用来与旧奖励版本比较。

## Reward 设计与训练曲线

目标每次采样 70% 翻面、30% 旋转。旋转成功一次性奖励权重 `+5`，翻面成功一次性奖励 `+40`；目标进度奖励要求至少一个指尖接触，并按单个目标进度封顶。接近奖励和单/双指接触里程碑分别为 `+1`。其他项包括 torque `-0.02`（按 0.5 N·m 限制归一）、cage escape `-0.2`、cage drop `-10`。

Stage 3 还把 action smoothness 惩罚设为 `-0.10`，finger self-collision 惩罚设为 `-0.25`；随课程升阶逐渐从较轻权重增强。原始权重与奖励定义见 [env_cfgs.py](../../src/wuji_mjlab/tasks/reorient/config/luwu_hand/env_cfgs.py)、[continuous_env_cfgs.py](../../src/wuji_mjlab/tasks/reorient/config/luwu_hand/continuous_env_cfgs.py) 和 [rewards.py](../../src/wuji_mjlab/tasks/reorient/mdp/rewards.py)。

| Episode reward 项 | iteration 350 | iteration 2999 |
|---|---:|---:|
| approach_progress | 0.0218 | 0.0325 |
| contact_milestones | 0.0360 | 0.0400 |
| goal_progress | 0.2472 | 9.4464 |
| rotation_success | 0.0705 | 1.9273 |
| flip_success | 0.0348 | 22.3469 |
| action_rate | -0.0001 | -0.0047 |
| torque | -0.0001 | -0.0039 |
| finger_collision | -0.00004 | -0.0052 |
| cage_escape | -0.00020 | -0.00012 |
| cage_drop | -0.0266 | -0.0128 |

CSV 保存所有 `Episode_Reward/*` 项、`Train/mean_reward` 和 episode length 在 2,650 个迭代点上的变化；完整 TensorBoard event 文件也保留在 run 目录中。CSV 是便于表格分析的导出，event 文件是原始数据。

## Stage 3 Domain Randomization

以下是最终阶段 profile，不代表前 3 阶段从训练开始就使用最大范围。模型达到 Stage 3 后，后续 reset 使用这些范围。

| DR 项目 | Stage 3 范围 |
|---|---|
| 方块 COM | 每轴 `±0.003 m`（±3 mm） |
| 方块尺寸 scale | `0.97–1.03` |
| 方块质量 scale | `0.85–1.15` |
| 机器人几何摩擦 scale | `0.5–1.5` |
| PD 增益 scale | kp、kd 均为 `0.75–1.25` |
| 编码器零偏 | `±0.02 rad` |
| 接触 solref time constant | `0.7–1.4` |
| 接触 solref dampratio | `0.7–1.4` |
| 接触 solimp width | `0.8–2.5` |
| 接触 solimp dmin | `0.5–1.0` |
| 关节 damping scale | `0.6–1.6` |
| 关节 armature scale | `0.75–1.25` |
| 关节 frictionloss scale | `0.6–1.6` |
| 连杆惯量 scale | `0.85–1.15` |
| 连杆质量 scale | `0.85–1.15` |
| action command delay | 最多 2 个控制步（100 ms） |
| action command noise std | `0.01`（policy action 单位） |
| 视觉姿态噪声 | `±5°` |
| 视觉慢变 bias 上限 / 单步 | `±5°` / `0.3°` |
| 姿态标定误差 | `±5°` |
| 视觉延迟 | 1–6 个观测步（50–300 ms） |
| 视觉 dropout / 旧帧 hold | `5%` / 最多 4 步（200 ms） |

Actor 关节角 observation 另有归一化均匀噪声 `[-0.06, 0.06]`；姿态 observation 保留 `2%` 随机方向注入。视觉 Stage 3 profile 定义在 [vision_observations.py](../../src/wuji_mjlab/tasks/reorient/mdp/vision_observations.py)，物理与控制 Stage 3 profile 定义在 [flip_curriculum.py](../../src/wuji_mjlab/tasks/reorient/mdp/flip_curriculum.py)。`params/env.yaml` 是初始化 Stage 0 配置快照；动态升阶后的实际范围以 curriculum 源码和 TensorBoard stage 指标为准。

实现中 robot link mass 与 inertia 是两个独立 DR 项；训练日志曾提示 `dr.body_mass` 单独改变质量而不同比例改变惯量。方块质量使用项目定义的 mass-and-inertia randomizer。实际硬件质量/惯量误差还应与这些仿真假设区分。

## 相关项目文档

- 当前四阶段运行手册：[2026-09-21-luwu-from-scratch-curriculum-runbook.md](2026-09-21-luwu-from-scratch-curriculum-runbook.md)
- 当前课程摘要：[2026-09-22-staged-flip-curriculum.md](../curriculum/2026-09-22-staged-flip-curriculum.md)
- 30/70 课程与 5/40 奖励设计：[2026-09-23-s0-continuous-yaw-flip-design.md](../plans/2026-09-23-s0-continuous-yaw-flip-design.md)
- Sim-to-real 观测与 DR 设计：[2026-09-23-sim2real-robust-continuous-task-design.md](../plans/2026-09-23-sim2real-robust-continuous-task-design.md)
- 旧 model_2000 能力交接和历史本机预览：[2026-09-23-model2000-capability-handoff.md](2026-09-23-model2000-capability-handoff.md)
- 历史 model_500 Native Viewer 手册：[2026-09-22-model500-native-preview-runbook.md](2026-09-22-model500-native-preview-runbook.md)；该手册固定旧任务与 CPU 环境，不是本模型的 checkpoint 记录。
