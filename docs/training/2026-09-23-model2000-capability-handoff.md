# Luwu `model_2000.pt` 能力与本地交接说明

更新时间：2026-09-23

## 1. 模型身份与状态

本文件对应训练任务 `LuwuHand_Reorient_FromScratch_Curriculum`，run 名称为 `Luwu_FromScratch_Curriculum`。

本地 checkpoint：

`/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-22_16-06-26_Luwu_FromScratch_Curriculum/model_2000.pt`

- 保存轮次：`2000/3000`
- SHA-256：`a69a5ef7eacf3fff05457a1e0dbab57a35d13c0bac5327029e171a3fbed4d075`
- 文件大小：10,296,373 bytes
- 这是该 run 在 2028 轮出现 `policy observation contains NaN` 并停止前，实际保存的最新 checkpoint；不是 3000 轮最终模型。
- 本地原始 TensorBoard 事件文件和环境参数也已同步，见同目录下的 `events.out.tfevents.*`、`params/env.yaml`、`params/agent.yaml`。

完整机器可读 bundle：

`/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-22_16-06-26_Luwu_FromScratch_Curriculum/model_2000.analysis.json`

## 2. 它实际学会了什么

在训练日志最后一个有效标量步 `2028`，课程已经进入 Stage 3：`all_faces_continuous`。这表示模型不再只执行固定的 A→B，而是在已知当前方块姿态的情况下连续选择下一个目标面：

- 约 70% 目标为翻到其它面；
- 约 30% 目标为小角度 yaw 旋转；
- 翻面成功不会结束 episode，成功后继续生成下一个目标；
- 目标可以覆盖相邻面和对面，不再固定 A→B。

日志指标显示它已经具备明显的连续操作能力，但不是“六面完全可靠”的成品控制器：

| 指标（step 2028） | 数值 | 含义 |
| --- | ---: | --- |
| flip episode success rate | 73.91% | 最近课程窗口内完成至少一次翻面的 episode 比例 |
| mean consecutive flip count | 34.74 | 连续翻面次数的窗口均值 |
| mean face coverage | 4.68 / 6 | 一段 episode 平均覆盖的不同面数 |
| adjacent flip count | 29.69 | 相邻面翻转次数 |
| opposite flip count | 7.45 | 对面翻转次数 |
| curriculum cage-drop rate | 5.18% | 课程窗口内方块脱离可操作区域的比例 |
| fingertip contact count | 3.24 | 指尖接触指标，说明策略已形成多指接触行为 |
| orientation error | 0.705 rad | 最后日志时刻的平均姿态误差，约 40.4°；不是成功窗口内误差 |
| finger collision frequency | 47.39% | 手指自碰撞仍然偏高，不能忽略 |

因此当前模型的正确定位是：**已经学到连续翻面/旋转的策略候选，可用于本地预览、故障分析和后续 sim-to-real 前的受控评估；尚不能直接视为已经通过真实硬件安全验证的策略。**

## 3. 训练环境

- 机械臂不参与策略控制，固定臂姿态烘焙为 `J4=+1.5 rad`。
- 可控制对象只有 Luwu 四指灵巧手的 16 个关节。
- reset 时方块 A 面朝上，方块线速度和角速度为零，位置噪声为零；手使用标定后的 home 姿态。
- 方块名义边长 `54 mm`，名义质量 `0.120 kg`。
- MuJoCo 物理步长 `0.01 s`，控制步长 `0.05 s`，episode 上限 `50 s`。
- 当前 Luwu MJCF 使用手的原始碰撞网格；本任务不是凸包碰撞体版本。
- 训练从 Stage 0 开始，按翻面成功率、连续翻面链、面覆盖和掉落率晋级；最后日志已经达到 Stage 3。

Stage 3 的随机化范围：

- 方块质心偏移：`[-1, +1] mm`
- 方块尺寸缩放：`[0.98, 1.02]`
- 方块质量缩放：`[0.90, 1.10]`
- 手部摩擦缩放：`[0.65, 1.35]`
- PD `kp` 缩放：`[0.75, 1.25]`
- PD `kd` 缩放：`[0.75, 1.25]`
- 编码器偏置：`[-0.015, +0.015] rad`

## 4. Actor 输入、Critic 输入和动作输出

### Actor：162 维

Actor 只接收可部署的策略观测，历史长度为 3：

1. 归一化关节角：`3 × 16 = 48`，附加均匀噪声 `±0.06`；
2. 当前关节角与处理后位置目标的误差：`3 × 16 = 48`；
3. 方块相对当前目标的 6D 姿态误差：`3 × 6 = 18`，高斯噪声标准差 `0.05`，并有 2% 随机注入；
4. 原始动作历史：`3 × 16 = 48`。

Actor **不接收**方块位置、接触状态、方块线速度、方块角速度或指尖位置。这些量只存在于 Critic 的 privileged observation 或奖励/指标中，因此不能直接把当前仿真 Critic 输入照搬到真实部署。

### Critic：344 维

Critic 在训练时额外看到：方块在 wrist-Tag 坐标系的位置、方块线/角速度、指尖位置、干净姿态误差、真实关节状态、扰动状态、课程状态、DR 参数和掌部姿态。它用于 PPO 的价值估计，不是部署时的网络输入。

本次本地 smoke test 已确认实际加载维度为：

`actor=162, critic=344, action=16`。

### Action：16 维归一化位置目标

策略输出是 16 个归一化动作，每个动作先裁剪到 `[-1, 1]`，随后映射为相对于 calibrated home 的**绝对关节位置目标**：

```text
action < 0: home + action * (home - lower_limit)
action >= 0: home + action * (upper_limit - home)
```

因此它不是直接输出力矩，也不是简单的固定 `±0.5 rad` 增量。动作使用 MJCF/URDF 的每关节限位，环境侧做 `EMA alpha=0.5` 平滑，episode 开始前 `0.4 s` 保持 home。

底层是位置执行器：`kp=1`、`kv=0.15`，每个关节的力矩范围为 `-0.5…+0.5 Nm`。sim-to-real 时需要在硬件侧实现同等的动作解释、home 偏置、关节顺序、限位裁剪、平滑和力矩保护。

## 5. 奖励和成功判定

| 项 | 权重 | 作用 |
| --- | ---: | --- |
| action rate | -0.02 | 抑制目标动作变化过快 |
| torque | -0.02 | 以 `0.5 Nm` 为参考抑制力矩 |
| cage escape | -0.2 | 惩罚手指/方块离开操作笼范围 |
| finger collision | -0.05 | 惩罚手指自碰撞 |
| approach progress | +1.0 | 鼓励指尖靠近方块 |
| contact milestones | +1.0 | 鼓励形成接触 |
| goal progress | +1.0 | 只有满足接触门槛后才给姿态进展信号 |
| rotation success | +20.0 | 小角度旋转目标成功 |
| flip success | +30.0 | 翻面目标成功 |
| cage drop | -10.0 | 方块掉出操作区域 |

当前连续 Stage 3 要求至少一个指尖接触，并在姿态误差阈值内保持 10 个控制步；翻面成功不会触发 episode 结束。只有 timeout 或 cage drop 结束 episode。

## 6. 当前明确的限制和风险

1. 训练 run 在 2028 轮因为 actor policy observation 出现 NaN 而中止；目前只确认 NaN 出现在观测组，尚未定位到具体观测项。因此该 checkpoint 可加载，但训练稳定性问题仍未闭环。
2. `finger_collision_frequency=47.39%` 较高，策略可能使用了不理想的手指互碰方式获得部分操作效果；真实手上必须重点检查自碰撞和舵机负载。
3. `orientation_error=0.705 rad` 是日志末端平均状态，不等于成功样本的误差；不要把它解释为 4° 的整体精度保证。
4. 连续翻面指标来自仿真接触传感器和仿真姿态真值。真实系统仍需将相机/魔方姿态变换到训练约定的 wrist-Tag 坐标系，并处理检测噪声、延迟和丢帧。
5. 训练时 Critic 使用 privileged state；部署只使用 162 维 Actor 输入，不能把 Critic 的额外状态误接入 Actor。

## 7. 复现本地原始 MuJoCo 预览

在本机执行：

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"
export MUJOCO_GL=glfw
export LD_LIBRARY_PATH="/home/benmao/mambaforge/envs/wheel_leg/lib:${LD_LIBRARY_PATH:-}"
/home/benmao/mambaforge/envs/wheel_leg/bin/python \
  scripts/play/play_rsl_rl.py \
  --task LuwuHand_Reorient_FromScratch_Curriculum \
  --checkpoint-file \
  /home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-22_16-06-26_Luwu_FromScratch_Curriculum/model_2000.pt \
  --viewer native --num-envs 8 --device cpu
```

这条命令已经实际烟测通过，成功构建并加载了上述 162/344/16 网络接口。预览时使用的是 `play=True` 的最终 Stage 3 配置：课程管理器关闭、目标为 all-faces continuous；这与训练过程的“从 Stage 0 晋级”不同，但观测维度和动作接口相同。

## 8. 原始交接材料

- [model_2000.pt](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-22_16-06-26_Luwu_FromScratch_Curriculum/model_2000.pt)
- [训练参数 env.yaml](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-22_16-06-26_Luwu_FromScratch_Curriculum/params/env.yaml)
- [训练参数 agent.yaml](/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-22_16-06-26_Luwu_FromScratch_Curriculum/params/agent.yaml)
- [交接元数据 JSON](/home/benmao/桌面/Luwu_Hand_Wuji/docs/training/2026-09-23-model2000-handoff-metadata.json)
