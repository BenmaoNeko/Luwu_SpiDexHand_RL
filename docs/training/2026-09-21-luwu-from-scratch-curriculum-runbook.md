# Luwu 从头训练：四阶段连续翻面课程运行手册

日期：2026-09-21  
任务：`LuwuHand_Reorient_FromScratch_Curriculum`  
运行名：`Luwu_FromScratch_Curriculum`

## 目的

从随机初始化的 PPO 开始训练，不加载已有模型，也不改变 actor/critic 的观测维度和 16 维动作接口。S0 起每次新目标按 70% 翻面、30% 旋转采样；S0/S1 只翻相邻面，S2/S3 开放到当前面之外的任意面，并逐步增加物理和视觉随机化。每次成功后继续当前 episode。

机械臂仍是固定物体：训练中没有机械臂执行器或机械臂动作；手部动作使用 URDF 关节限位，`action_scale=1.0`。

## 固定接口

| 项目 | 当前约定 |
|---|---|
| actor observation | 166 维 |
| critic observation | 344 维，包含 privileged 状态 |
| action | 16 维关节位置目标，按 URDF 限位约束 |
| 任务比例 | 所有课程阶段均为 70% 翻面，30% 连续 `+15°` 旋转 |
| 成功奖励 | 旋转 `+5`，翻面 `+40`；仅此从头训练课程覆写，独立迁移和基线配置不变 |
| 翻面成功 | 指尖接触、姿态误差在当前阶段容差内、连续保持当前阶段步数 |
| episode 终止 | `cage_drop` 或 `time_out` |
| 控制器 | 既有 PPO 网络、EMA action term、舵机力矩限制 0.5 N·m |
| 初始化 | Luwu home 姿态，方块静止放在掌心，正面朝上 |

## 四阶段参数

### Stage 0：相邻面获取期

- 翻面目标只从当前顶面四个相邻面中采样；不会抽到相对面。
- 成功容差：`8°`。
- 接触保持：`5` 个控制步。
- 物理为 nominal：质量、尺寸、摩擦、PD、编码器偏置、COM 均不做课程 DR。
- 动作平滑权重：`-0.04`；指间碰撞权重：`-0.08`。
- 翻面成功后继续采样下一个目标，不终止 episode。

### Stage 1：相邻面连续操作和轻度 DR

- 成功容差：`6°`。
- 接触保持：`8` 步；翻面成功后继续 episode。
- 翻面目标仍只从当前面四个相邻面采样。
- 动作平滑权重：`-0.06`；指间碰撞权重：`-0.12`。
- 接触、关节动力学和连杆参数加入轻度 DR。

### Stage 2：全其他面连续操作和中度 DR

- 翻面目标从当前面之外的五个面均匀采样。
- 成功容差：`6°`，接触保持 `8` 步；翻面成功后继续 episode。
- 动作平滑权重：`-0.08`；指间碰撞权重：`-0.18`。
- 加入中度接触、关节动力学和指令延迟/噪声 DR。

### Stage 3：全其他面连续操作和完整 DR

- 翻面目标从当前面之外的五个面均匀采样。
- 成功容差：`4°`，接触保持 `10` 步；翻面成功后继续 episode。
- 动作平滑权重：`-0.10`；指间碰撞权重：`-0.25`。
- 使用课程中的最高物理、视觉和控制 DR 强度。

课程训练从 `curriculum_stage=0` 开始；play 配置直接选用 Stage 3。

课程阶段由环境中的 `flip_curriculum` 在 reset 前更新。它使用最近最多 256 个已完成 episode 的翻面统计；旋转任务不会计入翻面成功分子，因此旋转成功不能掩盖翻面失败。

## 晋级条件

- Stage 0 → Stage 1、Stage 1 → Stage 2：至少 256 个 episode，翻面成功率 `≥0.25` 且 cage-drop 率 `≤0.20`。
- Stage 2 → Stage 3：至少 256 个 episode，翻面成功率 `≥0.20`、平均连续翻面数 `≥1.5`、平均面覆盖数 `≥2.5`，且 cage-drop 率 `≤0.20`。
- 不满足条件时保持原阶段；不会因为 iteration 数自动晋级。

## 推荐训练命令

```bash
cd /root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/lib:${LD_LIBRARY_PATH:-}"

/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/bin/python \
  scripts/train/train_rsl_rl.py \
  --task LuwuHand_Reorient_FromScratch_Curriculum \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.logger tensorboard \
  --gpu-ids '[0]'
```

注意：`--gpu-ids` 是 Tyro 的整数列表参数，服务器命令中必须使用 `'[0]'`，不能写成字符串 `'0'`。

## 必看 TensorBoard 指标

- `Curriculum/flip_curriculum/stage`
- `Curriculum/flip_curriculum/flip_episode_success_rate`
- `Curriculum/flip_curriculum/mean_consecutive_flip_count`
- `Curriculum/flip_curriculum/mean_face_coverage_count`
- `Curriculum/flip_curriculum/cage_drop_rate`
- `Metrics/reorient_command/flip_target_angle`
- `Metrics/reorient_command/flip_success_count`
- `Metrics/reorient_command/consecutive_flip_count`
- `Metrics/reorient_command/face_coverage_count`
- `Episode_Metrics/finger_collision_frequency`

预期现象是：每次新目标约 30% 为旋转、70% 为翻面；S0/S1 翻面目标为相邻面，S2/S3 可采样当前面之外的任意面。旋转和翻面成功奖励分别为 5 和 40。翻面成功后不因“翻面完成”终止 episode，而是继续采样下一个目标。

## 验证边界

上一版 50/50 课程通过 158 项 reorient 测试。本次 30/70 目标比例和 5/40 奖励调整尚未运行测试；真实训练还需检查阶段晋级、翻面成功率和接触稳定性。
