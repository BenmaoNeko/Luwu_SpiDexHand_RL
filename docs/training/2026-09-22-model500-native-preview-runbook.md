# model_500 本机 MuJoCo 原生预览手册

日期：2026-09-22  
用途：复现历史 `model_500.pt` 的本机策略预览，重点观察旋转、翻面和手指自碰撞。  
预览后端：MuJoCo Native Viewer，不使用网页/Viser。

## 固化对象

本手册固定的是远端运行 `2026-09-20_18-01-59_Luwu_Reorient` 拉取的 `model_500.pt`，不是后续课程训练或 `model_1500.pt` 的自动替代品。

| 项目 | 固定值 |
|---|---|
| 代码目录 | `/home/benmao/桌面/Luwu_Hand_Wuji` |
| 模型文件 | `/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-20_18-01-59/model_500.pt` |
| SHA-256 | `adedb6197496dd01e5d728d9b92cf77fdd6d79e6da0e21ba8413466f834ca303` |
| 任务 | `LuwuHand_Reorient` |
| 推理环境 | `/home/benmao/mambaforge/envs/wheel_leg` |
| 推理设备 | `cpu` |
| Viewer | MuJoCo Native Viewer |
| 并行环境 | `8` |
| 物理步长 | `0.01 s` |
| 环境步长 | `0.05 s` |

模型文件存在时，启动前可用以下命令核对哈希：

```bash
sha256sum \
  /home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-20_18-01-59/model_500.pt
```

## 预览任务分配

当前 `LuwuHand_Reorient` 的本地预览策略包装器固定要求 8 个环境：

- Env 0–3：旋转任务；
- Env 4–7：翻面任务。

翻面完成不会触发 `flip_completed` 终止；episode 仍由 `time_out` 或 `cage_drop` 结束。预览中应重点观察翻面环境是否完成真实的姿态变化，而不是仅观察奖励或目标计数。

## 模型接口

启动日志必须出现以下接口，否则应停止预览并检查任务或 checkpoint 是否不匹配：

| 接口 | 尺寸 | 内容 |
|---|---:|---|
| Actor observation | 162 | 最近 3 帧关节角、关节目标误差、方块姿态误差、动作历史 |
| Critic observation | 344 | 训练时 privileged 状态；仅用于加载/结构校验，部署 actor 不读取 |
| Action | 16 | 16 个灵巧手关节位置控制量 |

当前奖励项与权重由任务配置加载，预览不重新训练、不修改奖励：

```text
action_rate          -0.02
torque               -0.02
cage_escape          -0.2
finger_collision     -0.05
approach_progress     1.0
contact_milestones    1.0
goal_progress         1.0
rotation_success     20.0
flip_success         30.0
cage_drop           -10.0
```

## 标准启动命令

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"
export MUJOCO_GL=glfw
export LD_LIBRARY_PATH="/home/benmao/mambaforge/envs/wheel_leg/lib:${LD_LIBRARY_PATH:-}"

/home/benmao/mambaforge/envs/wheel_leg/bin/python \
  scripts/play/play_rsl_rl.py \
  --task LuwuHand_Reorient \
  --checkpoint-file \
  "/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-20_18-01-59/model_500.pt" \
  --viewer native \
  --num-envs 8 \
  --device cpu
```

启动成功的关键日志包括：

```text
[INFO]: Preview uses envs 0-3 for rotation and 4-7 for flip
[INFO]: Loading checkpoint: model_500.pt
Number of environments | 8
Environment device     | cpu
Active Action Terms (shape: 16)
policy (shape: (162,))
critic (shape: (344,))
```

## 为什么不能改成 `--num-envs 1`

当前代码中的 `_PreviewFixedTaskPolicy` 为了保证旋转/翻面各占 4 个环境，会对 `LuwuHand_Reorient` 强制检查环境数量为 8。使用 1 个环境会在策略包装阶段报错：

```text
ValueError: LuwuHand_Reorient preview requires exactly 8 environments: 4 rotation and 4 flip.
```

因此本手册以 8 环境命令为准。该约束位于：

```text
src/wuji_mjlab/utils/play_runner.py
```

## 预览检查清单

打开原生 MuJoCo 窗口后，依次检查：

1. 手掌和手指是否处于训练使用的 Home 姿态；
2. Env 0–3 是否主要执行小角度旋转；
3. Env 4–7 是否出现翻面目标并发生真实方块姿态变化；
4. 翻面后是否进入下一个目标，而不是立即关闭窗口或结束整个预览；
5. 是否存在手指互撞、方块穿透、方块脱离掌心或持续抖动；
6. 是否出现 `Warning: opt.ccd_iterations...`。该信息是 MuJoCo/Warp 的终端警告，不代表 checkpoint 加载失败；若只需安静观察窗口，可在终端过滤输出，但不能据此忽略实际碰撞异常。

## 当前资产边界

本手册复现的是当时的 MJLab 训练任务预览，加载：

```text
src/wuji_mjlab/assets/robots/luwu_hand/mjcf/luwu_mjlab.xml
```

它与项目根目录的凸包碰撞验证预览不是同一个场景：

- 本手册：加载 PPO、运行训练环境和策略；
- `tools/convex_preview.py`：使用 `/home/benmao/桌面/Luwu_Hand_Wuji/generated/luwu_arm_hand_002_convex/scene.xml` 做独立碰撞几何验证，不加载 PPO。

如果需要验证凸包碰撞体积，应单独运行：

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji
export MUJOCO_GL=glfw
export LD_LIBRARY_PATH="/home/benmao/mambaforge/envs/wheel_leg/lib:${LD_LIBRARY_PATH:-}"

/home/benmao/mambaforge/envs/wheel_leg/bin/python \
  tools/convex_preview.py \
  --output generated/luwu_arm_hand_002_convex \
  --viewer
```

这两个预览不能混称：前者用于策略行为，后者用于碰撞网格和固定机械臂姿态检查。

## 停止与故障处理

- 关闭原生 MuJoCo 窗口，或在启动终端按 `Ctrl+C`；
- 若看到 `policy (shape: (162,))`、`action shape: 16` 以外的尺寸，停止运行；
- 若 checkpoint 加载报维度错误，不要放宽 `strict=True`，先确认任务 ID、模型文件和 `PYTHONPATH`；
- 若只想重新打开相同环境，直接重复“标准启动命令”，不要使用网页预览入口；
- 当前命令使用 CPU 是为了复现历史本机预览环境。若改用 GPU，必须先确认运行时 PyTorch 与 checkpoint/依赖环境一致，并重新记录设备信息。

## 运行时版本

本次验证使用：

```text
Python 3.11.15
torch 2.7.1+cpu
mujoco 3.6.0
warp 1.12.1
```

## 记录来源

- 历史预览命令：`scripts/play/play_rsl_rl.py --task LuwuHand_Reorient --viewer native`；
- 当前任务包装器：`src/wuji_mjlab/utils/play_runner.py`；
- 当前环境配置：`src/wuji_mjlab/tasks/reorient/`；
- 模型来源 run：`remote-2026-09-20_18-01-59`；
- 本文档记录的实际启动验证：8 环境、CPU、162/344/16 接口均成功构建，Native MuJoCo viewer 进程保持运行。
