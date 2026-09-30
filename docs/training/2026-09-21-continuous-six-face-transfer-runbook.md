# Luwu Hand 连续六面迁移训练交接

## 目的

从第一成功基线模型迁移训练一个独立任务：70% 连续随机翻面、30% 连续 15° 旋转。翻面成功不再触发 episode 终止，而是在奖励事件被读取后切换到下一个目标面；只有 `time_out` 和 `cage_drop` 可以结束 episode。

## 基线模型

- 任务：`LuwuHand_Reorient`
- 文件：`/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/model_2999.pt`
- source iteration：`2999`
- SHA-256：`17ca59396093228b87b9488b1b6d4ac3576b095df4187d8c4387211de6dc4d3d`
- actor 输入：162
- critic 输入：344
- action：16

迁移时只加载 `actor_state_dict`，包括 actor 的 observation normalizer；不加载 critic、optimizer、iteration、环境状态或旧 curriculum。每个新 run 会写 `transfer_source.json` 记录来源和 SHA-256。

## 新任务配置

- task：`LuwuHand_Reorient_ContinuousFaces`
- 训练环境：4096 个并行环境
- 训练轮次：3000
- 学习率：`5e-5`
- 初始立方体：A 面朝上、掌心中心、零初速度
- 任务分布：`yaw_goal_probability=0.30`，即 30% 旋转、70% 翻面
- 旋转目标：每次绕当前连续目标前进约 15°
- 翻面目标：从当前顶面之外的其余五面中随机选择，按最短立方体姿态旋转选择面内朝向
- 成功容差：4°
- 成功保持：10 个控制步 = 0.5 秒
- 接触要求：至少 1 个指尖接触
- action：继续使用 URDF 全关节限位，动作尺度 1.0
- torque limit：0.5 N·m
- 终止：仅 `time_out`、`cage_drop`；不存在 `flip_completed` 终止

## 奖励与监控

保持原有接近、接触里程碑和目标进展信号，同时迁移任务使用：

| 项目 | 权重/含义 |
| --- | --- |
| `approach_progress` | `+1.0` |
| `contact_milestones` | `+1.0` |
| `goal_progress` | `+1.0`，接触门控 |
| `rotation_success` | `+20.0` |
| `flip_success` | `+30.0`，每个连续翻面目标一次性发放 |
| `finger_collision` | `-0.10`，比基线 `-0.05` 更严格 |
| `action_rate` | `-0.02` |
| `torque` | `-0.02`，按 0.5 N·m 限制 |
| `cage_escape` | `-0.2` |
| `cage_drop` | `-10.0` |

新增指标：

- `consecutive_flip_count`：当前 episode 连续完成翻面数
- `face_coverage_count`：当前 episode 已包含的不同顶面数，范围 1–6（初始顶面计入）
- `adjacent_flip_count`：90° 相邻面转移数
- `opposite_flip_count`：180° 对向面转移数

训练期间重点看 `goal_reach_count`、`flip_success_count`、`consecutive_flip_count`、`face_coverage_count`、`finger_collision_frequency`、`cage_drop` 和 `ori_error`。如果成功数上涨但覆盖长期为 1，说明策略可能只重复少数面；如果碰撞率和 `cage_drop` 同时上涨，应先降低迁移难度或检查接触几何。

## 训练命令

```bash
cd /root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/lib:${LD_LIBRARY_PATH:-}"

/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/bin/python \
  scripts/train/train_rsl_rl.py \
  --task LuwuHand_Reorient_ContinuousFaces \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.logger tensorboard \
  --gpu-ids '[0]' \
  --transfer-checkpoint-file \
  /root/gpufree-data/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/baseline-2026-09-21-first-success/model_2999.pt
```

如果服务器上的基线仍未同步到上述路径，应先把本地基线复制到服务器对应数据盘位置；迁移程序会在启动时校验文件存在，并把 SHA-256 写入新 run。

## 本地预览

新任务的 play 配置固定使用 8 个环境：0–3 为旋转、4–7 为连续随机翻面。翻面成功后不等待固定时间，也不结束 episode，而是直接进入下一个翻面目标；状态面板显示目标面、翻面链长度和六面覆盖数。

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"
export MUJOCO_GL=egl

/home/benmao/mambaforge/envs/wheel_leg/bin/python \
  scripts/play/play_rsl_rl.py \
  --task LuwuHand_Reorient_ContinuousFaces \
  --checkpoint-file /path/to/transfer/model_2999.pt \
  --viewer native
```

## 已完成的验证

- 24 个立方体对称姿态：6 面 × 4 个面内朝向，均为唯一 proper rotation，并按设备缓存。
- 10 步接触保持门槛、无接触不得成功、掉出 cage 不得成功。
- 连续翻面成功后目标面变化、`flip_completed=False`、episode 不被完成逻辑终止。
- 连续翻面链、六面覆盖、相邻/对向转移和部分 reset 清零逻辑。
- 旧版 `LuwuHand_Reorient` 的 A→B 终止行为和奖励配置回归。
- 真实 CPU MJLab 配置：policy 162、critic 344、action 16，新增任务仅保留 `time_out/cage_drop` 终止。
- 真实 runner 加载基线 actor-only 成功。
- 1 轮 CPU 迁移训练烟测成功；日志目录：
  `logs/rsl_rl/wuji_reorient/2026-09-21_16-39-08_Luwu_ContinuousFaces_Transfer`

完整测试命令：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src \
LD_LIBRARY_PATH=/home/benmao/mambaforge/envs/wheel_leg/lib:$LD_LIBRARY_PATH \
/home/benmao/mambaforge/envs/wheel_leg/bin/python \
  -m pytest src/wuji_mjlab/tasks/reorient/tests -q
```

当前结果：135 passed；pytest 外部插件关闭是为了绕过本机 ROS `launch_testing` 缺少 `lark` 的环境问题。长时间 GPU 训练和真实硬件闭环尚未在本次实现中验证。
