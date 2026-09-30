# Luwu 接触引导奖励实现验证记录

日期：2026-09-20  
任务：`LuwuHand_Reorient`  
范围：奖励、成功状态机、翻面终止、确定性 MuJoCo 接触链路。未进行长时间训练。

## 2026-09-20 V2 奖励尺度落地（最新结果）

根据 [奖励尺度 V2 设计](plans/2026-09-20-reward-scale-v2-design.md)，已完成代码落地：

- Luwu 删除 `hand_pose`、`tip_slide`、旧姿态对齐和无限 hold 奖励；home 仍只作为 reset/action=0 参考。
- 接近奖励改为四根指尖独立历史最佳增量，预算为每回合最多 2；稳定单指/双指接触各一次、各 1 分。
- 目标进展预算为每目标最多 10；旋转/翻面完成权重分别为 20/30。
- 动作平滑惩罚读取 EMA 后的实际关节目标，按 URDF 全限位归一化；不直接惩罚 raw PPO 探索噪声。
- 力矩、自碰撞、cage 越界和掉落改为 V2 首轮尺度；cage 状态每个控制步只更新一次。
- cage 状态使用逐环境步号，局部 reset 只使被 reset 实例重新计算，不会让同一步的其他实例重复累加掉落计数。

本机 reorient 测试集结果：`99 passed, 11 warnings`；`compileall` 通过。警告为已有 pytest marker、ONNX 弃用提示和 MuJoCo asset attach 提示。

这只证明代码路径、时序和数值预算已通过自动化验证；第 7 节要求的 10-seed 真实关节轨迹收益排序、刷 reset 防套利测试和 PPO 短训练尚未完成。

## 2026-09-20 V1 审核后修复（历史记录）

使用本机 Ponytail 4.9.0 的最小根因修复原则，不增加依赖、不编辑第三方 mjlab。此前“计划全部落地”的结论撤回；以下是本轮实际验证范围。

修复了审核发现的五项问题：

1. `contact_task_state.py`：旋转四元数显式扩展到每个指尖，解决 B>1、K>1 时的维度错误。
2. `commands.py`：Luwu 目标误差基准读取已写入的 free-joint qpos，不读取 reset 后尚未 forward 的旧 xquat。双实例实测基准与实际误差均为 `0.261799365 rad`（15°）。
3. `commands.py` / `terminations.py` / `rewards.py`：Luwu 显式启用 `synchronized_contact_task`；`cage_drop` 后的终止项计算当前步成功，奖励与课程在 reset 前获得当前成功结果；旋转新目标在奖励结算后的 command 阶段推进。原版 Wuji 不启用此路径。
4. `curriculums.py` / Luwu 配置：正常 `flip_completed` 从 early-drop 惩罚中排除。
5. `contact_task_state.py` / `rewards.py`：单指、双指接触分别连续计时，删除 command 中重复累加。持续单指接触不能让刚加入的第二指立即拿奖励。

同时锁定同一步重复读取、局部 reset、reset 后尚未动作的奖励读取、目标切换事件清除，以及掉落时不给目标进展奖励的边界。奖励按控制步缓存，reset 只清除被重置实例的缓存和状态，其他实例不受影响。

复现完整验证：

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji
PYTHONNOUSERSITE=1 PYTHONPATH=src:src/wuji_rl_libs/rsl_rl MUJOCO_GL=egl \
  /home/benmao/mambaforge/envs/wheel_leg/bin/python -m pytest -q
/home/benmao/mambaforge/envs/wheel_leg/bin/python -m compileall -q src/wuji_mjlab/tasks/reorient
```

本轮结果：`94 passed, 11 warnings`；compileall 通过。警告包括已有 pytest marker、ONNX 弃用提示，以及新增真实环境测试触发的 MuJoCo asset attach 提示。Ruff、Pyright 不在当前运行环境中，未安装依赖，未宣称静态 lint/typecheck 通过。

新增可重复验证：

- `test_contact_geometry_regression.py`：真实四元数计算的批量几何回归，修复前失败、修复后通过。
- `test_contact_step_regression.py`：按 manager 顺序验证成功/奖励/课程/局部 reset；使用可控姿态和接触输入，属于逻辑测试，不是物理操纵成功证明。
- `test_contact_environment_regression.py`：真实 CPU MJWarp，分别 2 和 4 实例，各运行 20 个控制步，中途只 reset 一个实例；基准角正确、回报有限、未重置实例计时不受影响。没有注入成功或接触，也没有加载 PPO。

未同步服务器、未启动训练。GPU 大批量、有效物理操纵轨迹相对静止策略的回报对比及新策略能否学会翻面，仍需后续验证；不能将本轮修复通过等同于训练效果已经达标。

## 已落地的行为

- Luwu 目标按 50% 增量旋转、50% A→B 翻面抽样。
- 旋转目标每次为正向 15°，训练只在有效成功后推进；预览仍可按 2 秒推进目标。
- 完整姿态成功容差为 4°，连续保持 5 个控制步（当前 `step_dt=0.05`，即 0.25 秒）。
- Luwu 成功必须同时满足姿态误差和至少一个指尖接触；翻面成功只发一次奖励并触发 `flip_completed` terminal。
- 移除 Luwu 的 `orientation_alignment`、`hold_escalation`、`palm_detach`；加入有限预算的接近、接触里程碑、接触门控目标进展、旋转成功、翻面成功。
- 保留 actor/critic 观测维度、URDF 关节限位和 0.5 N·m 硬限力矩；V2 已移除 Luwu 的 `hand_pose` 惩罚，并改用新的动作、力矩和越界尺度。
- 成功课程按任务类型统计：旋转完成至少 3 个目标，翻面完成 1 次即可计入成功。

## 确定性接触测试

探针文件：`tools/contact_test.py`。它直接构造一个 Luwu 环境，不创建 PPO/RSL-RL runner，也不加载 checkpoint。测试驱动指定手指的 `joint2`，读取同一控制步的三层结果：

1. MJWarp 的原始接触缓冲 `env.sim.data.contact`；
2. `tip_object_contact.data.found/force`；
3. 项目已有的 `fingertip_contact_count` metric。

注意：`env.sim.mj_data` 是初始化时的单世界 MuJoCo 数据，不会被 MJWarp 批量 stepping 回写；训练仿真中不能用它的 `ncon` 判断当前接触。使用 `env.sim.data.nacon` 和 `env.sim.data.contact` 才是当前批量仿真的 raw contact。

可复现命令：

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji
PYTHONNOUSERSITE=1 \
PYTHONPATH="src:src/wuji_rl_libs/rsl_rl" \
MUJOCO_GL=egl \
/home/benmao/mambaforge/envs/wheel_leg/bin/python \
tools/contact_test.py --finger 1 --steps 20
```

关键实测输出：

```text
[BASELINE] step=0 ncon=4 raw_tip_cube=none sensor_found=0 sensor_force=0.000000 metric=0
[STEP 09] ncon=2 raw_tip_cube=object/cube↔robot/right_finger1_link4_col dist=-0.003276 sensor_found=1 sensor_force=2.062876 metric=1
[PASS] raw contact -> tip_object_contact -> fingertip_contact_count=1+
```

这确认了“脚本驱动指尖接触 → MuJoCo raw contact → sensor found/force → metric=1”的链路，未以此证明肉眼验收或有效物理翻面。`ncon=4` 的 reset 基线接触是掌心与方块支撑，不是目标指尖接触，因此不会被接触 metric 计为指尖。

## 前一轮自动化验证（历史记录，不作为当前验收结果）

已通过：

```text
tests/test_contact_test.py + contact reward tests: 7 passed
Luwu design/contact success focused tests: 17 passed
full reorient suite: 77 passed, 9 warnings
python -m compileall -q src/wuji_mjlab/tasks/reorient: passed
```

警告为已有的 pytest 自定义 marker 和 ONNX 弃用提示；没有测试失败。

## 当前未验证项

### 2026-09-20 V2 审核修复（Ponytail）

本次只修复已经复现的三处实现错误：

- `drop_penalty_sparse` 返回事件值除以控制周期，经过真实 RewardManager 的 dt 积分后，掉落贡献为 -10。测试覆盖 dt=0.02/0.05/0.1，以及未掉落实例贡献为零。
- 删除 Luwu 的 `valid_channel_count=1` 覆盖，复用传感器固定通道维度归一化；当前 16 通道下，单通道碰撞为 1/16，全通道为 1。
- 接近奖励在掉落步仅更新历史最佳，不付款；后续读数不能补领同一次进展，其他实例仍可正常获得奖励。

新增的 5 个回归用例在修改前全部失败，修改后完整 reorient 测试为 **104 passed, 11 warnings**，包含 CPU 批量环境 reset/step 回归。奖励尺度测试调用实际 RewardManager.compute，而不是仅断言配置权重。

本次未改 gamma、任务采样、reset、网络或成功定义；未同步服务器、未启动 PPO。参数不同的 cage 缓存调用属于尚未在当前配置触发的接口风险，未扩展为多配置缓存。Episode_Return/分任务诊断日志、10-seed 真实操作轨迹收益排序和 CUDA 验证仍待完成，不能将本次单元与环境回归通过等同于训练效果验证。

- 尚未证明 PPO 在新奖励下已经学会真实旋转或翻面；需要后续短训练和有效关节操作轨迹对比。
- 当前确定性探针验证的是 CPU MJWarp 路径；训练服务器的 CUDA MJWarp 路径仍需执行同一命令并记录结果。
- 探针确认接触链路，不等价于确认方块能长期稳定停在掌心；稳定性仍需用 reset 预览和 rollout 检查。

## 2026-09-29 Rubik 接触链路复核

把 `tools/contact_test.py` 扩展为可选 Rubik 迁移任务后，首次确定性探针复现了真实漏检：第 9 步 raw contact 已出现，`right_finger1_link4_col` 穿入 `cubelet:pos_x_pos_y_pos_z` 约 `4.97 mm`，但 `tip_object_contact.found=0`、`fingertip_contact_count=0`。原因是 `ContactSensorCfg` 对多个 secondary 几何默认用 `secondary_policy="first"`，传感器只监听第一个 cubelet。

Rubik 的 fingertip sensor 现在用 `ContactMatch(mode="subtree", pattern="middle", entity="object")` 匹配整颗魔方主体子树，一份传感器覆盖所有 cubelet，同时不会把指尖自碰撞算成物体接触。较大步长探针在第 9 步报告 raw contact、`sensor_found=1`、`fingertip_contact_count=1`，但穿透约 `4.97 mm`、force 约 `2.03 N`。随后用 `--step-size 0.25` 复测，在第 16 步与 `cubelet:neg_x_pos_y_neg_z` 接触，穿透约 `0.45 mm`、force 约 `0.23 N`，三层 contact 信号仍同步为 1。旧整体方块探针也通过。

这证明 Rubik 的“raw contact → fingertip sensor → metric”链路能够工作；探针使用脚本关节目标、存在明显网格穿透，不证明 PPO 策略已经学会稳定接触或真实力控。
