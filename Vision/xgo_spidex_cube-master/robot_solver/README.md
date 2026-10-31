# 魔方还原与机器人动作规划

复用 cube.js / Kociemba 两阶段算法生成候选解，默认采用掌心 A/B 风险感知规划。网页与 v2 出口的实际转面**只使用逆时针 90°**（从操作面外侧看）：`R → R' R' R'`，`R2 → R' R'`，`R' → R'`。原始符号解可以包含 R、R2，但 `actions` 不会输出顺时针转面。

掌心参考为白上、蓝右、红前。整体翻面只用 A（上→前→下→后）与 B（上→左→下→右，2026-10-28 按实机反馈反向为 Ry(-90°)），优先减少实际基本翻面次数，其次减少实际逆时针转动次数。网页是离线预览，不连接硬件，也不代表实机验证。B 的规划矩阵、目标姿态与预览轴统一反向；A 与逆时针转层不变。

## v2 使用

```sh
node scripts/plan-risk.mjs --scramble "R U F2" --initial-quat-xyzw 1,0,0,0 --calibration-id demo-palm-v1 --model-profile-id preview --demo-center-map
npm test
node scripts/benchmark-risk.mjs --all-orientations --output docs/risk-benchmark-20261026.json
```

四元数顺序为 xyzw；示例中心映射只适用于本项目演示色系。实际设备须提供自己的标定、中心映射和动作资格 profile。`ideal_preview` 为理想几何预览；`shadow` 不下发动作；`validated_execution` 要求 profile 中对应姿态/动作边已经验证，并通过动作级反馈执行器验收。未知边不自动放行。

v2 计划按动作记录状态前后、姿态、原符号索引与展开转动索引。`ExecutionSession` 只通过显式 adapter 接入：转动逐个确认，A/B 翻面须验证完整朝向，不能以“目标面朝上”代替。状态过期、意外转层、MOVE 序号缺口、超时或进程重启均暂停，不自动重放。当前尚未接入用户实机网页。

`proxy` 模式不输出预测成功率。多候选搜索只保证在已找到候选中择优，不保证全局最优。旧版 `scripts/plan.mjs` 和 v1 训练求解接口保持兼容，仍保留原版动作能力；不要将其视作当前 v2 的逆时针执行出口。

## 启动

需要 Node.js 22.12+：

```sh
npm ci
npm run dev
```

打开 <http://127.0.0.1:5178>。顶部提供随机打乱、规划、逐步还原、连续还原；右侧显示面转与机器人动作序列。一次单步执行一个机器人动作，连续播放可在当前动作结束后暂停。

支持播放速度、初始偏航、54 面片状态导入和 JSON 导出。拖动三维视图只改变观察相机。求解在 Web Worker 中执行。

```sh
npm run build
npm run preview
```

`dist/` 是可托管的静态网站，运行时无需后端或外部 CDN；开发与预览服务使用同一端口。

## 原版结构与理念（兼容保留）

| 文件 | 用途 |
|---|---|
| `src/cube-model.js` | 中心颜色、坐标、面片表示和合法性校验 |
| `src/solver-core.js`、`src/solver.worker.js` | 成熟求解器调用与后台计算 |
| `src/planner.js` | 最短重定向、顶层指令和桌面几何 |
| `src/scene.js`、`src/main.js`、`src/style.css` | 三维动画和交互界面 |
| `scripts/plan.mjs` | 命令行导出机器人计划 |
| `docs/MATH.md` | 数学定义、坐标展开和最小性证明 |
| `vite.config.js` | 构建配置与 cube.js 旧 UMD 模块的 ESM 适配 |

内部状态采用角 / 棱块置换与朝向，外部输入为 **URFDLB 顺序的 54 面片字符串**，各面按标准展开图逐行排列。中心身份固定，输入会检查块组合、朝向和置换奇偶性。

整体姿态与内部状态分离。目标面法向 `n` 在世界中为 `a=R n`，最小重定向角为 `acos(a·[0,0,1])`。对给定还原序列，该映射使累计整体旋转角最小；不保证在全部还原序列中使面转数或机器人成本全局最小。

## 原版机器人接口（非 v2）

```sh
node scripts/plan.mjs --scramble "R U F2 L' D B2" --yaw 25 --output outputs/plan.json
node scripts/plan.mjs --state-file state.txt --output outputs/plan.json
node scripts/plan.mjs --random --output outputs/plan.json
```

也可直接调用 `solveFacelets(facelets54)` 和 `compileRobotPlan(facelets54, solution, quaternionXYZW)`。JSON 的 `actions` 只包含两类：

- `reorient`：目标四元数、按行存储的旋转矩阵、世界旋转轴、最小角及目标中心位置。`requiresMotion=false` 表示无需改变朝向。
- `turn_top`：世界 +Z 轴上的 `CW_90`、`CCW_90` 或 `HALF_180`；右手角分别为 −90°、+90°、+180°。

边长 56 mm，桌面 `z=0`，每个动作结束时平放，中心 `[0,0,28] mm`。重定向采用抬起、旋转、放平；实际抓取、关节轨迹、可达性和碰撞由硬件驱动负责。

中心轴与视觉模块一致：蓝 +X、红 +Y、黄 +Z、绿 −X、紫 −Y、白 −Z。标准面符号采用 `R=+X、U=+Z、F=−Y`，保持右手系；红色对应 B。

输入需要完整魔方状态；视觉模块的位姿识别不提供不可见面片。坐标及推导见 [数学说明](docs/MATH.md)，依赖来源和许可证见 [第三方说明](THIRD_PARTY_NOTICES.md)。
