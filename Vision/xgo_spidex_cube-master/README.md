# Rubiks Cube

面向灵巧手拧魔方的视觉识别与动作规划项目，包含两个独立模块：

- **视觉识别**：输入 RGB 图像，估计魔方六维位姿；单层偏转时，以其余两层为主体估计位姿，并输出该层的转角。
- **还原规划**：输入完整魔方状态，生成外层还原序列，映射为机器人可执行的“整体重定向、转动顶层”，提供 Web 三维演示和 JSON 导出。

## 设计理念

1. **几何确定形状，中心颜色确认朝向。** 利用角点、棱线和网格拟合位姿，不依赖打乱后非中心色块的排列。
2. **复用成熟方法。** 视觉使用 OpenCV / SciPy；还原复用 cube.js 的 Kociemba 两阶段算法；三维显示使用 Three.js。
3. **状态、位姿、执行分离。** 内部魔方状态与整体姿态独立；每次面转前，以最小整体旋转角将目标面法向对齐世界 +Z，放平后只转顶层。

## 目录结构

```text
cube_pose/          视觉特征、几何模型、位姿与转角估计、命令行入口
config/             魔方尺寸、颜色阈值和相机参数
raw_figure/         原始示例图片：1_ 为完整魔方，2_ 为单层偏转
docs/VISION.md      视觉接口、坐标约定和使用边界
robot_solver/
  src/              魔方模型、机器人动作映射和 Web 界面
  scripts/plan.mjs   生成机器人 JSON 计划的命令行入口
  docs/MATH.md      数学描述与最小重定向推导
  licenses/         第三方依赖许可证
requirements.txt    Python 运行依赖
```

## 快速开始

视觉模块需要 Python 3.10+，在仓库根目录执行：

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m cube_pose raw_figure/1_8.jpg --output outputs
python -m cube_pose raw_figure/2_7.jpg --output outputs
```

Web 界面需要 Node.js 22.12+：

```sh
cd robot_solver
npm ci
npm run dev
```

打开 <http://127.0.0.1:5178>，依次使用随机打乱、规划、逐步或连续还原。安装与导出说明见 [robot_solver/README.md](robot_solver/README.md)。

## 约定与边界

魔方边长默认 56 mm，蓝 / 红 / 黄 / 绿 / 紫 / 白分别对应 +X / +Y / +Z / −X / −Y / −Z。视觉颜色和相机参数在 [config/default.json](config/default.json)，三维显示色值在 `robot_solver/src/cube-model.js`。

默认相机内参是 D435i 量级的占位值，实际使用需标定。仅靠中心颜色和几何，单帧层角只能确定到模 90°。视觉模块不恢复完整 54 面片状态；两个模块目前通过独立接口使用，尚未连接真实机械手。重定向最小性针对给定还原序列及整体旋转角代价。

详细接口见 [视觉说明](docs/VISION.md) 与 [数学模型](robot_solver/docs/MATH.md)，依赖来源及许可证见 [第三方说明](robot_solver/THIRD_PARTY_NOTICES.md)。本地依赖、运行输出和构建目录由 `.gitignore` 排除。
