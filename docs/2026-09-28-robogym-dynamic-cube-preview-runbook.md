# Robogym 魔方连续力矩预览与后续接入说明

更新日期：2026-09-28  
用途：在本机查看 Robogym 三阶魔方模型，并用六面控制面板连续施加转动力矩，观察关节运动和地面碰撞。

## 1. 当前预览是什么

预览加载 Robogym 的 `rubik_perpendicular.xml`，以 MuJoCo 3.6 编译；地面通过运行脚本时的内存 XML 修改临时加入，不会改动 Robogym 原始文件。当前场景只有魔方和地面，没有 Shadow Hand、Luwu 手或机械臂。

预览控制是连续力矩模式：按住某一面 CW/CCW 按钮时，对该面的 9 个小块关节施加广义力矩；松开按钮后撤销外加力矩，剩余运动仍由惯性和碰撞动力学继续演化。代码路径是 `qfrc_applied → mj_step`，面转过程不直接写 qpos。每次按下时锁定当时选中的小块集合，避免转动中小块偏移后重新选层或让驱动线程退出。

面板力矩单位是**每块小块的 N·m**：默认 `0.005 N·m`，范围 `0–0.01 N·m`，步进 `0.001 N·m`。它是用于检查动力学的理想化外加力矩，不是手指或真实电机的力矩。

## 2. 模型加载路径与运行环境

模型源文件：

`/home/benmao/桌面/Luwu_Hand_Wuji/URDF/robogym-master/robogym/assets/xmls/rubik/rubik_perpendicular.xml`

相关几何和纹理目录：

- 网格：`/home/benmao/桌面/Luwu_Hand_Wuji/URDF/robogym-master/robogym/assets/stls`
- 纹理：`/home/benmao/桌面/Luwu_Hand_Wuji/URDF/robogym-master/robogym/assets/textures`

本地可复用启动器：

`/home/benmao/桌面/Luwu_Hand_Wuji/tools/robogym_cube_face_preview.py`

当前验证环境是 `/home/benmao/mambaforge/envs/wheel_leg`，Python 3.11.15、MuJoCo 3.6.0、Tk 8.6。Robogym 仓库本身依赖旧版 `mujoco-py==2.0.2.13`；本预览**不导入**旧 Robogym 环境，只直接加载 MJCF 并调用其中纯 Python 的旋转变换工具。

## 3. 模型自由度和碰撞参数

由当前预览启动器实际编译出的模型：

| 属性 | 数值 | 说明 |
| --- | ---: | --- |
| `nbody` | 54 | 包含魔方各小块和地面场景 |
| `njnt` | 90 | 关节元素数 |
| `nq` | 93 | qpos 维度，球关节占 4 个 qpos |
| `nv` | 92 | qvel / 广义力维度 |
| `nu` | 0 | 魔方模型没有内建 actuator |
| 总质量 | 约 0.09217 kg | 由当前 XML 的 mesh 与密度编译得出 |

关节组成：

- 魔方根部：3 个平移 slide + 1 个 ball joint，共 6 个速度自由度；`qpos` 占 7 维。
- 六个中心小块：6 个 `cubelet:driver:*` hinge，对应六个面角。
- 20 个边块/角块：每块 `rotx/roty/rotz` 三个 hinge，共 60 个小块姿态自由度。
- 20 个 `cubelet:spring:*` slide 关节，用于约束小块相对结构。

魔方小块碰撞几何默认 `condim=6`、摩擦参数约 `(0.2, 0.0025, 0.0005)`，密度为 `521 kg/m³`。预览地面是 `z=0` 的 plane，`condim=6`、摩擦 `(0.8, 0.005, 0.001)`；reset 时将魔方中心置于 `z=0.029 m`。地面是无限平面，魔方可滑移/滚动，但不会从桌边掉落。

注意质量：此前实际魔方质量约 80 g；Robogym 模型当前编译质量约 92.17 g，约重 15%。正式训练或 sim-to-real 前应校准质量/惯量。

## 4. 六面方向和力矩符号

面名按 Robogym cube-local 坐标定义：

| 面 | 局部轴 | 顺时针力矩符号（绕正轴） |
| --- | --- | --- |
| L | −X | 正 |
| R | +X | 负 |
| F | −Y | 正 |
| B | +Y | 负 |
| D | −Z | 正 |
| U | +Z | 负 |

CW/CCW 是从所选面的外侧朝魔方看时的顺/逆时针。面板按下时对该层选出的九个 cubie 施加广义力矩；力矩会投影到当前 MuJoCo hinge 轴上。释放后不再施加新力矩，但不会清零速度，因此魔方可能继续靠惯性运动。Reset 才会清零 qvel 和外加力矩。

## 5. 启动本机预览

从 Luwu 项目根目录运行：

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
export MUJOCO_GL=glfw
export LD_LIBRARY_PATH="/home/benmao/mambaforge/envs/wheel_leg/lib:${LD_LIBRARY_PATH:-}"

/home/benmao/mambaforge/envs/wheel_leg/bin/python \
  tools/robogym_cube_face_preview.py
```

会打开两个窗口：MuJoCo viewer 和 `Robogym Rubik - Dynamic Face Torque` 控制面板。按住 `Hold CW` 或 `Hold CCW` 才会持续施加力矩；松开即停止施力。滑杆可以调每块小块的力矩。`RESET CUBE + VIEW` 会把魔方 qpos、spring 偏移、速度和外加力矩归零，并恢复三分之四视角，以便看清不同颜色。关闭面板会同时关闭 viewer。

## 6. 测试与复验

运行定向测试：

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji

LD_LIBRARY_PATH="/home/benmao/mambaforge/envs/wheel_leg/lib:${LD_LIBRARY_PATH:-}" \
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/benmao/mambaforge/envs/wheel_leg/bin/python -m pytest \
  -p no:cacheprovider tests/test_robogym_cube_face_preview.py -q
```

当前验证覆盖：六面 CW/CCW 方向、力矩只写广义力而不直接改 qpos、持续力矩经 `mj_step` 后推动关节、地面接触下的小块运动、按住时锁定九块、reset 恢复 spring qpos 和相机视角。

2026-09-28 在 `wheel_leg` 的 MuJoCo 3.6.0 环境中运行，结果：`14 passed`。

## 7. 接入训练前的边界

这只是**无手动力学检查器**：外加力矩直接施加给魔方小块，不模拟手指接触、抓握力、摩擦不足导致的手指滑移，也没有手部 actuator。无限地面不能模拟掉出台面。面层的多个小块以独立关节表示，动态时出现的非标准立方体形状可能反映模型自由度/弹簧近似，而非真实魔方内部机构。

Robogym 的 `FullPerpendicularEnv` 会把 Shadow Hand 与魔方组合，并在运行时移除 spring joints；本预览使用原始 `rubik_perpendicular.xml`，保留其 20 个 spring joints，但仍然不包含 Shadow Hand。后续接入 Luwu 训练时，应使用 Luwu 手的接触力驱动魔方，而不是保留本预览的 `qfrc_applied` 面板输入；还需单独评估质量、惯量、接触和合法面层联动。原始 Robogym XML 未修改。
