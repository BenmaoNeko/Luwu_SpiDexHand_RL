# 本地 Luwu 模型 MuJoCo 预览实现

日期：2026-09-24  
范围：使用本机 checkpoint 和现有 MJLab 环境，在浏览器中实时预览 Luwu 手部策略。  
验证模型：`model_2999.pt`，任务 `LuwuHand_Reorient_FromScratch_Curriculum`。

## 目的与边界

本流程用于检查 checkpoint 是否能严格加载、策略能否驱动实时仿真，以及观察手部和方块的运动。它不是训练复现或量化评测；预览中某个环境成功翻面，不能代替训练统计或多种子评估。

预览复用项目根目录下 `scripts/play/play_rsl_rl.py` 使用的配置、环境和 runner 代码。入口本身没有“从观测配置删除条目”的 CLI 参数，因此旧模型兼容由一个进程内的启动适配完成，不改项目源码，也不放宽 checkpoint 的严格加载。

Ponytail 原则：先复用现有路径，只处理已确认的 actor 输入差异；不加依赖、不改奖励或训练代码、不改第三方文件。Viser 明确绑定 `127.0.0.1`，避免默认监听所有网卡。

## 本次核验的模型

| 项目 | 值 |
|---|---|
| checkpoint | `logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/model_2999.pt` |
| SHA-256 | `399b3876a714fc9ed19e9305742942f6c7595a10672d11b9ce46bb955d888313` |
| 文件大小 | 10,296,373 bytes |
| task ID | `LuwuHand_Reorient_FromScratch_Curriculum` |
| actor / critic / action | 162 / 344 / 16 |
| 环境数 / 设备 | 8 / CPU |
| 仿真 / 控制步长 | 0.01 s / 0.05 s |

训练结果和模型接口以 [model_2999 训练交接](training/2026-09-24-model2999-final-training-handoff.md) 为准。该交接记录的滚动翻面成功率为 70.2%、平均连续翻面 27.3 次、平均面覆盖 4.48；指间碰撞频率约 44.8%。

## 本地入口与环境

- 主入口：[scripts/play/play_rsl_rl.py](../scripts/play/play_rsl_rl.py)
- 配置加载：`wuji_mjlab.utils.task_cfg_utils.prepare_task_cfgs`
- 推理和 viewer 分发：`wuji_mjlab.utils.play_runner.run_play_with_cfg`
- 本地环境：`/home/benmao/mambaforge/envs/wheel_leg`
- 本次版本：Python 3.11.15、PyTorch 2.7.1+cpu、MuJoCo 3.6.0、Warp 1.12.1

不要用 `src/wuji_mjlab/tasks/reorient/scripts/view_task.py` 预览 checkpoint：它使用 zero/random dummy policy，不加载训练模型。根目录的 `tools/convex_preview.py` 是碰撞几何检查，也不运行 PPO 策略。

## 为不同 checkpoint 选择观测配置

先读模型对应的 training handoff，确认 task ID、actor observation 维度、action 维度和 checkpoint 路径。不要默认所有 checkpoint 都是 162 维。

本次模型的 actor 输入是 162 维；较新的本地策略配置会在 actor 末尾增加 4 个视觉 metadata 项，成为 166 维：

- `vision_valid`
- `vision_age`
- `vision_confidence`
- `vision_innovation`

启动时根据 checkpoint 的 actor 维度设置 `EXPECTED_ACTOR_DIM`：

- 162：从预览 policy terms 中移除上述 metadata 项。
- 166：保留全部项。
- 其他维度：先检查 handoff 和训练时配置，补齐经过确认的映射后再启动。

运行日志必须与 checkpoint 接口匹配。保留 `strict=True`；维度或权重不匹配时停止并查明原因，不能通过关闭严格加载来绕过。

## 启动实时预览

在一个终端运行以下命令。只需按当前模型修改 `TASK_ID`、`CHECKPOINT` 和 `EXPECTED_ACTOR_DIM`。本示例显式把 Viser 限制在 loopback 地址；若 8080 已占用，Viser 会选择其他空闲端口，使用日志里实际打印的地址。

```bash
cd /home/benmao/桌面/Luwu_Hand_Wuji

PYTHONNOUSERSITE=1 \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl" \
MUJOCO_GL=glfw \
MPLCONFIGDIR=/tmp/mpl-luwu-preview \
LD_LIBRARY_PATH="/home/benmao/mambaforge/envs/wheel_leg/lib:${LD_LIBRARY_PATH:-}" \
/home/benmao/mambaforge/envs/wheel_leg/bin/python - <<'PY'
import viser
import wuji_mjlab.tasks  # noqa: F401
import wuji_mjlab.utils.play_runner as play_runner
from mjlab.scripts.play import PlayConfig
from mjlab.viewer import ViserPlayViewer
from wuji_mjlab.utils.play_runner import run_play_with_cfg
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK_ID = "LuwuHand_Reorient_FromScratch_Curriculum"
CHECKPOINT = "/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/model_2999.pt"
EXPECTED_ACTOR_DIM = 162
PREVIEW_METADATA_TERMS = (
    "vision_valid", "vision_age", "vision_confidence", "vision_innovation",
)


class LoopbackViserPlayViewer(ViserPlayViewer):
    """Use the standard MJLab viewer with a loopback-only Viser server."""

    def __init__(self, env, policy):
        self._loopback_server = viser.ViserServer(
            host="127.0.0.1", port=8080, label="mjlab"
        )
        super().__init__(env, policy, viser_server=self._loopback_server)

    def close(self):
        try:
            super().close()
        finally:
            self._loopback_server.stop()

    def run(self, num_steps=None):
        try:
            super().run(num_steps=num_steps)
        except KeyboardInterrupt:
            print("[INFO] Preview stopped by Ctrl+C.", flush=True)


# run_play_with_cfg uses this module-level viewer class. Replace it only in
# this process so its normal environment, actor loader, and strict=True path
# remain in use while the Viser bind address stays local.
play_runner.ViserPlayViewer = LoopbackViserPlayViewer

env_cfg, agent_cfg = prepare_task_cfgs(TASK_ID, [], play=True)
policy_terms = env_cfg.observations["policy"].terms
if EXPECTED_ACTOR_DIM == 162:
    for name in PREVIEW_METADATA_TERMS:
        policy_terms.pop(name, None)
elif EXPECTED_ACTOR_DIM != 166:
    raise RuntimeError(
        f"No documented observation mapping for actor dim {EXPECTED_ACTOR_DIM}; "
        "inspect the checkpoint handoff before previewing."
    )

print(f"[INFO] Task: {TASK_ID}", flush=True)
print(f"[INFO] Checkpoint: {CHECKPOINT}", flush=True)
print(f"[INFO] Expected actor dim: {EXPECTED_ACTOR_DIM}", flush=True)
print(f"[INFO] Policy terms: {list(policy_terms)}", flush=True)

config = PlayConfig(
    viewer="viser", num_envs=8, device="cpu", checkpoint_file=CHECKPOINT
)
run_play_with_cfg(TASK_ID, config, env_cfg, agent_cfg)
PY
```

Viser 会打印 `HTTP` 地址，例如 `http://127.0.0.1:8080`。在本机浏览器打开日志里打印的地址。保持启动终端和浏览器页面打开，预览是实时仿真，不是静态截图。

### Codex 受限环境说明

受限执行环境可能禁止创建 socket，并报 `PermissionError: [Errno 1] Operation not permitted`，或导致 Viser 立即停止。不要改成公开监听或添加绕过代码；对启动命令申请本机执行权限，并保持脚本中的 `host="127.0.0.1"`。只把最终打印的 loopback 地址交给本机浏览器。

## 启动后检查

启动日志至少应显示：

1. 加载的是指定 checkpoint 文件。
2. Environment device 为 `cpu`，environment count 为 8。
3. policy observation shape 为 `(162,)`，critic 为 `(344,)`，action shape 为 16（对本次 model_2999）。
4. actor checkpoint 严格加载成功，未出现权重维度错误。
5. Viser 显示 `listening` 且地址以 `127.0.0.1` 开头。
6. 浏览器中的状态为 `Running`，step 计数持续增加；策略状态和手/方块场景持续更新。

面板中的 `ori_error`、`flip_chain`、`face_coverage` 是所选仿真环境的当前状态。它们只用于确认场景和策略在推进，不应当作独立的模型评测结果。

## 预览配置差异与已知提示

- `play=True` 使用任务注册的 play 环境配置。对 FromScratch 任务，它设置 Stage 3 目标模式并关闭课程管理器；这不是原训练过程中从 Stage 0 晋级的重放。
- 训练时的随机化、奖励、终止和统计以训练交接与训练日志为准。预览日志中出现的 reward 权重不能替代训练时动态课程参数；actor 推理只使用 policy observation。
- 本机 PyTorch 是 CPU build。本次按 `device="cpu"` 预览；Warp 虽报告本机 GPU 可用，也不代表 PyTorch 推理可使用 CUDA。CPU 环境速度约为 0.6 倍实时。
- 本次场景打印过 `opt.ccd_iterations ... needs to be increased` 和 MJLab asset attach 提示。预览可启动，但若观察到穿透或碰撞异常，应单独检查碰撞求解与资产，不要把警告当成 checkpoint 加载失败。
- 交接文档报告了训练结束在 iteration 2999/3000。预览只核实 checkpoint 加载和实时行为，不修复或重新评估训练过程。

## 停止与故障处理

- 在启动终端按 `Ctrl+C`。适配 viewer 会关闭本地 Viser server 并返回命令行。
- 若 observation 维度不符，停止并对照 handoff、task ID、policy terms；不要改 `strict=True`。
- 若 8080 被占用，查看启动输出中的实际 Viser 端口，浏览器打开对应的 `127.0.0.1` 地址。
- 若本机浏览器无法连接，先确认启动进程仍在运行、页面 URL 使用日志中端口、socket 权限允许；不要把监听地址改为 `0.0.0.0`。
- 若加载了错误 run，检查 `CHECKPOINT` 的绝对路径和 SHA-256，再启动。

## 本次实际观察记录

model_2999 在 8 环境 CPU preview 中严格加载成功，actor/critic/action 为 162/344/16。Viser 面板显示过 `Running`、策略 `success=True`、姿态误差约 1.7°、翻面链 50、六面覆盖 6/6。这些是预览运行中的单环境快照，不是可复现的总体成功率。
