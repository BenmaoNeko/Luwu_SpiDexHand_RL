"""夹一根小方棍，依次在 x/y/z/roll/yaw 上做小幅周期运动。

夹持配置与 ik_demo.py 相同：finger0 / finger2，theta=85°，init_l=0.02，z=0.16。
两指尖刚连在棍子两侧，棍子位姿在手根坐标系下变化，IK 跟踪指尖位置。
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from finger_ik import FingerIK

XML_PATH = Path(__file__).resolve().parent / "xmls" / "scene_stick.xml"

THETA = 85 / 180 * np.pi
INIT_L = 0.02
Z0 = 0.16
CENTER0 = np.array([0.0, 0.0, Z0])
OFFSET0 = INIT_L * np.array([np.cos(THETA), -np.sin(THETA), 0.0])
OFFSET2 = -OFFSET0

LONG_DIR = np.array([0.0, 0.0, 1.0])  # 细长方向：掌心向上（base_link +Z）
PINCH_DIR = OFFSET0 / (np.linalg.norm(OFFSET0) + 1e-12)
WIDTH_DIR = np.cross(LONG_DIR, PINCH_DIR)
R_REST = np.column_stack([LONG_DIR, PINCH_DIR, WIDTH_DIR])

# 平移 4 mm；转动 10 deg。roll=绕长轴（竖直），yaw=绕横向。
CHANNELS = (
    ("x", "trans", np.array([1.0, 0.0, 0.0]), 0.004),
    ("y", "trans", np.array([0.0, 1.0, 0.0]), 0.004),
    ("z", "trans", np.array([0.0, 0.0, 1.0]), 0.004),
    ("roll", "rot", LONG_DIR, np.deg2rad(10.0)),
    ("yaw", "rot", WIDTH_DIR, np.deg2rad(10.0)),
)
PERIOD = 2.0
N_CYCLES = 2
PAUSE = 0.4
IK_KW = dict(multiseed=False, tol=5e-5, max_iters=80, nullspace=0.005)


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ]
    )


def stick_pose(channel_idx: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """返回棍子中心（base 系）以及作用在 rest offset 上的附加旋转。"""
    _, kind, axis, amp = CHANNELS[channel_idx]
    s = amp * np.sin(phase)
    t = CENTER0.copy()
    r_extra = np.eye(3)
    if kind == "trans":
        t = t + s * axis
    else:
        r_extra = axis_angle_matrix(axis, s)
    return t, r_extra


def tip_targets(center: np.ndarray, r_extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return center + r_extra @ OFFSET0, center + r_extra @ OFFSET2


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot, dtype=np.float64).reshape(9))
    return quat


def main():
    ik = FingerIK()
    confs = [np.zeros(4) for _ in range(4)]
    p0, p2 = tip_targets(CENTER0, np.eye(3))
    confs[0] = ik.finger_ik(0, p0, q_init=confs[0], **IK_KW)
    confs[2] = ik.finger_ik(2, p2, q_init=confs[2], **IK_KW)
    for i, p in ((0, p0), (2, p2)):
        got = ik.finger_fk(i, confs[i])
        print(f"finger{i} rest err={np.linalg.norm(got - p) * 1000:.3f} mm")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")])
    q_home = np.array([0.0, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0])
    p_base = q_home[:3]

    seg = PERIOD * N_CYCLES + PAUSE
    t0 = time.time()
    last_name = None
    print("关闭窗口结束；依次 x → y → z → roll → yaw")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            elapsed = time.time() - t0
            ch = int(elapsed / seg) % len(CHANNELS)
            local = elapsed - int(elapsed / seg) * seg
            name = CHANNELS[ch][0]
            if name != last_name:
                print(f"channel: {name}")
                last_name = name
            if local >= PERIOD * N_CYCLES:
                phase = 0.0
            else:
                phase = 2.0 * np.pi * (local / PERIOD)

            center, r_extra = stick_pose(ch, phase)
            t0_tip, t2_tip = tip_targets(center, r_extra)
            confs[0] = ik.finger_ik(0, t0_tip, q_init=confs[0], q_pref=confs[0], **IK_KW)
            confs[2] = ik.finger_ik(2, t2_tip, q_init=confs[2], q_pref=confs[2], **IK_KW)

            data.qpos[:7] = q_home
            data.qpos[7:] = np.concatenate(confs)
            data.ctrl[:] = data.qpos[7:]
            data.mocap_pos[mocap_id] = p_base + center
            data.mocap_quat[mocap_id] = mat_to_quat(r_extra @ R_REST)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
