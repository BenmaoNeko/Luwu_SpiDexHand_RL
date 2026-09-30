"""Spidex 单指位置逆解。

每根手指 4 个 hinge，只约束指尖在手根坐标系（base_link）下的 3D 位置。
多出来的 1 个自由度用零空间拉向参考姿态（默认全零）。

用法::

    from finger_ik import finger_ik
    finger_conf = finger_ik(index, pos)   # index: 0~3, pos: (x, y, z) in base_link
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

XML_PATH = Path(__file__).resolve().parent / "xmls" / "spidex.xml"

# index 0..3 → finger1..finger4
FINGER_JOINTS = (
    ("joint7", "joint8", "joint9", "joint10"),
    ("joint11", "joint12", "joint13", "joint14"),
    ("joint15", "joint16", "joint17", "joint18"),
    ("joint19", "joint20", "joint21", "joint22"),
)
FINGER_TIPS = ("finger1_tip", "finger2_tip", "finger3_tip", "finger4_tip")


def _normalize_index(index: int) -> int:
    if not 0 <= int(index) <= 3:
        raise ValueError(f"finger index must be 0..3 (finger1..finger4), got {index}")
    return int(index)


class FingerIK:
    """基于 MJCF 正运动学 + 阻尼最小二乘雅可比的单指 IK。"""

    def __init__(self, xml_path: str | Path | None = None):
        path = Path(xml_path) if xml_path is not None else XML_PATH
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._tip_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FINGER_TIPS
        ]
        self._qposadr = []
        self._dofadr = []
        self._q_lo = []
        self._q_hi = []
        for joints in FINGER_JOINTS:
            qadr, dadr, lo, hi = [], [], [], []
            for name in joints:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qadr.append(int(self.model.jnt_qposadr[jid]))
                dadr.append(int(self.model.jnt_dofadr[jid]))
                lo.append(float(self.model.jnt_range[jid, 0]))
                hi.append(float(self.model.jnt_range[jid, 1]))
            self._qposadr.append(np.array(qadr, dtype=int))
            self._dofadr.append(np.array(dadr, dtype=int))
            self._q_lo.append(np.array(lo))
            self._q_hi.append(np.array(hi))
        mujoco.mj_forward(self.model, self.data)

    def _set_finger(self, index: int, conf: np.ndarray) -> None:
        qadr = self._qposadr[index]
        self.data.qpos[qadr] = np.clip(conf, self._q_lo[index], self._q_hi[index])
        mujoco.mj_forward(self.model, self.data)

    def _tip_in_base(self, index: int) -> np.ndarray:
        p_base = self.data.xpos[self._base_id]
        r_base = self.data.xmat[self._base_id].reshape(3, 3)
        p_tip = self.data.xpos[self._tip_ids[index]]
        return r_base.T @ (p_tip - p_base)

    def _jac_in_base(self, index: int) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, self._tip_ids[index])
        r_base = self.data.xmat[self._base_id].reshape(3, 3)
        return r_base.T @ jacp[:, self._dofadr[index]]

    def finger_fk(self, index: int, conf) -> np.ndarray:
        """正运动学：关节角 → 指尖在 base_link 下的位置。"""
        index = _normalize_index(index)
        self._set_finger(index, np.asarray(conf, dtype=float).reshape(4))
        return self._tip_in_base(index).copy()

    def finger_ik(
        self,
        index: int,
        pos,
        q_init=None,
        q_pref=None,
        tol: float = 1e-4,
        max_iters: int = 120,
        damping: float = 1e-5,
        nullspace: float = 0.02,
        multiseed: bool = True,
    ) -> np.ndarray:
        """单指位置逆解。

        Parameters
        ----------
        index : int
            手指编号，0=finger1, 1=finger2, 2=finger3, 3=finger4。
        pos : array_like, shape (3,)
            指尖目标位置，手根坐标系 base_link。
        q_init : array_like, optional
            初值，默认全零。
        q_pref : array_like, optional
            冗余自由度参考姿态，默认与 q_init 相同。

        Returns
        -------
        np.ndarray, shape (4,)
            该手指 4 个关节角（弧度）。
        """
        index = _normalize_index(index)
        target = np.asarray(pos, dtype=float).reshape(3)
        q = np.zeros(4) if q_init is None else np.asarray(q_init, dtype=float).reshape(4).copy()
        q_pref = q.copy() if q_pref is None else np.asarray(q_pref, dtype=float).reshape(4)
        q = np.clip(q, self._q_lo[index], self._q_hi[index])

        seeds = [q]
        if multiseed:
            if not np.allclose(q, 0.0):
                seeds.append(np.zeros(4))
            rng = np.random.default_rng(0)
            seeds.extend(rng.uniform(-0.8, 0.8, size=(4, 4)))

        best_q, best_err = q.copy(), np.inf
        eye3 = np.eye(3)
        eye4 = np.eye(4)

        for seed in seeds:
            q = np.clip(seed.copy(), self._q_lo[index], self._q_hi[index])
            for _ in range(max_iters):
                self._set_finger(index, q)
                err = target - self._tip_in_base(index)
                nerr = float(np.linalg.norm(err))
                if nerr < best_err:
                    best_err, best_q = nerr, q.copy()
                if nerr < tol:
                    return q
                jac = self._jac_in_base(index)
                jjt = jac @ jac.T + damping * eye3
                dq_pri = jac.T @ np.linalg.solve(jjt, err)
                j_pinv = jac.T @ np.linalg.solve(jjt, jac)
                dq_sec = (eye4 - j_pinv) @ (q_pref - q)
                dq = dq_pri + nullspace * dq_sec
                step = min(1.0, 0.4 / (np.linalg.norm(dq) + 1e-9))
                q = np.clip(q + step * dq, self._q_lo[index], self._q_hi[index])
            if best_err < tol:
                break

        if best_err >= tol:
            print(f"[finger_ik] finger {index} residual {best_err*1000:.2f} mm (target {target})")
        return best_q


_SOLVER: FingerIK | None = None


def _solver() -> FingerIK:
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = FingerIK()
    return _SOLVER


def finger_ik(index: int, pos, **kwargs) -> np.ndarray:
    """finger_conf = finger_ik(index, pos)"""
    return _solver().finger_ik(index, pos, **kwargs)


def finger_fk(index: int, conf) -> np.ndarray:
    return _solver().finger_fk(index, conf)


if __name__ == "__main__":
    ik = FingerIK()
    print("rest tip positions in base_link:")
    for i in range(4):
        p0 = ik.finger_fk(i, [0, 0, 0, 0])
        print(f"  finger{i} {p0}")

    print("\nround-trip (offset -8 mm in z, +5 mm radial):")
    for i in range(4):
        p0 = ik.finger_fk(i, [0, 0, 0, 0])
        radial = np.array([p0[0], p0[1], 0.0])
        radial = radial / (np.linalg.norm(radial) + 1e-12)
        target = p0 + np.array([0.0, 0.0, -0.008]) - 0.005 * radial
        q = ik.finger_ik(i, target)
        p1 = ik.finger_fk(i, q)
        err = np.linalg.norm(p1 - target)
        print(f"  finger{i} q={np.round(q, 3)}  err={err*1000:.2f} mm")
