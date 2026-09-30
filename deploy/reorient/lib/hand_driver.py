# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""HandDriver interface + MockHandDriver for testing.

Interface contract:
  - home() — block until home pose is reached
  - write_target(qpos: np.ndarray) — (20,) joint targets in radians
  - read_encoders() -> np.ndarray — (20,) current joint angles
  - joint_names_in_encoder_order() -> tuple[str, ...] — verify real-hand vs sim order

MockHandDriver is for testing without hardware (can run the full RealHandEnv pipeline).
The real-hardware WujiHandDriver wraps wujihandpy.
"""
from __future__ import annotations

import abc
import re
from typing import Tuple

import numpy as np


# ────────────────── 20-joint naming order (same order as mjcf right_mjlab.xml) ──────────────────

JOINT_NAMES_20: Tuple[str, ...] = (
    "right_finger1_joint1", "right_finger1_joint2", "right_finger1_joint3", "right_finger1_joint4",
    "right_finger2_joint1", "right_finger2_joint2", "right_finger2_joint3", "right_finger2_joint4",
    "right_finger3_joint1", "right_finger3_joint2", "right_finger3_joint3", "right_finger3_joint4",
    "right_finger4_joint1", "right_finger4_joint2", "right_finger4_joint3", "right_finger4_joint4",
    "right_finger5_joint1", "right_finger5_joint2", "right_finger5_joint3", "right_finger5_joint4",
)


def _resolve_home_qpos() -> np.ndarray:
    """Build the (20,) home qpos array from REORIENT_JOINT_POS regex dict."""
    from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_JOINT_POS

    qpos = np.zeros(20, dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES_20):
        for pattern, val in REORIENT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                qpos[i] = val
                break
        else:
            raise ValueError(f"REORIENT_JOINT_POS has no match for {name}")
    return qpos


# ────────────────── interface ──────────────────

class HandDriverBase(abc.ABC):
    """Driver interface for HandDriver implementations.

    Context manager protocol (__enter__/__exit__) gives implementations a
    chance to acquire/release hardware controllers (wujihandpy.realtime_controller).
    MockHandDriver's enter/exit are no-ops; WujiHandDriver opens the realtime
    controller in __enter__ and closes it in __exit__.

    play_real.py wraps the env loop:
        with WujiHandDriver() as drv:
            env = RealHandEnv(cfg, hand_base_pose, hand_driver=drv)
            ... loop ...
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # don't suppress

    @abc.abstractmethod
    def home(self) -> None: ...

    @abc.abstractmethod
    def write_target(self, joint_targets: np.ndarray) -> None: ...

    @abc.abstractmethod
    def read_encoders(self) -> np.ndarray: ...

    @abc.abstractmethod
    def joint_names_in_encoder_order(self) -> Tuple[str, ...]: ...


# ────────────────── Mock driver ──────────────────

class MockHandDriver(HandDriverBase):
    """In-memory mock: write→read echo with REORIENT home pose initialization.

    - `home()` resets encoder to REORIENT_JOINT_POS (strictly consistent with sim).
    - After `write_target(qpos)`, `read_encoders()` returns the last write value (ideal tracking).
    - Guards against NaN / Inf / shape errors.
    """

    def __init__(self) -> None:
        self._home_qpos = _resolve_home_qpos()
        self._encoders = self._home_qpos.copy()

    def home(self) -> None:
        self._encoders = self._home_qpos.copy()

    def write_target(self, joint_targets: np.ndarray) -> None:
        assert joint_targets.shape == (20,), (
            f"expected (20,), got {joint_targets.shape}"
        )
        assert np.isfinite(joint_targets).all(), "MockHandDriver refuses NaN/Inf"
        self._encoders[:] = joint_targets

    def read_encoders(self) -> np.ndarray:
        return self._encoders.copy()

    def joint_names_in_encoder_order(self) -> Tuple[str, ...]:
        return JOINT_NAMES_20


# ────────────────── Real driver ──────────────────

class WujiHandDriver(HandDriverBase):
    """Real-hardware driver via wujihandpy.

    wujihandpy uses a (5, 4) tensor layout for the 20-joint hand (5 fingers ×
    4 joints), in the same logical order as the mjcf naming convention
    (finger1_joint1 first, ..., finger5_joint4 last). We reshape between
    (20,) flat (deploy convention) and (5, 4) at the boundary.

    Must be used as a context manager:
        with WujiHandDriver() as drv:
            env = RealHandEnv(cfg, hand_driver=drv)
            env.reset()
            ... loop ...

    Args:
        effort_limit: per-joint torque limit (Nm). ``None`` reads
            ``hardware.effort_limit_nm`` from control.yaml; pass explicit
            value to override.
        lowpass_cutoff: realtime_controller lowpass cutoff (Hz). ``None``
            reads ``hardware.lowpass_cutoff_hz`` from control.yaml; pass
            explicit value to override.
        home_duration_s: time to ramp to home pose in home().
    """

    def __init__(
        self,
        effort_limit: float | None = None,
        lowpass_cutoff: float | None = None,
        home_duration_s: float = 3.0,
    ):
        import wujihandpy  # lazy: MockHandDriver works without wujihandpy installed

        from .config_loader import effort_limit_nm, lowpass_cutoff_hz

        if effort_limit is None:
            effort_limit = effort_limit_nm()
        if lowpass_cutoff is None:
            lowpass_cutoff = lowpass_cutoff_hz()

        self._wujihandpy = wujihandpy
        self._hand = None
        self._ctrl = None
        self._ctrl_cm = None
        self._effort_limit = float(effort_limit)
        self._lowpass_cutoff = float(lowpass_cutoff)
        self._home_duration_s = home_duration_s
        self._home_qpos = _resolve_home_qpos()

    def __enter__(self):
        self._hand = self._wujihandpy.Hand()
        self._hand.write_joint_effort_limit(self._effort_limit)
        self._hand.write_joint_enabled(True)
        self._ctrl_cm = self._hand.realtime_controller(
            enable_upstream=True,
            filter=self._wujihandpy.filter.LowPass(cutoff_freq=self._lowpass_cutoff),
        )
        self._ctrl = self._ctrl_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._ctrl_cm is not None:
                self._ctrl_cm.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._ctrl = None
            self._ctrl_cm = None
            if self._hand is not None:
                try:
                    self._hand.write_joint_enabled(False)
                except Exception:
                    pass
                self._hand = None
        return False

    def home(self) -> None:
        """Smoothly ramp from current pose to REORIENT_JOINT_POS."""
        import time

        assert self._ctrl is not None, "WujiHandDriver must be used as context manager"
        current = self._hand.read_joint_actual_position()  # (5, 4)
        target = self._home_qpos.reshape(5, 4)

        steps = int(self._home_duration_s * 50)  # 50 Hz smoothing
        dt = self._home_duration_s / steps
        for i in range(steps):
            t = (i + 1) / steps
            t_smooth = t * t * (3 - 2 * t)  # ease in-out
            interp = current + t_smooth * (target - current)
            self._ctrl.set_joint_target_position(interp)
            time.sleep(dt)

    def write_target(self, joint_targets: np.ndarray) -> None:
        assert joint_targets.shape == (20,), (
            f"expected (20,), got {joint_targets.shape}"
        )
        assert np.isfinite(joint_targets).all(), "WujiHandDriver refuses NaN/Inf"
        assert self._ctrl is not None, "WujiHandDriver must be used as context manager"
        self._ctrl.set_joint_target_position(joint_targets.reshape(5, 4))

    def read_encoders(self) -> np.ndarray:
        assert self._hand is not None, "WujiHandDriver must be used as context manager"
        return self._hand.read_joint_actual_position().flatten().astype(np.float64)

    def joint_names_in_encoder_order(self) -> Tuple[str, ...]:
        return JOINT_NAMES_20
