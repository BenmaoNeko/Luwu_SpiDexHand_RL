# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Quat math helpers used by deploy scripts.

The ``load_hand_base_pose`` + ``hand_base_calibration.yaml`` pair was
removed once cube / goal observations were moved onto the wrist-AprilTag
frame. Robot root pose is baked into the mjcf at compile time.

Convention:
  - All quaternions are in wxyz order (consistent across MuJoCo / mjlab stack).
"""
from __future__ import annotations

import numpy as np


# ────────────────── quat math (wxyz convention) ──────────────────

def quat_apply(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` by `quat_wxyz`. Returns the rotated vector.

    Uses the standard rotation matrix derivation from a unit quaternion
    (Hamilton convention, wxyz order).
    """
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R @ vec


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quats: q1 ∘ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quat) of a wxyz quat."""
    w, x, y, z = quat_wxyz
    return np.array([w, -x, -y, -z])
