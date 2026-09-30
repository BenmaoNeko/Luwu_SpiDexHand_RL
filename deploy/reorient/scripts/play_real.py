# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Real-hand policy loop with cube + goal ZMQ wiring.

Pipeline:
    ONNXPolicy(ckpt) -> make_real_hand_env_cfg(policy_config=policy.config)
                     -> RealHandEnv(WujiHandDriver) ← CubeReceiver(ZMQ:5555)
                                                    ← GoalReceiver(ZMQ:5556) / _GoalStub
                     -> step (~20Hz) → cube tracking goal → success monitor

The env is built FROM the policy's sidecar JSON (model-intrinsic params:
action_scale, ema_alpha, warmup_time_s, ctrl_dt, history_len, control_mode).
Hardware params (effort_limit_nm, lowpass_cutoff_hz) come from control.yaml
unless overridden by CLI.

Goal modes:
  external  Goal driven by toreal_viewer mocap drag (subscribes ZMQ:5556).
  fixed     User-specified --goal-quat held constant. Bypasses ZMQ via stub.
  random    Uniform-SO(3) random goal rotated every --goal-period sec. Stub.
  auto      Uniform-SO(3) goal; switches when cube achieves it (geodesic <
            --success-threshold sustained for --auto-hold-sec) OR per-goal
            wall-clock exceeds --auto-timeout-sec. New goals reject
            candidates closer than --auto-min-angle-deg to the cube quat.

Benchmark mode (``--benchmark-trials N``):
  Force --goal-mode=auto + face-aligned goal sampling (or ``--benchmark-hard``
  for uniform SO(3)), run N independent trials, track per-trial outcome
  (``success`` / ``timeout`` / ``stuck_reset``), then write
  ``logs/benchmark_<timestamp>.json``.  Trials end on achievement, per-trial
  timeout, or stuck detection (no joint movement for ``STUCK_TIMEOUT_SEC``).
  On stuck/timeout we run the recovery sequence and sample the next goal.

Success: geodesic(cube_quat, goal_quat) < --success-threshold (default 0.2 rad ≈ 11°).

Use Ctrl+C any time to abort. The context manager disables joints on exit.

Usage:
    pixi run -e deploy python deploy/reorient/scripts/play_real.py \\
        [--ckpt PATH] [--duration 30] \\
        [--goal-mode external|fixed|random|auto] \\
        [--goal-quat W,X,Y,Z] [--goal-period 5] [--effort-limit N] \\
        [--lowpass-cutoff HZ] [--no-cube-zmq] [--success-threshold 0.2] \\
        [--auto-hold-sec 0.5] [--auto-min-angle-deg 90] \\
        [--auto-timeout-sec 14] [--log-file PATH] \\
        [--benchmark-trials N] [--benchmark-timeout 14] \\
        [--benchmark-threshold 0.35] [--benchmark-hold-sec 2.5] \\
        [--benchmark-hard] [--benchmark-output PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Silence GLFW/EGL teardown noise. When mujoco's passive viewer shuts down
# (process exit or sim_viewer.close()) the EGL display may already be
# destroyed by another part of the GL stack, and GLFW emits a warning like
# "Failed to clear current context: An EGLDisplay argument does not name a
# valid EGL display connection" via warnings.warn(.., GLFWError). It's
# cosmetic — the viewer itself behaves correctly — but it clutters the
# benchmark stdout right when the operator wants to read the JSON summary.
try:
    from glfw import GLFWError as _GLFWError
    warnings.filterwarnings("ignore", category=_GLFWError)
except ImportError:
    pass

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from deploy.reorient.lib._paths import find_logs_root
from deploy.reorient.lib.hand_driver import WujiHandDriver
from deploy.reorient.lib.ood_diagnostics import AsyncPrinter, format_ood_lines
from deploy.reorient.lib.onnx_policy import ONNXPolicy
from deploy.reorient.lib.real_hand_env import RealHandEnv
from deploy.reorient.lib.real_hand_env_cfg import make_real_hand_env_cfg
from deploy.reorient.lib.zmq_bridge import CubeReceiver, GoalReceiver


# ────────────────── benchmark + stuck/recovery constants ──────────────────
# These constants are calibrated to specific failure modes — do not drift
# without a hardware sweep.
BENCHMARK_DEFAULT_THRESHOLD = 0.35   # rad (~20°), loose
BENCHMARK_DEFAULT_HOLD_SEC = 2.5     # cumulative seconds of hold
BENCHMARK_DEFAULT_TIMEOUT = 14.0     # per-trial wall clock

STUCK_JOINT_THRESHOLD = 0.4          # rad; max |Δjoint| below this counts as "not moving"
STUCK_TIMEOUT_SEC = 5.0              # seconds of no movement before stuck
RECOVERY_F1J1 = 0.50                 # finger1 joint1 (thumb) override
RECOVERY_F5J2 = -0.37                # finger5 joint2 (pinky sideways) override
RECOVERY_DURATION_SEC = 2.0          # smooth recovery move duration


# 6 canonical face-axis orientations.  Used by --benchmark mode (without
# --benchmark-hard) to sample easier goals than uniform SO(3): pick a base
# face quat, then compose a random yaw about Z.
_FACE_BASE_QUATS = (
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float64),
    np.array([0.7071068, 0.0, -0.7071068, 0.0], dtype=np.float64),
    np.array([0.7071068, -0.7071068, 0.0, 0.0], dtype=np.float64),
    np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float64),
)


# DEFAULT_CKPT is intentionally None: public users must supply their own
# trained policy via --ckpt <path-to-your-policy.onnx>. main() reports a
# clear error if --ckpt is omitted.
DEFAULT_CKPT = None


# ────────────────── helpers ──────────────────


def quat_geodesic(q1_wxyz, q2_wxyz):
    """Angle (rad) between two unit quats (wxyz)."""
    dot = abs(float(np.dot(q1_wxyz, q2_wxyz)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * np.arccos(dot)


def _compute_ori_error_6d(cube_quat_wxyz, goal_quat_wxyz) -> np.ndarray:
    """6D rotation-matrix error (last 6 elements of row-major mat_diff[3:]).

    Matches training-side ``goal_rot_err_6d``.  Both quats are in
    the tag frame in deploy.  The training-side obs computes:
        quat_diff = cube_quat * goal_quat^(-1)
        mat_diff  = quat2Mat(quat_diff)  # 9 elements, row-major
        ori_err   = mat_diff[3:]         # last 6
    """
    goal_inv = np.zeros(4)
    mujoco.mju_negQuat(goal_inv, np.asarray(goal_quat_wxyz, dtype=np.float64))
    quat_diff = np.zeros(4)
    mujoco.mju_mulQuat(
        quat_diff,
        np.asarray(cube_quat_wxyz, dtype=np.float64),
        goal_inv,
    )
    mat_diff = np.zeros(9)
    mujoco.mju_quat2Mat(mat_diff, quat_diff)
    return mat_diff[3:].astype(np.float64)


def _parse_quat_wxyz(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--goal-quat expects 4 comma-separated floats (w,x,y,z), got {s!r}"
        )
    q = np.array(parts, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        raise argparse.ArgumentTypeError("--goal-quat has zero norm")
    return q / n


def _random_unit_quat_wxyz() -> np.ndarray:
    """Uniform random rotation over SO(3) via scipy.

    Rotation.random() draws uniformly over SO(3); naive euler-angle sampling
    would bias the poles. The returned scipy quat is (x, y, z, w); we re-pack
    to MuJoCo's (w, x, y, z) convention used everywhere else in this stack.
    """
    q_xyzw = Rotation.random().as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _normalize_quat_wxyz(quat) -> np.ndarray:
    """Normalize a (w, x, y, z) quat and enforce canonical sign (w >= 0)."""
    q = np.asarray(quat, dtype=np.float64).copy()
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q /= n
    if q[0] < 0:
        q = -q
    return q


def _quat_mul_wxyz(q1, q2) -> np.ndarray:
    """Multiply quaternions in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def random_face_aligned_quaternion(rng=None) -> np.ndarray:
    """Sample a face-aligned benchmark goal: a canonical face quat plus a
    random yaw about Z.  Easier than uniform SO(3); the legacy benchmark
    uses this as the default (``--benchmark-hard`` opts into uniform).

    Args:
        rng: ``np.random.Generator``, ``np.random.RandomState`` or the
            ``np.random`` module.  Anything with ``integers``/``randint`` +
            ``uniform`` works.  Defaults to ``np.random`` (process-global).
    """
    if rng is None:
        rng = np.random
    if hasattr(rng, "integers"):
        base_idx = int(rng.integers(len(_FACE_BASE_QUATS)))
    else:
        base_idx = int(rng.randint(len(_FACE_BASE_QUATS)))
    base = _FACE_BASE_QUATS[base_idx]
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    z_rot = np.array(
        [np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)],
        dtype=np.float64,
    )
    return _normalize_quat_wxyz(_quat_mul_wxyz(z_rot, base))


def _run_recovery_sequence(env, drv, duration: float = RECOVERY_DURATION_SEC) -> None:
    """Smoothly move the hand to a recovery pose, then back to REORIENT home.

    Two-leg cubic-ease interpolation, written directly via the driver's
    ``write_target`` (bypasses the policy + env.step entirely).  The
    recovery pose is REORIENT_JOINT_POS with two finger overrides
    (``RECOVERY_F1J1`` on finger1_joint1, ``RECOVERY_F5J2`` on
    finger5_joint2) that un-wedge the most common finger-collision modes.
    After the override-leg we interpolate back to clean REORIENT home so
    the next policy tick starts from the same pose as a fresh reset.

    Routes through the ``HandDriverBase.write_target((20,))`` abstraction
    so MockHandDriver can exercise this code path in tests.  Joint-limit
    clipping happens inside the env's action pipeline; the recovery
    sequence sends only interpolated home-pose values which are well
    within limits.
    """
    from wuji_mjlab.tasks.reorient.reorient_constants import REORIENT_JOINT_POS
    from deploy.reorient.lib.hand_driver import JOINT_NAMES_20

    import re

    # Build (20,) home qpos in JOINT_NAMES_20 order from the regex dict.
    home = np.zeros(20, dtype=np.float64)
    for i, name in enumerate(JOINT_NAMES_20):
        for pattern, val in REORIENT_JOINT_POS.items():
            if re.fullmatch(pattern, name):
                home[i] = val
                break

    # Recovery pose: home + 2 overrides.  Indices are JOINT_NAMES_20-order.
    # finger1_joint1 = index 0; finger5_joint2 = index 17.
    f1j1_idx = JOINT_NAMES_20.index("right_finger1_joint1")
    f5j2_idx = JOINT_NAMES_20.index("right_finger5_joint2")
    recovery = home.copy()
    recovery[f1j1_idx] = RECOVERY_F1J1
    recovery[f5j2_idx] = RECOVERY_F5J2

    current = drv.read_encoders().astype(np.float64)
    # 50 Hz update inside the recovery move.
    steps = max(1, int(duration * 50))
    dt = duration / steps

    def _leg(start: np.ndarray, target: np.ndarray) -> None:
        for i in range(steps):
            t = (i + 1) / steps
            t_smooth = t * t * (3.0 - 2.0 * t)  # cubic ease-in-out
            interp = start + t_smooth * (target - start)
            drv.write_target(interp.astype(np.float64))
            time.sleep(dt)

    print(f"    [stuck-reset] moving to recovery pose over {duration:.1f}s ...")
    _leg(current, recovery)
    print(f"    [stuck-reset] interpolating back to home over {duration:.1f}s ...")
    _leg(recovery, home)
    print("    [stuck-reset] recovery complete.")


def _viz_mj_data(env):
    """Return the MjData instance that's updated each step (for the viewer).

    RealHandEnv._fast_forward() (lib/real_hand_env.py) copies wp_data.qpos
    into a private ``env._mj_data`` buffer and runs mj_forward. The public
    ``env.sim.mj_data`` is a separate host-side template that is NOT
    refreshed each step — passing it to the viewer would render a frozen
    pose. The private buffer is the one the viewer must point at.
    """
    return env._mj_data if hasattr(env, "_mj_data") else env.sim.mj_data


def _write_cube_to_sim_for_viz(env, cube_pos_tag, cube_quat_tag_wxyz):
    """Write cube pose into mj_data for digital-twin visualization.

    This is VIZ-ONLY: the obs pipeline (cube_pos_in_tag_from_zmq) bypasses
    scene["object"] entirely, so the policy never reads this. The viewer
    just needs SOMETHING in mj_data.qpos[cube_qpos_adr:] to render.

    Tag-frame coords from the observer are anchored to the wrist AprilTag.
    Lifting them into mjworld for the viewer = compose with the current
    palm pose (palm is read from the already-FK-updated mj_data):
        cube_pos_mjworld  = palm_pos_w + R(palm_quat_w) @ cube_pos_tag
        cube_quat_mjworld = palm_quat_w * cube_quat_tag

    Without the palm offset the cube would render near mjworld origin
    while the hand sits at z≈0.5, looking detached. The math is the same
    "mjworld dance" the policy path was simplified out of — fine here
    because it's viz-only (approximate physical-tag location is OK).

    Returns:
        np.ndarray (3,) — ``cube_pos_mjworld`` (the position we wrote), so
        the caller can place overlay geoms (e.g. the goal-cube overlay)
        relative to where the observed cube is actually rendered.
        ``None`` if the cube body / free joint could not be located.
    """
    mj_model = env.sim.mj_model
    mj_data = _viz_mj_data(env)
    cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object/cube")
    if cube_body_id < 0:
        for candidate in ("object", "cube", "free_cube"):
            cube_body_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, candidate
            )
            if cube_body_id >= 0:
                break
    if cube_body_id < 0:
        return None  # silently skip if we can't find the cube body — viz best-effort
    joint_id = int(mj_model.body_jntadr[cube_body_id])
    if joint_id < 0:
        return None  # body has no free joint
    qpos_adr = int(mj_model.jnt_qposadr[joint_id])

    palm_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_palm_link")
    if palm_body_id < 0:
        for candidate in ("right_palm_link", "palm_link", "palm"):
            palm_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if palm_body_id >= 0:
                break

    if palm_body_id >= 0:
        palm_pos_w = np.asarray(mj_data.xpos[palm_body_id], dtype=np.float64)
        palm_quat_w = np.asarray(mj_data.xquat[palm_body_id], dtype=np.float64)
        from deploy.reorient.lib.frame_transform import quat_apply, quat_mul
        # Lift tag-frame cube to mjworld via palm * TAG_IN_PALM. Viz only — policy obs bypasses this.
        from wuji_mjlab.tasks.reorient.reorient_constants import (
            TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ,
        )
        tip_pos = np.asarray(TAG_IN_PALM_POS, dtype=np.float64)
        tip_quat = np.asarray(TAG_IN_PALM_QUAT_WXYZ, dtype=np.float64)
        tag_pos_w = palm_pos_w + quat_apply(palm_quat_w, tip_pos)
        tag_quat_w = quat_mul(palm_quat_w, tip_quat)
        cube_pos_w = tag_pos_w + quat_apply(
            tag_quat_w, np.asarray(cube_pos_tag, dtype=np.float64)
        )
        cube_quat_w = quat_mul(tag_quat_w, np.asarray(cube_quat_tag_wxyz, dtype=np.float64))
    else:
        cube_pos_w = np.asarray(cube_pos_tag, dtype=np.float64)
        cube_quat_w = np.asarray(cube_quat_tag_wxyz, dtype=np.float64)

    mj_data.qpos[qpos_adr:qpos_adr + 3] = cube_pos_w
    mj_data.qpos[qpos_adr + 3:qpos_adr + 7] = cube_quat_w
    mujoco.mj_forward(mj_model, mj_data)
    return np.asarray(cube_pos_w, dtype=np.float64)


def _draw_goal_overlay(viewer, env, cube_pos_mjworld, goal_quat_tag, cube_size_m: float = 0.054) -> None:
    """Overlay a translucent goal cube 10 cm above the observed cube.

    The training mjcf has no goal-mocap body built in; instead we use
    ``mujoco.viewer.user_scn`` to inject an ad-hoc geom each frame. Drawn
    10 cm above the observed cube so the operator can compare orientations
    without occlusion.

    Args:
        viewer: ``mujoco.viewer`` passive viewer (must expose ``user_scn``).
        env: RealHandEnv (used to locate the palm body for tag→mjworld lift).
        cube_pos_mjworld: mjworld position of the *observed* cube, as
            returned by ``_write_cube_to_sim_for_viz``.
        goal_quat_tag: target orientation in TAG frame (deploy convention).
            We compose with the current palm orientation to push it into
            mjworld — same composition the viz cube write uses for position.
        cube_size_m: cube edge length in meters; consumer should pass
            ``cube_recv.cube_size`` when available so the overlay matches
            the physical tag set in use.
    """
    from deploy.reorient.lib.frame_transform import quat_mul

    mj_data = _viz_mj_data(env)
    palm_body_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_palm_link")
    if palm_body_id < 0:
        for cand in ("right_palm_link", "palm_link", "palm"):
            palm_body_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if palm_body_id >= 0:
                break
    if palm_body_id < 0:
        return

    palm_quat_w = np.asarray(mj_data.xquat[palm_body_id], dtype=np.float64)
    # Lift goal_quat from tag-frame to mjworld via the same composition the
    # cube-position lift uses (tag_quat_w = palm_quat_w * TAG_IN_PALM_QUAT_WXYZ):
    from wuji_mjlab.tasks.reorient.reorient_constants import TAG_IN_PALM_QUAT_WXYZ
    tag_quat_w = quat_mul(palm_quat_w, np.asarray(TAG_IN_PALM_QUAT_WXYZ, dtype=np.float64))
    goal_quat_mj = quat_mul(tag_quat_w, np.asarray(goal_quat_tag, dtype=np.float64))

    # mjv_initGeom needs a 9-element rotation matrix (row-major flatten).
    goal_quat_xyzw = np.array(
        [goal_quat_mj[1], goal_quat_mj[2], goal_quat_mj[3], goal_quat_mj[0]],
        dtype=np.float64,
    )
    mat = Rotation.from_quat(goal_quat_xyzw).as_matrix().flatten()

    offset = np.array([0.0, 0.0, 0.10], dtype=np.float64)
    goal_pos = np.asarray(cube_pos_mjworld, dtype=np.float64) + offset

    # Goal-cube draw mirrors mdp/command_visualization.py; matid/dataid setup is the non-obvious part.
    mj_model = env.sim.mj_model
    cube_mesh_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MESH, "object/cube_mesh")
    ghost_mat_id = -1
    if cube_mesh_id >= 0:
        for geom_idx in range(mj_model.ngeom):
            if (
                mj_model.geom_type[geom_idx] == mujoco.mjtGeom.mjGEOM_MESH
                and mj_model.geom_dataid[geom_idx] == cube_mesh_id
            ):
                ghost_mat_id = int(mj_model.geom_matid[geom_idx])
                break

    viewer.user_scn.ngeom = 0
    g = viewer.user_scn.geoms[0]
    if cube_mesh_id >= 0:
        # _GHOST_MESH_SIZE mirrors mjlab's training-side goal cube (3 cm half-edge,
        # slightly larger than the 2.8 cm physical cube so it visibly enframes it).
        ghost_mesh_size = np.array([0.030, 0.030, 0.030], dtype=np.float64)
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_MESH.value,
            ghost_mesh_size,
            goal_pos,
            mat,
            np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32),  # white + translucent
        )
        g.dataid = 2 * cube_mesh_id  # MuJoCo: mesh renderable slot is 2*mesh_id
        if ghost_mat_id >= 0:
            g.matid = ghost_mat_id
            g.texcoord = 1
    else:
        half = float(cube_size_m) / 2.0
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_BOX.value,
            np.array([half, half, half], dtype=np.float64),
            goal_pos,
            mat,
            np.array([1.0, 0.5, 0.0, 0.5], dtype=np.float32),
        )
    viewer.user_scn.ngeom = 1


# ────────────────── stubs ──────────────────


class _GoalStub:
    """Drop-in for GoalReceiver — exposes .latest() returning the current target quat.

    Used for --goal-mode={fixed,random}: bypasses ZMQ entirely and lets the
    main loop (or a background thread for random) update the target directly.
    """

    def __init__(self, initial_quat_wxyz: np.ndarray):
        self._quat = initial_quat_wxyz.astype(np.float64)
        self._lock = threading.Lock()

    def set_quat(self, quat_wxyz: np.ndarray) -> None:
        with self._lock:
            self._quat = quat_wxyz.astype(np.float64)

    def latest(self) -> np.ndarray:
        with self._lock:
            return self._quat.copy()


class _RandomGoalDriver:
    """Background thread that rotates a _GoalStub to a fresh uniform-SO(3)
    random quat every ``period`` seconds.

    Daemon thread; exits when the main process exits.
    """

    def __init__(self, stub: _GoalStub, period: float):
        self._stub = stub
        self._period = max(period, 0.01)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # Daemon thread would self-terminate at process exit, but an explicit
        # join with a finite timeout gives deterministic shutdown order so
        # cleanup of downstream objects (ZMQ sockets, hand driver) happens
        # after the goal generator has actually stopped writing.
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            self._stub.set_quat(_random_unit_quat_wxyz())
            # Sleep in small chunks so stop() is responsive.
            t_end = time.monotonic() + self._period
            while self._running and time.monotonic() < t_end:
                time.sleep(0.05)


class _AutoGoalDriver:
    """Goal driver that switches on achievement OR per-goal timeout.

    Behaviour:

      * **Achievement**: cube geodesic error < ``success_threshold_rad``
        sustained continuously for ``hold_sec`` seconds (counter resets
        whenever error climbs back above threshold).
      * **Timeout**: per-goal wall-clock exceeds ``timeout_sec`` without
        achievement. Prevents the loop getting stuck on an infeasible goal.
      * **New goal**: uniformly sampled SO(3); rejection-sampled to ensure
        angular distance from the current cube quat is ≥ ``min_goal_angle_rad``
        (up to 1000 retries; falls back to the last candidate if none clear
        the gate).

    Unlike ``_RandomGoalDriver``, this is **not** a background thread — its
    state advances on each ``tick()`` call from the main control loop. That
    keeps the achievement check tightly coupled to the per-step ori_error
    the loop already computes.

    Exposes:
      ``.latest() -> np.ndarray (4,) wxyz`` — current goal (thread-safe).
      ``.tick(cube_quat_wxyz, ori_error_rad, now_sec) -> dict | None``
        Called once per control step. Returns a switch event dict on the
        step the goal flipped, else None.
        Event keys::
          {"reason": "achievement" | "timeout",
           "duration_sec": float,   # wall-clock since this goal was set
           "goal_idx": int}         # 0-indexed switch count, pre-increment
    """

    def __init__(
        self,
        initial_quat_wxyz: np.ndarray,
        success_threshold_rad: float,
        hold_sec: float,
        min_goal_angle_rad: float,
        timeout_sec: float,
        ctrl_dt: float,
        sampler=None,
        cumulative_hold: bool = False,
    ):
        """
        Args:
            sampler: optional ``() -> np.ndarray (4,) wxyz`` to pick the next
                goal.  Defaults to ``_random_unit_quat_wxyz`` (uniform SO(3)).
                Benchmark mode passes ``random_face_aligned_quaternion`` (or
                uniform-SO(3) for ``--benchmark-hard``).
            cumulative_hold: when True the hold counter never resets on a
                miss (cumulative benchmark semantics).
                When False the counter resets the moment error rises above
                threshold (consecutive hold — auto-mode default).
        """
        self._quat = initial_quat_wxyz.astype(np.float64)
        self._success_threshold = float(success_threshold_rad)
        self._hold_sec = float(hold_sec)
        self._min_goal_angle = float(min_goal_angle_rad)
        self._timeout_sec = float(timeout_sec)
        self._ctrl_dt = float(ctrl_dt)
        self._sampler = sampler if sampler is not None else _random_unit_quat_wxyz
        self._cumulative_hold = bool(cumulative_hold)
        self._hold_counter = 0
        # Derive integer step count from seconds so achievement uses the
        # same counter convention as old deploy (_check_hold).
        self._hold_steps_target = max(1, int(round(hold_sec / ctrl_dt)))
        # Goal timer: lazily armed on first tick so .latest()-only consumers
        # don't accidentally start the clock.
        self._goal_start_t: float | None = None
        self._goal_idx = 0
        self._lock = threading.Lock()

    def latest(self) -> np.ndarray:
        with self._lock:
            return self._quat.copy()

    def tick(self, cube_quat_wxyz, ori_error_rad, now_sec):
        """Advance hold counter + per-goal timer; emit switch event when due."""
        if self._goal_start_t is None:
            self._goal_start_t = now_sec

        # Hold logic.  Consecutive-hold (default) resets the moment error
        # crosses back above threshold.  Cumulative-hold (benchmark mode)
        # never resets (cumulative semantics for the relaxed benchmark
        # threshold).
        if ori_error_rad < self._success_threshold:
            self._hold_counter += 1
        elif not self._cumulative_hold:
            self._hold_counter = 0
        achieved = self._hold_counter >= self._hold_steps_target

        elapsed = now_sec - self._goal_start_t
        timed_out = elapsed >= self._timeout_sec

        if not (achieved or timed_out):
            return None

        reason = "achievement" if achieved else "timeout"
        event = {
            "reason": reason,
            "duration_sec": float(elapsed),
            "goal_idx": int(self._goal_idx),
        }
        self._switch_to_new_goal(cube_quat_wxyz)
        return event

    def force_switch(self, cube_quat_wxyz) -> None:
        """Externally-driven switch (benchmark stuck/timeout). Resets hold
        counter and goal timer; samples a new far-enough goal. The main loop
        uses this on benchmark stuck/timeout events; achievement-driven
        switches go through ``tick`` → ``_switch_to_new_goal``."""
        self._switch_to_new_goal(cube_quat_wxyz)

    def _switch_to_new_goal(self, cube_quat_wxyz) -> None:
        """Sample a new SO(3) goal far enough from the current cube quat.

        Up to 1000 retries; if none clear the ``min_goal_angle`` gate we
        accept the last candidate (matches old deploy's fallback). With a
        90° minimum the rejection rate is ~50%, so 1000 tries is overkill
        in practice but cheap.

        The sampler is parameterised so benchmark mode can swap in
        ``random_face_aligned_quaternion`` for face-aligned goals while
        normal auto mode keeps uniform SO(3).
        """
        candidate = self._sampler()
        for _ in range(1000):
            angle = quat_geodesic(candidate, cube_quat_wxyz)
            if angle >= self._min_goal_angle:
                break
            candidate = self._sampler()
        with self._lock:
            self._quat = candidate
        self._hold_counter = 0
        self._goal_start_t = None  # rearmed on next tick
        self._goal_idx += 1


# ────────────────── main ──────────────────


def _build_goal_source(args):
    """Return (goal_obj, random_driver_or_none, description).

    goal_obj must expose .latest() → np.ndarray (4,) wxyz.
    """
    if args.goal_mode == "external":
        recv = GoalReceiver(port=5556)
        # CubeReceiver/GoalReceiver now expose .latest() directly — no adapter.
        return recv, None, recv, "GoalReceiver(ZMQ:5556)"

    if args.goal_mode == "fixed":
        if args.goal_quat is None:
            raise SystemExit(
                "ERROR: --goal-mode=fixed requires --goal-quat W,X,Y,Z"
            )
        stub = _GoalStub(args.goal_quat)
        return stub, None, None, f"fixed quat={args.goal_quat.tolist()}"

    if args.goal_mode == "random":
        stub = _GoalStub(_random_unit_quat_wxyz())
        driver = _RandomGoalDriver(stub, args.goal_period)
        driver.start()
        return stub, driver, None, f"random (period={args.goal_period}s)"

    if args.goal_mode == "auto":
        # Benchmark mode opts into:
        #   * face-aligned sampling (unless --benchmark-hard)
        #   * cumulative hold semantics
        # Both stay off in plain auto mode.
        is_benchmark = getattr(args, "benchmark_trials", None) is not None
        face_aligned = is_benchmark and not getattr(args, "benchmark_hard", False)
        sampler = (
            random_face_aligned_quaternion
            if face_aligned
            else _random_unit_quat_wxyz
        )
        cumulative_hold = is_benchmark
        initial = sampler()
        driver = _AutoGoalDriver(
            initial_quat_wxyz=initial,
            success_threshold_rad=args.success_threshold,
            hold_sec=args.auto_hold_sec,
            min_goal_angle_rad=np.deg2rad(args.auto_min_angle_deg),
            timeout_sec=args.auto_timeout_sec,
            ctrl_dt=args.ctrl_dt,
            sampler=sampler,
            cumulative_hold=cumulative_hold,
        )
        sampler_desc = "face-aligned" if face_aligned else "uniform-SO(3)"
        desc = (
            f"auto (thresh={args.success_threshold:.2f}rad, "
            f"hold={args.auto_hold_sec:.1f}s, "
            f"min_angle={args.auto_min_angle_deg:.0f}°, "
            f"timeout={args.auto_timeout_sec:.1f}s, "
            f"sampler={sampler_desc})"
        )
        # No background driver, no external receiver — auto cycles in-loop.
        return driver, None, None, desc

    raise SystemExit(f"unknown --goal-mode: {args.goal_mode}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-hand policy loop with cube + goal ZMQ.",
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="loop duration in seconds (default 30)")
    parser.add_argument("--ctrl-dt", type=float, default=0.05,
                        help="control step in seconds (default 0.05 = 20Hz, "
                             "used only for monitoring/log timing; the actual "
                             "control rate comes from the policy sidecar JSON)")
    parser.add_argument("--effort-limit", type=float, default=None,
                        help="per-joint Nm (default: control.yaml "
                             "hardware.effort_limit_nm)")
    parser.add_argument("--lowpass-cutoff", type=float, default=None,
                        help="hand LowPass cutoff Hz (default: control.yaml "
                             "hardware.lowpass_cutoff_hz)")
    parser.add_argument("--goal-mode",
                        choices=["external", "fixed", "random", "auto"],
                        default="external")
    parser.add_argument("--goal-quat", type=_parse_quat_wxyz, default=None,
                        help="W,X,Y,Z wxyz (required for --goal-mode=fixed)")
    parser.add_argument("--goal-period", type=float, default=5.0,
                        help="seconds between random goal updates")
    parser.add_argument("--no-cube-zmq", action="store_true",
                        help="disable CubeReceiver, fall back to zeros")
    parser.add_argument("--success-threshold", type=float, default=0.2,
                        help="success geodesic threshold in radians "
                             "(default 0.2 rad ≈ 11°)")
    parser.add_argument("--auto-hold-sec", type=float, default=0.5,
                        help="(auto) seconds cube must stay below "
                             "--success-threshold to count as achievement "
                             "(default 0.5)")
    parser.add_argument("--auto-min-angle-deg", type=float, default=90.0,
                        help="(auto) min angular distance for the next goal "
                             "from the current cube quat (default 90°)")
    parser.add_argument("--auto-timeout-sec", type=float, default=14.0,
                        help="(auto) per-goal timeout — force a switch if "
                             "no achievement within this many seconds "
                             "(default 14)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="optional .npz dump path")
    parser.add_argument(
        "--show-sim", action=argparse.BooleanOptionalAction, default=True,
        help="Open a mujoco passive viewer showing the live digital twin "
             "(real hand joints + observed cube pose). Default ON; pass "
             "--no-show-sim for headless / CI runs."
    )
    # ── Benchmark mode. The default
    # invocation IS a benchmark (20 trials); pass --no-benchmark to skip
    # and run in normal duration-bounded mode. When benchmark is active it
    # forces --goal-mode=auto, overrides success-threshold/hold/timeout
    # with benchmark defaults (unless the user explicitly overrode them),
    # and writes a JSON trial summary on exit. ──
    parser.add_argument(
        "--benchmark-trials", type=int, default=20,
        help="Run N benchmark trials and write a JSON summary. Default 20. "
             "Pass --no-benchmark to disable benchmark mode and run in the "
             "normal duration-bounded path instead.",
    )
    parser.add_argument(
        "--no-benchmark", dest="no_benchmark", action="store_true",
        help="Skip benchmark mode; run for --duration with the goal-mode "
             "specified by --goal-mode (default 'external').",
    )
    parser.add_argument(
        "--benchmark-timeout", type=float, default=BENCHMARK_DEFAULT_TIMEOUT,
        help=f"per-trial timeout in seconds (default: {BENCHMARK_DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--benchmark-threshold", type=float, default=BENCHMARK_DEFAULT_THRESHOLD,
        help=f"benchmark success angle threshold in radians "
             f"(default: {BENCHMARK_DEFAULT_THRESHOLD}, ~20°)",
    )
    parser.add_argument(
        "--benchmark-hold-sec", type=float, default=BENCHMARK_DEFAULT_HOLD_SEC,
        help=f"cumulative seconds near goal in benchmark mode "
             f"(default: {BENCHMARK_DEFAULT_HOLD_SEC})",
    )
    parser.add_argument(
        "--benchmark-hard", action="store_true",
        help="Use uniform-SO(3) benchmark goals instead of the default "
             "face-aligned sampling.",
    )
    parser.add_argument(
        "--benchmark-output", type=str, default=None,
        help="JSON output path (default: logs/benchmark_<timestamp>.json)",
    )
    args = parser.parse_args()

    if args.ckpt is None:
        print(
            "ERROR: no --ckpt provided and DEFAULT_CKPT could not be resolved "
            "(WUJI_LOGS_ROOT unset and no logs/ ancestor found)."
        )
        return 1
    if not Path(args.ckpt).exists():
        print(f"ERROR: ckpt not found: {args.ckpt}")
        return 1

    # --no-benchmark sentinel: nullify benchmark_trials so downstream code
    # (which uses ``args.benchmark_trials is not None`` as the "in benchmark"
    # signal) falls through to the duration-bounded path.
    if getattr(args, "no_benchmark", False):
        args.benchmark_trials = None

    # Benchmark mode: force --goal-mode=auto (warn if user passed something
    # else), then bend the success threshold + hold + per-goal timeout to the
    # benchmark-* defaults *unless* the user explicitly overrode them.
    if args.benchmark_trials is not None:
        if args.goal_mode != "auto":
            print(
                f"[benchmark] WARNING: --goal-mode={args.goal_mode!r} ignored "
                "(benchmark mode requires goal switching); forcing auto."
            )
            args.goal_mode = "auto"
        # Only override when the user is on the play_real defaults; if
        # they passed --success-threshold/--auto-hold-sec/--auto-timeout-sec
        # explicitly, respect their values.
        if args.success_threshold == 0.2:
            args.success_threshold = args.benchmark_threshold
        if args.auto_hold_sec == 0.5:
            args.auto_hold_sec = args.benchmark_hold_sec
        if args.auto_timeout_sec == 14.0:
            args.auto_timeout_sec = args.benchmark_timeout
        # First-line sanity check on the constants: stuck timeout must safely
        # exceed the max hold wall-clock, else a legitimate success hold would
        # be misclassified.  The in-loop guard ``hold_counter > 0`` is the
        # second line of defense (suppresses stuck-during-hold no matter what).
        min_stuck_timeout = args.auto_hold_sec + 0.5
        if STUCK_TIMEOUT_SEC <= min_stuck_timeout:
            print(
                f"[benchmark] WARNING: STUCK_TIMEOUT_SEC={STUCK_TIMEOUT_SEC}s "
                f"not safely greater than hold_sec={args.auto_hold_sec:.2f}s. "
                f"Raise STUCK_TIMEOUT_SEC above {min_stuck_timeout:.1f}s."
            )
        sampler_desc = "uniform-SO(3)" if args.benchmark_hard else "face-aligned"
        print(
            f"[benchmark] trials={args.benchmark_trials}  "
            f"threshold={args.success_threshold:.3f}rad "
            f"({np.rad2deg(args.success_threshold):.1f}°)  "
            f"hold_sec={args.auto_hold_sec:.2f}s  "
            f"timeout={args.auto_timeout_sec:.1f}s  "
            f"sampler={sampler_desc}  "
            f"stuck_timeout={STUCK_TIMEOUT_SEC:.1f}s "
            f"stuck_thresh={STUCK_JOINT_THRESHOLD:.2f}rad"
        )

    # Build goal source up-front so an arg error fails before hardware init.
    goal_obj, random_driver, external_recv, goal_desc = _build_goal_source(args)

    # Cube source — CubeReceiver.latest() returns (pos, quat_wxyz) directly.
    cube_recv: CubeReceiver | None = None
    if not args.no_cube_zmq:
        cube_recv = CubeReceiver(port=5555)
        cube_desc = "CubeReceiver(ZMQ:5555)"
    else:
        cube_desc = "DISABLED (--no-cube-zmq, cube=zeros)"

    print("=" * 60)
    print(f"play_real: {args.duration:.1f}s @ {1.0/args.ctrl_dt:.0f}Hz "
          f"with {Path(args.ckpt).parent.name}")
    print(f"  goal:  {goal_desc}")
    print(f"  cube:  {cube_desc}")
    effort_disp = (
        f"{args.effort_limit} Nm" if args.effort_limit is not None
        else "control.yaml default"
    )
    print(f"  effort_limit={effort_disp}  "
          f"success<{args.success_threshold:.2f}rad")
    print("=" * 60)

    # Load policy FIRST so we can build the env from its sidecar JSON
    # (model-intrinsic params: action_scale, ema_alpha, warmup_time_s,
    # ctrl_dt, history_len, control_mode).
    policy = ONNXPolicy(args.ckpt)
    print(f"  policy config: {policy.config}")
    cfg = make_real_hand_env_cfg(
        robot_variant="wuji_hand",
        policy_config=policy.config,
    )
    cfg.observations["policy"].enable_corruption = False  # determinism

    log_records: list[dict] = []
    rc = 0
    sim_viewer = None

    try:
        with WujiHandDriver(
            effort_limit=args.effort_limit,
            lowpass_cutoff=args.lowpass_cutoff,
        ) as drv:
            env = RealHandEnv(cfg=cfg, hand_driver=drv)
            # RealHandEnv exposes _cube_zmq / _goal_zmq as post-construction
            # attrs (see real_hand_env.py). Set them now so step() picks up
            # the wiring before reset/step. Each receiver/stub exposes
            # .latest() with the (pos, quat) / quat ABI step() expects.
            if cube_recv is not None:
                env._cube_zmq = cube_recv
            env._goal_zmq = goal_obj

            try:
                expected_action_dim = env.action_manager.total_action_dim
                if policy.action_dim != expected_action_dim:
                    raise RuntimeError(
                        "Policy action dimension does not match the deployment "
                        f"environment: policy={policy.action_dim}, "
                        f"env={expected_action_dim}"
                    )
                # Belt-and-suspenders sanity check — env was already built
                # from policy.config, so this will trivially pass. Kept to
                # catch surprise mutations to cfg between build + validate.
                policy.validate_against_env(env)

                print("\n[1] Reset (homing real hand)...")
                obs, _ = env.reset()
                print(f"    obs shape: {obs['policy'].shape}")
                expected_obs_dim = int(obs["policy"].shape[-1])
                if policy.input_dim != expected_obs_dim:
                    raise RuntimeError(
                        "Policy observation dimension does not match the "
                        f"deployment environment: policy={policy.input_dim}, "
                        f"env={expected_obs_dim}"
                    )

                # ── Optional digital-twin viewer ──
                # Open after reset so hand qpos and the FK-refreshed mj_data
                # are populated before the first render. Pass the env's
                # private mj_data (refreshed each step via _fast_forward),
                # NOT env.sim.mj_data — that stays at the compile-time
                # default and would render a frozen pose. See _viz_mj_data.
                if args.show_sim:
                    # Disable shadows + MSAA (the only render knobs reachable via passive viewer).
                    env.sim.mj_model.vis.quality.shadowsize = 0
                    env.sim.mj_model.vis.quality.offsamples = 0
                    sim_viewer = mujoco.viewer.launch_passive(
                        env.sim.mj_model,
                        _viz_mj_data(env),
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                    print("    [show-sim] mujoco passive viewer launched (lightweight render)")

                n_steps_planned = int(args.duration / args.ctrl_dt)
                is_benchmark = args.benchmark_trials is not None
                if is_benchmark:
                    print(
                        f"\n[2] Benchmark: running {args.benchmark_trials} "
                        f"trials (per-trial timeout {args.auto_timeout_sec:.1f}s; "
                        f"stuck triggers recovery)..."
                    )
                else:
                    print(f"\n[2] Running ~{n_steps_planned} steps "
                          f"(duration={args.duration:.1f}s)...")

                geodesics: list[float] = []
                successes: list[bool] = []
                # Auto-mode bookkeeping (only used when goal_obj is _AutoGoalDriver).
                auto_achievement_count = 0
                auto_timeout_count = 0

                # ── Rich diagnostics: AsyncPrinter consumes 6-line snapshots
                # on a daemon thread so its stdout cost stays off the hot
                # control loop. Gated to one snapshot per second. ──
                ansi_enabled = sys.stdout.isatty()
                if ansi_enabled:
                    ood_printer = AsyncPrinter(formatter=format_ood_lines)
                else:
                    # Non-tty (log file / pipe): strip ANSI before printing
                    # so the captured output stays grep-friendly.
                    import re as _re
                    _ansi_re = _re.compile(r"\033\[[0-9;]*m")
                    def _plain_formatter(snap):
                        return [_ansi_re.sub("", ln) for ln in format_ood_lines(snap)]
                    ood_printer = AsyncPrinter(formatter=_plain_formatter)
                ood_printer.start()
                # Derive cube_ref_pos in tag frame from default spawn; zero would falsely flag in-palm cubes as OOD.
                from deploy.reorient.lib.frame_transform import (
                    quat_apply, quat_conjugate, quat_mul,
                )
                from wuji_mjlab.tasks.reorient.reorient_constants import (
                    TAG_IN_PALM_POS, TAG_IN_PALM_QUAT_WXYZ,
                )
                _mj_data_ref = _viz_mj_data(env)
                _cube_body_id = mujoco.mj_name2id(
                    env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object/cube"
                )
                _palm_body_id = mujoco.mj_name2id(
                    env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_palm_link"
                )
                if _cube_body_id >= 0 and _palm_body_id >= 0:
                    _cube_pos_w = np.asarray(
                        _mj_data_ref.xpos[_cube_body_id], dtype=np.float64
                    )
                    _palm_pos_w = np.asarray(
                        _mj_data_ref.xpos[_palm_body_id], dtype=np.float64
                    )
                    _palm_quat_w = np.asarray(
                        _mj_data_ref.xquat[_palm_body_id], dtype=np.float64
                    )
                    _tag_pos_w = _palm_pos_w + quat_apply(
                        _palm_quat_w, np.asarray(TAG_IN_PALM_POS, dtype=np.float64)
                    )
                    _tag_quat_w = quat_mul(
                        _palm_quat_w,
                        np.asarray(TAG_IN_PALM_QUAT_WXYZ, dtype=np.float64),
                    )
                    # cube_pos_in_tag = R(tag_quat).T @ (cube_pos_w - tag_pos_w)
                    cube_ref_pos = quat_apply(
                        quat_conjugate(_tag_quat_w),
                        _cube_pos_w - _tag_pos_w,
                    )
                else:
                    cube_ref_pos = np.zeros(3, dtype=np.float64)
                print(
                    f"    [diag] cube_ref_pos (tag frame): "
                    f"({cube_ref_pos[0]:+.4f}, {cube_ref_pos[1]:+.4f}, {cube_ref_pos[2]:+.4f})"
                )
                hold_counter = 0
                completed_goals = 0
                hold_steps_target = max(
                    1, int(round(args.auto_hold_sec / args.ctrl_dt))
                )
                last_print_time = 0.0
                step_count = 0

                # Benchmark / stuck-detection state.
                benchmark_results: list[dict] = []
                bench_min_ori_error = float("inf")
                trial_start_t: float | None = None
                stuck_ref_positions: np.ndarray | None = None
                stuck_ref_time: float | None = None

                t_start = time.perf_counter()
                step_idx = 0
                try:
                    while True:
                        elapsed = time.perf_counter() - t_start
                        if is_benchmark:
                            if len(benchmark_results) >= args.benchmark_trials:
                                break
                        else:
                            if elapsed >= args.duration:
                                break

                        obs_vec = obs["policy"][0].cpu().numpy()
                        action_np = policy(obs_vec)

                        action = torch.from_numpy(action_np).float().unsqueeze(0)
                        action = torch.clamp(action, -1.0, 1.0)
                        obs, *_ = env.step(action)

                        # Pull latest tracked state for monitoring/logging.
                        if cube_recv is not None:
                            cube_pos, cube_quat = cube_recv.latest()
                        else:
                            cube_pos = np.zeros(3, dtype=np.float64)
                            cube_quat = np.array([1.0, 0.0, 0.0, 0.0])
                        goal_quat = goal_obj.latest()

                        # ── Digital-twin viz: inject observed cube into the
                        # viewer's mj_data (env.step skips this write since
                        # the policy reads cube_pos from ZMQ directly), draw
                        # the goal overlay 10cm above it, and sync the
                        # viewer. Sub-ms in practice. ──
                        if args.show_sim and sim_viewer is not None:
                            cube_pos_mjworld = None
                            if cube_recv is not None:
                                cube_pos_mjworld = _write_cube_to_sim_for_viz(
                                    env, cube_pos, cube_quat
                                )
                            if sim_viewer.is_running():
                                if cube_pos_mjworld is not None:
                                    cube_size_m = (
                                        getattr(cube_recv, "cube_size", None)
                                        or 0.054
                                    )
                                    _draw_goal_overlay(
                                        sim_viewer,
                                        env,
                                        cube_pos_mjworld,
                                        goal_quat_tag=goal_quat,
                                        cube_size_m=cube_size_m,
                                    )
                                sim_viewer.sync()
                        geo = quat_geodesic(cube_quat, goal_quat)
                        ok = geo < args.success_threshold
                        geodesics.append(geo)
                        successes.append(ok)
                        step_count += 1

                        # Hold counter: cumulative in benchmark mode, consecutive
                        # otherwise. _AutoGoalDriver maintains its own internal
                        # counter; we maintain a parallel one for diagnostic
                        # display + benchmark trial accounting.
                        if geo < args.success_threshold:
                            hold_counter += 1
                        elif not is_benchmark:
                            hold_counter = 0
                        if geo < bench_min_ori_error:
                            bench_min_ori_error = float(geo)

                        # Benchmark per-trial timer (lazy-armed on first valid tick).
                        if is_benchmark and trial_start_t is None:
                            trial_start_t = time.perf_counter()
                        benchmark_timed_out = (
                            is_benchmark
                            and trial_start_t is not None
                            and (time.perf_counter() - trial_start_t)
                                > args.auto_timeout_sec
                        )

                        # ── Stuck detection (auto + benchmark goal modes
                        # only; quietly skipped for fixed/external/random).
                        # Uses encoder reads to detect "joints not moving"
                        # over STUCK_TIMEOUT_SEC.  Hold-phase suppression:
                        # when hold_counter > 0 we re-baseline so a
                        # legitimate success hold cannot fire the stuck
                        # timer.  Without this guard cumulative-hold +
                        # stuck-window collide on ~2.5s scales. ──
                        benchmark_stuck = False
                        if isinstance(goal_obj, _AutoGoalDriver):
                            now_mono = time.perf_counter()
                            try:
                                actual_qpos = drv.read_encoders()
                            except Exception:
                                actual_qpos = None
                            if actual_qpos is not None:
                                if stuck_ref_positions is None:
                                    stuck_ref_positions = actual_qpos.copy()
                                    stuck_ref_time = now_mono
                                elif hold_counter > 0:
                                    # Holding success pose: re-baseline so
                                    # the stuck timer cannot fire.
                                    stuck_ref_positions = actual_qpos.copy()
                                    stuck_ref_time = now_mono
                                else:
                                    joint_movement = float(
                                        np.max(np.abs(actual_qpos - stuck_ref_positions))
                                    )
                                    if joint_movement > STUCK_JOINT_THRESHOLD:
                                        stuck_ref_positions = actual_qpos.copy()
                                        stuck_ref_time = now_mono
                                    elif (
                                        stuck_ref_time is not None
                                        and now_mono - stuck_ref_time
                                        > STUCK_TIMEOUT_SEC
                                    ):
                                        benchmark_stuck = True

                        # Auto-mode: advance achievement/timeout state machine.
                        # Driver returns a switch event dict on the step that
                        # flipped the goal; we log it and bump the right
                        # counter for the exit summary.
                        achievement_event = None
                        if isinstance(goal_obj, _AutoGoalDriver):
                            achievement_event = goal_obj.tick(
                                cube_quat, geo, time.monotonic()
                            )

                        # ── Goal-transition handling.  Three outcomes:
                        # achievement (success), timeout (no hold in
                        # auto_timeout_sec), or stuck (no joint movement).
                        # Each emits a multi-line "GOAL …" message and,
                        # in benchmark mode, appends a trial record. ──
                        if achievement_event is not None:
                            prev_goal = goal_quat.copy()
                            new_goal = goal_obj.latest()  # updated by tick()
                            if achievement_event["reason"] == "achievement":
                                auto_achievement_count += 1
                                completed_goals += 1
                                print(
                                    f"\n  >>> GOAL REACHED #{completed_goals}: "
                                    f"err={np.rad2deg(geo):.1f}° for "
                                    f"{hold_counter} steps"
                                )
                            else:  # timeout
                                auto_timeout_count += 1
                                print(
                                    f"\n  >>> GOAL TIMEOUT #{achievement_event['goal_idx']} "
                                    f"(no hold for {achievement_event['duration_sec']:.1f}s): "
                                    f"err={np.rad2deg(geo):.1f}°"
                                )
                            print(
                                f"      old=[{prev_goal[0]:+.3f}, {prev_goal[1]:+.3f}, "
                                f"{prev_goal[2]:+.3f}, {prev_goal[3]:+.3f}]"
                            )
                            print(
                                f"      new=[{new_goal[0]:+.3f}, {new_goal[1]:+.3f}, "
                                f"{new_goal[2]:+.3f}, {new_goal[3]:+.3f}]"
                            )
                            if is_benchmark:
                                trial_dur = time.perf_counter() - (
                                    trial_start_t or time.perf_counter()
                                )
                                outcome = (
                                    "success"
                                    if achievement_event["reason"] == "achievement"
                                    else "timeout"
                                )
                                benchmark_results.append({
                                    "success": outcome == "success",
                                    "outcome": outcome,
                                    "min_ori_error_rad": float(bench_min_ori_error),
                                    "final_ori_error_rad": float(geo),
                                    "duration_sec": float(trial_dur),
                                })
                                idx = len(benchmark_results)
                                print(
                                    f"  [benchmark] trial {idx}/{args.benchmark_trials} "
                                    f"{outcome.upper()}  "
                                    f"min_err={np.rad2deg(bench_min_ori_error):.1f}°  "
                                    f"dur={trial_dur:.1f}s"
                                )
                                trial_start_t = None
                                bench_min_ori_error = float("inf")
                                stuck_ref_positions = None
                            hold_counter = 0
                            goal_quat = new_goal

                        elif benchmark_timed_out:
                            # Force a switch (auto-driver hasn't fired its own
                            # timeout because cumulative-hold + non-arming
                            # corner cases may differ — we drive it directly).
                            prev_goal = goal_quat.copy()
                            trial_dur = time.perf_counter() - trial_start_t
                            trial_final_err = float(geo)
                            trial_min_err = float(bench_min_ori_error)
                            benchmark_results.append({
                                "success": False,
                                "outcome": "timeout",
                                "min_ori_error_rad": trial_min_err,
                                "final_ori_error_rad": trial_final_err,
                                "duration_sec": float(trial_dur),
                            })
                            idx = len(benchmark_results)
                            auto_timeout_count += 1
                            print(
                                f"\n  >>> GOAL TIMEOUT #{idx} "
                                f"(no hold for {trial_dur:.1f}s): "
                                f"err={np.rad2deg(trial_final_err):.1f}° "
                                f"(min={np.rad2deg(trial_min_err):.1f}°)"
                            )
                            if idx < args.benchmark_trials:
                                goal_obj.force_switch(cube_quat)
                                new_goal = goal_obj.latest()
                                print(
                                    f"      old=[{prev_goal[0]:+.3f}, {prev_goal[1]:+.3f}, "
                                    f"{prev_goal[2]:+.3f}, {prev_goal[3]:+.3f}]"
                                )
                                print(
                                    f"      new=[{new_goal[0]:+.3f}, {new_goal[1]:+.3f}, "
                                    f"{new_goal[2]:+.3f}, {new_goal[3]:+.3f}]"
                                )
                                goal_quat = new_goal
                            trial_start_t = None
                            bench_min_ori_error = float("inf")
                            stuck_ref_positions = None
                            hold_counter = 0

                        elif benchmark_stuck and is_benchmark:
                            prev_goal = goal_quat.copy()
                            trial_dur = time.perf_counter() - (
                                trial_start_t or time.perf_counter()
                            )
                            trial_final_err = float(geo)
                            trial_min_err = float(bench_min_ori_error)
                            benchmark_results.append({
                                "success": False,
                                "outcome": "stuck_reset",
                                "min_ori_error_rad": trial_min_err,
                                "final_ori_error_rad": trial_final_err,
                                "duration_sec": float(trial_dur),
                            })
                            idx = len(benchmark_results)
                            print(
                                f"\n  >>> GOAL STUCK #{idx} "
                                f"(no movement for {STUCK_TIMEOUT_SEC:.1f}s, "
                                f"total trial {trial_dur:.1f}s): "
                                f"err={np.rad2deg(trial_final_err):.1f}° "
                                f"(min={np.rad2deg(trial_min_err):.1f}°)"
                            )
                            if idx < args.benchmark_trials:
                                print(f"  >>> GOAL STUCK #{idx} — running recovery sequence")
                                _run_recovery_sequence(env, drv, RECOVERY_DURATION_SEC)
                                # Recovery blocks ~2 * RECOVERY_DURATION_SEC s;
                                # reset t_start so the overrun check (if any)
                                # doesn't misfire on the recovery move itself.
                                t_start = time.perf_counter()
                                goal_obj.force_switch(cube_quat)
                                new_goal = goal_obj.latest()
                                print(
                                    f"      old=[{prev_goal[0]:+.3f}, {prev_goal[1]:+.3f}, "
                                    f"{prev_goal[2]:+.3f}, {prev_goal[3]:+.3f}]"
                                )
                                print(
                                    f"      new=[{new_goal[0]:+.3f}, {new_goal[1]:+.3f}, "
                                    f"{new_goal[2]:+.3f}, {new_goal[3]:+.3f}]"
                                )
                                goal_quat = new_goal
                            trial_start_t = None
                            bench_min_ori_error = float("inf")
                            stuck_ref_positions = None
                            hold_counter = 0

                        elif benchmark_stuck and not is_benchmark:
                            # Auto-only stuck (no benchmark trial tracking):
                            # log + recover + sample next goal, then keep
                            # cycling.
                            prev_goal = goal_quat.copy()
                            print(
                                f"\n  >>> GOAL STUCK "
                                f"(no movement for {STUCK_TIMEOUT_SEC:.1f}s): "
                                f"err={np.rad2deg(geo):.1f}° — running recovery sequence"
                            )
                            _run_recovery_sequence(env, drv, RECOVERY_DURATION_SEC)
                            t_start = time.perf_counter()
                            goal_obj.force_switch(cube_quat)
                            new_goal = goal_obj.latest()
                            print(
                                f"      old=[{prev_goal[0]:+.3f}, {prev_goal[1]:+.3f}, "
                                f"{prev_goal[2]:+.3f}, {prev_goal[3]:+.3f}]"
                            )
                            print(
                                f"      new=[{new_goal[0]:+.3f}, {new_goal[1]:+.3f}, "
                                f"{new_goal[2]:+.3f}, {new_goal[3]:+.3f}]"
                            )
                            goal_quat = new_goal
                            stuck_ref_positions = None
                            hold_counter = 0

                        if args.log_file is not None:
                            enc = drv.read_encoders()
                            log_records.append({
                                "t": elapsed,
                                "action": action_np.copy(),
                                "encoder": enc.copy(),
                                "cube_quat": cube_quat.copy(),
                                "goal_quat": goal_quat.copy(),
                                "geodesic": float(geo),
                                "success": bool(ok),
                            })

                        # Per-second 6-line OOD diagnostics print.  Off the
                        # hot path via AsyncPrinter — drops if the worker
                        # falls behind.  ANSI colors gated via the
                        # formatter swap above so log files stay clean.
                        now_wall = time.time()
                        if now_wall - last_print_time > 1.0:
                            try:
                                actual_qpos = drv.read_encoders()
                            except Exception:
                                actual_qpos = np.zeros(20, dtype=np.float64)
                            joint_error_rad = float(
                                np.abs(actual_qpos - action_np).max()
                                if actual_qpos.shape == action_np.shape
                                else 0.0
                            )
                            ood_printer.submit({
                                "step_count": step_count,
                                "ori_error_rad": float(geo),
                                "success_threshold_rad": float(args.success_threshold),
                                "joint_error_rad": joint_error_rad,
                                "hold_counter": int(hold_counter),
                                "hold_steps": int(hold_steps_target),
                                "completed_goals": int(completed_goals),
                                "cube_pos_error": cube_ref_pos - cube_pos,
                                "cube_ori_error_6d": _compute_ori_error_6d(
                                    cube_quat, goal_quat
                                ),
                                "cube_pos": cube_pos,
                                "cube_ref_pos": cube_ref_pos,
                                "cube_quat": cube_quat,
                                "goal_quat": goal_quat,
                                "action": action_np,
                            })
                            last_print_time = now_wall

                        step_idx += 1
                except KeyboardInterrupt:
                    print("\n  Ctrl+C — aborting loop")
                finally:
                    ood_printer.stop()

                elapsed = time.perf_counter() - t_start
                n_actual = len(geodesics)
                if n_actual == 0:
                    print("\n[3] No steps recorded.")
                else:
                    geo_arr = np.array(geodesics)
                    n_succ = int(sum(successes))
                    print(f"\n[3] Final summary "
                          f"({n_actual} steps in {elapsed:.2f}s, "
                          f"{n_actual / max(elapsed, 1e-9):.1f} Hz):")
                    print(f"    geodesic mean: {geo_arr.mean():.3f} rad "
                          f"({np.rad2deg(geo_arr.mean()):.1f}°)")
                    print(f"    geodesic peak: {geo_arr.max():.3f} rad "
                          f"({np.rad2deg(geo_arr.max()):.1f}°)")
                    # time<thr fraction = "% of loop time near the goal."
                    # Not the policy's success rate; biased by goal
                    # difficulty + dwell.  The meaningful policy metric is
                    # the auto-mode achievement rate below.
                    print(
                        f"    time<thr "
                        f"(geo < {args.success_threshold:.2f} rad): "
                        f"{n_succ} / {n_actual} steps "
                        f"({100.0 * n_succ / n_actual:.1f}% of loop time)"
                    )
                    if isinstance(goal_obj, _AutoGoalDriver):
                        total_switches = (
                            auto_achievement_count + auto_timeout_count
                        )
                        if total_switches > 0:
                            achievement_rate = (
                                100.0 * auto_achievement_count / total_switches
                            )
                            achievement_rate_str = f"{achievement_rate:.1f}%"
                        else:
                            achievement_rate_str = "n/a"
                        print(
                            f"    auto: {total_switches} goal switches "
                            f"({auto_achievement_count} achieved, "
                            f"{auto_timeout_count} timed out); "
                            f"achievement rate "
                            f"{auto_achievement_count}/"
                            f"({auto_achievement_count}+{auto_timeout_count}) "
                            f"= {achievement_rate_str}"
                        )

                # Benchmark mode: print success rate + dump JSON. Safe no-op
                # when not in benchmark mode.  Wrap in try/except so a JSON
                # write failure cannot block env.close() / driver teardown.
                if is_benchmark:
                    try:
                        n = len(benchmark_results)
                        succ = sum(1 for r in benchmark_results if r["success"])
                        rate = (succ / n) if n > 0 else 0.0
                        print()
                        print(f"[benchmark] {succ}/{n} success = {rate * 100:.1f}%")
                        if args.benchmark_output is not None:
                            out_path = Path(args.benchmark_output)
                        else:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            try:
                                logs_root = find_logs_root()
                            except FileNotFoundError:
                                logs_root = Path.cwd() / "logs"
                            out_path = logs_root / f"benchmark_{ts}.json"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        payload = {
                            "success_rate": rate,
                            "num_trials": args.benchmark_trials,
                            "num_completed": n,
                            "trials": benchmark_results,
                        }
                        with out_path.open("w") as f:
                            json.dump(payload, f, indent=2)
                        print(f"[benchmark] wrote {out_path}")
                    except Exception as exc:
                        print(f"[benchmark] ERROR writing summary: {exc}")

                if args.log_file is not None and log_records:
                    out = Path(args.log_file)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        out,
                        t=np.array([r["t"] for r in log_records]),
                        action=np.stack([r["action"] for r in log_records]),
                        encoder=np.stack([r["encoder"] for r in log_records]),
                        cube_quat=np.stack([r["cube_quat"] for r in log_records]),
                        goal_quat=np.stack([r["goal_quat"] for r in log_records]),
                        geodesic=np.array([r["geodesic"] for r in log_records]),
                        success=np.array([r["success"] for r in log_records]),
                    )
                    print(f"\n    log written to: {out}")
            finally:
                env.close()
    finally:
        # Tear down viewer first so its background thread stops touching
        # mj_data before we release downstream resources.
        if sim_viewer is not None:
            try:
                sim_viewer.close()
            except Exception:
                pass
        # Tear down ZMQ resources regardless of how we exited.
        if random_driver is not None:
            random_driver.stop()
        if external_recv is not None:
            external_recv.close()
        if cube_recv is not None:
            cube_recv.close()

    print("\nplay_real complete; joints disabled.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
