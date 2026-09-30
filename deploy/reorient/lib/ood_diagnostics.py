# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Out-of-distribution diagnostics for sim2real deployment.

Split into two pieces so the hot control loop never blocks on stdout:

- :func:`format_ood_lines` is a pure function that takes a numeric snapshot
  dict and returns the six formatted, ANSI-colored lines that would be printed.
  Pure → trivially testable.
- :class:`AsyncPrinter` runs a daemon thread consuming snapshots from a tiny
  queue, formatting them off the critical path, and emitting each snapshot as
  a single coalesced ``print`` call (one syscall instead of six).
"""
from __future__ import annotations

import queue
import sys
import threading
from typing import Any, Callable, TextIO

import numpy as np

_C_RST = "\033[0m"
_C_RED = "\033[91m"
_C_YEL = "\033[93m"
_C_GRN = "\033[92m"


def _pos_tag(pos_mag: float) -> tuple[str, str]:
    # Standard OOD thresholds for cube position drift (meters).
    POS_WARN, POS_CRIT = 0.05, 0.10
    if pos_mag > POS_CRIT:
        return _C_RED, "OOD!"
    if pos_mag > POS_WARN:
        return _C_YEL, "WARN"
    return _C_GRN, " OK "


def _ori_tag(cube_ori_error_6d: np.ndarray) -> tuple[str, str]:
    ori_max = float(np.max(np.abs(cube_ori_error_6d)))
    if ori_max > 1.05:
        return _C_RED, "OOD!"
    return _C_GRN, " OK "


def _action_tag(act_mag: float) -> tuple[str, str]:
    if act_mag < 0.01:
        return _C_RED, "DEAD"
    if act_mag < 0.05:
        return _C_YEL, "LOW "
    return _C_GRN, " OK "


def format_ood_lines(snap: dict[str, Any]) -> list[str]:
    """Format a single OOD diagnostic snapshot as six ANSI-colored lines.

    Expected snapshot keys (all numpy arrays or scalars):
        step_count, ori_error_rad, success_threshold_rad, joint_error_rad,
        hold_counter, hold_steps, completed_goals,
        cube_pos_error, cube_ori_error_6d, cube_pos, cube_ref_pos,
        cube_quat, goal_quat, action
    """
    cube_pos_error = np.asarray(snap["cube_pos_error"])
    cube_ori_error_6d = np.asarray(snap["cube_ori_error_6d"])
    cube_pos = np.asarray(snap["cube_pos"])
    cube_ref_pos = np.asarray(snap["cube_ref_pos"])
    cube_quat = np.asarray(snap["cube_quat"])
    goal_quat = np.asarray(snap["goal_quat"])
    action = np.asarray(snap["action"])

    pos_mag = float(np.linalg.norm(cube_pos_error))
    act_mag = float(np.mean(np.abs(action)))

    pos_c, pos_tag = _pos_tag(pos_mag)
    ori_c, ori_tag = _ori_tag(cube_ori_error_6d)
    act_c, act_tag = _action_tag(act_mag)

    thresh_deg = float(np.rad2deg(snap["success_threshold_rad"]))
    ori_deg = float(np.rad2deg(snap["ori_error_rad"]))
    below_thresh = snap["ori_error_rad"] < snap["success_threshold_rad"]
    thresh_mark = "✓<" if below_thresh else "✗>"

    return [
        f"\n[{int(snap['step_count']):5d}] ---- OOD Diagnostics ----",
        (
            f"  OrientErr: {ori_deg:6.2f}° {thresh_mark}thresh({thresh_deg:.2f}°) | "
            f"JointErr: {np.rad2deg(snap['joint_error_rad']):5.2f}° | "
            f"Hold: {int(snap['hold_counter'])}/{int(snap['hold_steps'])} | "
            f"GoalsDone: {int(snap['completed_goals'])}"
        ),
        (
            f"  CubePosErr  {pos_c}[{pos_tag}]{_C_RST} "
            f"xyz=[{cube_pos_error[0]:+.4f}, {cube_pos_error[1]:+.4f}, {cube_pos_error[2]:+.4f}] "
            f"mag={pos_mag:.4f}m  (train: <0.05m)"
        ),
        (
            f"  CubeOriErr  {ori_c}[{ori_tag}]{_C_RST} "
            f"[{cube_ori_error_6d[0]:+.3f}, {cube_ori_error_6d[1]:+.3f}, {cube_ori_error_6d[2]:+.3f}, "
            f"{cube_ori_error_6d[3]:+.3f}, {cube_ori_error_6d[4]:+.3f}, {cube_ori_error_6d[5]:+.3f}]"
        ),
        (
            f"  RawCubePos  "
            f"xyz=[{cube_pos[0]:+.4f}, {cube_pos[1]:+.4f}, {cube_pos[2]:+.4f}]  "
            f"RefPos=[{cube_ref_pos[0]:+.4f}, {cube_ref_pos[1]:+.4f}, {cube_ref_pos[2]:+.4f}]"
        ),
        (
            f"  RawCubeQuat "
            f"wxyz=[{cube_quat[0]:+.3f}, {cube_quat[1]:+.3f}, {cube_quat[2]:+.3f}, {cube_quat[3]:+.3f}]  "
            f"GoalQuat=[{goal_quat[0]:+.3f}, {goal_quat[1]:+.3f}, {goal_quat[2]:+.3f}, {goal_quat[3]:+.3f}]"
        ),
        (
            f"  Action      {act_c}[{act_tag}]{_C_RST} "
            f"mean|a|={act_mag:.4f}  "
            f"range=[{float(action.min()):+.3f}, {float(action.max()):+.3f}]"
        ),
    ]


class AsyncPrinter:
    """Consume snapshots on a daemon thread and print them off the hot path.

    The hot control loop calls :meth:`submit` which is a non-blocking
    ``put_nowait`` into a tiny queue (``maxsize=2``). When the queue is full
    (printer fell behind) the snapshot is silently dropped so the loop never
    blocks on stdout.
    """

    def __init__(
        self,
        formatter: Callable[[dict[str, Any]], list[str]],
        maxsize: int = 2,
        stream: TextIO | None = None,
    ) -> None:
        self._formatter = formatter
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=maxsize)
        self._stream = stream if stream is not None else sys.stdout
        self._thread: threading.Thread | None = None
        self._stopped = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="ood-printer")
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stopped = True
        try:
            self._queue.put_nowait(None)  # sentinel
        except queue.Full:
            # Sentinel dropped — fine. The worker will also notice
            # self._stopped on its next get() timeout (0.2s).
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, snapshot: dict[str, Any]) -> None:
        """Non-blocking. Drops if the queue is full."""
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stopped:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:  # sentinel
                break
            try:
                lines = self._formatter(item)
            except Exception as exc:  # formatter bugs must not kill the thread
                print(f"[ood-printer] formatter raised: {exc!r}", file=self._stream, flush=True)
                continue
            # Single syscall: one print with embedded newlines.
            print("\n".join(lines), file=self._stream, flush=True)
