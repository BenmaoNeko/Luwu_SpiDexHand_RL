#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Static calibration check for cube pose vs. hand pose.

What this does:
  1. Initialises ``WujiHandDriver`` and rams the hand to REORIENT home pose.
  2. Opens a MuJoCo passive viewer showing the digital twin (hand + cube).
  3. Loops at ~20 Hz reading joint encoders + cube pose over ZMQ, redrawing
     the viewer each tick.

Use this to eyeball whether the ArUco-based cube pose estimate (anchored
to the wrist AprilTag world frame) matches the physical cube position.
The hand stays in home after step 1 — there is no policy and no control
beyond the initial homing ramp.

Run:
    pixi run -e deploy vision &                                # publisher
    pixi run -e deploy python deploy/reorient/tools/calib_check.py

Press Ctrl+C or close the viewer window to exit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow ``deploy.reorient.*`` imports when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mujoco
import mujoco.viewer
import numpy as np
import torch

from deploy.reorient.lib.hand_driver import WujiHandDriver
from deploy.reorient.lib.real_hand_env import RealHandEnv
from deploy.reorient.lib.real_hand_env_cfg import make_real_hand_env_cfg
from deploy.reorient.lib.zmq_bridge import CubeReceiver

# ────────────────── viz helpers (copied from play_real.py) ──────────────────


def _viz_mj_data(env: RealHandEnv):
  """The private MjData buffer that ``_fast_forward`` keeps in sync.

  ``env.sim.mj_data`` is a separate host-side template that does NOT track
  per-step state — passing it to the viewer would render a frozen pose.
  """
  return env._mj_data if hasattr(env, "_mj_data") else env.sim.mj_data


def _write_cube_to_sim_for_viz(env: RealHandEnv, cube_pos_tag, cube_quat_tag_wxyz):
  """Inject the observed cube pose into mj_data so the viewer can render it.

  Tag-frame coords are anchored to the wrist AprilTag; lift into mjworld by
  composing palm pose * TAG_IN_PALM_* offsets. VIZ-ONLY — the policy obs
  pipeline bypasses scene["object"] entirely.
  """
  from wuji_mjlab.tasks.reorient.reorient_constants import (
    TAG_IN_PALM_POS,
    TAG_IN_PALM_QUAT_WXYZ,
  )

  from deploy.reorient.lib.frame_transform import quat_apply, quat_mul

  mj_model = env.sim.mj_model
  mj_data = _viz_mj_data(env)
  cube_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "object/cube")
  if cube_body_id < 0:
    return
  joint_id = int(mj_model.body_jntadr[cube_body_id])
  if joint_id < 0:
    return
  qpos_adr = int(mj_model.jnt_qposadr[joint_id])

  palm_body_id = mujoco.mj_name2id(
    mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_palm_link"
  )
  if palm_body_id < 0:
    return

  palm_pos_w = np.asarray(mj_data.xpos[palm_body_id], dtype=np.float64)
  palm_quat_w = np.asarray(mj_data.xquat[palm_body_id], dtype=np.float64)
  tip_pos = np.asarray(TAG_IN_PALM_POS, dtype=np.float64)
  tip_quat = np.asarray(TAG_IN_PALM_QUAT_WXYZ, dtype=np.float64)
  tag_pos_w = palm_pos_w + quat_apply(palm_quat_w, tip_pos)
  tag_quat_w = quat_mul(palm_quat_w, tip_quat)
  cube_pos_w = tag_pos_w + quat_apply(
    tag_quat_w, np.asarray(cube_pos_tag, dtype=np.float64)
  )
  cube_quat_w = quat_mul(tag_quat_w, np.asarray(cube_quat_tag_wxyz, dtype=np.float64))

  mj_data.qpos[qpos_adr : qpos_adr + 3] = cube_pos_w
  mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = cube_quat_w
  # Recompute derived fields (xpos/xquat) — the viewer renders from those,
  # not from qpos. Without this the cube would stay frozen at whatever
  # mj_data state ``_fast_forward()`` left behind.
  mujoco.mj_forward(mj_model, mj_data)


# ────────────────── main ──────────────────


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    "--cube-port", type=int, default=5555, help="ZMQ port for cube poses"
  )
  parser.add_argument(
    "--no-cube-zmq",
    action="store_true",
    help="skip CubeReceiver — render only the hand (useful when "
    "diagnosing publisher issues separately).",
  )
  parser.add_argument(
    "--rate-hz", type=float, default=20.0, help="viewer refresh rate (default 20)"
  )
  parser.add_argument(
    "--effort-limit",
    type=float,
    default=None,
    help="per-joint Nm (default: control.yaml hardware.effort_limit_nm)",
  )
  parser.add_argument(
    "--lowpass-cutoff",
    type=float,
    default=None,
    help="hand LowPass cutoff Hz (default: control.yaml hardware.lowpass_cutoff_hz)",
  )
  parser.add_argument(
    "--show-sim",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="open the MuJoCo passive viewer (default ON; pass --no-show-sim for "
    "a tty-only check, useful in CI/headless).",
  )
  args = parser.parse_args()

  dt = 1.0 / max(args.rate_hz, 1.0)

  cube_recv: CubeReceiver | None = None
  if not args.no_cube_zmq:
    cube_recv = CubeReceiver(port=args.cube_port)
    print(f"[calib-check] CubeReceiver listening on ZMQ:{args.cube_port}")
  else:
    print("[calib-check] cube ZMQ disabled (--no-cube-zmq)")

  cfg = make_real_hand_env_cfg(robot_variant="wuji_hand")
  cfg.observations["policy"].enable_corruption = False

  with WujiHandDriver(
    effort_limit=args.effort_limit,
    lowpass_cutoff=args.lowpass_cutoff,
  ) as drv:
    env = RealHandEnv(cfg=cfg, hand_driver=drv)
    # Wire the cube receiver in case any obs term reads it during reset.
    if cube_recv is not None:
      env._cube_zmq = cube_recv

    print("[calib-check] homing hand ...")
    env.reset()
    print("[calib-check] homed.")

    sim_viewer = None
    if args.show_sim:
      env.sim.mj_model.vis.quality.shadowsize = 0
      env.sim.mj_model.vis.quality.offsamples = 0
      sim_viewer = mujoco.viewer.launch_passive(
        env.sim.mj_model,
        _viz_mj_data(env),
        show_left_ui=False,
        show_right_ui=False,
      )
      print(
        "[calib-check] viewer up — compare the rendered cube against the "
        "real cube. Ctrl+C or close the window to exit."
      )

    env_ids = torch.tensor([0], device=env.device)
    try:
      while True:
        if sim_viewer is not None and not sim_viewer.is_running():
          break

        encoder_qpos = drv.read_encoders()
        env.scene["robot"].write_joint_position_to_sim(
          torch.from_numpy(encoder_qpos).float().unsqueeze(0).to(env.device),
          env_ids=env_ids,
        )
        env._fast_forward()

        if cube_recv is not None:
          cube_pos, cube_quat = cube_recv.latest()
          _write_cube_to_sim_for_viz(env, cube_pos, cube_quat)

        if sim_viewer is not None:
          sim_viewer.sync()

        time.sleep(dt)
    except KeyboardInterrupt:
      print("\n[calib-check] interrupted by user.")
    finally:
      if sim_viewer is not None:
        sim_viewer.close()

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
