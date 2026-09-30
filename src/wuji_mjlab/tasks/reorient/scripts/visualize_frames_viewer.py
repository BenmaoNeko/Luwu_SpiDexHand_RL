# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Interactive MuJoCo viewer showing palm-frame axes across wrist orientations.

Builds the ``WujiHand_Reorient`` env with a few parallel envs, overrides each
env's wrist (robot root) quaternion to a different orientation so the four
hands sit side-by-side in the scene, then opens the Viser viewer with a
debug-vis overlay that draws:

  * the world-frame axes at the world origin (RGB triad, length 0.20 m),
  * the **palm-frame axes** at every env's palm body (solid RGB triad,
    length 0.10 m),
  * a black world-frame **gravity** arrow from each palm pointing world -Z
    (length 0.15 m),
  * a tiny gray sphere at ``palm + 0.05 * (-Zp)`` showing where gravity sits
    in palm-local coordinates -- it overlays the gravity arrow tail only when
    palm-Z is parallel to world-Z.

Each env's wrist orientation is fixed at startup; a zero-action policy keeps
the simulation quiet. Spin / zoom the viewer to inspect 3D placement.

Usage::

  pixi run python -m wuji_mjlab.tasks.reorient.scripts.visualize_frames_viewer
  pixi run python -m wuji_mjlab.tasks.reorient.scripts.visualize_frames_viewer \\
      --num-envs 8 --viewer viser
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import wuji_mjlab.tasks  # noqa: F401  (registers WujiHand_Reorient)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_from_angle_axis,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from wuji_mjlab.tasks.reorient.mdp.observations import _palm_pose_to_tag_pose
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_CUBE_INIT_POS,
  REORIENT_ROBOT_ROOT_POS,
  REORIENT_ROBOT_ROOT_ROT,
)
from wuji_mjlab.utils.math import random_quat_uniform

TASK_ID = "WujiHand_Reorient"


@dataclass(frozen=True)
class WristCase:
  name: str
  quat_wxyz: tuple[float, float, float, float]


def _quat_axis_angle_wxyz(
  axis: tuple[float, float, float], deg: float
) -> tuple[float, float, float, float]:
  axis_t = torch.tensor([axis], dtype=torch.float32)
  axis_t = axis_t / axis_t.norm(dim=-1, keepdim=True)
  angle_t = torch.tensor([math.radians(deg)], dtype=torch.float32)
  q = quat_from_angle_axis(angle_t, axis_t).squeeze(0).tolist()
  return (q[0], q[1], q[2], q[3])


def build_wrist_cases(num_envs: int) -> list[WristCase]:
  """Return ``num_envs`` distinct wrist orientations covering the SO(3) sweep."""
  torch.manual_seed(0)
  random_q = random_quat_uniform(1, device="cpu").squeeze(0).tolist()
  random_quat = (random_q[0], random_q[1], random_q[2], random_q[3])

  pool: list[WristCase] = [
    WristCase("identity (Zp = Zw)", (1.0, 0.0, 0.0, 0.0)),
    WristCase(
      "+90 pitch (palm-up, Xp = -Zw)",
      _quat_axis_angle_wxyz((0.0, 1.0, 0.0), 90.0),
    ),
    WristCase(
      "+90 roll (Yp = +Zw)",
      _quat_axis_angle_wxyz((1.0, 0.0, 0.0), 90.0),
    ),
    WristCase(
      "+90 yaw (Xp = +Yw)",
      _quat_axis_angle_wxyz((0.0, 0.0, 1.0), 90.0),
    ),
    WristCase(
      "180 pitch (palm-down, Zp = -Zw)",
      _quat_axis_angle_wxyz((0.0, 1.0, 0.0), 180.0),
    ),
    WristCase(
      "+30 pitch",
      _quat_axis_angle_wxyz((0.0, 1.0, 0.0), 30.0),
    ),
    WristCase(
      "default reorient root rot",
      tuple(REORIENT_ROBOT_ROOT_ROT),  # type: ignore[arg-type]
    ),
    WristCase("random SO(3)", random_quat),
  ]
  if num_envs <= len(pool):
    return pool[:num_envs]
  # Fall back to repeating; not really useful but keep things robust.
  cases = list(pool)
  while len(cases) < num_envs:
    cases.append(pool[(len(cases)) % len(pool)])
  return cases


def make_palm_axes_visualizer(
  env: ManagerBasedRlEnv,
  case_names: list[str],
  base_update_visualizers: Callable[..., None] | None,
):
  """Return a function that draws per-env world frame, gravity, and palm frame.

  The returned callable matches ``ManagerBasedRlEnv.update_visualizers``: it
  receives a ``DebugVisualizer`` and draws the overlay. It also delegates to
  ``base_update_visualizers`` first so any task-specific debug viz (e.g. the
  goal ghost mesh) keeps working.

  At each env_origin we draw:
    * a world-frame RGB triad (length 0.20m, alpha 0.5),
    * a black gravity arrow pointing world -Z (length 0.15m),
    * a thin gray cylinder from env_origin straight up to that env's palm
      body (visualises the palm offset from the per-env world reference).

  At each palm body (per env) we draw:
    * a solid RGB triad (length 0.10m) — the wrist orientation.

  At each tag pose (per env, computed via the same rigid transform the obs
  functions use) we draw:
    * a CMY triad (cyan = X_tag, magenta = Y_tag, yellow = Z_tag,
      length 0.10m) — the frame in which policy obs are computed. Distinct
      from RGB so it cannot be confused with the world or palm triads.
  """
  scene = env.scene
  robot = scene["robot"]
  palm_ids, _ = robot.find_bodies(".*_palm_link")
  if not palm_ids:
    raise RuntimeError("could not find palm body via pattern '.*_palm_link'")
  palm_body_id = int(palm_ids[0])

  env_origins_np = scene.env_origins.detach().cpu().numpy().astype(np.float32)
  world_R = np.eye(3, dtype=np.float32)

  palm_axis_len = 0.10
  tag_axis_len = 0.10
  world_axis_len = 0.20
  gravity_len = 0.15

  # CMY triad colors for the tag frame (cyan / magenta / yellow), distinct from
  # the RGB triads used for world (dashed) and palm (solid).
  tag_axis_colors = ((0.0, 0.85, 0.85), (0.85, 0.0, 0.85), (0.9, 0.85, 0.0))

  def update(visualizer) -> None:
    if base_update_visualizers is not None:
      base_update_visualizers(visualizer)

    # Palm world poses for every env.
    palm_pose = robot.data.body_link_pose_w[:, palm_body_id, :]
    palm_pos_t = palm_pose[:, 0:3]
    palm_quat_t = palm_pose[:, 3:7]
    palm_pos = palm_pos_t.detach().cpu().numpy()
    palm_R = matrix_from_quat(palm_quat_t).detach().cpu().numpy()  # (N, 3, 3)

    # Tag world poses derived from the same rigid transform the obs use.
    tag_pos_t, tag_quat_t = _palm_pose_to_tag_pose(palm_pos_t, palm_quat_t)
    tag_pos = tag_pos_t.detach().cpu().numpy()
    tag_R = matrix_from_quat(tag_quat_t).detach().cpu().numpy()

    for env_idx in range(palm_pos.shape[0]):
      origin = env_origins_np[env_idx]
      pos = palm_pos[env_idx].astype(np.float32)
      R = palm_R[env_idx].astype(np.float32)
      tag_p = tag_pos[env_idx].astype(np.float32)
      tag_M = tag_R[env_idx].astype(np.float32)

      # Per-env world-frame axes (dashed-style: thin + 0.5 alpha).
      visualizer.add_frame(
        position=origin,
        rotation_matrix=world_R,
        scale=world_axis_len,
        axis_radius=0.0035,
        alpha=0.5,
        label=f"world_env{env_idx}",
      )

      # Per-env gravity arrow anchored at env_origin, pointing world -Z.
      grav_end = origin + np.array([0.0, 0.0, -gravity_len], dtype=np.float32)
      visualizer.add_arrow(
        start=origin,
        end=grav_end,
        color=(0.05, 0.05, 0.05, 0.95),
        width=0.005,
        label=f"gravity_env{env_idx}",
      )

      # Thin gray connector from env_origin straight up to the palm body so
      # the palm offset (~0.5m in z) is visible relative to the per-env world
      # reference, not just floating in space.
      visualizer.add_cylinder(
        start=origin,
        end=np.array([origin[0], origin[1], pos[2]], dtype=np.float32),
        radius=0.0015,
        color=(0.6, 0.6, 0.6, 0.6),
        label=f"palm_offset_env{env_idx}",
      )

      visualizer.add_frame(
        position=pos,
        rotation_matrix=R,
        scale=palm_axis_len,
        axis_radius=0.005,
        alpha=1.0,
        label=f"palm_env{env_idx}",
      )

      # Tag-frame triad at the actual tag world pose (CMY) — this is the
      # frame the policy obs are computed in.
      visualizer.add_frame(
        position=tag_p,
        rotation_matrix=tag_M,
        scale=tag_axis_len,
        axis_radius=0.005,
        alpha=1.0,
        axis_colors=tag_axis_colors,
        label=f"tag_env{env_idx}",
      )

      # A small label-cue "pin" along +Zp so the user can see palm-up axis
      # direction at a glance even from below.
      zp_end = pos + 0.06 * R[:, 2]
      visualizer.add_sphere(
        center=zp_end,
        radius=0.006,
        color=(0.0, 0.0, 0.95, 0.85),
        label=f"zp_marker_env{env_idx}",
      )

  return update


def override_wrist_orientations(
  env: ManagerBasedRlEnv,
  cases: list[WristCase],
) -> None:
  """Set each env's robot wrist pose to its target wrist quaternion.

  The reorient robot is fixed-base + mocap, so root pose is set via
  ``write_mocap_pose_to_sim``. Position is kept at the per-env scene origin
  plus the configured root pos offset; only the orientation differs across envs.
  """
  scene = env.scene
  robot = scene["robot"]
  num_envs = scene.num_envs

  env_origins = scene.env_origins.to(env.device)
  base_pos = torch.tensor(REORIENT_ROBOT_ROOT_POS, device=env.device).unsqueeze(0)
  positions = env_origins + base_pos  # (N, 3)
  quats = torch.tensor(
    [c.quat_wxyz for c in cases], dtype=torch.float32, device=env.device
  )
  pose = torch.cat([positions, quats], dim=-1)
  env_ids = torch.arange(num_envs, device=env.device, dtype=torch.int)

  if robot.is_fixed_base and robot.is_mocap:
    robot.write_mocap_pose_to_sim(pose, env_ids=env_ids)
  else:
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    zero_vel = torch.zeros((num_envs, 6), device=env.device)
    robot.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)

  # Park the cube near each palm so the scene is recognisable. The cube has a
  # free joint; without an active policy it will fall under gravity, but at
  # least the initial frame shows the scene clearly.
  if "object" in scene.entities:
    cube = scene["object"]
    cube_offset = torch.tensor(REORIENT_CUBE_INIT_POS, device=env.device).unsqueeze(0)
    cube_pos = env_origins + cube_offset
    cube_quat = torch.tensor(
      [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device
    ).expand(num_envs, -1)
    cube_pose = torch.cat([cube_pos, cube_quat], dim=-1)
    cube.write_root_link_pose_to_sim(cube_pose, env_ids=env_ids)
    cube_zero_vel = torch.zeros((num_envs, 6), device=env.device)
    cube.write_root_link_velocity_to_sim(cube_zero_vel, env_ids=env_ids)


def disable_dynamic_resets(env_cfg) -> None:
  """Disable terminations and randomization that would clobber our overrides."""
  env_cfg.terminations = {}
  if hasattr(env_cfg, "events") and isinstance(env_cfg.events, dict):
    for key in (
      "object_disturbance_force",
      "reset_object_disturbance_force",
    ):
      env_cfg.events.pop(key, None)


def run_visualization(
  num_envs: int,
  viewer_choice: str,
  device: str | None,
) -> None:
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  disable_dynamic_resets(env_cfg)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  # Inject `_debug_vis_enabled` on visualizable reward terms that don't have it.
  # The Viser viewer reads this attribute when wiring the per-reward debug-vis
  # GUI; some class-based terms in this repo (e.g. CageEscapePenalty) don't
  # set it, which would crash setup. Local workaround scoped to this viewer.
  for _, func in env.reward_manager.get_visualizable_terms():
    if not hasattr(func, "_debug_vis_enabled"):
      func._debug_vis_enabled = False  # leave their viz off; ours is separate

  cases = build_wrist_cases(num_envs)
  print("\n[viz] wrist orientations per env:")
  for i, c in enumerate(cases):
    print(
      f"  env {i}: {c.name}  wxyz=({c.quat_wxyz[0]:+.3f}, {c.quat_wxyz[1]:+.3f}, "
      f"{c.quat_wxyz[2]:+.3f}, {c.quat_wxyz[3]:+.3f})"
    )
  print()

  # First reset to drive normal init (joints, contacts, ...) then override.
  env.reset()
  override_wrist_orientations(env, cases)
  # Forward to commit pose into sim derived state without taking a real step.
  env.sim.forward()

  # Wrap the user-facing update_visualizers so our palm axes are drawn after
  # any task-defined debug viz (e.g. the reorient goal ghost mesh).
  base_visualizer = getattr(env, "update_visualizers", None)
  env.update_visualizers = make_palm_axes_visualizer(  # type: ignore[method-assign]
    env, [c.name for c in cases], base_visualizer
  )

  # Wrap for the viewer protocol; clip_actions=None to keep things simple.
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=None)

  action_shape: tuple[int, ...] = wrapped_env.unwrapped.action_space.shape

  class PolicyZero:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros(action_shape, device=wrapped_env.unwrapped.device)

  policy = PolicyZero()

  if viewer_choice == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = viewer_choice

  print(f"[viz] launching {resolved_viewer} viewer with num_envs={num_envs}")
  print("[viz] tip (viser): in the GUI panel turn on 'Debug Viz' and 'All envs'")
  print("[viz] tip (native): press 'V' to toggle debug vis, 'A' to show all envs")

  if resolved_viewer == "native":
    NativeMujocoViewer(wrapped_env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(wrapped_env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  wrapped_env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--num-envs",
    type=int,
    default=4,
    help="number of envs / wrist orientations (1-8)",
  )
  parser.add_argument(
    "--viewer",
    type=str,
    default="auto",
    choices=("auto", "native", "viser"),
    help="viewer backend",
  )
  parser.add_argument("--device", type=str, default=None)
  args = parser.parse_args()

  if not (1 <= args.num_envs <= 8):
    print(f"[err] --num-envs must be in [1, 8], got {args.num_envs}", file=sys.stderr)
    sys.exit(2)

  run_visualization(
    num_envs=args.num_envs,
    viewer_choice=args.viewer,
    device=args.device,
  )


if __name__ == "__main__":
  main()
