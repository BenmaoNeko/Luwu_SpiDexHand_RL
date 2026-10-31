# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Thin local play adapter that accepts already-prepared cfg objects."""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.keys import KEY_R

from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
  aligned_target_state,
  apply_face_move,
  apply_face_turn_angle,
  decode_actual_facelets,
  face_id_for_driver_joint,
  face_id_for_move,
  face_turn_parameters,
  invert_moves,
  make_solver_plan,
)
from wuji_mjlab.utils.rubik_layer_lock import RubikLayerLock


class _RubikResettableNativeMujocoViewer(NativeMujocoViewer):
  """Map R to the viewer's queued full-environment reset."""

  def _safe_key_callback(self, key: int) -> None:
    if key == KEY_R:
      self.request_reset()
      return
    super()._safe_key_callback(key)


class _RubikSolverPolicy:
  """Execute the solver sequence generated once from the episode scramble."""

  _FLIP_UNLOCK_ANGLE_RAD = math.radians(20.0)
  _PHASE_STEP_LIMIT = 2_000

  def __init__(self, policy, env, options: dict[str, Any]):
    self._policy = policy
    self._raw_env = env.unwrapped
    self._command = self._raw_env.command_manager.get_term("reorient_command")
    self._options = options
    self._object = self._raw_env.scene["object"]
    self._target = self._raw_env.scene["target"]
    if self._raw_env.num_envs != 1:
      raise ValueError("Solver preview is intentionally single-environment.")
    if not self._command.cfg.solver_mode:
      raise ValueError("Solver preview requires RubikUFaceCommandCfg.solver_mode=True")
    self._joint_names = self._object.joint_names
    self._target_joint_names = self._target.joint_names
    self._layer_lock = RubikLayerLock(self._raw_env.sim, self._joint_names)
    self._solution: tuple[str, ...] = ()
    self._planned_state: dict[str, float] = {}
    self._move_index = 0
    self._phase_steps = 0
    self._flip_boundary_pending = False
    self._turn_boundary_pending = False
    self._pending_flip_face: int | None = None
    self._pending_turn: tuple[str, str] | None = None
    self._active_turn_move: str | None = None
    self._phase = "flip"
    self._status = "not-started"
    self._result = ""
    self._last_legal_state: dict[str, float] = {}
    self._phase_start_facelets: str | None = None
    self._candidate_facelets: str | None = None
    self._candidate_duration = 0.0
    self._invalid_duration = 0.0
    self._monitor_accumulator = 0.0
    self._recovery_count = 0
    self._replan_count = 0
    self._recovery_delta = 0.0

  def __call__(self, obs):
    return self._policy(obs)

  def _joint_state(self, positions: torch.Tensor | None = None) -> dict[str, float]:
    if positions is None:
      positions = self._object.data.joint_pos[0]
    values = positions.detach().cpu().tolist()
    if len(values) != len(self._joint_names):
      raise RuntimeError("Rubik entity joint state and joint names do not align")
    return dict(zip(self._joint_names, map(float, values), strict=True))

  def _target_tensor(self, state: dict[str, float]) -> torch.Tensor:
    return torch.tensor(
      [[state[name] for name in self._target_joint_names]],
      device=self._raw_env.device,
      dtype=self._target.data.default_joint_pos.dtype,
    )

  def _refresh_object_observation_history(self) -> None:
    manager = self._raw_env.observation_manager
    manager._obs_buffer = None
    object_terms = {
      "rubik_cube_orientation",
      "rubik_compat_cube_ori_error",
      "rubik_u_face_angle",
    }
    env_ids = torch.arange(self._raw_env.num_envs, device=self._raw_env.device)
    for group_name, buffers in manager._group_obs_term_history_buffer.items():
      for term_name in object_terms & buffers.keys():
        buffers[term_name].reset(batch_ids=env_ids)
        delay = manager._group_obs_term_delay_buffer[group_name].get(term_name)
        if delay is not None:
          delay.reset(batch_ids=env_ids)
    self._raw_env.obs_buf = manager.compute(update_history=False)

  def _install_goal(
    self,
    face_id: int,
    phase: int,
    target_angle: float,
    target_state: dict[str, float],
  ) -> None:
    target_state = aligned_target_state(target_state, self._target_joint_names)
    if phase == 1:
      # The requested endpoint is an absolute legal detent, not start error + 90°.
      target_angle = round(target_angle / (math.pi / 2)) * (math.pi / 2)
    self._layer_lock.select(self._joint_state(), face_id if phase == 1 else None)
    self._command.set_solver_goal(
      face_id,
      phase,
      target_angle,
      self._target_tensor(target_state),
    )
    self._raw_env.sim.forward()
    self._refresh_object_observation_history()

  def reset(self) -> None:
    """Build a fresh solver state and write its scramble only at episode reset."""
    scramble_plan = make_solver_plan(
      scramble=self._options.get("scramble"),
      random_state=bool(self._options.get("random_state", False)),
    )
    self._solution = ()
    self._move_index = 0
    self._phase_steps = 0
    self._flip_boundary_pending = False
    self._turn_boundary_pending = False
    self._phase = "initializing"
    self._status = "running"
    self._result = ""
    self._pending_turn = None
    self._active_turn_move = None

    defaults = self._object.data.default_joint_pos[0].clone()
    solved_state = self._joint_state(defaults)
    scrambled_state = solved_state
    for token in invert_moves(tuple(scramble_plan["solution"])):
      scrambled_state = apply_face_move(scrambled_state, token, self._joint_names)

    joint_pos = torch.tensor(
      [[scrambled_state[name] for name in self._joint_names]],
      device=self._raw_env.device,
      dtype=self._object.data.joint_pos.dtype,
    )
    self._object.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    object_pose = self._object.data.root_link_pose_w.clone()
    palm_id = self._command.palm_body_id
    palm_pose = self._command.robot.data.body_link_pose_w[:, palm_id]
    object_pose[:, 3:7] = palm_pose[:, 3:7]
    self._object.write_root_link_pose_to_sim(object_pose)
    self._object.write_root_link_velocity_to_sim(
      torch.zeros((1, 6), device=self._raw_env.device)
    )
    from wuji_mjlab.tasks.reorient.mdp.vision_observations import (
      reset_visual_pose_state,
    )

    reset_visual_pose_state(self._raw_env)
    self._raw_env.sim.forward()
    self._last_legal_state = dict(scrambled_state)
    self._phase_start_facelets = scramble_plan["initialFacelets"]
    self._candidate_facelets = self._phase_start_facelets
    self._candidate_duration = 0.0
    self._invalid_duration = 0.0
    self._monitor_accumulator = 0.0
    self._install_scramble_plan(scramble_plan, scrambled_state)

  def _start_flip(self, face_id: int, state: dict[str, float]) -> None:
    self._pending_flip_face = None
    planned_state = getattr(self, "_planned_state", state)
    top_face = int(self._command._read_top_face_id()[0])
    if top_face == (face_id ^ 1):
      self._pending_flip_face = face_id
      face_id = next(face for face in range(6) if face // 2 != face_id // 2)
      print("[Rubik solver] bottom flip: adjacent waypoint 1/2")
    self._install_goal(
      face_id, 0, float(self._command.u_angle[0]), planned_state
    )

  def _advance_flip_waypoint(self, state: dict[str, float]) -> bool:
    if self._pending_flip_face is None:
      return False
    face_id = self._pending_flip_face
    self._pending_flip_face = None
    self._flip_boundary_pending = False
    self._phase_steps = 0
    self._install_goal(
      face_id, 0, float(self._command.u_angle[0]),
      getattr(self, "_planned_state", state),
    )
    print("[Rubik solver] bottom flip: final face to U 2/2")
    return True

  def _start_turn(
    self, move: str, state: dict[str, float], facelets: str | None = None
  ) -> None:
    half_turn = move.endswith("2")
    # face_turn_parameters uses positive 180 degrees for half turns.
    step_move = move[0] + "'" if half_turn else move
    _axis, side, raw_degrees = face_turn_parameters(step_move)
    planned_state = getattr(self, "_planned_state", state)
    expected = apply_face_move(planned_state, step_move, self._joint_names)
    expected_facelets = decode_actual_facelets(expected, self._joint_names)
    self._pending_turn = (step_move, expected_facelets) if half_turn else None
    target_angle = float(self._command.u_angle[0]) + math.radians(raw_degrees * side)
    # Do not expose a turn phase with the old flip goal if unlocking fails.
    self._install_goal(face_id_for_move(step_move), 1, target_angle, expected)
    self._active_turn_move = step_move
    self._phase = "turn"
    self._phase_steps = 0
    self._flip_boundary_pending = False
    self._turn_boundary_pending = False
    print(
      f"[Rubik solver] turn {step_move}" + (f" (1/2 of {move})" if half_turn else "")
    )

  def _install_scramble_plan(
    self, plan: dict[str, Any], scrambled_state: dict[str, float]
  ) -> None:
    """Install the already-computed solve sequence and its expected state."""
    self._flip_boundary_pending = False
    self._turn_boundary_pending = False
    self._pending_flip_face = None
    self._pending_turn = None
    self._solution = tuple(plan["solution"])
    self._planned_state = dict(scrambled_state)
    self._move_index = 0
    self._phase_steps = 0
    self._active_turn_move = None
    if not self._solution:
      self._status = "complete"
      self._phase = "complete"
      self._result = "Scramble is already solved"
      self._install_goal(
        face_id=4,
        phase=0,
        target_angle=float(self._command.u_angle[0]),
        target_state=self._planned_state,
      )
      print("[Rubik solver] scramble is already solved")
      return

    self._status = "running"
    self._phase = "flip"
    self._start_flip(face_id_for_move(self._solution[0]), self._planned_state)
    print(
      f"[Rubik solver] reset: {plan['initialFacelets']}; "
      f"{len(self._solution)} planned moves; next {self._solution[0]}: flip selected face to U"
    )

  def _fail(self, reason: str, viewer) -> None:
    if self._status != "running":
      return
    self._status = "failed"
    self._result = reason
    move = (
      self._solution[getattr(self, "_move_index", 0)]
      if getattr(self, "_move_index", 0) < len(self._solution)
      else "final verification"
    )
    print(
      f"[Rubik solver] PAUSED at move {getattr(self, '_move_index', 0) + 1}/"
      f"{len(self._solution)} ({move}): {reason}"
    )
    if viewer is not None:
      viewer.pause()

  def _advance_completed_turn(self, viewer) -> None:
    active_move = getattr(self, "_active_turn_move", None)
    if active_move is None and getattr(self, "_planned_state", None) is None:
      pending = getattr(self, "_pending_turn", None)
      if pending is not None:
        move, expected = pending
        self._pending_turn = None
        self._start_turn(move, self._joint_state(), expected)
        return
    if active_move is None and getattr(self, "_pending_turn", None) is not None:
      active_move = self._pending_turn[0]
    if active_move is None:
      state = self._joint_state()
      facelets = decode_actual_facelets(state, self._joint_names)
      if facelets is not None and facelets != getattr(self, "_phase_start_facelets", None):
        self._accept_actual_state(state, facelets, viewer=viewer)
      return
    self._planned_state = apply_face_move(
      getattr(self, "_planned_state", self._joint_state()), active_move, self._joint_names
    )
    self._last_legal_state = dict(self._planned_state)
    if self._pending_turn is not None:
      move, expected = self._pending_turn
      self._pending_turn = None
      print(f"[Rubik solver] half turn: second 90 degrees 2/2 of {move}")
      self._start_turn(move, self._joint_state(), expected)
      return

    self._move_index += 1
    self._active_turn_move = None
    if self._move_index >= len(self._solution):
      self._status = "complete"
      self._phase = "complete"
      self._result = "Precomputed solve sequence completed"
      self._install_goal(
        face_id=4,
        phase=0,
        target_angle=float(self._command.u_angle[0]),
        target_state=self._planned_state,
      )
      print("[Rubik solver] precomputed solve sequence completed")
      if viewer is not None:
        viewer.pause()
      return

    next_move = self._solution[self._move_index]
    self._phase = "flip"
    self._phase_steps = 0
    self._flip_boundary_pending = False
    self._turn_boundary_pending = False
    self._start_flip(face_id_for_move(next_move), self._joint_state())
    print(
      f"[Rubik solver] planned move {self._move_index + 1}/{len(self._solution)}: "
      f"{next_move}; flip selected face to U"
    )

  def after_step(self, viewer) -> None:
    if self._status != "running":
      return
    if self._raw_env.termination_manager.get_term("cage_drop")[0].item():
      self._fail("cube dropped out of the palm cage", viewer)
      return
    self._phase_steps += 1
    if self._phase_steps > self._PHASE_STEP_LIMIT:
      self._fail(f"{self._phase} stage timed out on the fixed plan", viewer)
      return

    if self._phase == "flip":
      self._flip_boundary_pending = self._geometric_goal_reached(turn=False)
    if self._phase == "flip" and self._flip_boundary_pending:
      state = self._joint_state()
      if self._advance_flip_waypoint(state):
        return
      if getattr(self, "_move_index", 0) >= len(getattr(self, "_solution", ())):
        self._fail("fixed plan has no move after flip", viewer)
        return
      self._start_turn(
        self._solution[getattr(self, "_move_index", 0)],
        state,
        decode_actual_facelets(state, self._joint_names),
      )
      return

    if self._phase == "turn":
      self._turn_boundary_pending = self._geometric_goal_reached(turn=True)
    if self._phase == "turn" and self._turn_boundary_pending:
      self._advance_completed_turn(viewer)
    self._monitor_actual_state(viewer)

  def _stationary(self) -> bool:
    """Return whether the cube has settled enough for state inspection."""
    data = self._object.data
    joint_vel = getattr(data, "joint_vel", None)
    root_vel = getattr(data, "root_link_vel_w", None)
    if joint_vel is not None and float(joint_vel.abs().max()) > 0.12:
      return False
    if root_vel is not None and float(root_vel.abs().max()) > 0.08:
      return False
    return True

  def _accept_actual_state(
    self, state: dict[str, float], facelets: str, *, viewer=None
  ) -> None:
    """Replace the stale plan with one generated from the observed cube state."""
    plan = make_solver_plan(facelets=facelets)
    self._replan_count += 1
    self._solution = tuple(plan["solution"])
    self._planned_state = dict(state)
    self._last_legal_state = dict(state)
    self._move_index = 0
    self._phase_steps = 0
    self._active_turn_move = None
    self._pending_turn = None
    self._flip_boundary_pending = False
    self._turn_boundary_pending = False
    self._phase_start_facelets = facelets
    self._candidate_facelets = facelets
    self._candidate_duration = 0.0
    self._invalid_duration = 0.0
    self._status = "complete" if not self._solution else "running"
    self._phase = "complete" if not self._solution else "flip"
    if not self._solution:
      self._result = "Observed cube is already solved"
      self._install_goal(
        face_id=4,
        phase=0,
        target_angle=float(self._command.u_angle[0]),
        target_state=self._planned_state,
      )
      return
    self._start_flip(face_id_for_move(self._solution[0]), self._planned_state)
    print(
      f"[Rubik solver] replanned from observed state; "
      f"{len(self._solution)} moves remain; next {self._solution[0]}"
    )

  def _begin_recovery(self, state: dict[str, float], viewer) -> None:
    """Recover one partially turned layer before requesting a new plan."""
    if self._phase_steps > self._PHASE_STEP_LIMIT:
      self._fail("cube recovery exceeded phase step limit; alignment deltas deg", viewer)
      return
    if not self._last_legal_state:
      return
    deltas = {}
    for name in (
      "cubelet:driver:pos_x", "cubelet:driver:neg_x",
      "cubelet:driver:pos_y", "cubelet:driver:neg_y",
      "cubelet:driver:pos_z", "cubelet:driver:neg_z",
    ):
      face_id = face_id_for_driver_joint(name)
      delta = math.atan2(
        math.sin(float(state[name]) - float(self._last_legal_state.get(name, 0.0))),
        math.cos(float(state[name]) - float(self._last_legal_state.get(name, 0.0))),
      )
      if abs(delta) > math.radians(8.0) and abs(abs(delta) - math.pi / 2) > math.radians(12.0):
        deltas[face_id] = delta
    if len(deltas) != 1:
      return
    face_id, delta = next(iter(deltas.items()))
    recovery_state = apply_face_turn_angle(
      state, face_id, -delta, self._joint_names
    )
    self._recovery_delta = -delta
    self._recovery_count += 1
    self._phase = "recover_flip"
    self._phase_steps = 0
    self._install_goal(
      face_id=face_id,
      phase=0,
      target_angle=float(self._command.u_angle[0]),
      target_state=recovery_state,
    )

  def _monitor_actual_state(self, viewer) -> None:
    """Inspect settled boundaries and recover/replan only when state is stable."""
    dt = float(getattr(self._raw_env, "step_dt", 0.05))
    self._monitor_accumulator += dt
    if self._monitor_accumulator < dt:
      return
    self._monitor_accumulator = 0.0
    state = self._joint_state()
    facelets = decode_actual_facelets(state, self._joint_names)
    if facelets is None:
      self._invalid_duration += dt
      if self._turn_boundary_pending or self._flip_boundary_pending:
        self._begin_recovery(state, viewer)
      return
    self._invalid_duration = 0.0
    if facelets != self._phase_start_facelets:
      if self._candidate_facelets == facelets:
        self._candidate_duration += dt
      else:
        self._candidate_facelets = facelets
        self._candidate_duration = 0.0
      if (
        self._candidate_duration >= dt
        and self._phase in {"flip", "recover_flip"}
        and (
          getattr(self, "_flip_boundary_pending", False)
          or self._phase == "recover_flip"
          or self._phase == "flip"
        )
      ):
        self._accept_actual_state(state, facelets, viewer=viewer)
    else:
      self._candidate_facelets = facelets
      self._candidate_duration = 0.0

  def _geometric_goal_reached(self, *, turn: bool) -> bool:
    """Preview completion uses current geometry only, never training reward gates."""
    command = self._command
    upright_limit = (
      command.cfg.upright_tolerance_rad
      if turn else self._FLIP_UNLOCK_ANGLE_RAD
    )
    if float(command._read_flip_target_dot()[0]) < math.cos(upright_limit):
      return False
    if not turn:
      return True
    delta = float((command.u_angle - command.u_target_angle)[0])
    error = abs(math.atan2(math.sin(delta), math.cos(delta)))
    return error <= command.cfg.turn_tolerance_rad


class _RubikSolverNativeViewer(_RubikResettableNativeMujocoViewer):
  def setup(self) -> None:
    super().setup()
    if self.policy._status != "running":
      self.pause()

  def reset_environment(self) -> None:
    super().reset_environment()
    if self.policy._status == "running":
      self.resume()

  def _set_status_overlay(self, viewer) -> None:
    status = self.get_status()
    solver = self.policy
    if solver._status == "running" and solver._move_index < len(solver._solution):
      move = solver._solution[solver._move_index]
    else:
      move = "-"
    angle_error = math.degrees(float(solver._command.u_angle_error[0]))
    face_error = math.degrees(
      math.acos(float(solver._command.u_up_dot[0].clamp(-1.0, 1.0)))
    )
    labels = (
      "Env",
      "Step",
      "Status",
      "Speed",
      "Target RT",
      "Actual RT",
      "Move",
      "Notation",
      "Phase",
      "Face err",
      "Layer err",
      "Result",
    )
    values = (
      f"{self.env_idx + 1}/{self.env.num_envs}",
      str(status.step_count),
      "PAUSED" if status.paused else "RUNNING",
      status.speed_label,
      f"{status.target_realtime:.2f}x",
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)",
      f"{min(solver._move_index + 1, len(solver._solution))}/{len(solver._solution)}",
      move,
      solver._phase if solver._status == "running" else solver._status,
      f"{face_error:.1f} deg",
      f"{angle_error:.1f} deg",
      solver._result[:48] or "-",
    )
    viewer.set_texts(
      (
        mujoco.mjtFontScale.mjFONTSCALE_150.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        "\n".join(labels),
        "\n".join(values),
      )
    )

  def _execute_step(self) -> bool:
    success = super()._execute_step()
    if success:
      try:
        self.policy.after_step(self)
      except Exception as error:
        self.policy._fail(f"execution stopped: {error}", self)
        return False
    return success


class _PreviewFixedTaskPolicy:
  """Keep four rotation and four flip tasks fixed in the local preview."""

  def __init__(self, policy, env):
    self._policy = policy
    self._env = env.unwrapped
    self._command = self._env.command_manager.get_term("reorient_command")
    if self._env.num_envs != 8:
      raise ValueError(
        "LuwuHand_Reorient preview requires exactly 8 environments: "
        "4 rotation and 4 flip."
      )
    env_ids = torch.arange(self._env.num_envs, device=self._env.device)
    self._rotation_env_ids = env_ids[:4]
    self._flip_env_ids = env_ids[4:]

  def __call__(self, obs):
    self._command.evaluate_contact_task()
    reset_mask = self._env.episode_length_buf == 0
    if reset_mask.any():
      rotation_ids = self._rotation_env_ids[reset_mask[self._rotation_env_ids]]
      flip_ids = self._flip_env_ids[reset_mask[self._flip_env_ids]]
      self._command.preview_set_fixed_task_modes(rotation_ids, flip_ids)
    return self._policy(obs)


def _resolve_downloaded_motion_path(download_dir: Path) -> Path:
  preferred_names = ("trajectories_dyn.npz", "trajectories_kin.npz", "motion.npz")
  for name in preferred_names:
    candidate = download_dir / name
    if candidate.exists():
      return candidate

  motion_files = sorted(
    list(download_dir.glob("*.pt")) + list(download_dir.glob("*.npz"))
  )
  if len(motion_files) == 1:
    return motion_files[0]
  if len(motion_files) > 1:
    raise RuntimeError(
      f"Multiple motion files found under artifact download: {download_dir}"
    )
  raise FileNotFoundError(
    f"No motion file found under artifact download: {download_dir}"
  )


def _get_motion_path_cfg(motion_cmd: Any) -> tuple[str | None, str]:
  motion_files = getattr(motion_cmd, "motion_files", None)
  if motion_files is not None:
    return motion_files[0] if motion_files else None, "motion_files"

  # fallback for single-file API
  motion_file = getattr(motion_cmd, "motion_file", None)
  if motion_file is not None:
    return motion_file, "motion_file"

  dataset_cfg = getattr(motion_cmd, "dataset_cfg", None)
  if dataset_cfg is not None:
    return getattr(dataset_cfg, "data_path", None), "dataset_cfg.data_path"

  raise AttributeError(
    "Unsupported tracking command cfg: missing motion_files, motion_file, and dataset_cfg.data_path"
  )


def _set_motion_path_cfg(motion_cmd: Any, motion_path: str) -> None:
  # motion_files is our plural API (used by the NPZ loader); motion_file is
  # the mjlab base-class compat. Keep both in sync.
  set_any = False
  if hasattr(motion_cmd, "motion_files"):
    motion_cmd.motion_files = (motion_path,)
    set_any = True
  if hasattr(motion_cmd, "motion_file"):
    motion_cmd.motion_file = motion_path
    set_any = True
  if set_any:
    return

  dataset_cfg = getattr(motion_cmd, "dataset_cfg", None)
  if dataset_cfg is not None and hasattr(dataset_cfg, "data_path"):
    dataset_cfg.data_path = motion_path
    return

  raise AttributeError(
    "Unsupported tracking command cfg: missing motion_files, motion_file, and dataset_cfg.data_path"
  )


def run_play_with_cfg(
  task_id: str,
  cfg: PlayConfig,
  env_cfg: Any,
  agent_cfg: Any,
  *,
  rubik_solver_options: dict[str, Any] | None = None,
):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  fixed_preview_tasks = task_id in {
    "LuwuHand_Reorient",
    "LuwuHand_Reorient_ContinuousFaces",
  }
  if fixed_preview_tasks:
    print("[INFO]: Preview uses envs 0-3 for rotation and 4-7 for flip")

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    bundled_motion_file, motion_field = _get_motion_path_cfg(motion_cmd)
    bundled_path = Path(bundled_motion_file) if bundled_motion_file else None
    has_bundled_motion_file = bundled_path is not None and bundled_path.exists()
    motion_path = Path(cfg.motion_file) if cfg.motion_file is not None else None

    if motion_path is not None:
      if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
      print(f"[INFO]: Using local motion file: {motion_path}")
      _set_motion_path_cfg(motion_cmd, str(motion_path))
    elif has_bundled_motion_file:
      print(f"[INFO]: Using bundled motion file from {motion_field}: {bundled_path}")
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion-file (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      _set_motion_path_cfg(
        motion_cmd,
        str(_resolve_downloaded_motion_path(Path(artifact.download()))),
      )
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        _set_motion_path_cfg(motion_cmd, cfg.motion_file)
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          _set_motion_path_cfg(
            motion_cmd,
            str(_resolve_downloaded_motion_path(Path(art.download()))),
          )

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      assert resume_path is not None
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    assert resume_path is not None
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if rubik_solver_options is not None or task_id == "LuwuHand_Rubik_FullCube_SolverTransfer":
    for name, term in env.unwrapped.reward_manager.get_visualizable_terms():
      if name == "cage_escape":
        term._debug_vis_enabled = False

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    checkpoint_infos = runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    if task_id == "LuwuHand_Rubik_FlipGentleTransfer":
      from wuji_mjlab.tasks.reorient.mdp.gentle_flip import restore_gentle_checkpoint
      restore_gentle_checkpoint(env.unwrapped, checkpoint_infos, preview=True)
    policy = runner.get_inference_policy(device=device)

  if rubik_solver_options is not None:
    if task_id != "LuwuHand_Rubik_UFace_ContinuousTransfer":
      raise ValueError("Rubik solver preview requires the continuous-transfer actor")
    if cfg.viewer != "native":
      raise ValueError("Rubik solver preview requires the native MuJoCo viewer")
    policy = _RubikSolverPolicy(policy, env, rubik_solver_options)
    termination_manager = env.unwrapped.termination_manager
    original_compute = termination_manager.compute

    def compute_without_automatic_reset():
      done = original_compute()
      return torch.zeros_like(done)

    termination_manager.compute = compute_without_automatic_reset
    env.reset()
    policy.reset()

  if fixed_preview_tasks:
    policy = _PreviewFixedTaskPolicy(policy, env)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    if rubik_solver_options is not None:
      print(
        "[INFO] Solver preview: R/Enter starts a new scramble; failures pause in place"
      )
      _RubikSolverNativeViewer(env, policy).run()
    elif task_id == "LuwuHand_Rubik_UFace_ContinuousTransfer":
      print("[INFO] Rubik preview: press R or Enter to reset the environment")
      _RubikResettableNativeMujocoViewer(env, policy).run()
    else:
      NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()
