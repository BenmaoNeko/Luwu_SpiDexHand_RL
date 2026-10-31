# SPDX-License-Identifier: Apache-2.0
"""Continuous Rubik flip, U-layer turn, and whole-cube yaw commands."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_mul,
)

from .cube_faces import FACE_NORMALS, opposite_face_ids, sample_adjacent_face
from .event_utils import resolve_env_ids


def bounded_u_turn_progress(
  angle_error: torch.Tensor,
  initial_error: torch.Tensor,
  best_score: torch.Tensor,
  phase: torch.Tensor,
  contact: torch.Tensor,
  upright: torch.Tensor,
  *,
  same_goal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Pay each target's new U-angle progress once, only under valid manipulation."""
  score = (1.0 - angle_error / initial_error.clamp_min(0.1)).clamp(0.0, 1.0)
  turn_phase = phase == 1
  improvement = (score - best_score).clamp_min(0.0)
  new_best = torch.where(turn_phase, torch.maximum(best_score, score), best_score)
  eligible = turn_phase & contact & upright
  if same_goal is not None:
    eligible &= same_goal
  return torch.where(eligible, improvement, torch.zeros_like(improvement)), new_best


class RubikUFaceCommand(CommandTerm):
  cfg: RubikUFaceCommandCfg

  def __init__(self, cfg: RubikUFaceCommandCfg, env):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.robot: Entity = env.scene[cfg.robot_entity_name]
    palm_ids, _ = self.robot.find_bodies(cfg.palm_body_pattern)
    joint_ids, _ = self.object.find_joints(cfg.u_face_joint_name)
    if len(palm_ids) != 1 or len(joint_ids) != 1:
      raise ValueError(
        "Rubik U-face task requires exactly one palm body and one U hinge; "
        f"found {len(palm_ids)} and {len(joint_ids)}."
      )
    self.palm_body_id = int(palm_ids[0])
    self.u_face_joint_id = int(joint_ids[0])
    face_driver_names = (
      "cubelet:driver:pos_x",
      "cubelet:driver:neg_x",
      "cubelet:driver:pos_y",
      "cubelet:driver:neg_y",
      "cubelet:driver:pos_z",
      "cubelet:driver:neg_z",
    )
    face_driver_ids = []
    for name in face_driver_names:
      ids, _ = self.object.find_joints(name)
      if len(ids) != 1:
        raise ValueError(f"Rubik cube is missing face driver joint {name!r}.")
      face_driver_ids.append(int(ids[0]))
    self._face_driver_ids = torch.tensor(face_driver_ids, device=self.device)
    self._solver_target_joint_pos: torch.Tensor | None = None
    self.target = None
    self._target_sync_step = -1
    if cfg.target_entity_name:
      self.target: Entity = env.scene[cfg.target_entity_name]
      target_joint_ids, _ = self.target.find_joints(cfg.u_face_joint_name)
      target_linked_ids, _ = self.target.find_joints(r"cubelet:rotz:.*_pos_z")
      if len(target_joint_ids) != 1 or len(target_linked_ids) != 8:
        raise ValueError("Rubik target cube is missing its U-layer joints.")
      self.target_u_face_joint_id = int(target_joint_ids[0])
      self.target_linked_joint_ids = target_linked_ids

    n, device = self.num_envs, self.device
    self.task_mode = torch.zeros(n, dtype=torch.long, device=device)
    self.phase = torch.zeros(n, dtype=torch.long, device=device)
    self.target_face_id = torch.full((n,), 4, dtype=torch.long, device=device)
    self.root_target_quat_w = torch.zeros((n, 4), device=device)
    self.root_target_quat_w[:, 0] = 1.0
    self.u_initial_angle = torch.zeros(n, device=device)
    self.u_target_angle = torch.zeros(n, device=device)
    self.flip_start_dot = torch.zeros(n, device=device)
    self.turn_start_error = torch.ones(n, device=device)
    self._flip_goal_bottom = torch.zeros(n, dtype=torch.bool, device=device)
    self._turn_goal_180 = torch.zeros_like(self._flip_goal_bottom)
    self._pending_goal_advance = torch.zeros_like(self._flip_goal_bottom)
    self._upright_hold = torch.zeros(n, dtype=torch.int32, device=device)
    self._turn_hold = torch.zeros_like(self._upright_hold)
    self._yaw_hold = torch.zeros_like(self._upright_hold)
    self._flip_event = torch.zeros(n, dtype=torch.bool, device=device)
    self._turn_event = torch.zeros_like(self._flip_event)
    self._yaw_event = torch.zeros_like(self._flip_event)
    self._contact_hold_time = torch.zeros((n, 2), device=device)
    self._contact_first_awarded = torch.zeros_like(self._flip_event)
    self._contact_second_awarded = torch.zeros_like(self._flip_event)
    self._approach_best_score = torch.zeros(
      (n, cfg.approach_fingertip_count), device=device
    )
    self._approach_initialized = torch.zeros_like(self._flip_event)
    self._previous_cube_pos_palm = torch.zeros((n, 3), device=device)
    self.cube_position_delta = torch.zeros(n, device=device)
    self.cube_position_delta_valid = torch.zeros(n, dtype=torch.bool, device=device)
    self.smart_move_pending = torch.zeros(n, dtype=torch.bool, device=device)
    self._smart_move_state_step = torch.full((n,), -1, dtype=torch.long, device=device)
    self._last_update_step = torch.full((n,), -1, dtype=torch.long, device=device)
    self._previous_phase = torch.full_like(self.phase, -1)
    self._previous_potential = torch.zeros(n, device=device)
    self.progress_delta = torch.zeros(n, device=device)
    self.turn_progress_delta = torch.zeros(n, device=device)
    self._turn_progress_best = torch.zeros(n, device=device)
    self.u_up_dot = torch.zeros(n, device=device)
    self.u_angle_error = torch.zeros(n, device=device)
    self.yaw_error = torch.zeros(n, device=device)
    self.tip_contact_count = torch.zeros(n, device=device)
    self.flip_to_top_count = torch.zeros(n, device=device)
    self.turn_success_count = torch.zeros(n, device=device)
    self.yaw_success_count = torch.zeros(n, device=device)
    self.goal_reach_count = torch.zeros(n, device=device)

    for name in (
      "rubik_task_phase",
      "rubik_turn_task_fraction",
      "rubik_yaw_task_fraction",
      "rubik_u_up_dot",
      "rubik_u_angle_error_rad",
      "rubik_yaw_error_rad",
      "rubik_tip_contact_count",
      "rubik_flip_to_top_count",
      "rubik_turn_success_count",
      "rubik_yaw_success_count",
      "rubik_goal_reach_count",
      "rubik_flip_stage",
      "rubik_turn_stage",
    ):
      self.metrics[name] = torch.zeros(n, device=device)

  @property
  def command(self) -> torch.Tensor:
    return torch.stack(
      (self.u_up_dot, self.u_angle, self.u_target_angle, self.phase.float()), dim=-1
    )

  @property
  def task_reward_active(self) -> torch.Tensor:
    return self.phase <= 2

  @property
  def success_achieved(self) -> torch.Tensor:
    return self._flip_event | self._turn_event | self._yaw_event

  @property
  def flip_to_top_achieved(self) -> torch.Tensor:
    return self._flip_event

  @property
  def turn_completed(self) -> torch.Tensor:
    return self._turn_event

  @property
  def yaw_completed(self) -> torch.Tensor:
    return self._yaw_event

  @property
  def u_angle(self) -> torch.Tensor:
    """Continuous physics angle for rewards, skill success and privileged critic.

    Smart-cube MOVE quantization belongs exclusively in the actor observation.
    """
    if self.cfg.solver_mode:
      ids = self._face_driver_ids[self.target_face_id]
      raw = self.object.data.joint_pos.gather(1, ids.unsqueeze(-1)).squeeze(-1)
      sign = torch.where(
        (self.target_face_id % 2) == 1, -torch.ones_like(raw), torch.ones_like(raw)
      )
      return raw * sign
    return self.object.data.joint_pos[:, self.u_face_joint_id]

  @property
  def u_angle_velocity(self) -> torch.Tensor:
    if self.cfg.solver_mode:
      ids = self._face_driver_ids[self.target_face_id]
      raw = self.object.data.joint_vel.gather(1, ids.unsqueeze(-1)).squeeze(-1)
      sign = torch.where(
        (self.target_face_id % 2) == 1, -torch.ones_like(raw), torch.ones_like(raw)
      )
      return raw * sign
    return self.object.data.joint_vel[:, self.u_face_joint_id]

  def _read_tip_contact_count(self) -> torch.Tensor:
    sensor = self._env.scene[self.cfg.tip_contact_sensor_name]
    found = sensor.data.found
    if found is None:
      return torch.zeros(self.num_envs, device=self.device)
    if found.ndim == 1:
      found = found.unsqueeze(-1)
    return (found > 0).float().sum(dim=-1)

  def _read_upright_dot(self) -> torch.Tensor:
    dtype = self.object.data.root_link_quat_w.dtype
    if self.cfg.solver_mode:
      up_local = FACE_NORMALS.to(device=self.device, dtype=dtype)[self.target_face_id]
    else:
      up_local = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
      up_local[:, 2] = 1.0
    u_normal_w = quat_apply(self.object.data.root_link_quat_w, up_local)
    return (u_normal_w * self._palm_normal_w()).sum(dim=-1).clamp(-1.0, 1.0)

  def _palm_normal_w(self) -> torch.Tensor:
    quat = self.robot.data.body_link_pose_w[:, self.palm_body_id, 3:7]
    up = torch.zeros((self.num_envs, 3), device=self.device, dtype=quat.dtype)
    up[:, 2] = 1.0
    return quat_apply(quat, up)

  def _read_flip_target_dot(self) -> torch.Tensor:
    normal_local = FACE_NORMALS.to(
      device=self.device, dtype=self.object.data.root_link_quat_w.dtype
    )[self.target_face_id]
    normal_w = quat_apply(self.object.data.root_link_quat_w, normal_local)
    return (normal_w * self._palm_normal_w()).sum(dim=-1).clamp(-1.0, 1.0)

  def _read_top_face_id(self) -> torch.Tensor:
    normals = FACE_NORMALS.to(
      device=self.device, dtype=self.object.data.root_link_quat_w.dtype
    ).unsqueeze(0).expand(self.num_envs, -1, -1)
    quats = self.object.data.root_link_quat_w.unsqueeze(1).expand(-1, 6, -1)
    dots = (quat_apply(quats, normals) * self._palm_normal_w().unsqueeze(1)).sum(-1)
    return dots.argmax(dim=-1)

  def _potential(
    self,
    phase: torch.Tensor,
    flip_dot: torch.Tensor,
    upright_dot: torch.Tensor,
    angle_error: torch.Tensor,
    yaw_error: torch.Tensor,
  ) -> torch.Tensor:
    flip = (flip_dot - self.flip_start_dot) / (
      1.0 - self.flip_start_dot
    ).clamp_min(0.1) - 0.5 * angle_error / torch.pi
    turn = 1.0 - angle_error / self.turn_start_error.clamp_min(0.1)
    turn += 0.5 * (upright_dot - 1.0)
    yaw = 1.0 - yaw_error / (torch.pi / 12.0)
    yaw += 0.5 * (upright_dot - 1.0) - 0.5 * angle_error / torch.pi
    return torch.where(phase == 0, flip, torch.where(phase == 1, turn, yaw))

  def _sync_preview_target(self) -> None:
    if self.target is None:
      return
    cube_pose = self._env.sim.data.qpos[:, self.object.indexing.free_joint_q_adr]
    # Anchor orientation to the installed goal, rather than following the cube.
    cube_quat = (
      self.root_target_quat_w if self.cfg.target_fixed_pose else cube_pose[:, 3:7]
    )
    palm_quat = self.robot.data.body_link_pose_w[:, self.palm_body_id, 3:7]
    local_up = torch.zeros((self.num_envs, 3), device=self.device)
    local_up[:, 2] = 1.0
    face_local = FACE_NORMALS.to(device=self.device, dtype=cube_quat.dtype)[
      self.target_face_id
    ]
    u_normal = quat_apply(cube_quat, face_local)
    palm_normal = quat_apply(palm_quat, local_up)
    dot = (u_normal * palm_normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    axis = torch.cross(u_normal, palm_normal, dim=-1)
    align_quat = torch.cat((1.0 + dot, axis), dim=-1)
    basis_x = torch.zeros_like(axis)
    basis_y = torch.zeros_like(axis)
    basis_x[:, 0] = 1.0
    basis_y[:, 1] = 1.0
    basis = torch.where(
      (u_normal[:, :1].abs() < 0.9), basis_x, basis_y
    )
    fallback_axis = torch.cross(u_normal, basis, dim=-1)
    fallback_axis = fallback_axis / fallback_axis.norm(
      dim=-1, keepdim=True
    ).clamp_min(1e-8)
    opposite = dot.squeeze(-1) < -0.9999
    align_quat[opposite, 1:] = fallback_axis[opposite]
    align_quat[opposite, 0] = 0.0
    align_quat = align_quat / align_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    flip_target_quat = quat_mul(align_quat, cube_quat)
    target_quat = torch.where(
      (self.phase == 0).unsqueeze(-1),
      flip_target_quat,
      torch.where((self.phase == 2).unsqueeze(-1), self.root_target_quat_w, cube_quat),
    )

    offset = torch.tensor(
      self.cfg.target_offset_palm_m, device=self.device, dtype=cube_quat.dtype
    ).expand(self.num_envs, 3)
    target_pos = cube_pose[:, :3] + quat_apply(palm_quat, offset)
    if self.cfg.target_fixed_pose:
      target_pos = (
        self.target.data.default_root_state[:, :3] + self._env.scene.env_origins
      )
    self.target.write_root_link_pose_to_sim(
      torch.cat((target_pos, target_quat), dim=-1)
    )
    self.target.write_root_link_velocity_to_sim(
      torch.zeros((self.num_envs, 6), device=self.device)
    )

    if self.cfg.solver_mode and self._solver_target_joint_pos is not None:
      target_joint_pos = self._solver_target_joint_pos.clone()
    else:
      target_joint_pos = self.target.data.default_joint_pos.clone()
    target_joint_vel = torch.zeros_like(target_joint_pos)
    if not (self.cfg.solver_mode and self._solver_target_joint_pos is not None):
      target_joint_pos[:, self.target_u_face_joint_id] = self.u_target_angle
      target_joint_pos[:, self.target_linked_joint_ids] = self.u_target_angle.unsqueeze(-1)
    self.target.write_joint_state_to_sim(target_joint_pos, target_joint_vel)

  def set_solver_goal(
    self,
    face_id: int | torch.Tensor,
    phase: int | torch.Tensor,
    target_angle: float | torch.Tensor,
    target_joint_pos: torch.Tensor,
    *,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Install planned goals for all environments or an explicit subset."""
    if not self.cfg.solver_mode:
      raise RuntimeError("set_solver_goal is only available in solver preview mode")
    ids = (
      torch.arange(self.num_envs, device=self.device)
      if env_ids is None
      else env_ids.to(device=self.device, dtype=torch.long).flatten()
    )
    if ids.numel() == 0:
      return

    def values(value, *, dtype):
      tensor = torch.as_tensor(value, device=self.device, dtype=dtype).flatten()
      if tensor.numel() == 1:
        return tensor.expand(ids.numel())
      if tensor.numel() != ids.numel():
        raise ValueError("Solver goal values must be scalar or match env_ids")
      return tensor

    face_ids = values(face_id, dtype=torch.long)
    phases = values(phase, dtype=torch.long)
    target_angles = values(target_angle, dtype=self.object.data.joint_pos.dtype)
    if bool(((face_ids < 0) | (face_ids >= 6)).any()) or bool(
      ((phases < 0) | (phases > 1)).any()
    ):
      raise ValueError("Solver face ids must be in [0, 5] and phases in {0, 1}")
    target_state = torch.as_tensor(
      target_joint_pos, device=self.device, dtype=self.object.data.joint_pos.dtype
    )
    if target_state.ndim == 1:
      target_state = target_state.unsqueeze(0)
    if target_state.shape[0] == self.num_envs:
      target_state = target_state[ids]
    elif target_state.shape[0] == 1:
      target_state = target_state.expand(ids.numel(), -1)
    elif target_state.shape[0] != ids.numel():
      raise ValueError("Solver target joint state batch must match env_ids")

    self.target_face_id[ids] = face_ids
    self.task_mode[ids] = 0
    self.phase[ids] = phases
    self.root_target_quat_w[ids] = self.object.data.root_link_quat_w[ids]
    self.u_initial_angle[ids] = self.u_angle[ids]
    self.u_target_angle[ids] = torch.where(
      phases == 0, self.u_angle[ids], target_angles
    )
    delta = self.u_initial_angle[ids] - self.u_target_angle[ids]
    self.turn_start_error[ids] = torch.atan2(
      torch.sin(delta), torch.cos(delta)
    ).abs()
    self.flip_start_dot[ids] = self._read_flip_target_dot()[ids]
    self._flip_goal_bottom[ids] = False
    self._turn_goal_180[ids] = self.turn_start_error[ids] > (math.pi * 0.75)
    self._pending_goal_advance[ids] = False
    self._upright_hold[ids] = 0
    self._turn_hold[ids] = 0
    self._yaw_hold[ids] = 0
    self._flip_event[ids] = False
    self._turn_event[ids] = False
    self._yaw_event[ids] = False
    self._previous_phase[ids] = -1
    self._previous_potential[ids] = 0.0
    self._turn_progress_best[ids] = 0.0
    if self._solver_target_joint_pos is None or (
      self._solver_target_joint_pos.shape != self.object.data.joint_pos.shape
    ):
      self._solver_target_joint_pos = self.object.data.joint_pos.clone()
    self._solver_target_joint_pos[ids] = target_state
    self._last_update_step[ids] = -1
    self._target_sync_step = -1
    if self.target is not None:
      self._sync_preview_target()
      self._target_sync_step = int(self._env.common_step_counter)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    env_ids = resolve_env_ids(self._env, env_ids)
    if env_ids.numel() == 0:
      return
    self.task_mode[env_ids] = self._env._rubik_reset_mode[env_ids]
    self.phase[env_ids] = self.task_mode[env_ids]
    self.target_face_id[env_ids] = 4
    self.root_target_quat_w[env_ids] = self._env._rubik_reset_yaw_goal_quat[env_ids]
    self.u_initial_angle[env_ids] = self._env._rubik_reset_initial_angle[env_ids]
    self.u_target_angle[env_ids] = self._env._rubik_reset_goal_angle[env_ids]
    self.flip_start_dot[env_ids] = self._env._rubik_reset_flip_start_dot[env_ids]
    self.turn_start_error[env_ids] = torch.where(
      self._env._rubik_reset_turn180[env_ids], torch.pi, torch.pi / 2.0
    )
    self._flip_goal_bottom[env_ids] = self._env._rubik_reset_bottom[env_ids]
    self._turn_goal_180[env_ids] = self._env._rubik_reset_turn180[env_ids]
    self._pending_goal_advance[env_ids] = False
    self._upright_hold[env_ids] = 0
    self._turn_hold[env_ids] = 0
    self._yaw_hold[env_ids] = 0
    self._flip_event[env_ids] = False
    self._turn_event[env_ids] = False
    self._yaw_event[env_ids] = False
    self._contact_hold_time[env_ids] = 0.0
    self._contact_first_awarded[env_ids] = False
    self._contact_second_awarded[env_ids] = False
    self._approach_best_score[env_ids] = 0.0
    self._approach_initialized[env_ids] = False
    palm_pose = self.robot.data.body_link_pose_w[env_ids, self.palm_body_id]
    cube_pos_palm = quat_apply_inverse(
      palm_pose[:, 3:7], self.object.data.root_link_pos_w[env_ids] - palm_pose[:, :3]
    )
    self._previous_cube_pos_palm[env_ids] = cube_pos_palm
    self.cube_position_delta[env_ids] = 0.0
    self.cube_position_delta_valid[env_ids] = False
    self.smart_move_pending[env_ids] = False
    self._smart_move_state_step[env_ids] = -1
    self.flip_to_top_count[env_ids] = 0.0
    self.turn_success_count[env_ids] = 0.0
    self.yaw_success_count[env_ids] = 0.0
    self.goal_reach_count[env_ids] = 0.0
    self.progress_delta[env_ids] = 0.0
    self.turn_progress_delta[env_ids] = 0.0
    self._turn_progress_best[env_ids] = 0.0
    self.u_up_dot[env_ids] = 0.0
    initial_error = self.u_initial_angle[env_ids] - self.u_target_angle[env_ids]
    self.u_angle_error[env_ids] = torch.atan2(
      torch.sin(initial_error), torch.cos(initial_error)
    ).abs()
    self.yaw_error[env_ids] = torch.where(
      self.task_mode[env_ids] == 2, torch.pi / 12.0, 0.0
    )
    self.tip_contact_count[env_ids] = 0.0
    self._last_update_step[env_ids] = int(self._env.common_step_counter)
    self._previous_phase[env_ids] = -1
    self._previous_potential[env_ids] = 0.0
    self._target_sync_step = -1
    if self.target is not None:
      self._sync_preview_target()
      self._target_sync_step = int(self._env.common_step_counter)

  def evaluate_task(self) -> None:
    self._evaluate_current_step()
    step = int(self._env.common_step_counter)
    if self.target is not None and self._target_sync_step != step:
      self._sync_preview_target()
      self._target_sync_step = step

  def update_smart_move_pending(self) -> None:
    """Track a SmartCube MOVE until continuous physics confirms the turn."""
    state = getattr(self._env, "_smart_cube_state", None)
    if state is None:
      return
    step = int(self._env.common_step_counter)
    fresh = self._smart_move_state_step != step
    if not fresh.any():
      return
    self.smart_move_pending |= state.move_event & fresh
    self.smart_move_pending &= ~self._turn_event
    self._smart_move_state_step[fresh] = step

  def _update_metrics(self) -> None:
    curriculum = getattr(self._env, "_rubik_curriculum", None)
    self.metrics["rubik_task_phase"] = self.phase.float()
    self.metrics["rubik_turn_task_fraction"] = (self.task_mode == 1).float()
    self.metrics["rubik_yaw_task_fraction"] = (self.task_mode == 2).float()
    self.metrics["rubik_u_up_dot"] = self.u_up_dot
    self.metrics["rubik_u_angle_error_rad"] = self.u_angle_error
    self.metrics["rubik_yaw_error_rad"] = self.yaw_error
    self.metrics["rubik_tip_contact_count"] = self.tip_contact_count
    self.metrics["rubik_flip_to_top_count"] = self.flip_to_top_count
    self.metrics["rubik_turn_success_count"] = self.turn_success_count
    self.metrics["rubik_yaw_success_count"] = self.yaw_success_count
    self.metrics["rubik_goal_reach_count"] = self.goal_reach_count
    if curriculum is None:
      self.metrics["rubik_flip_stage"].zero_()
      self.metrics["rubik_turn_stage"].zero_()
    else:
      self.metrics["rubik_flip_stage"] = curriculum["flip_stage"].float().expand(self.num_envs)
      self.metrics["rubik_turn_stage"] = curriculum["turn_stage"].float().expand(self.num_envs)

  def _evaluate_current_step(self) -> None:
    step = int(self._env.common_step_counter)
    fresh = self._last_update_step != step
    if not fresh.any():
      return

    self._flip_event[fresh] = False
    self._turn_event[fresh] = False
    self._yaw_event[fresh] = False
    angle_delta = self.u_angle - self.u_target_angle
    angle_error = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta)).abs()
    upright_dot = self._read_upright_dot()
    flip_dot = self._read_flip_target_dot()
    yaw_error = quat_error_magnitude(
      self.object.data.root_link_quat_w, self.root_target_quat_w
    )
    contact_count = self._read_tip_contact_count()
    cage_drop = self._env.termination_manager.get_term("cage_drop")
    contact = (contact_count >= self.cfg.min_contact_fingertips) & ~cage_drop
    palm_pose = self.robot.data.body_link_pose_w[:, self.palm_body_id]
    cube_pos_palm = quat_apply_inverse(
      palm_pose[:, 3:7], self.object.data.root_link_pos_w - palm_pose[:, :3]
    )
    position_delta = (cube_pos_palm - self._previous_cube_pos_palm).norm(dim=-1)
    was_position_initialized = self.cube_position_delta_valid.clone()
    self.cube_position_delta = torch.where(
      fresh & was_position_initialized, position_delta, self.cube_position_delta
    )
    self.cube_position_delta_valid = torch.where(
      fresh, torch.ones_like(self.cube_position_delta_valid), self.cube_position_delta_valid
    )
    self._previous_cube_pos_palm = torch.where(
      fresh.unsqueeze(-1), cube_pos_palm, self._previous_cube_pos_palm
    )
    root_speed = self.object.data.root_link_vel_w[:, 3:6].norm(dim=-1)
    flip_episode = getattr(self, "episode_kind", None)
    flip_episode = (
      flip_episode is not None
      and flip_episode == getattr(self.cfg, "continuous_flip_task_kind", -1)
    )
    upright_tolerance = getattr(self.cfg, "continuous_flip_upright_tolerance_rad", self.cfg.upright_tolerance_rad)
    flip_angle_tolerance = getattr(self.cfg, "continuous_flip_angle_tolerance_rad", self.cfg.flip_angle_tolerance_rad)
    flip_hold_steps = getattr(self.cfg, "continuous_flip_hold_steps", self.cfg.upright_hold_steps)
    upright_tolerance = torch.where(
      flip_episode,
      torch.as_tensor(upright_tolerance, device=self.device, dtype=upright_dot.dtype),
      torch.as_tensor(self.cfg.upright_tolerance_rad, device=self.device, dtype=upright_dot.dtype),
    )
    flip_angle_tolerance = torch.where(
      flip_episode,
      torch.as_tensor(flip_angle_tolerance, device=self.device, dtype=angle_error.dtype),
      torch.as_tensor(self.cfg.flip_angle_tolerance_rad, device=self.device, dtype=angle_error.dtype),
    )
    flip_hold_steps = torch.where(
      flip_episode,
      torch.as_tensor(flip_hold_steps, device=self.device),
      torch.as_tensor(self.cfg.upright_hold_steps, device=self.device),
    )
    upright_ok = upright_dot >= torch.cos(upright_tolerance)

    potential = self._potential(
      self.phase, flip_dot, upright_dot, angle_error, yaw_error
    )
    delta = (potential - self._previous_potential).clamp(-0.1, 0.1)
    self.progress_delta = torch.where(
      fresh & (self.phase == self._previous_phase) & contact,
      delta,
      torch.zeros_like(delta),
    )
    turn_improvement, turn_best = bounded_u_turn_progress(
      angle_error,
      self.turn_start_error,
      self._turn_progress_best,
      self.phase,
      contact,
      upright_ok,
      same_goal=fresh & (self.phase == self._previous_phase),
    )
    self.turn_progress_delta = torch.where(
      fresh, turn_improvement, self.turn_progress_delta
    )
    self._turn_progress_best = torch.where(fresh, turn_best, self._turn_progress_best)

    flip_valid = (
      contact & (flip_dot >= torch.cos(upright_tolerance))
      & (angle_error <= flip_angle_tolerance)
      & (root_speed <= self.cfg.max_flip_root_speed)
    )
    if getattr(self.cfg, 'flip_specialization', False):
      # A selected driver may be20 degrees off: geometric arrival must still
      # start the alignment window, rather than waiting for internal alignment.
      flip_valid = (contact & (flip_dot >= torch.cos(upright_tolerance))
                    & (root_speed <= self.cfg.max_flip_root_speed))
    self._upright_hold = torch.where(
      fresh & (self.phase == 0),
      torch.where(flip_valid, self._upright_hold + 1, 0),
      self._upright_hold,
    )
    flip_event = fresh & (self.phase == 0) & (
      self._upright_hold >= flip_hold_steps
    )

    turn_valid = (
      contact & upright_ok & (angle_error <= self.cfg.turn_tolerance_rad)
      & (self.u_angle_velocity.abs() <= self.cfg.max_u_face_speed)
    )
    self._turn_hold = torch.where(
      fresh & (self.phase == 1),
      torch.where(turn_valid, self._turn_hold + 1, 0),
      self._turn_hold,
    )
    turn_event = fresh & (self.phase == 1) & (
      self._turn_hold >= self.cfg.turn_hold_steps
    )

    yaw_valid = (
      contact & upright_ok & (yaw_error <= self.cfg.yaw_tolerance_rad)
      & (angle_error <= self.cfg.flip_angle_tolerance_rad)
      & (root_speed <= self.cfg.max_yaw_root_speed)
    )
    self._yaw_hold = torch.where(
      fresh & (self.phase == 2),
      torch.where(yaw_valid, self._yaw_hold + 1, 0),
      self._yaw_hold,
    )
    yaw_event = fresh & (self.phase == 2) & (
      self._yaw_hold >= self.cfg.yaw_hold_steps
    )

    self._flip_event |= flip_event
    self._turn_event |= turn_event
    self._yaw_event |= yaw_event
    self._pending_goal_advance |= flip_event | turn_event | yaw_event
    self.flip_to_top_count += flip_event.float()
    self.turn_success_count += turn_event.float()
    self.yaw_success_count += yaw_event.float()
    self.goal_reach_count += (flip_event | turn_event | yaw_event).float()

    if not self.cfg.solver_mode:
      curriculum = self._env._rubik_curriculum
      curriculum["side_success"] += (flip_event & ~self._flip_goal_bottom).sum()
      curriculum["bottom_success"] += (flip_event & self._flip_goal_bottom).sum()
      curriculum["turn90_success"] += (turn_event & ~self._turn_goal_180).sum()
      curriculum["turn180_success"] += (turn_event & self._turn_goal_180).sum()
      curriculum["yaw_success"] += yaw_event.sum()
      self._promote_curriculum()

    self.u_up_dot = torch.where(fresh, upright_dot, self.u_up_dot)
    self.u_angle_error = torch.where(fresh, angle_error, self.u_angle_error)
    self.yaw_error = torch.where(fresh, yaw_error, self.yaw_error)
    self.tip_contact_count = torch.where(fresh, contact_count, self.tip_contact_count)
    self._last_update_step[fresh] = step
    self._previous_phase = torch.where(fresh, self.phase, self._previous_phase)
    self._previous_potential = torch.where(fresh, potential, self._previous_potential)

  def _promote_curriculum(self) -> None:
    state = self._env._rubik_curriculum
    side_rate = state["side_success"].float() / state["side_trials"].clamp_min(1)
    bottom_rate = state["bottom_success"].float() / state["bottom_trials"].clamp_min(1)
    turn90_rate = state["turn90_success"].float() / state["turn90_trials"].clamp_min(1)
    turn180_rate = state["turn180_success"].float() / state["turn180_trials"].clamp_min(1)
    flip_stage = state["flip_stage"]
    turn_stage = state["turn_stage"]
    flip_stage = torch.where(
      (flip_stage == 0) & (state["side_trials"] >= 256) & (side_rate >= 0.70),
      1, flip_stage,
    )
    state["flip_stage"] = torch.where(
      (flip_stage == 1) & (state["bottom_trials"] >= 64)
      & (bottom_rate >= 0.60),
      2, flip_stage,
    )
    turn_stage = torch.where(
      (turn_stage == 0) & (state["turn90_trials"] >= 256)
      & (turn90_rate >= 0.70),
      1, turn_stage,
    )
    state["turn_stage"] = torch.where(
      (turn_stage == 1) & (state["turn180_trials"] >= 64)
      & (turn180_rate >= 0.60),
      2, turn_stage,
    )

  def _sample_next_flip_goal(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    current = self._read_top_face_id()[env_ids]
    adjacent = sample_adjacent_face(current)
    opposite = opposite_face_ids(current)
    stage = self._env._rubik_curriculum["flip_stage"]
    opposite_probability = torch.where(
      stage == 0, 0.0, torch.where(stage == 1, 0.20, 0.40)
    )
    choose_opposite = torch.rand(env_ids.numel(), device=self.device) < opposite_probability
    self.target_face_id[env_ids] = torch.where(choose_opposite, opposite, adjacent)
    self._flip_goal_bottom[env_ids] = choose_opposite
    self.flip_start_dot[env_ids] = self._read_flip_target_dot()[env_ids]
    self.u_target_angle[env_ids] = self.u_angle[env_ids]
    state = self._env._rubik_curriculum
    state["side_trials"] += (~choose_opposite).sum()
    state["bottom_trials"] += choose_opposite.sum()

  def _sample_next_turn_goal(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    stage = self._env._rubik_curriculum["turn_stage"]
    probability = torch.where(stage == 0, 0.0, torch.where(stage == 1, 0.15, 0.40))
    override = getattr(self.cfg, 'turn_180_probability', None)
    if override is not None:
      if not 0 <= override <= 1:
        raise ValueError('turn_180_probability must be in [0, 1]')
      probability = override
    use_180 = torch.rand(env_ids.numel(), device=self.device) < probability
    angle_step = torch.where(use_180, torch.pi, torch.pi / 2.0)
    sign = torch.where(torch.rand(env_ids.numel(), device=self.device) < 0.5, -1.0, 1.0)
    current_angle = self.u_angle[env_ids]
    if getattr(self.cfg, 'turn_specialization', False):
      # Repeated turns aim at detents rather than accumulating a 5-degree offset.
      current_angle = torch.round(current_angle / (math.pi / 2)) * (math.pi / 2)
    next_angle = current_angle + sign * angle_step
    self.u_target_angle[env_ids] = torch.atan2(torch.sin(next_angle), torch.cos(next_angle))
    self.turn_start_error[env_ids] = angle_step
    self._turn_goal_180[env_ids] = use_180
    state = self._env._rubik_curriculum
    state["turn90_trials"] += (~use_180).sum()
    state["turn180_trials"] += use_180.sum()

  def _sample_next_yaw_goal(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    sign = torch.where(torch.rand(env_ids.numel(), device=self.device) < 0.5, -1.0, 1.0)
    half_angle = sign * (torch.pi / 24.0)
    normal = self._palm_normal_w()[env_ids]
    delta = torch.cat(
      (torch.cos(half_angle).unsqueeze(-1), normal * torch.sin(half_angle).unsqueeze(-1)),
      dim=-1,
    )
    self.root_target_quat_w[env_ids] = quat_mul(
      delta, self.object.data.root_link_quat_w[env_ids]
    )
    self.u_target_angle[env_ids] = self.u_angle[env_ids]
    self._env._rubik_curriculum["yaw_trials"] += env_ids.numel()

  def _advance_pending_goals(self) -> None:
    if self.cfg.solver_mode:
      self._pending_goal_advance.zero_()
      return
    env_ids = self._pending_goal_advance.nonzero().flatten()
    if env_ids.numel() == 0:
      return
    flip_ids = env_ids[self.phase[env_ids] == 0]
    turn_ids = env_ids[self.phase[env_ids] == 1]
    yaw_ids = env_ids[self.phase[env_ids] == 2]
    combine = int(self._env.common_step_counter) >= self.cfg.combine_after_steps
    combine_ids = flip_ids if combine else flip_ids[:0]
    continuing_flip_ids = flip_ids[:0] if combine else flip_ids
    if combine_ids.numel():
      self.phase[combine_ids] = 1
      self._sample_next_turn_goal(combine_ids)
    self._sample_next_flip_goal(continuing_flip_ids)
    self._sample_next_turn_goal(turn_ids)
    self._sample_next_yaw_goal(yaw_ids)
    self._upright_hold[env_ids] = 0
    self._turn_hold[env_ids] = 0
    self._yaw_hold[env_ids] = 0
    self._previous_phase[env_ids] = -1
    self._previous_potential[env_ids] = 0.0
    self._turn_progress_best[env_ids] = 0.0
    self._pending_goal_advance[env_ids] = False
    self._target_sync_step = -1
    if self.target is not None:
      self._sync_preview_target()
      self._target_sync_step = int(self._env.common_step_counter)

  def _update_command(self) -> None:
    self._evaluate_current_step()
    self._advance_pending_goals()


@dataclass(kw_only=True)
class RubikUFaceCommandCfg(CommandTermCfg):
  entity_name: str
  debug_vis: bool = False
  target_entity_name: str = ""
  target_offset_palm_m: tuple[float, float, float] = (0.09, 0.0, 0.0)
  target_fixed_pose: bool = False
  robot_entity_name: str = "robot"
  palm_body_pattern: str = ".*_palm_link"
  u_face_joint_name: str = "cubelet:driver:pos_z"
  tip_contact_sensor_name: str = "tip_object_contact"
  min_contact_fingertips: int = 1
  approach_fingertip_count: int = 4
  upright_tolerance_rad: float = 0.20
  upright_hold_steps: int = 8
  flip_angle_tolerance_rad: float = 0.1745329252
  turn_tolerance_rad: float = 0.0872664626
  turn_hold_steps: int = 8
  max_u_face_speed: float = 0.45
  yaw_tolerance_rad: float = 0.0523598776
  yaw_hold_steps: int = 8
  max_flip_root_speed: float = 0.8
  max_yaw_root_speed: float = 0.45
  combine_after_steps: int = 160_000
  solver_mode: bool = False

  def build(self, env) -> RubikUFaceCommand:
    return RubikUFaceCommand(self, env)
