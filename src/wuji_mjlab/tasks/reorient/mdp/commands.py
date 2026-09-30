# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_error_magnitude, quat_mul

from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
)
from wuji_mjlab.tasks.reorient.mdp.cube_faces import (
  FACE_NORMALS,
  face_id_from_quat,
  sample_adjacent_face,
  sample_curriculum_face,
  sample_other_face,
  select_min_rotation_target,
)
from wuji_mjlab.tasks.reorient.mdp.flip_curriculum import stage_parameters
from wuji_mjlab.tasks.reorient.reorient_constants import (
  TAG_IN_PALM_QUAT_WXYZ as _TAG_IN_PALM_QUAT_WXYZ,
)
from wuji_mjlab.utils.math import random_quat_uniform

if TYPE_CHECKING:
  import viser


class InHandReorientCommand(CommandTerm):
  """SO(3) reorientation command with hold-and-advance goal progression."""

  cfg: InHandReorientCommandCfg

  def __init__(self, cfg: InHandReorientCommandCfg, env):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.robot: Entity = env.scene[cfg.robot_entity_name]
    palm_ids, _ = self.robot.find_bodies(cfg.palm_body_pattern)
    if not palm_ids:
      raise ValueError(
        f"InHandReorientCommand: no palm body matched pattern "
        f"'{cfg.palm_body_pattern}' on entity '{cfg.robot_entity_name}'."
      )
    self.palm_body_id = int(palm_ids[0])

    self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.goal_quat_w[:, 0] = 1.0
    self._rotation_mode = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._active_curriculum_stage = torch.full(
      (self.num_envs,),
      int(cfg.curriculum_stage),
      dtype=torch.int32,
      device=self.device,
    )

    self.hold_counter = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self.goal_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    self.goal_reach_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._rotation_goal_reach_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._flip_success_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    # Two-phase state machine
    self.in_success_window = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.window_timer = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    # Pre-reset snapshot of window_timer for reward computation
    self._reward_window_timer = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    # Reward hold counter: only increments when in_success_window AND within_threshold
    self.reward_hold_counter = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._reward_hold_counter_snapshot = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )

    # One-shot event flags (overwritten every step in _update_command)
    self._success_achieved = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._goal_switched = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._within_threshold = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._success_valid = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._flip_completed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._tip_contact_count = torch.zeros(self.num_envs, device=self.device)

    # Per-goal shaping state. These are reset for the relevant envs on every
    # goal change; episode-scoped counters are reset only by _resample_command.
    self._goal_initial_error = torch.zeros(self.num_envs, device=self.device)
    self._goal_progress_best = torch.zeros(self.num_envs, device=self.device)
    self._approach_best_score = torch.zeros(
      (self.num_envs, self.cfg.approach_fingertip_count), device=self.device
    )
    self._approach_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._contact_hold_time = torch.zeros((self.num_envs, 2), device=self.device)
    self._task_evaluated_step = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._task_reset_step = self._task_evaluated_step.clone()
    self._pending_rotation_advance = torch.zeros_like(self._rotation_mode)
    self._pending_goal_advance = torch.zeros_like(self._rotation_mode)
    self._current_face_id = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._target_face_id = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._consecutive_flip_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._face_visited_mask = torch.zeros(
      self.num_envs, dtype=torch.int64, device=self.device
    )
    self._adjacent_flip_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._opposite_flip_count = torch.zeros(
      self.num_envs, dtype=torch.int32, device=self.device
    )
    self._flip_target_angle = torch.zeros(self.num_envs, device=self.device)
    self._task_reward_cache: dict[str, tuple[int, torch.Tensor]] = {}
    self._contact_first_awarded = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._contact_second_awarded = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.metrics["goal_reach_count"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["rotation_goal_reach_count"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["flip_success_count"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["ori_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["hold_counter"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_timer"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["in_success_window"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["window_timer"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["tip_contact_count"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["success_valid"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["flip_completed"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["consecutive_flip_count"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["face_coverage_count"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["adjacent_flip_count"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["opposite_flip_count"] = torch.zeros(
      self.num_envs, device=self.device
    )

    self._visualization = ReorientCommandVisualization(entity_name=self.cfg.entity_name)

  @property
  def task_reward_active(self) -> torch.Tensor:
    if not self.cfg.synchronized_contact_task:
      return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    return self._task_reset_step != self._env.common_step_counter

  @property
  def active_curriculum_stage(self) -> torch.Tensor:
    """Current task-difficulty stage, one value per environment."""
    return self._active_curriculum_stage

  def _continuous_task_mask(self) -> torch.Tensor:
    """Return which environments advance to another goal after success."""
    if self.cfg.goal_sampling_mode == "continuous_yaw_or_random_face":
      return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    if self.cfg.goal_sampling_mode != "curriculum_flip":
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    values = torch.tensor(
      [stage_parameters(stage).continuous for stage in range(4)],
      device=self.device,
      dtype=torch.bool,
    )
    return values[self._active_curriculum_stage.long()]

  def _flip_termination_mask(self) -> torch.Tensor:
    """Return which curriculum environments end after a valid flip."""
    if self.cfg.goal_sampling_mode != "curriculum_flip":
      return ~self._continuous_task_mask()
    values = torch.tensor(
      [stage_parameters(stage).flip_terminates for stage in range(4)],
      device=self.device,
      dtype=torch.bool,
    )
    return values[self._active_curriculum_stage.long()]

  def _sample_curriculum_target_face(
    self, current_face_id: torch.Tensor, stage: int
  ) -> torch.Tensor:
    """Sample the next face using the active stage's difficulty contract."""
    stage_cfg = stage_parameters(stage)
    if stage_cfg.target_sampling_mode == "adjacent_only":
      return sample_adjacent_face(current_face_id)
    if stage_cfg.target_sampling_mode == "all_other":
      return sample_other_face(current_face_id)
    raise ValueError(
      "A fixed A→B curriculum stage does not sample a target face; "
      f"got target_sampling_mode={stage_cfg.target_sampling_mode!r}."
    )

  @property
  def success_threshold(self) -> torch.Tensor:
    """Active orientation tolerance, including staged curriculum overrides."""
    if not self.cfg.curriculum_enabled:
      return torch.full(
        (self.num_envs,),
        float(self.cfg.success_threshold),
        device=self.device,
        dtype=self.goal_quat_w.dtype,
      )
    values = torch.tensor(
      [stage_parameters(int(stage)).success_threshold_rad for stage in range(4)],
      device=self.device,
      dtype=self.goal_quat_w.dtype,
    )
    return values[self._active_curriculum_stage.long()]

  @property
  def success_hold_steps(self) -> torch.Tensor:
    """Active number of valid contact steps required for success."""
    if not self.cfg.curriculum_enabled:
      return torch.full(
        (self.num_envs,),
        int(self.cfg.success_hold_steps),
        device=self.device,
        dtype=torch.int32,
      )
    values = torch.tensor(
      [stage_parameters(int(stage)).hold_steps for stage in range(4)],
      device=self.device,
      dtype=torch.int32,
    )
    return values[self._active_curriculum_stage.long()]

  @property
  def command(self) -> torch.Tensor:
    return self.goal_quat_w

  @property
  def goal_quat(self) -> torch.Tensor:
    return self.goal_quat_w

  def _update_metrics(self) -> None:
    self.metrics["goal_reach_count"] = self.goal_reach_count.float()
    self.metrics["rotation_goal_reach_count"] = self._rotation_goal_reach_count.float()
    self.metrics["flip_success_count"] = self._flip_success_count.float()
    ori_err = quat_error_magnitude(self.object.data.root_link_quat_w, self.goal_quat_w)
    self.metrics["ori_error"] = ori_err
    self.metrics["hold_counter"] = self.hold_counter.float()
    self.metrics["goal_timer"] = self.goal_timer.float()
    self.metrics["in_success_window"] = self.in_success_window.float()
    self.metrics["window_timer"] = self.window_timer.float()
    self.metrics["tip_contact_count"] = self._tip_contact_count
    self.metrics["success_valid"] = self._success_valid.float()
    self.metrics["flip_completed"] = self._flip_completed.float()
    self.metrics["consecutive_flip_count"] = self._consecutive_flip_count.float()
    self.metrics["face_coverage_count"] = self.face_coverage_count.float()
    self.metrics["adjacent_flip_count"] = self._adjacent_flip_count.float()
    self.metrics["opposite_flip_count"] = self._opposite_flip_count.float()
    self.metrics["flip_target_angle"] = self._flip_target_angle
    self._visualization.update_status_gui(
      num_envs=self.num_envs,
      policy_status_for_env=self._policy_status_for_env,
    )

  def _sample_goal_in_world(self, env_ids: torch.Tensor) -> None:
    """Sample new goal_quat_w for the given envs.

    The legacy ``uniform_so3`` mode samples in tag frame to match deploy
    semantics. The Luwu modes compose their relative rotations with the cube
    reset orientation, so "face A remains up" is defined against the actual
    reset pose rather than the tag's arbitrary base rotation.
    """
    n = len(env_ids)
    if self.cfg.goal_sampling_mode == "uniform_so3":
      tag_rel_goal = random_quat_uniform(n, self.device)
      self._rotation_mode[env_ids] = False
    elif self.cfg.goal_sampling_mode == "continuous_yaw_or_random_face":
      dtype = self.robot.data.body_link_pose_w.dtype
      reset_quat_w = self.object.data.default_root_state[env_ids, 3:7]
      yaw_mask = torch.rand(n, device=self.device) < self.cfg.yaw_goal_probability
      current_face_id = face_id_from_quat(reset_quat_w)
      stage = int(self._active_curriculum_stage[env_ids[0]].item())
      if stage >= 3:
        flip_face_id = sample_other_face(current_face_id)
      else:
        flip_face_id = sample_curriculum_face(
          current_face_id,
          stage=stage,
          adjacent_probability=stage_parameters(stage).adjacent_probability,
        )
      rotation_goal = quat_mul(
        reset_quat_w,
        self._rotation_step_quat(dtype).expand(n, 4),
      )
      flip_goal = select_min_rotation_target(reset_quat_w, flip_face_id)
      self._rotation_mode[env_ids] = yaw_mask
      self._current_face_id[env_ids] = current_face_id
      self._face_visited_mask[env_ids] |= (
        torch.ones_like(current_face_id, dtype=torch.int64) << current_face_id
      )
      self._target_face_id[env_ids] = torch.where(
        yaw_mask, current_face_id, flip_face_id
      )
      normals = FACE_NORMALS.to(device=self.device, dtype=dtype)
      flip_angle = torch.acos(
        (normals[current_face_id] * normals[flip_face_id]).sum(dim=-1).clamp(-1.0, 1.0)
      )
      self._flip_target_angle[env_ids] = torch.where(
        yaw_mask,
        torch.zeros_like(flip_angle),
        flip_angle,
      )
      self.goal_quat_w[env_ids] = torch.where(
        yaw_mask.unsqueeze(-1), rotation_goal, flip_goal
      )
      self._initialize_goal_progress(env_ids)
      return
    elif self.cfg.goal_sampling_mode == "curriculum_flip":
      dtype = self.robot.data.body_link_pose_w.dtype
      reset_quat_w = self.object.data.default_root_state[env_ids, 3:7]
      stage = int(self._active_curriculum_stage[env_ids[0]].item())
      stage_cfg = stage_parameters(stage)
      current_face_id = face_id_from_quat(reset_quat_w)
      if stage_cfg.target_sampling_mode == "fixed_ab":
        yaw_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        flip_quat = torch.tensor(
          self.cfg.flip_goal_quat_wxyz,
          device=self.device,
          dtype=dtype,
        ).expand(n, 4)
        flip_goal = quat_mul(reset_quat_w, flip_quat)
        flip_face_id = face_id_from_quat(flip_goal)
        goal_quat = flip_goal
      else:
        yaw_mask = torch.rand(n, device=self.device) < stage_cfg.yaw_goal_probability
        flip_face_id = self._sample_curriculum_target_face(current_face_id, stage)
        rotation_goal = quat_mul(
          reset_quat_w,
          self._rotation_step_quat(dtype).expand(n, 4),
        )
        flip_goal = select_min_rotation_target(reset_quat_w, flip_face_id)
        goal_quat = torch.where(
          yaw_mask.unsqueeze(-1), rotation_goal, flip_goal
        )
      self._rotation_mode[env_ids] = yaw_mask
      self._current_face_id[env_ids] = current_face_id
      self._face_visited_mask[env_ids] |= (
        torch.ones_like(current_face_id, dtype=torch.int64) << current_face_id
      )
      self._target_face_id[env_ids] = torch.where(
        yaw_mask, current_face_id, flip_face_id
      )
      normals = FACE_NORMALS.to(device=self.device, dtype=dtype)
      flip_angle = torch.acos(
        (normals[current_face_id] * normals[flip_face_id])
        .sum(dim=-1)
        .clamp(-1.0, 1.0)
      )
      self._flip_target_angle[env_ids] = torch.where(
        yaw_mask,
        torch.zeros_like(flip_angle),
        flip_angle,
      )
      self.goal_quat_w[env_ids] = goal_quat
      self._initialize_goal_progress(env_ids)
      return
    elif self.cfg.goal_sampling_mode in {"yaw_or_flip", "incremental_yaw_or_flip"}:
      dtype = self.robot.data.body_link_pose_w.dtype
      if self.cfg.goal_sampling_mode == "incremental_yaw_or_flip":
        yaw_quat = self._rotation_step_quat(dtype).expand(n, 4)
        yaw_mask = torch.rand(n, device=self.device) < (
          self.cfg.yaw_goal_probability
        )
        self._rotation_mode[env_ids] = yaw_mask
      else:
        yaw_quat = torch.tensor(
          self.cfg.yaw_goal_quat_wxyz, device=self.device, dtype=dtype
        ).expand(n, 4)
        yaw_mask = torch.rand(n, device=self.device) < self.cfg.yaw_goal_probability
        self._rotation_mode[env_ids] = False
      flip_quat = torch.tensor(
        self.cfg.flip_goal_quat_wxyz,
        device=self.device,
        dtype=dtype,
      ).expand(n, 4)
      tag_rel_goal = torch.where(
        yaw_mask.unsqueeze(-1), yaw_quat, flip_quat
      )
      reset_quat_w = self.object.data.default_root_state[env_ids, 3:7]
      self.goal_quat_w[env_ids] = quat_mul(reset_quat_w, tag_rel_goal)
      self._initialize_goal_progress(env_ids)
      return
    else:
      raise ValueError(
        f"Unsupported goal_sampling_mode={self.cfg.goal_sampling_mode!r}"
      )
    palm_quat_w = self.robot.data.body_link_pose_w[env_ids, self.palm_body_id, 3:7]
    tag_in_palm = torch.tensor(
      _TAG_IN_PALM_QUAT_WXYZ, device=self.device, dtype=palm_quat_w.dtype
    ).expand(n, 4)
    tag_quat_w = quat_mul(palm_quat_w, tag_in_palm)
    self.goal_quat_w[env_ids] = quat_mul(tag_quat_w, tag_rel_goal)
    self._initialize_goal_progress(env_ids)

  def _rotation_step_quat(self, dtype: torch.dtype) -> torch.Tensor:
    half_angle = math.radians(self.cfg.rotation_step_degrees) / 2.0
    return torch.tensor(
      [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
      device=self.device,
      dtype=dtype,
    )

  def _advance_rotation_goals(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    step_quat = self._rotation_step_quat(self.goal_quat_w.dtype).expand(
      env_ids.numel(), 4
    )
    self.goal_quat_w[env_ids] = quat_mul(self.goal_quat_w[env_ids], step_quat)
    self._initialize_goal_progress(env_ids)

  def _advance_continuous_goals(self, env_ids: torch.Tensor) -> None:
    """Advance a continuous episode without resetting the object or robot."""
    if env_ids.numel() == 0:
      return
    dtype = self.goal_quat_w.dtype
    current_quat = self.object.data.root_link_quat_w[env_ids]
    current_face_id = face_id_from_quat(current_quat)
    rotation_mask = self._rotation_mode[env_ids]
    stage = int(self._active_curriculum_stage[env_ids[0]].item())
    flip_face_id = self._sample_curriculum_target_face(current_face_id, stage)
    rotation_goal = quat_mul(
      self.goal_quat_w[env_ids],
      self._rotation_step_quat(dtype).expand(env_ids.numel(), 4),
    )
    flip_goal = select_min_rotation_target(current_quat, flip_face_id)
    self._current_face_id[env_ids] = current_face_id
    self._target_face_id[env_ids] = torch.where(
      rotation_mask, current_face_id, flip_face_id
    )
    normals = FACE_NORMALS.to(device=self.device, dtype=dtype)
    flip_angle = torch.acos(
      (normals[current_face_id] * normals[flip_face_id]).sum(dim=-1).clamp(-1.0, 1.0)
    )
    self._flip_target_angle[env_ids] = torch.where(
      rotation_mask,
      torch.zeros_like(flip_angle),
      flip_angle,
    )
    self.goal_quat_w[env_ids] = torch.where(
      rotation_mask.unsqueeze(-1), rotation_goal, flip_goal
    )
    self._initialize_goal_progress(env_ids)

  def _record_continuous_flip(self, flip_event: torch.Tensor) -> None:
    """Record a newly completed flip without affecting episode termination."""
    valid = (
      flip_event
      & self._continuous_task_mask()
      & (self._target_face_id >= 0)
      & (self._current_face_id >= 0)
    )
    target = self._target_face_id.clamp(0, 5)
    current = self._current_face_id.clamp(0, 5)
    bit = torch.ones_like(target, dtype=torch.int64) << target
    self._consecutive_flip_count += valid.int()
    self._face_visited_mask |= torch.where(
      valid, bit, torch.zeros_like(self._face_visited_mask)
    )
    normals = FACE_NORMALS.to(device=self.device, dtype=self.goal_quat_w.dtype)
    dot = (normals[current] * normals[target]).sum(dim=-1)
    self._adjacent_flip_count += (valid & (dot == 0.0)).int()
    self._opposite_flip_count += (valid & (dot == -1.0)).int()

  def _initialize_goal_progress(self, env_ids: torch.Tensor) -> None:
    # Reset events have written qpos, but forward() has not refreshed xquat.
    # Read the free-joint state directly for Luwu, including randomized resets.
    quat = self.object.data.root_link_quat_w
    if self.cfg.synchronized_contact_task:
      quat = self._env.sim.data.qpos[:, self.object.indexing.free_joint_q_adr[3:7]]
    ori_error = quat_error_magnitude(
      quat[env_ids], self.goal_quat_w[env_ids]
    )
    self._goal_initial_error[env_ids] = ori_error
    self._goal_progress_best[env_ids] = 0.0

  def _reset_episode_task_state(self, env_ids: torch.Tensor) -> None:
    self.goal_reach_count[env_ids] = 0
    self._rotation_goal_reach_count[env_ids] = 0
    self._flip_success_count[env_ids] = 0
    self._flip_completed[env_ids] = False
    self._approach_best_score[env_ids] = 0.0
    self._approach_initialized[env_ids] = False
    self._contact_hold_time[env_ids] = 0.0
    self._contact_first_awarded[env_ids] = False
    self._contact_second_awarded[env_ids] = False
    self._pending_rotation_advance[env_ids] = False
    self._pending_goal_advance[env_ids] = False
    self._current_face_id[env_ids] = -1
    self._target_face_id[env_ids] = -1
    self._flip_target_angle[env_ids] = 0.0
    self._consecutive_flip_count[env_ids] = 0
    self._face_visited_mask[env_ids] = 0
    self._adjacent_flip_count[env_ids] = 0
    self._opposite_flip_count[env_ids] = 0
    self._task_evaluated_step[env_ids] = getattr(self._env, "common_step_counter", 0)
    self._task_reset_step[env_ids] = self._task_evaluated_step[env_ids]
    for _, value in self._task_reward_cache.values():
      value[env_ids] = 0

  def _reset_goal_progress(self, env_ids: torch.Tensor) -> None:
    self.hold_counter[env_ids] = 0
    self.goal_timer[env_ids] = 0
    self.in_success_window[env_ids] = False
    self.window_timer[env_ids] = 0
    self._reward_window_timer[env_ids] = 0
    self.reward_hold_counter[env_ids] = 0
    self._reward_hold_counter_snapshot[env_ids] = 0
    self._success_achieved[env_ids] = False
    self._goal_switched[env_ids] = False
    self._within_threshold[env_ids] = False
    self._success_valid[env_ids] = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._reset_episode_task_state(env_ids)
    self._reset_goal_progress(env_ids)
    if self.cfg.curriculum_enabled:
      stage = int(getattr(self._env, "_flip_curriculum_stage", self.cfg.curriculum_stage))
      stage = min(max(stage, 0), 3)
    else:
      stage = int(self.cfg.curriculum_stage)
    self._active_curriculum_stage[env_ids] = stage
    self._sample_goal_in_world(env_ids)

  def preview_resample_goals(self, env_ids: torch.Tensor) -> None:
    """Resample goals for the local preview without changing training logic.

    The normal training path advances a goal only after the policy reaches and
    holds it. The zero/random preview policies are deliberately not expected
    to succeed, so the viewer uses this explicit seam to demonstrate the
    target cube changing over simulation time.
    """
    self._resample_command(env_ids)

  def preview_set_fixed_task_modes(
    self,
    rotation_env_ids: torch.Tensor,
    flip_env_ids: torch.Tensor,
  ) -> None:
    """Assign fixed rotation/flip goals for the local preview only.

    Goals are composed from the cube pose that exists immediately after a
    reset. This keeps the preview groups deterministic without changing the
    training-time sampler or episode state machine.
    """
    if rotation_env_ids.numel():
      self._preview_set_relative_goal(rotation_env_ids, rotation=True)
    if flip_env_ids.numel():
      self._preview_set_relative_goal(flip_env_ids, rotation=False)

  def _preview_set_relative_goal(
    self, env_ids: torch.Tensor, *, rotation: bool
  ) -> None:
    current_quat_w = self.object.data.root_link_quat_w[env_ids].clone()
    current_face_id = face_id_from_quat(current_quat_w)
    self._rotation_mode[env_ids] = rotation
    self._reset_episode_task_state(env_ids)
    self._reset_goal_progress(env_ids)
    if rotation:
      relative_goal = self._rotation_step_quat(current_quat_w.dtype).expand(
        env_ids.numel(), 4
      )
      target_face_id = current_face_id
      goal_quat_w = quat_mul(current_quat_w, relative_goal)
    elif self.cfg.goal_sampling_mode in {
      "continuous_yaw_or_random_face",
      "curriculum_flip",
    }:
      stage = int(self._active_curriculum_stage[env_ids[0]].item())
      stage_cfg = stage_parameters(stage)
      if (
        self.cfg.goal_sampling_mode == "curriculum_flip"
        and stage_cfg.target_sampling_mode == "fixed_ab"
      ):
        relative_goal = torch.tensor(
          self.cfg.flip_goal_quat_wxyz,
          device=self.device,
          dtype=current_quat_w.dtype,
        ).expand(env_ids.numel(), 4)
        goal_quat_w = quat_mul(current_quat_w, relative_goal)
        target_face_id = face_id_from_quat(goal_quat_w)
      else:
        if self.cfg.goal_sampling_mode == "curriculum_flip":
          target_face_id = self._sample_curriculum_target_face(
            current_face_id, stage
          )
        else:
          target_face_id = sample_curriculum_face(
            current_face_id,
            stage=stage,
            adjacent_probability=stage_cfg.adjacent_probability,
          )
        goal_quat_w = select_min_rotation_target(current_quat_w, target_face_id)
    else:
      relative_goal = torch.tensor(
        self.cfg.flip_goal_quat_wxyz,
        device=self.device,
        dtype=current_quat_w.dtype,
      ).expand(env_ids.numel(), 4)
      target_face_id = current_face_id
      goal_quat_w = quat_mul(current_quat_w, relative_goal)
    self._current_face_id[env_ids] = current_face_id
    self._target_face_id[env_ids] = target_face_id
    normals = FACE_NORMALS.to(device=self.device, dtype=current_quat_w.dtype)
    self._flip_target_angle[env_ids] = torch.where(
      torch.full((env_ids.numel(),), rotation, device=self.device, dtype=torch.bool),
      torch.zeros(env_ids.numel(), device=self.device, dtype=current_quat_w.dtype),
      torch.acos(
        (normals[current_face_id] * normals[target_face_id]).sum(dim=-1).clamp(-1.0, 1.0)
      ),
    )
    self._face_visited_mask[env_ids] |= (
      torch.ones_like(current_face_id, dtype=torch.int64) << current_face_id
    )
    self.goal_quat_w[env_ids] = goal_quat_w
    self._initialize_goal_progress(env_ids)

  def preview_set_rotation_mode(self, env_ids: torch.Tensor) -> None:
    """Reset preview goals to a visible one-step rotation target."""
    self._rotation_mode[env_ids] = True
    reset_quat_w = self.object.data.default_root_state[env_ids, 3:7]
    step_quat = self._rotation_step_quat(reset_quat_w.dtype).expand(
      env_ids.numel(), 4
    )
    self.goal_quat_w[env_ids] = quat_mul(reset_quat_w, step_quat)
    self._reset_goal_progress(env_ids)
    self._initialize_goal_progress(env_ids)

  def preview_advance_goal(self, env_ids: torch.Tensor) -> None:
    """Advance preview targets by one rotation step without requiring success."""
    self._rotation_mode[env_ids] = True
    self._advance_rotation_goals(env_ids)
    self._reset_goal_progress(env_ids)

  def _read_tip_contact_count(self) -> torch.Tensor:
    try:
      sensor = self._env.scene[self.cfg.tip_contact_sensor_name]
    except (KeyError, TypeError):
      sensor = None
    found = None if sensor is None else getattr(sensor.data, "found", None)
    if found is None:
      return torch.zeros(self.num_envs, device=self.device)
    if found.ndim == 1:
      found = found.unsqueeze(-1)
    return (found > 0).float().sum(dim=-1)

  def _min_goal_steps(self) -> int:
    return max(int(self.cfg.min_goal_interval / self._env.step_dt), 0)

  def evaluate_contact_task(self) -> None:
    """Snapshot Luwu success before rewards; repeated reads never tick time.

    Called by flip_completed after cage_drop. Reset marks only the reset envs
    as evaluated at this step, so a post-reset read cannot count as an action.
    Goal changes are deferred to the normal command phase after reward/reset.
    """
    if not self.cfg.synchronized_contact_task:
      return
    step = self._env.common_step_counter
    fresh = self._task_evaluated_step != step
    if not fresh.any():
      return
    self._goal_switched[fresh] = False
    error = quat_error_magnitude(self.object.data.root_link_quat_w, self.goal_quat_w)
    count = self._read_tip_contact_count()
    drop = self._env.termination_manager.get_term("cage_drop")
    threshold = self.success_threshold
    valid = (error <= threshold) & ~drop & (
      count >= self.cfg.min_contact_fingertips
    ) & (self._goal_initial_error > threshold)
    hold = torch.where(valid, self.hold_counter + 1, 0)
    self.hold_counter = torch.where(fresh, hold, self.hold_counter)
    event = fresh & ~self.in_success_window & (
      self.hold_counter >= self.success_hold_steps
    )
    self._success_achieved = torch.where(fresh, event, self._success_achieved)
    self._success_valid = torch.where(fresh, valid, self._success_valid)
    self._within_threshold = torch.where(
      fresh, error <= threshold, self._within_threshold
    )
    self._tip_contact_count = torch.where(fresh, count, self._tip_contact_count)
    self.goal_reach_count += event.int()
    self._rotation_goal_reach_count += (event & self._rotation_mode).int()
    self._flip_success_count += (event & ~self._rotation_mode).int()
    self._record_continuous_flip(event & ~self._rotation_mode)
    continuous = self._continuous_task_mask()
    self._flip_completed |= event & ~self._rotation_mode & self._flip_termination_mask()
    self.in_success_window |= event
    self.goal_timer += fresh.int()
    self.window_timer += (fresh & self.in_success_window).int()
    ready_to_advance = (
      fresh & self.in_success_window
      & (self.window_timer >= self.cfg.goal_switch_delay)
      & (self.goal_timer >= self._min_goal_steps())
    )
    self._pending_goal_advance |= ready_to_advance & continuous
    self._pending_rotation_advance |= ready_to_advance & ~continuous & self._rotation_mode
    self._task_evaluated_step[fresh] = step
    self._update_metrics()

  def _update_command(self) -> None:
    if self.cfg.synchronized_contact_task:
      continuous_ids = self._pending_goal_advance.nonzero().flatten()
      rotation_ids = self._pending_rotation_advance.nonzero().flatten()
      if continuous_ids.numel():
        self._advance_continuous_goals(continuous_ids)
      if rotation_ids.numel():
        self._advance_rotation_goals(rotation_ids)
      ids = torch.cat((continuous_ids, rotation_ids))
      if ids.numel():
        self._reset_goal_progress(ids)
        self._goal_switched[ids] = True
        self._pending_rotation_advance[ids] = False
        self._pending_goal_advance[ids] = False
      self._update_metrics()
      return
    goal_timer = self.goal_timer + 1
    ori_err = quat_error_magnitude(self.object.data.root_link_quat_w, self.goal_quat_w)
    within_threshold = ori_err < self.success_threshold
    tip_contact_count = self._read_tip_contact_count()
    contact_valid = tip_contact_count >= self.cfg.min_contact_fingertips
    self._tip_contact_count = tip_contact_count

    # APPROACHING phase
    approaching_mask = ~self.in_success_window

    try:
      cage_drop = self._env.termination_manager.get_term("cage_drop")
    except (AttributeError, KeyError, ValueError):
      cage_drop = torch.zeros_like(contact_valid)
    success_conditions = within_threshold & ~cage_drop & (
      contact_valid if self.cfg.require_tip_contact else torch.ones_like(contact_valid)
    )
    hold_counter = torch.where(
      approaching_mask & success_conditions,
      self.hold_counter + 1,
      torch.where(
        approaching_mask, torch.zeros_like(self.hold_counter), self.hold_counter
      ),
    )

    just_succeeded = approaching_mask & (hold_counter >= self.success_hold_steps)

    in_success_window = self.in_success_window | just_succeeded
    window_timer = torch.where(
      just_succeeded, torch.zeros_like(self.window_timer), self.window_timer
    )
    hold_counter = torch.where(
      just_succeeded, torch.zeros_like(hold_counter), hold_counter
    )
    self.goal_reach_count += just_succeeded.long()
    rotation_success = just_succeeded & self._rotation_mode
    flip_success = just_succeeded & ~self._rotation_mode
    self._rotation_goal_reach_count += rotation_success.long()
    self._flip_success_count += flip_success.long()
    self._record_continuous_flip(flip_success)
    self._flip_completed |= flip_success & self._flip_termination_mask()

    self._success_achieved = just_succeeded
    self._success_valid = success_conditions

    # SUCCESS_WINDOW phase
    window_mask = in_success_window
    window_timer = torch.where(window_mask, window_timer + 1, window_timer)

    reward_hold_counter = torch.where(
      window_mask & within_threshold,
      self.reward_hold_counter + 1,
      self.reward_hold_counter,
    )
    reward_hold_counter = torch.where(
      just_succeeded, torch.zeros_like(reward_hold_counter), reward_hold_counter
    )

    min_goal_steps = self._min_goal_steps()
    should_switch = (
      window_mask
      & (window_timer >= self.cfg.goal_switch_delay)
      & (goal_timer >= min_goal_steps)
    )

    if self.cfg.goal_sampling_mode == "incremental_yaw_or_flip":
      # Rotation episodes progress through adjacent +15° targets only after
      # the current target has been reached and held. The flip episode is a
      # one-shot A→B goal and remains in its success window after completion.
      advance_ids = (should_switch & self._rotation_mode).nonzero(
        as_tuple=False
      ).squeeze(-1)
      if advance_ids.numel() > 0:
        self._advance_rotation_goals(advance_ids)
      should_switch = should_switch & self._rotation_mode
    elif self.cfg.goal_sampling_mode == "continuous_yaw_or_random_face":
      advance_ids = should_switch.nonzero(as_tuple=False).squeeze(-1)
      if advance_ids.numel() > 0:
        self._advance_continuous_goals(advance_ids)
    elif self.cfg.goal_sampling_mode == "curriculum_flip":
      continuous_ids = (should_switch & self._continuous_task_mask()).nonzero(
        as_tuple=False
      ).squeeze(-1)
      rotation_ids = (
        should_switch
        & ~self._continuous_task_mask()
        & self._rotation_mode
      ).nonzero(as_tuple=False).squeeze(-1)
      if continuous_ids.numel() > 0:
        self._advance_continuous_goals(continuous_ids)
      if rotation_ids.numel() > 0:
        self._advance_rotation_goals(rotation_ids)
      should_switch = continuous_ids.new_zeros(self.num_envs, dtype=torch.bool)
      if continuous_ids.numel() > 0:
        should_switch[continuous_ids] = True
      if rotation_ids.numel() > 0:
        should_switch[rotation_ids] = True
    else:
      advance_ids = should_switch.nonzero(as_tuple=False).squeeze(-1)
      if advance_ids.numel() > 0:
        self._sample_goal_in_world(advance_ids)

    self._goal_switched = should_switch

    self._reward_window_timer = window_timer.clone()
    self._reward_hold_counter_snapshot = reward_hold_counter.clone()

    in_success_window = torch.where(
      should_switch, torch.zeros_like(in_success_window), in_success_window
    )
    window_timer = torch.where(
      should_switch, torch.zeros_like(window_timer), window_timer
    )
    hold_counter = torch.where(
      should_switch, torch.zeros_like(hold_counter), hold_counter
    )
    goal_timer = torch.where(should_switch, torch.zeros_like(goal_timer), goal_timer)
    reward_hold_counter = torch.where(
      should_switch, torch.zeros_like(reward_hold_counter), reward_hold_counter
    )

    self._within_threshold = within_threshold

    self.hold_counter = hold_counter
    self.goal_timer = goal_timer
    self.in_success_window = in_success_window
    self.window_timer = window_timer
    self.reward_hold_counter = reward_hold_counter

  def _debug_vis_impl(self, visualizer) -> None:
    self._visualization.draw_debug_visuals(
      visualizer=visualizer,
      num_envs=self.num_envs,
      palm_pose_w=self.robot.data.body_link_pose_w[:, self.palm_body_id, :],
      object_pos_w=self.object.data.root_link_pos_w,
      goal_quat_w=self.goal_quat_w,
      policy_status_for_env=self._policy_status_for_env,
    )

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx,
  ) -> None:
    self._visualization.create_gui(
      name=name,
      server=server,
      get_env_idx=get_env_idx,
    )
    self._visualization.update_status_gui(
      num_envs=self.num_envs,
      policy_status_for_env=self._policy_status_for_env,
    )

  def _policy_status_for_env(self, env_idx: int) -> tuple:
    ori_error_rad = float(
      quat_error_magnitude(
        self.object.data.root_link_quat_w[env_idx : env_idx + 1],
        self.goal_quat_w[env_idx : env_idx + 1],
      )[0].item()
    )
    threshold = float(self.success_threshold[env_idx].item())
    is_success = ori_error_rad < threshold
    if bool(self._continuous_task_mask()[env_idx].item()):
      return (
        ori_error_rad,
        is_success,
        int(self._target_face_id[env_idx].item()),
        int(self._consecutive_flip_count[env_idx].item()),
        int(self.face_coverage_count[env_idx].item()),
      )
    return ori_error_rad, is_success

  @property
  def within_threshold(self) -> torch.Tensor:
    """Instantaneous: ori_err < success_threshold this step."""
    return self._within_threshold

  @property
  def success_achieved(self) -> torch.Tensor:
    """One-shot: True only on the step hold_counter first reaches success_hold_steps."""
    return self._success_achieved

  @property
  def goal_switched(self) -> torch.Tensor:
    """One-shot: True only on the step the goal switches."""
    return self._goal_switched

  @property
  def reward_window_timer(self) -> torch.Tensor:
    """Window timer value before goal-switch reset, for reward computation."""
    return self._reward_window_timer

  @property
  def reward_hold_counter_snapshot(self) -> torch.Tensor:
    """Hold counter for reward scaling (pre-reset snapshot). Only increments when within_threshold."""
    return self._reward_hold_counter_snapshot

  @property
  def tip_contact_count(self) -> torch.Tensor:
    return self._tip_contact_count

  @property
  def success_valid(self) -> torch.Tensor:
    return self._success_valid

  @property
  def flip_completed(self) -> torch.Tensor:
    return self._flip_completed

  @property
  def rotation_mode(self) -> torch.Tensor:
    return self._rotation_mode

  @property
  def current_face_id(self) -> torch.Tensor:
    return self._current_face_id

  @property
  def target_face_id(self) -> torch.Tensor:
    return self._target_face_id

  @property
  def consecutive_flip_count(self) -> torch.Tensor:
    return self._consecutive_flip_count

  @property
  def face_coverage_count(self) -> torch.Tensor:
    bits = torch.arange(6, device=self.device, dtype=torch.int64)
    return ((self._face_visited_mask.unsqueeze(-1) >> bits) & 1).sum(dim=-1)

  @property
  def adjacent_flip_count(self) -> torch.Tensor:
    return self._adjacent_flip_count

  @property
  def opposite_flip_count(self) -> torch.Tensor:
    return self._opposite_flip_count

  @property
  def goal_initial_error(self) -> torch.Tensor:
    return self._goal_initial_error

  @property
  def goal_progress_best(self) -> torch.Tensor:
    return self._goal_progress_best

  @property
  def approach_best_score(self) -> torch.Tensor:
    return self._approach_best_score


@dataclass(kw_only=True)
class InHandReorientCommandCfg(CommandTermCfg):
  entity_name: str
  goal_sampling_mode: str = "uniform_so3"
  # From-scratch curriculum is opt-in so the established tasks retain their
  # exact command semantics.  The new Luwu task enables it and starts at 0.
  curriculum_enabled: bool = False
  curriculum_stage: int = 2
  yaw_goal_probability: float = 0.7
  rotation_step_degrees: float = 15.0
  yaw_goal_quat_wxyz: tuple[float, float, float, float] = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865476,
  )
  flip_goal_quat_wxyz: tuple[float, float, float, float] = (
    0.7071067811865476,
    0.7071067811865476,
    0.0,
    0.0,
  )
  success_threshold: float = 0.2
  success_hold_steps: int = 5
  goal_switch_delay: int = 20
  min_goal_interval: float = 0.0
  require_tip_contact: bool = False
  synchronized_contact_task: bool = False
  min_contact_fingertips: int = 1
  approach_fingertip_count: int = 4
  tip_contact_sensor_name: str = "tip_object_contact"
  debug_vis: bool = False
  robot_entity_name: str = "robot"
  palm_body_pattern: str = ".*_palm_link"

  def build(self, env) -> InHandReorientCommand:
    return InHandReorientCommand(self, env)
