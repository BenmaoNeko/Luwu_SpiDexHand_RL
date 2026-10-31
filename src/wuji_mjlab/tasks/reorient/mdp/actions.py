# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


def map_normalized_actions_to_joint_limits(
  actions: torch.Tensor,
  default_joint_pos: torch.Tensor,
  lower_limits: torch.Tensor,
  upper_limits: torch.Tensor,
) -> torch.Tensor:
  """Map normalized actions to the full per-joint range around home.

  Zero remains the calibrated home position. Negative and positive actions
  use separate spans because home is not generally the midpoint of a joint's
  URDF limits.
  """
  clamped = torch.clamp(actions, -1.0, 1.0)
  negative_span = default_joint_pos - lower_limits
  positive_span = upper_limits - default_joint_pos
  span = torch.where(clamped < 0.0, negative_span, positive_span)
  return default_joint_pos + clamped * span


def apply_smart_cube_action_dr(
  command_queue: torch.Tensor, alpha: float, *, joint_dr_progress: float | None = None,
) -> torch.Tensor:
  """One-step whole-command delay and bounded uniform target noise in radians."""
  delayed = torch.rand(command_queue.shape[0], device=command_queue.device) < .7 * alpha
  target = torch.where(delayed[:, None], command_queue[:, 1], command_queue[:, 0])
  if joint_dr_progress is not None:
    p = joint_dr_progress
    lower, upper = math.radians(5) * p, .02 * alpha * (1-p) + math.radians(15) * p
    draw = torch.empty_like(target).uniform_(-1., 1.)
    sign = torch.where(draw < 0, -1., 1.)
    target = target + sign * (lower + draw.abs() * (upper - lower))
  elif alpha > 0:
    target = target + torch.empty_like(target).uniform_(-.02 * alpha, .02 * alpha)
  return target


def sample_realistic_action_bias(num_envs: int, num_joints: int, progress: float, device) -> torch.Tensor:
  bound = math.radians(3) * progress
  return torch.empty((num_envs, num_joints), device=device).uniform_(-bound, bound)


def apply_realistic_action_bias(target: torch.Tensor, bias: torch.Tensor,
                                lower=None, upper=None) -> torch.Tensor:
  result = target + bias
  return torch.clamp(result, min=lower, max=upper) if lower is not None else result


class JointPositionOffsetEMAAction(JointPositionAction):
  """Joint position action: default_pos + action * scale, with EMA smoothing and warmup hold."""

  cfg: "JointPositionOffsetEMAActionCfg"

  def __init__(self, cfg: "JointPositionOffsetEMAActionCfg", env):
    super().__init__(cfg, env)

    self._action_scale = cfg.action_scale
    self._ema_alpha = cfg.ema_alpha
    self._warmup_time_s = cfg.warmup_time_s

    self._default_joint_pos = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()

    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower_limits = soft_limits[..., 0]
    self._upper_limits = soft_limits[..., 1]

    self._prev_target = self._default_joint_pos.clone()
    self._previous_processed_action = self._default_joint_pos.clone()
    self._previous_previous_processed_action = self._default_joint_pos.clone()
    self._max_command_delay_steps = max(int(cfg.max_command_delay_steps), 0)
    self._command_queue = self._default_joint_pos.unsqueeze(1).expand(
      -1, self._max_command_delay_steps + 1, -1
    ).clone()
    self._realistic_bias = torch.zeros_like(self._default_joint_pos)

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._previous_previous_processed_action[:] = self._previous_processed_action
    self._previous_processed_action[:] = self._prev_target
    clamped = torch.clamp(actions, -1.0, 1.0)

    if self.cfg.use_joint_limits:
      raw_target = map_normalized_actions_to_joint_limits(
        clamped,
        self._default_joint_pos,
        self._lower_limits,
        self._upper_limits,
      )
    else:
      raw_target = self._default_joint_pos + clamped * self._action_scale
    raw_target = torch.clamp(raw_target, self._lower_limits, self._upper_limits)

    if self._max_command_delay_steps > 0:
      self._command_queue[:, 1:] = self._command_queue[:, :-1].clone()
    self._command_queue[:, 0] = raw_target
    delay_steps = min(
      max(int(self.cfg.command_delay_steps), 0),
      self._max_command_delay_steps,
    )
    commanded_target = self._command_queue[:, delay_steps]
    if self.cfg.realistic_dr:
      commanded_target = apply_realistic_action_bias(
        commanded_target, self._realistic_bias, self._lower_limits, self._upper_limits)
    elif self.cfg.smart_cube_dr:
      from .smart_cube_curriculum import get_smart_cube_curriculum
      controller = get_smart_cube_curriculum(self._env)
      commanded_target = apply_smart_cube_action_dr(
        self._command_queue, controller.alpha,
        joint_dr_progress=(controller.joint_dr_progress
          if controller.cfg.joint_dr_transition_steps else None))
    noise_std = max(float(self.cfg.command_noise_std), 0.0)
    if noise_std > 0.0:
      commanded_target = commanded_target + torch.randn_like(commanded_target) * noise_std
    commanded_target = torch.clamp(
      commanded_target, self._lower_limits, self._upper_limits
    )

    smoothed = (
      self._ema_alpha * commanded_target
      + (1.0 - self._ema_alpha) * self._prev_target
    )

    in_warmup = (
      self._env.episode_length_buf * self._env.step_dt < self._warmup_time_s
    ).unsqueeze(-1)
    self._processed_actions = torch.where(in_warmup, self._default_joint_pos, smoothed)
    self._prev_target = self._processed_actions.clone()

  @property
  def processed_action(self) -> torch.Tensor:
    return self._processed_actions

  @property
  def previous_processed_action(self) -> torch.Tensor:
    return self._previous_processed_action

  @property
  def previous_previous_processed_action(self) -> torch.Tensor:
    return self._previous_previous_processed_action

  def reset(self, env_ids: torch.Tensor) -> None:
    super().reset(env_ids)
    self._prev_target[env_ids] = self._default_joint_pos[env_ids]
    self._previous_processed_action[env_ids] = self._default_joint_pos[env_ids]
    self._previous_previous_processed_action[env_ids] = self._default_joint_pos[env_ids]
    self._command_queue[env_ids] = self._default_joint_pos[env_ids].unsqueeze(1)
    if self.cfg.realistic_dr:
      from .smart_cube_curriculum import (
        get_smart_cube_curriculum, realistic_parameter_progress, realistic_stage_progress,
      )
      controller = get_smart_cube_curriculum(self._env)
      progress = (realistic_parameter_progress(controller.realistic_progress)
        if controller.cfg.realistic_success_gated else
        realistic_stage_progress(controller.realistic_steps)[0])
      sampled = sample_realistic_action_bias(env_ids.numel(), self._realistic_bias.shape[1],
                                              progress, self._realistic_bias.device)
      nominal = getattr(self._env, '_realistic_nominal_mask', None)
      if nominal is not None:
        sampled[nominal[env_ids]] = 0
      self._realistic_bias[env_ids] = sampled


@dataclass(kw_only=True)
class JointPositionOffsetEMAActionCfg(JointPositionActionCfg):
  """Configuration for offset + EMA action with warmup."""

  action_scale: float = 0.5
  use_joint_limits: bool = False
  ema_alpha: float = 0.5
  warmup_time_s: float = 0.4
  max_command_delay_steps: int = 2
  command_delay_steps: int = 0
  command_noise_std: float = 0.0
  smart_cube_dr: bool = False
  realistic_dr: bool = False

  def build(self, env) -> JointPositionOffsetEMAAction:
    return JointPositionOffsetEMAAction(self, env)
