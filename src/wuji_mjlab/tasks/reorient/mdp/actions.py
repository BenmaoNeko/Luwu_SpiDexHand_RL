# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

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

  def build(self, env) -> JointPositionOffsetEMAAction:
    return JointPositionOffsetEMAAction(self, env)
