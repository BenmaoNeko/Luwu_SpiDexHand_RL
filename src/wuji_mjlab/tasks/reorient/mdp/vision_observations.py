# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Temporally correlated visual-pose corruption for policy observations.

The physics state and privileged critic remain clean.  This module only owns
the actor-side measurement model: a short delayed pose buffer, bounded
axis-angle noise, slowly varying bias, and packet holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_inv, quat_mul


@dataclass(frozen=True, slots=True)
class VisionPoseStage:
  """Visual measurement uncertainty for one curriculum stage."""

  stage: int
  noise_range_rad: float
  bias_limit_rad: float
  bias_step_rad: float
  calibration_limit_rad: float
  delay_steps: tuple[int, int]
  dropout_probability: float
  max_hold_steps: int


_VISION_STAGES = (
  VisionPoseStage(
    stage=0,
    noise_range_rad=math.pi / 180.0,
    bias_limit_rad=0.5 * math.pi / 180.0,
    bias_step_rad=0.05 * math.pi / 180.0,
    calibration_limit_rad=0.5 * math.pi / 180.0,
    delay_steps=(0, 1),
    dropout_probability=0.0,
    max_hold_steps=0,
  ),
  VisionPoseStage(
    stage=1,
    noise_range_rad=2.0 * math.pi / 180.0,
    bias_limit_rad=1.5 * math.pi / 180.0,
    bias_step_rad=0.10 * math.pi / 180.0,
    calibration_limit_rad=1.5 * math.pi / 180.0,
    delay_steps=(0, 2),
    dropout_probability=0.01,
    max_hold_steps=1,
  ),
  VisionPoseStage(
    stage=2,
    noise_range_rad=3.0 * math.pi / 180.0,
    bias_limit_rad=3.0 * math.pi / 180.0,
    bias_step_rad=0.20 * math.pi / 180.0,
    calibration_limit_rad=3.0 * math.pi / 180.0,
    delay_steps=(1, 4),
    dropout_probability=0.03,
    max_hold_steps=2,
  ),
  VisionPoseStage(
    stage=3,
    noise_range_rad=5.0 * math.pi / 180.0,
    bias_limit_rad=5.0 * math.pi / 180.0,
    bias_step_rad=0.30 * math.pi / 180.0,
    calibration_limit_rad=5.0 * math.pi / 180.0,
    delay_steps=(1, 6),
    dropout_probability=0.05,
    max_hold_steps=4,
  ),
  # Transfer-only profile: visual uncertainty is stronger than the final
  # curriculum stage, while the slowly varying bias and calibration model
  # remain unchanged to isolate the requested sim-to-real robustness change.
  VisionPoseStage(
    stage=4,
    noise_range_rad=7.0 * math.pi / 180.0,
    bias_limit_rad=5.0 * math.pi / 180.0,
    bias_step_rad=0.30 * math.pi / 180.0,
    calibration_limit_rad=5.0 * math.pi / 180.0,
    delay_steps=(1, 8),
    dropout_probability=0.08,
    max_hold_steps=6,
  ),
)

_MAX_VISION_DELAY_STEPS = max(profile.delay_steps[1] for profile in _VISION_STAGES)


@dataclass(slots=True)
class VisualPoseState:
  """GPU state for all parallel environments."""

  pose_history: torch.Tensor
  delay_steps: torch.Tensor
  bias_rotvec: torch.Tensor
  calibration_rotvec: torch.Tensor
  last_output: torch.Tensor
  initialized: torch.Tensor
  hold_remaining: torch.Tensor
  stale_age_steps: torch.Tensor
  vision_valid: torch.Tensor
  vision_age: torch.Tensor
  vision_confidence: torch.Tensor
  vision_innovation: torch.Tensor
  last_update_step: torch.Tensor


def vision_stage_parameters(stage: int) -> VisionPoseStage:
  """Return the validated visual DR profile for ``stage``."""
  if stage not in range(len(_VISION_STAGES)):
    raise ValueError(f"stage must be in [0, {len(_VISION_STAGES) - 1}], got {stage}")
  return _VISION_STAGES[stage]


def create_visual_pose_state(
  num_envs: int,
  device: torch.device | str,
  *,
  max_delay_steps: int = _MAX_VISION_DELAY_STEPS,
  dtype: torch.dtype = torch.float32,
) -> VisualPoseState:
  """Allocate visual state without touching the simulation state."""
  if max_delay_steps < 0:
    raise ValueError("max_delay_steps must be non-negative")
  return VisualPoseState(
    pose_history=torch.zeros(
      (num_envs, max_delay_steps + 1, 4), device=device, dtype=dtype
    ),
    delay_steps=torch.zeros((num_envs,), device=device, dtype=torch.long),
    bias_rotvec=torch.zeros((num_envs, 3), device=device, dtype=dtype),
    calibration_rotvec=torch.zeros((num_envs, 3), device=device, dtype=dtype),
    last_output=torch.zeros((num_envs, 4), device=device, dtype=dtype),
    initialized=torch.zeros((num_envs,), device=device, dtype=torch.bool),
    hold_remaining=torch.zeros((num_envs,), device=device, dtype=torch.long),
    stale_age_steps=torch.zeros((num_envs,), device=device, dtype=torch.long),
    vision_valid=torch.ones((num_envs,), device=device, dtype=dtype),
    vision_age=torch.zeros((num_envs,), device=device, dtype=dtype),
    vision_confidence=torch.ones((num_envs,), device=device, dtype=dtype),
    vision_innovation=torch.zeros((num_envs,), device=device, dtype=dtype),
    last_update_step=torch.full(
      (num_envs,), -1, device=device, dtype=torch.long
    ),
  )


def _all_env_ids(state: VisualPoseState) -> torch.Tensor:
  return torch.arange(
    state.pose_history.shape[0], device=state.pose_history.device, dtype=torch.long
  )


def reset_visual_pose_state(
  state_or_env,
  env_ids: torch.Tensor | None = None,
) -> VisualPoseState:
  """Reset visual state, accepting either a state or an environment.

  The environment form is used by the mjlab reset event.  The state form keeps
  the tensor-level behavior directly testable.
  """
  if isinstance(state_or_env, VisualPoseState):
    state = state_or_env
  else:
    state = get_visual_pose_state(state_or_env)

  if env_ids is None:
    env_ids = _all_env_ids(state)
  else:
    env_ids = env_ids.to(device=state.pose_history.device, dtype=torch.long)

  state.pose_history[env_ids] = 0.0
  state.delay_steps[env_ids] = 0
  state.bias_rotvec[env_ids] = 0.0
  state.calibration_rotvec[env_ids] = 0.0
  state.last_output[env_ids] = 0.0
  state.initialized[env_ids] = False
  state.hold_remaining[env_ids] = 0
  state.stale_age_steps[env_ids] = 0
  state.vision_valid[env_ids] = 1.0
  state.vision_age[env_ids] = 0.0
  state.vision_confidence[env_ids] = 1.0
  state.vision_innovation[env_ids] = 0.0
  state.last_update_step[env_ids] = -1
  return state


def visual_pose_metadata(env) -> torch.Tensor:
  """Return actor-side visual status as ``(N, 4)`` normalized metadata."""
  state = get_visual_pose_state(env)
  initialized = state.initialized
  defaults = torch.stack(
    [
      torch.ones_like(state.vision_valid),
      torch.zeros_like(state.vision_age),
      torch.ones_like(state.vision_confidence),
      torch.zeros_like(state.vision_innovation),
    ],
    dim=-1,
  )
  values = torch.stack(
    [
      state.vision_valid,
      state.vision_age,
      state.vision_confidence,
      state.vision_innovation,
    ],
    dim=-1,
  )
  return torch.where(initialized.unsqueeze(-1), values, defaults).clamp(0.0, 1.0)


def vision_valid(env) -> torch.Tensor:
  return visual_pose_metadata(env)[:, 0:1]


def vision_age(env) -> torch.Tensor:
  return visual_pose_metadata(env)[:, 1:2]


def vision_confidence(env) -> torch.Tensor:
  return visual_pose_metadata(env)[:, 2:3]


def vision_innovation(env) -> torch.Tensor:
  return visual_pose_metadata(env)[:, 3:4]


def get_visual_pose_state(env) -> VisualPoseState:
  """Lazily allocate the environment-local visual observation state."""
  state = getattr(env, "_visual_pose_state", None)
  if state is None or state.pose_history.shape[0] != env.num_envs:
    state = create_visual_pose_state(
      env.num_envs, env.device, max_delay_steps=_MAX_VISION_DELAY_STEPS
    )
    env._visual_pose_state = state
  return state


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
  return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)


def _rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
  angle = torch.linalg.vector_norm(rotvec, dim=-1)
  axis = rotvec / angle.unsqueeze(-1).clamp_min(1e-8)
  quat = quat_from_angle_axis(angle, axis)
  identity = torch.zeros_like(quat)
  identity[..., 0] = 1.0
  return torch.where((angle > 1e-8).unsqueeze(-1), quat, identity)


def _sample_uniform_rotvec(
  shape: tuple[int, ...], limit: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
  if limit <= 0.0:
    return torch.zeros((*shape, 3), device=device, dtype=dtype)
  return torch.empty((*shape, 3), device=device, dtype=dtype).uniform_(-limit, limit)


def _select_visual_pose_state(
  state: VisualPoseState, env_ids: torch.Tensor
) -> VisualPoseState:
  return VisualPoseState(
    **{
      item.name: getattr(state, item.name).index_select(0, env_ids).clone()
      for item in fields(state)
    }
  )


def _scatter_visual_pose_state(
  state: VisualPoseState,
  env_ids: torch.Tensor,
  selected: VisualPoseState,
) -> None:
  for item in fields(state):
    getattr(state, item.name).index_copy_(
      0, env_ids, getattr(selected, item.name)
    )


def step_visual_pose_model(
  current_quat_w: torch.Tensor,
  state: VisualPoseState,
  profile: VisionPoseStage,
  *,
  noise_rotvec: torch.Tensor | None = None,
  dropout_mask: torch.Tensor | None = None,
  update_bias: bool = True,
  step_id: int | None = None,
) -> torch.Tensor:
  """Advance the measurement model and return a corrupted quaternion.

  ``current_quat_w`` is already expressed in the tag frame despite the
  historical variable name.  The first call backfills every delay slot with
  the current sample so reset cannot expose zero quaternions.
  """
  current_quat_w = _normalize_quat(current_quat_w)
  num_envs = current_quat_w.shape[0]
  if state.pose_history.shape[0] != num_envs:
    raise ValueError("visual state and quaternion batch sizes differ")
  if profile.delay_steps[1] >= state.pose_history.shape[1]:
    raise ValueError("visual state history is shorter than the selected delay")

  if step_id is not None:
    cached = state.last_update_step == int(step_id)
    pending = (~cached).nonzero(as_tuple=False).flatten()
    if pending.numel() == 0:
      return state.last_output.clone()
    if pending.numel() < num_envs:
      selected_state = _select_visual_pose_state(state, pending)
      selected_noise = (
        None if noise_rotvec is None else noise_rotvec.index_select(0, pending)
      )
      selected_dropout = (
        None if dropout_mask is None else dropout_mask.index_select(0, pending)
      )
      step_visual_pose_model(
        current_quat_w.index_select(0, pending),
        selected_state,
        profile,
        noise_rotvec=selected_noise,
        dropout_mask=selected_dropout,
        update_bias=update_bias,
      )
      selected_state.last_update_step.fill_(int(step_id))
      _scatter_visual_pose_state(state, pending, selected_state)
      return state.last_output.clone()

  fresh = ~state.initialized
  low, high = profile.delay_steps
  sampled_delays = torch.randint(
    low,
    high + 1,
    (num_envs,),
    device=current_quat_w.device,
  )
  state.delay_steps.copy_(
    torch.where(fresh, sampled_delays, state.delay_steps)
  )
  sampled_calibration = _sample_uniform_rotvec(
    (num_envs,),
    profile.calibration_limit_rad,
    current_quat_w.device,
    current_quat_w.dtype,
  )
  state.calibration_rotvec.copy_(
    torch.where(
      fresh.unsqueeze(-1), sampled_calibration, state.calibration_rotvec
    )
  )
  filled_history = current_quat_w.unsqueeze(1).expand_as(state.pose_history)
  state.pose_history.copy_(
    torch.where(fresh.view(num_envs, 1, 1), filled_history, state.pose_history)
  )
  state.last_output.copy_(
    torch.where(fresh.unsqueeze(-1), current_quat_w, state.last_output)
  )
  state.initialized |= fresh

  state.pose_history[:, 1:] = state.pose_history[:, :-1].clone()
  state.pose_history[:, 0] = current_quat_w
  delayed = state.pose_history.gather(
    1,
    state.delay_steps.view(num_envs, 1, 1).expand(-1, 1, 4),
  ).squeeze(1)

  holding = state.hold_remaining > 0
  previous_output = state.last_output.clone()
  state.hold_remaining.copy_(
    torch.where(
      holding,
      (state.hold_remaining - 1).clamp_min(0),
      state.hold_remaining,
    )
  )

  if dropout_mask is None:
    dropout_mask = torch.rand(
      (num_envs,), device=current_quat_w.device
    ) < profile.dropout_probability
  else:
    dropout_mask = dropout_mask.to(device=current_quat_w.device, dtype=torch.bool)

  can_hold = (~holding) & dropout_mask & (profile.max_hold_steps > 0)
  sampled_holds = torch.randint(
    1,
    profile.max_hold_steps + 1,
    (num_envs,),
    device=current_quat_w.device,
  ) if profile.max_hold_steps > 0 else torch.zeros_like(state.hold_remaining)
  state.hold_remaining.copy_(
    torch.where(can_hold, sampled_holds, state.hold_remaining)
  )
  holding = holding | can_hold

  if update_bias and profile.bias_limit_rad > 0.0:
    drift = _sample_uniform_rotvec(
      (num_envs,),
      profile.bias_step_rad,
      current_quat_w.device,
      current_quat_w.dtype,
    )
    state.bias_rotvec.mul_(0.995).add_(drift)
    state.bias_rotvec.clamp_(-profile.bias_limit_rad, profile.bias_limit_rad)
  else:
    state.bias_rotvec.clamp_(-profile.bias_limit_rad, profile.bias_limit_rad)

  if noise_rotvec is None:
    noise_rotvec = _sample_uniform_rotvec(
      (num_envs,),
      profile.noise_range_rad,
      current_quat_w.device,
      current_quat_w.dtype,
    )
  else:
    noise_rotvec = noise_rotvec.to(
      device=current_quat_w.device, dtype=current_quat_w.dtype
    )

  corrupted = _normalize_quat(
    quat_mul(
      delayed,
      _rotvec_to_quat(
        state.calibration_rotvec + state.bias_rotvec + noise_rotvec
      ),
    )
  )
  output = torch.where(holding.unsqueeze(-1), state.last_output, corrupted)

  delta_quat = quat_mul(output, quat_inv(_normalize_quat(previous_output)))
  delta_angle = 2.0 * torch.atan2(
    torch.linalg.vector_norm(delta_quat[..., 1:], dim=-1),
    delta_quat[..., 0].abs().clamp_min(1e-8),
  )
  fresh_packet = ~holding
  previous_age = state.stale_age_steps
  stale_age = torch.where(
    fresh_packet,
    torch.zeros_like(previous_age),
    (previous_age + 1).clamp(max=max(profile.max_hold_steps, 1)),
  )
  max_age = float(max(profile.max_hold_steps, 1))
  age_norm = stale_age.to(current_quat_w.dtype) / max_age
  innovation_norm = (delta_angle / math.radians(30.0)).clamp(0.0, 1.0)
  innovation_norm = torch.where(
    fresh,
    torch.zeros_like(innovation_norm),
    innovation_norm,
  )
  confidence = ((1.0 - age_norm) * (1.0 - 0.5 * innovation_norm)).clamp(0.0, 1.0)
  state.stale_age_steps.copy_(stale_age)
  state.vision_valid.copy_(fresh_packet.to(current_quat_w.dtype))
  state.vision_age.copy_(age_norm)
  state.vision_confidence.copy_(confidence)
  state.vision_innovation.copy_(innovation_norm)
  state.last_output.copy_(output)
  if step_id is not None:
    state.last_update_step.fill_(int(step_id))
  return _normalize_quat(output)
