# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Pure tensor helpers for contact-gated Luwu task shaping.

These functions deliberately do not read or mutate an environment. Episode and
goal state is owned by the command term so reward evaluation remains explicit
and can be reset per environment.
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


def box_surface_distance(
  site_pos_w: torch.Tensor,
  box_pos_w: torch.Tensor,
  box_quat_w: torch.Tensor,
  half_size: torch.Tensor,
) -> torch.Tensor:
  """Return outside Euclidean distance from each point to an oriented box.

  ``site_pos_w`` is ``(B, K, 3)`` or ``(B, 3)``. The box dimensions are
  expressed in the box frame. Points inside the box have distance zero and do
  not receive extra proximity credit.
  """
  point = site_pos_w if site_pos_w.ndim == 3 else site_pos_w.unsqueeze(1)
  center = box_pos_w.unsqueeze(1)
  quat = box_quat_w.unsqueeze(1).expand(-1, point.shape[1], -1)
  local = quat_apply_inverse(quat, point - center)
  half = half_size.unsqueeze(1) if half_size.ndim == 2 else half_size
  outside = torch.relu(local.abs() - half)
  distance = torch.linalg.vector_norm(outside, dim=-1)
  return distance if site_pos_w.ndim == 3 else distance.squeeze(1)


def surface_proximity_score(
  distances: torch.Tensor,
  length_scale: float = 0.02,
) -> torch.Tensor:
  """Average bounded proximity score for a batch of fingertip distances."""
  return per_finger_surface_proximity_score(distances, length_scale).mean(dim=-1)


def per_finger_surface_proximity_score(
  distances: torch.Tensor,
  length_scale: float = 0.02,
) -> torch.Tensor:
  """Return an independent bounded proximity score for each fingertip."""
  if length_scale <= 0.0:
    raise ValueError("length_scale must be positive")
  return torch.exp(-distances / length_scale).clamp(0.0, 1.0)


def bounded_monotonic_increment(
  value: torch.Tensor,
  best_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return only new progress and update the running best value."""
  value = value.clamp(0.0, 1.0)
  best_value = best_value.clamp(0.0, 1.0)
  increment = torch.relu(value - best_value)
  return increment, torch.maximum(best_value, value)


def mean_budgeted_increment(
  increment: torch.Tensor,
  episode_budget: float,
) -> torch.Tensor:
  """Allocate one episode budget across independent fingertip channels."""
  if episode_budget < 0.0:
    raise ValueError("episode_budget must be non-negative")
  if increment.ndim < 2:
    raise ValueError("increment must include a fingertip/channel dimension")
  return episode_budget * increment.clamp_min(0.0).mean(dim=-1)


def contact_milestone_events(
  contact_count: torch.Tensor,
  contact_hold_time: torch.Tensor,
  first_awarded: torch.Tensor,
  second_awarded: torch.Tensor,
  step_dt: float,
  stable_time: float = 0.1,
  single_reward: float = 1.0,
  double_reward: float = 1.0,
) -> tuple[
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
]:
  """Update one-shot contact milestones and return their actual reward.

  The first event is one fingertip; the second is two distinct fingertips.
  Both are episode-scoped by the caller through the award flags.
  """
  if step_dt <= 0.0:
    raise ValueError("step_dt must be positive")
  if stable_time <= 0.0:
    raise ValueError("stable_time must be positive")
  if single_reward < 0.0 or double_reward < 0.0:
    raise ValueError("contact milestone rewards must be non-negative")
  # Separate clocks: time with one finger cannot qualify a new second finger.
  in_contact = contact_count.unsqueeze(-1) >= torch.tensor(
    [1, 2], device=contact_count.device
  )
  new_hold_time = torch.where(
    in_contact,
    contact_hold_time + step_dt,
    torch.zeros_like(contact_hold_time),
  )
  stable = in_contact & (new_hold_time >= stable_time)
  first_event = (~first_awarded & stable[:, 0]).float() * single_reward
  second_event = (~second_awarded & stable[:, 1]).float() * double_reward
  new_first_awarded = first_awarded | (first_event > 0)
  new_second_awarded = second_awarded | (second_event > 0)
  return (
    first_event,
    second_event,
    new_hold_time,
    new_first_awarded,
    new_second_awarded,
  )


def goal_error_progress(
  initial_error: torch.Tensor,
  current_error: torch.Tensor,
  tolerance: float | torch.Tensor,
) -> torch.Tensor:
  """Normalize error reduction to [0, 1], with tolerance as full progress."""
  tolerance_tensor = torch.as_tensor(
    tolerance, device=initial_error.device, dtype=initial_error.dtype
  )
  if torch.any(tolerance_tensor <= 0.0):
    raise ValueError("tolerance must be positive")
  denominator = (initial_error - tolerance_tensor).clamp_min(1e-6)
  progress = (initial_error - current_error) / denominator
  return torch.where(
    initial_error <= tolerance_tensor,
    torch.zeros_like(progress),
    progress.clamp(0.0, 1.0),
  )
