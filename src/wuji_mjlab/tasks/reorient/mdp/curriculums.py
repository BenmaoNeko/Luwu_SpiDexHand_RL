# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch

from wuji_mjlab.tasks.reorient.mdp.event_utils import (
  get_linear_progress,
  resolve_env_ids,
)
from wuji_mjlab.utils.curriculum import get_curriculum_value  # noqa: F401 -- re-export

# Backwards-compatible re-exports for external callers that historically
# imported these from ``curriculums``. New code should import them from
# ``mdp.event_utils`` directly.
__all__ = [
  "adaptive_episode_curriculum",
  "friction_curriculum",
  "geom_size_curriculum",
  "get_curriculum_value",
  "get_linear_progress",
  "reorient_success_curriculum",
  "resolve_env_ids",
]


def reorient_success_curriculum(
  env,
  env_ids,
  command_name: str = "reorient_command",
  count_threshold: int = 3,
  delta_per_loop: float = 0.01,
  task_aware: bool = False,
  rotation_count_threshold: int | None = None,
  flip_count_threshold: int = 1,
) -> dict[str, float]:
  """Adaptive curriculum based on task-appropriate completion at reset."""
  if not hasattr(env, "_reorient_success_curriculum"):
    env._reorient_success_curriculum = 0.0

  env_ids = resolve_env_ids(env, env_ids)

  command = env.command_manager.get_term(command_name)
  goal_reach_count = command.metrics["goal_reach_count"][env_ids]
  if goal_reach_count.numel() == 0:
    return {"value": float(env._reorient_success_curriculum)}

  if task_aware:
    if rotation_count_threshold is None:
      rotation_count_threshold = count_threshold
    rotation_count = command.metrics.get("rotation_goal_reach_count")
    flip_count = command.metrics.get("flip_success_count")
    if rotation_count is None or flip_count is None:
      raise ValueError(
        "task_aware success curriculum requires rotation_goal_reach_count "
        "and flip_success_count command metrics"
      )
    completed = (rotation_count[env_ids] >= float(rotation_count_threshold)) | (
      flip_count[env_ids] >= float(flip_count_threshold)
    )
    completion_value = torch.where(
      flip_count[env_ids] >= float(flip_count_threshold),
      flip_count[env_ids].float(),
      rotation_count[env_ids].float(),
    )
  else:
    completed = goal_reach_count >= float(count_threshold)
    completion_value = goal_reach_count

  delta_per_env = float(delta_per_loop) / float(env.num_envs)
  delta = torch.where(
    completed,
    torch.full_like(goal_reach_count, delta_per_env, dtype=torch.float32),
    torch.full_like(goal_reach_count, -delta_per_env, dtype=torch.float32),
  )
  delta_sum = float(delta.sum().item())
  env._reorient_success_curriculum = float(
    max(min(env._reorient_success_curriculum + delta_sum, 1.0), 0.0)
  )
  return {
    "value": float(env._reorient_success_curriculum),
    "mean_goal_reach_count": float(completion_value.float().mean().item()),
    "success_env_frac": float(completed.float().mean().item()),
    "delta": delta_sum,
  }


def adaptive_episode_curriculum(
  env,
  env_ids,
  target_steps: int = 800,
  inc_rate: float = 0.1,
  dec_rate: float = 0.2,
  min_scale: float = 0.05,
  drop_gamma: float = 1.0,
  success_term: str | None = None,
) -> dict[str, float]:
  """Adaptive curriculum based on episode survival length."""
  if not hasattr(env, "_adaptive_episode_curriculum"):
    env._adaptive_episode_curriculum = float(min_scale)

  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return {"value": float(env._adaptive_episode_curriculum)}

  target_steps = max(int(target_steps), 1)
  episode_steps = env.episode_length_buf[env_ids].float()
  progress = torch.clamp(episode_steps / float(target_steps), 0.0, 1.0)
  reached_target = episode_steps >= float(target_steps)

  terminated = getattr(env.termination_manager, "terminated", None)
  if terminated is None:
    terminated = torch.zeros_like(reached_target)
  else:
    terminated = terminated[env_ids]
  early_drop = terminated & ~reached_target
  if success_term is not None:
    early_drop &= ~env.termination_manager.get_term(success_term)[env_ids]

  inc = reached_target.float() * float(inc_rate)
  pen = (
    early_drop.float() * torch.pow(1.0 - progress, float(drop_gamma)) * float(dec_rate)
  )

  delta = float((inc - pen).sum().item() / float(env.num_envs))
  env._adaptive_episode_curriculum = float(
    max(min(env._adaptive_episode_curriculum + delta, 1.0), float(min_scale))
  )
  return {
    "value": float(env._adaptive_episode_curriculum),
    "mean_episode_steps": float(episode_steps.mean().item()),
    "mean_progress": float(progress.mean().item()),
    "reached_target_frac": float(reached_target.float().mean().item()),
    "early_drop_frac": float(early_drop.float().mean().item()),
    "target_steps": float(target_steps),
    "delta": delta,
  }


def friction_curriculum(
  env,
  env_ids,
  friction_start: float = 1.0,
  friction_end: float = 0.5,
  warmup_frac: float = 0.05,
  rampup_frac: float = 0.60,
  total_steps: int | None = None,
) -> dict[str, float]:
  """Friction scale curriculum: high friction (easy) -> low friction (hard)."""
  del env_ids
  progress = get_linear_progress(env, warmup_frac, rampup_frac, total_steps=total_steps)
  scale = float(friction_start) + progress * (
    float(friction_end) - float(friction_start)
  )
  env._friction_curriculum = scale
  return {
    "value": float(scale),
    "progress": float(progress),
  }


def geom_size_curriculum(
  env,
  env_ids,
  geom_size_start: float = 1.0,
  geom_size_end: float = 1.15,
  warmup_frac: float = 0.05,
  rampup_frac: float = 0.60,
  total_steps: int | None = None,
) -> dict[str, float]:
  """Collision geom size curriculum: nominal -> enlarged (simulate glove/soft body)."""
  del env_ids
  progress = get_linear_progress(env, warmup_frac, rampup_frac, total_steps=total_steps)
  scale = float(geom_size_start) + progress * (
    float(geom_size_end) - float(geom_size_start)
  )
  env._geom_size_curriculum = scale
  return {
    "value": float(scale),
    "progress": float(progress),
  }
