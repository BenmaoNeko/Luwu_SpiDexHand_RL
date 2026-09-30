# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared reward decorators used across multiple tasks."""

from __future__ import annotations

from functools import wraps

import torch

from wuji_mjlab.utils.curriculum import get_curriculum_value


def curriculum_scaled(func):
  """Scale reward by a curriculum term value."""

  @wraps(func)
  def wrapper(env, *args, **kwargs):
    curriculum_term = kwargs.get("curriculum_term", "")
    curriculum_min = kwargs.get("curriculum_min", 0.0)

    reward = func(env, *args, **kwargs)
    if not curriculum_term:
      return reward

    scale = max(curriculum_min, get_curriculum_value(env, curriculum_term, 1.0))
    return scale * reward

  return wrapper


def warmup(func):
  """Gate reward during episode warmup period."""

  @wraps(func)
  def wrapper(env, *args, **kwargs):
    warmup_mode = kwargs.get("warmup_mode", "always")
    warmup_time_s = kwargs.get("warmup_time_s", 0.0)
    reward = func(env, *args, **kwargs)

    if warmup_mode == "always" or warmup_time_s <= 0.0:
      return reward

    in_warmup = env.episode_length_buf * env.step_dt < warmup_time_s
    if warmup_mode == "init":
      return torch.where(in_warmup, reward, torch.zeros_like(reward))
    if warmup_mode == "delay":
      return torch.where(in_warmup, torch.zeros_like(reward), reward)

    raise ValueError(
      f"Invalid warmup_mode '{warmup_mode}'. Must be 'always', 'init', or 'delay'."
    )

  return wrapper
