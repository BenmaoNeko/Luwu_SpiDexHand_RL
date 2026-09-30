# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared helpers consumed by both curriculum functions and event terms.

These utilities live in a neutral module so that
``mdp/event_impl/`` does not need to import ``mdp/curriculums.py`` for
unrelated reasons (input-normalization or step-counter math).
"""

from __future__ import annotations

import torch


def resolve_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  """Normalize ``env_ids`` to a 1-D long tensor on ``env.device``.

  ``None`` and ``slice`` inputs expand to ``torch.arange(num_envs)``; concrete
  tensor inputs are moved to the correct device and dtype but otherwise
  unchanged.
  """
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
  return env_ids.to(env.device, dtype=torch.long)


def get_linear_progress(
  env,
  warmup_frac: float = 0.05,
  rampup_frac: float = 0.60,
  total_steps: int | None = None,
) -> float:
  """Linear schedule based on ``env.common_step_counter`` over ``total_steps``.

  Returns 0.0 during ``[0, warmup_frac * total_steps]``, 1.0 once
  ``rampup_frac * total_steps`` is reached, and a linear interpolation in
  between.

  Args:
    total_steps: Expected total training steps (``max_iterations *
      num_steps_per_env``). Must be provided explicitly via CurriculumTermCfg
      params or caller. Can be overridden at runtime via the
      ``env.max_common_steps`` attribute.
  """
  if total_steps is None:
    raise ValueError(
      "total_steps must be provided explicitly. "
      "Compute it from rl_cfg: max_iterations * num_steps_per_env."
    )
  effective_total_steps = max(float(getattr(env, "max_common_steps", total_steps)), 1.0)
  step = float(getattr(env, "common_step_counter", 0))
  warmup = float(warmup_frac) * effective_total_steps
  rampup = float(rampup_frac) * effective_total_steps
  if step <= warmup:
    return 0.0
  if step >= rampup:
    return 1.0
  return float((step - warmup) / max(rampup - warmup, 1.0))
