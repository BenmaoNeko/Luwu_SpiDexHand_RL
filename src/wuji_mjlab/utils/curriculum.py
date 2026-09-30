# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared curriculum helpers used across multiple tasks."""

from __future__ import annotations

import torch


def get_curriculum_value(env, term_name: str, default: float = 1.0) -> float:
  """Read scalar curriculum state from manager, with a safe default."""
  manager = getattr(env, "curriculum_manager", None)
  if manager is None:
    return default

  state = getattr(manager, "_curriculum_state", {})
  if term_name not in state or state[term_name] is None:
    return default

  value = state[term_name]
  if isinstance(value, torch.Tensor):
    return value.mean().item() if value.numel() > 0 else default
  if isinstance(value, dict):
    if not value:
      return default
    first = next(iter(value.values()))
    if isinstance(first, torch.Tensor):
      return first.mean().item() if first.numel() > 0 else default
    return float(first)
  return float(value)
