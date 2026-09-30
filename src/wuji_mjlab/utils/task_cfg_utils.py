# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared task config loading/preparation helpers."""

from __future__ import annotations

from typing import Any

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from .override_utils import apply_overrides


def apply_modifications(
  task_id: str,
  env_cfg: Any,
  agent_cfg: Any,
  *,
  play: bool = False,
) -> None:
  del task_id, agent_cfg, play
  hook = getattr(env_cfg, "apply_modifications", None)
  if callable(hook):
    hook(env_cfg)


def load_task_cfgs(task_id: str, *, play: bool = False) -> tuple[Any, Any]:
  env_cfg = load_env_cfg(task_id, play=play)
  agent_cfg = load_rl_cfg(task_id)
  return env_cfg, agent_cfg


def prepare_task_cfgs(
  task_id: str,
  overrides: list[str],
  *,
  play: bool = False,
) -> tuple[Any, Any]:
  env_cfg, agent_cfg = load_task_cfgs(task_id, play=play)
  apply_overrides({"env": env_cfg, "agent": agent_cfg}, overrides)
  apply_modifications(task_id, env_cfg, agent_cfg, play=play)
  return env_cfg, agent_cfg
