# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch

from .cage import (
  CageEscapePenalty,
  _compute_cage_aabb_escape_in_palm,
  _compute_cage_escape_dist,
  cage_drop,
  compute_cage_escalation,
  read_cage_penalty_counter,
  update_cage_penalty_counter,
)


def flip_completed(env, command_name: str = "reorient_command") -> torch.Tensor:
  """Terminate Luwu flip episodes exactly once after a valid flip."""
  command = env.command_manager.get_term(command_name)
  command.evaluate_contact_task()
  return command.flip_completed.clone()


__all__ = [
  "_compute_cage_aabb_escape_in_palm",
  "_compute_cage_escape_dist",
  "update_cage_penalty_counter",
  "compute_cage_escalation",
  "CageEscapePenalty",
  "read_cage_penalty_counter",
  "cage_drop",
  "flip_completed",
]
