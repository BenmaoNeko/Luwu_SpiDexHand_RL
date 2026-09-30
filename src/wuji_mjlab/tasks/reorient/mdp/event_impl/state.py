# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class ReorientEventState:
  friction_dr_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
  geom_size_dr_snapshot: torch.Tensor | None = None
  pert_force_dir: torch.Tensor | None = None
  pert_velocity_cache: torch.Tensor | None = None
  prev_joint_vel: torch.Tensor | None = None


def get_reorient_event_state(env) -> ReorientEventState:
  state = getattr(env, "_reorient_event_state", None)
  if state is None:
    state = ReorientEventState(
      pert_force_dir=torch.zeros((env.num_envs, 3), device=env.device),
      pert_velocity_cache=torch.zeros((env.num_envs, 6), device=env.device),
    )
    env._reorient_event_state = state
  return state
