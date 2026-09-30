# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from typing import TypedDict


class CurriculumReport(TypedDict, total=False):
  value: float
  delta: float
  mean_goal_reach_count: float
  success_env_frac: float
  mean_episode_steps: float
  mean_progress: float
  reached_target_frac: float
  early_drop_frac: float
  target_steps: float
  progress: float
