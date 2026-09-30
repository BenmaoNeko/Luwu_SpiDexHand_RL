# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from dataclasses import asdict

from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)


def test_reorient_rl_cfg_sets_explicit_min_std():
  runner_cfg = wuji_hand_reorient_ppo_runner_cfg()

  dist_cfg = asdict(runner_cfg)["actor"]["distribution_cfg"]
  assert dist_cfg["min_std"] == 0.2
