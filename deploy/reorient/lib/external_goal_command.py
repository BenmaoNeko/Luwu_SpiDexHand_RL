# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""ExternalGoalCommandTerm: ZMQ-driven goal injection (no state machine).

Replaces the training-side InHandReorientCommand (which has state machine
fields such as hold_counter / reward_window_timer that are coupled to the
critic obs term). This class is a minimal no-op CommandTerm: the goal is
read from ZMQ by RealHandEnv.step() and injected via set_goal().

Quat convention: wxyz (consistent across mjlab/MuJoCo stack), world frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class ExternalGoalCommandTermCfg(CommandTermCfg):
    """Cfg for ExternalGoalCommandTerm.

    resampling_time_range=(1e9, 1e9) + __init__ filling time_left=1e9
    suppresses the base class _resample path (never expires)."""

    def build(self, env) -> "ExternalGoalCommandTerm":
        return ExternalGoalCommandTerm(self, env)


class ExternalGoalCommandTerm(CommandTerm):
    """No-op command term; goal set externally via .set_goal().

    Exposes both `.command` (mjlab CommandTerm base API) and `.goal_quat`
    (reorient obs term API in mdp/observations.py) — both point to the
    same underlying buffer (num_envs, 4) wxyz.
    """

    def __init__(self, cfg: ExternalGoalCommandTermCfg, env):
        super().__init__(cfg, env)
        self._goal_quat_w = torch.zeros(env.num_envs, 4, device=env.device)
        self._goal_quat_w[:, 0] = 1.0  # identity wxyz
        # Suppress base-class resample loop (compute() will subtract dt but
        # never reach 0 within reasonable wall time).
        self.time_left.fill_(1e9)

    @property
    def command(self) -> torch.Tensor:
        return self._goal_quat_w

    @property
    def goal_quat(self) -> torch.Tensor:
        return self._goal_quat_w

    def set_goal(self, quat_wxyz: torch.Tensor) -> None:
        """Inject goal quat (shape (num_envs, 4), world frame, wxyz)."""
        assert quat_wxyz.shape == self._goal_quat_w.shape, (
            f"expected {self._goal_quat_w.shape}, got {quat_wxyz.shape}"
        )
        self._goal_quat_w[:] = quat_wxyz.to(self._goal_quat_w.device)

    # ── Abstract methods from CommandTerm — all no-ops ──

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass
