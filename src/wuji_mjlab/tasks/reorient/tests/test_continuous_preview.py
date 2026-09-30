# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
  luwu_hand_continuous_faces_env_cfg,
)
from wuji_mjlab.tasks.reorient.tests.test_continuous_face_command import _make_command


def test_continuous_play_config_exposes_eight_preview_environments():
  cfg = luwu_hand_continuous_faces_env_cfg(play=True)

  assert cfg.scene.num_envs == 8
  assert cfg.commands["reorient_command"].goal_sampling_mode == (
    "continuous_yaw_or_random_face"
  )


def test_preview_assigns_four_rotations_and_four_continuous_flips():
  command, _ = _make_command(num_envs=8)
  command.preview_set_fixed_task_modes(
    torch.arange(4), torch.arange(4, 8)
  )

  assert command.rotation_mode.tolist() == [True, True, True, True, False, False, False, False]
  assert torch.all(command.target_face_id[:4] == command.current_face_id[:4])
  assert torch.all(command.target_face_id[4:] != command.current_face_id[4:])
