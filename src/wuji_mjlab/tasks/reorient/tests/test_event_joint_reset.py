# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from wuji_mjlab.tasks.reorient.mdp.events import reset_joints_within_limits_range
from wuji_mjlab.tasks.reorient.tests.fakes import make_fake_event_env


def test_reset_joints_within_limits_range_writes_values_inside_limits():
  env = make_fake_event_env(num_envs=2)
  env_ids = torch.tensor([0, 1], dtype=torch.long)
  robot_cfg = SceneEntityCfg("robot")
  robot_cfg.joint_ids = slice(None)

  reset_joints_within_limits_range(
    env,
    env_ids=env_ids,
    position_range={"finger.*": (-0.25, 0.25)},
    velocity_range={"finger.*": (-0.5, 0.5)},
    asset_cfg=robot_cfg,
  )

  joint_pos, joint_vel, written_env_ids = env.scene["robot"].last_written_joint_state
  assert torch.equal(written_env_ids, env_ids)
  assert torch.all(joint_pos <= 1.0)
  assert torch.all(joint_pos >= -1.0)
  assert torch.all(joint_vel <= 1.0)
  assert torch.all(joint_vel >= -1.0)


def test_reset_joints_within_limits_range_requires_velocity_limits_for_scale_mode():
  env = make_fake_event_env(num_envs=1)
  env.scene["robot"].data.soft_joint_vel_limits = None
  env.scene["robot"].data.joint_vel_limits = None
  robot_cfg = SceneEntityCfg("robot")
  robot_cfg.joint_ids = slice(None)

  with pytest.raises(ValueError, match="requires joint velocity limits"):
    reset_joints_within_limits_range(
      env,
      env_ids=torch.tensor([0], dtype=torch.long),
      position_range={"finger.*": (-0.25, 0.25)},
      velocity_range={"finger.*": (0.5, 1.0)},
      asset_cfg=robot_cfg,
      operation="scale",
    )
