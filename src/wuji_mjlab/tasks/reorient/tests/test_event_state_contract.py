# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp.event_impl.state import (
  ReorientEventState,
  get_reorient_event_state,
)
from wuji_mjlab.tasks.reorient.tests.fakes import make_fake_event_env


def test_get_reorient_event_state_lazily_attaches_one_state_object():
  env = make_fake_event_env(num_envs=3)

  state = get_reorient_event_state(env)
  object_asset = env.scene["object"]
  pose = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
  object_asset.write_root_link_pose_to_sim(pose, env_ids=torch.tensor([1]))

  assert isinstance(state, ReorientEventState)
  assert not hasattr(state, "__dict__")
  assert env._reorient_event_state is state
  assert get_reorient_event_state(env) is state
  assert state.pert_force_dir.device == env.device  # type: ignore[union-attr]
  assert state.pert_velocity_cache.device == env.device  # type: ignore[union-attr]
  assert env.curriculum_manager._curriculum_state == {}
  assert env.termination_manager._term_dones == {}
  assert torch.equal(
    env.termination_manager.terminated,
    torch.zeros(3, dtype=torch.bool, device=env.device),
  )
  assert torch.equal(object_asset.data.root_link_pose_w[1], pose[0])
  assert torch.equal(object_asset.data.root_link_pos_w[1], pose[0, :3])
  assert torch.equal(object_asset.data.root_link_quat_w[1], pose[0, 3:7])
  assert torch.equal(
    object_asset.data.default_root_state[:, 3], torch.ones(3, device=env.device)
  )
  assert torch.equal(
    object_asset.data.root_link_pose_w[:, 3], torch.ones(3, device=env.device)
  )
  with pytest.raises(AttributeError):
    state.unknown_attr = "boom"  # type: ignore[attr-defined]


def test_get_reorient_event_state_allocates_stable_zero_buffers():
  env = make_fake_event_env(num_envs=2)
  other_env = make_fake_event_env(num_envs=2)

  state = get_reorient_event_state(env)
  second_state = get_reorient_event_state(env)
  other_state = get_reorient_event_state(other_env)
  robot = env.scene["robot"]
  joint_ids, joint_names = robot.find_joints("finger.*")
  robot_geom_ids, robot_geom_names = robot.find_geoms("geom_.*")
  robot_body_ids, robot_body_names = robot.find_bodies("body_.*")

  assert state.friction_dr_snapshot == {}
  assert state.geom_size_dr_snapshot is None
  assert state.prev_joint_vel is None
  assert state.pert_force_dir.shape == (2, 3)  # type: ignore[union-attr]
  assert state.pert_velocity_cache.shape == (2, 6)  # type: ignore[union-attr]
  assert state.pert_force_dir.device == env.device  # type: ignore[union-attr]
  assert state.pert_velocity_cache.device == env.device  # type: ignore[union-attr]
  assert torch.count_nonzero(state.pert_force_dir) == 0
  assert torch.count_nonzero(state.pert_velocity_cache) == 0
  assert second_state is state
  assert second_state.pert_force_dir is state.pert_force_dir
  assert second_state.pert_velocity_cache is state.pert_velocity_cache
  assert other_state is not state
  assert other_state.pert_force_dir is not state.pert_force_dir
  assert other_state.pert_velocity_cache is not state.pert_velocity_cache

  state.friction_dr_snapshot["robot"] = torch.ones((2, 2), device=env.device)
  state.geom_size_dr_snapshot = torch.full((2, 3), 0.25, device=env.device)
  state.prev_joint_vel = torch.full((2, 4), 0.5, device=env.device)

  assert torch.equal(
    second_state.friction_dr_snapshot["robot"], torch.ones((2, 2), device=env.device)
  )
  assert torch.equal(
    second_state.geom_size_dr_snapshot, torch.full((2, 3), 0.25, device=env.device)
  )
  assert torch.equal(
    second_state.prev_joint_vel, torch.full((2, 4), 0.5, device=env.device)
  )
  assert joint_ids == [0, 1]
  assert joint_names == ["finger_0", "finger_1"]
  assert robot_geom_ids == [0, 1, 2]
  assert robot_geom_names == ["geom_0", "geom_1", "geom_2"]
  assert robot_body_ids == [0, 1]
  assert robot_body_names == ["body_0", "body_1"]
