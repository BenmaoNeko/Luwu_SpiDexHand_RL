# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from wuji_mjlab.tasks.reorient.mdp.commands import (
  InHandReorientCommand,
  InHandReorientCommandCfg,
)


class _FakeEntity:
  def __init__(self, *, num_envs: int, with_root: bool):
    pose = torch.zeros((num_envs, 1, 7))
    pose[:, :, 3] = 1.0
    data = SimpleNamespace(body_link_pose_w=pose)
    if with_root:
      default_root_state = torch.zeros((num_envs, 13))
      default_root_state[:, 3] = 1.0
      data.default_root_state = default_root_state
      data.root_link_quat_w = default_root_state[:, 3:7].clone()
      data.root_link_pos_w = default_root_state[:, :3].clone()
    self.data = data

  def find_bodies(self, _pattern):
    return [0], ["robot_palm_link"]


def _make_command(*, yaw_probability: float, num_envs: int = 1) -> tuple[InHandReorientCommand, SimpleNamespace]:
  robot = _FakeEntity(num_envs=num_envs, with_root=False)
  obj = _FakeEntity(num_envs=num_envs, with_root=True)
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.zeros((num_envs, 4))))
  env = SimpleNamespace(
    num_envs=num_envs,
    device=torch.device("cpu"),
    step_dt=0.05,
    scene={"robot": robot, "object": obj, "tip_object_contact": sensor},
  )
  command = InHandReorientCommand(
    InHandReorientCommandCfg(
      entity_name="object",
      robot_entity_name="robot",
      resampling_time_range=(1.0, 1.0),
      goal_sampling_mode="incremental_yaw_or_flip",
      yaw_goal_probability=yaw_probability,
      success_threshold=0.07,
      success_hold_steps=2,
      goal_switch_delay=0,
      require_tip_contact=True,
    ),
    env,
  )
  command._sample_goal_in_world(torch.arange(num_envs))
  return command, env


def test_rotation_cannot_succeed_without_tip_contact():
  command, env = _make_command(yaw_probability=1.0)
  env.scene["object"].data.root_link_quat_w[:] = command.goal_quat_w

  for _ in range(command.cfg.success_hold_steps + 2):
    command._update_command()

  assert not command.success_achieved.any()
  assert command.goal_reach_count.item() == 0


def test_flip_success_is_one_shot_and_sets_episode_completion():
  command, env = _make_command(yaw_probability=0.0)
  env.scene["object"].data.root_link_quat_w[:] = command.goal_quat_w
  env.scene["tip_object_contact"].data.found[:, 0] = 1.0

  for _ in range(command.cfg.success_hold_steps):
    command._update_command()

  assert command.success_achieved.all()
  assert command.flip_completed.all()
  assert command.goal_reach_count.item() == 1

  for _ in range(5):
    command._update_command()
  assert command.goal_reach_count.item() == 1
  assert command.flip_completed.all()


def test_drop_has_priority_over_orientation_and_contact_success():
  command, env = _make_command(yaw_probability=0.0)
  env.scene["object"].data.root_link_quat_w[:] = command.goal_quat_w
  env.scene["tip_object_contact"].data.found[:, 0] = 1.0
  env.termination_manager = SimpleNamespace(
    get_term=lambda _name: torch.tensor([True])
  )

  for _ in range(command.cfg.success_hold_steps + 1):
    command._update_command()

  assert not command.success_achieved.any()
  assert not command.flip_completed.any()
