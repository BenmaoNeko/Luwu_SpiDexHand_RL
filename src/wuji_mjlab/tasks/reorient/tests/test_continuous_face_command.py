# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from wuji_mjlab.tasks.reorient.mdp.commands import (
  InHandReorientCommand,
  InHandReorientCommandCfg,
)


class _Entity:
  def __init__(self, num_envs: int, *, root: bool):
    pose = torch.zeros((num_envs, 1, 7))
    pose[:, :, 3] = 1.0
    data = SimpleNamespace(body_link_pose_w=pose)
    if root:
      default_root_state = torch.zeros((num_envs, 13))
      default_root_state[:, 3] = 1.0
      data.default_root_state = default_root_state
      data.root_link_quat_w = default_root_state[:, 3:7].clone()
      data.root_link_pos_w = default_root_state[:, :3].clone()
      data.root_link_pose_w = default_root_state[:, :7].clone()
    self.data = data
    self.indexing = SimpleNamespace(free_joint_q_adr=torch.arange(7))

  def find_bodies(self, _pattern):
    return [0], ["robot_palm_link"]


def _make_command(
  *,
  num_envs: int = 1,
  goal_sampling_mode: str = "continuous_yaw_or_random_face",
  curriculum_stage: int = 2,
):
  robot = _Entity(num_envs, root=False)
  obj = _Entity(num_envs, root=True)
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.zeros((num_envs, 4))))
  env = SimpleNamespace(
    num_envs=num_envs,
    device=torch.device("cpu"),
    step_dt=0.05,
    common_step_counter=0,
    scene={"robot": robot, "object": obj, "tip_object_contact": sensor},
    sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros((num_envs, 7)))),
    termination_manager=SimpleNamespace(
      get_term=lambda _name: torch.zeros(num_envs, dtype=torch.bool)
    ),
  )
  env.sim.data.qpos[:, 3] = 1.0
  command = InHandReorientCommand(
    InHandReorientCommandCfg(
      entity_name="object",
      robot_entity_name="robot",
      resampling_time_range=(1.0, 1.0),
      goal_sampling_mode=goal_sampling_mode,
      curriculum_stage=curriculum_stage,
      yaw_goal_probability=0.30,
      success_threshold=0.07,
      success_hold_steps=10,
      goal_switch_delay=0,
      require_tip_contact=True,
      synchronized_contact_task=True,
    ),
    env,
  )
  return command, env


def test_continuous_mode_samples_rotation_below_probability_and_flip_above(monkeypatch):
  command, _ = _make_command(num_envs=2)

  def random_values(shape, device):
    shape = (shape,) if isinstance(shape, int) else shape
    values = torch.tensor([0.29, 0.30], device=device)
    return values[: shape[0]].reshape(-1, *([1] * (len(shape) - 1))).expand(shape)

  monkeypatch.setattr(
    torch,
    "rand",
    random_values,
  )

  command._resample_command(torch.arange(2))

  assert command.rotation_mode.tolist() == [True, False]
  assert command.target_face_id[0].item() == command.current_face_id[0].item()
  assert command.target_face_id[1].item() != command.current_face_id[1].item()


def test_curriculum_stage_zero_is_a_continuous_adjacent_flip(monkeypatch):
  command, env = _make_command(
    goal_sampling_mode="curriculum_flip",
    curriculum_stage=0,
  )
  command.cfg.curriculum_enabled = True
  env._flip_curriculum_stage = 0
  monkeypatch.setattr(torch, "rand", lambda n, device: torch.ones(n, device=device))
  command._resample_command(torch.tensor([0]))

  assert command.rotation_mode.item() is False
  assert command.target_face_id.item() != command.current_face_id.item()
  env.scene["object"].data.root_link_quat_w[:] = command.goal_quat_w
  env.sim.data.qpos[:, 3:7] = command.goal_quat_w
  env.scene["tip_object_contact"].data.found[:, 0] = 1.0

  for step in range(1, 6):
    env.common_step_counter = step
    command.evaluate_contact_task()
    if step == 5:
      assert command.success_achieved.item() is True
    command._update_command()

  assert command.flip_completed.item() is False
  assert command.goal_switched.item() is True


def test_curriculum_stage_two_is_continuous_all_faces_flip(monkeypatch):
  command, env = _make_command(
    goal_sampling_mode="curriculum_flip",
    curriculum_stage=2,
  )
  command.cfg.curriculum_enabled = True
  env._flip_curriculum_stage = 2
  monkeypatch.setattr(torch, "rand", lambda n, device: torch.ones(n, device=device))
  command._resample_command(torch.tensor([0]))
  first_target = command.target_face_id.clone()
  env.scene["object"].data.root_link_quat_w[:] = command.goal_quat_w
  env.sim.data.qpos[:, 3:7] = command.goal_quat_w
  env.scene["tip_object_contact"].data.found[:, 0] = 1.0

  for step in range(1, 8):
    env.common_step_counter = step
    command.evaluate_contact_task()
    command._update_command()

  env.common_step_counter = 8
  command.evaluate_contact_task()
  assert command.success_achieved.item() is True
  command._update_command()

  assert command.flip_completed.item() is False
  assert command.goal_switched.item() is True
  assert command.target_face_id.item() != first_target.item()


def test_curriculum_command_exposes_stage_specific_success_parameters():
  command, env = _make_command(goal_sampling_mode="curriculum_flip")
  command.cfg.curriculum_enabled = True
  env._flip_curriculum_stage = 0
  command._resample_command(torch.tensor([0]))
  assert command.active_curriculum_stage.item() == 0
  assert torch.allclose(
    command.success_threshold,
    torch.tensor([math.radians(8.0)]),
    atol=1e-6,
  )
  assert command.success_hold_steps.item() == 5

  env._flip_curriculum_stage = 2
  command._resample_command(torch.tensor([0]))
  assert command.active_curriculum_stage.item() == 2
  assert torch.allclose(
    command.success_threshold,
    torch.tensor([math.radians(6.0)]),
    atol=1e-6,
  )
  assert command.success_hold_steps.item() == 8

  env._flip_curriculum_stage = 3
  command._resample_command(torch.tensor([0]))
  assert command.active_curriculum_stage.item() == 3
  assert torch.allclose(
    command.success_threshold,
    torch.tensor([math.radians(4.0)]),
    atol=1e-6,
  )
  assert command.success_hold_steps.item() == 10


def _yaw_quat(degrees: float) -> torch.Tensor:
  half_angle = math.radians(degrees) / 2.0
  return torch.tensor(
    [[math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]],
    dtype=torch.float32,
  )


def test_continuous_yaw_uses_previous_target_as_the_next_goal_base():
  command, env = _make_command()
  command._rotation_mode[:] = True
  command.goal_quat_w[:] = _yaw_quat(15.0)
  env.scene["object"].data.root_link_quat_w[:] = _yaw_quat(12.0)
  env.sim.data.qpos[:, 3:7] = _yaw_quat(12.0)

  command._advance_continuous_goals(torch.tensor([0]))

  expected = _yaw_quat(30.0)
  assert torch.allclose(command.goal_quat_w, expected, atol=1e-5)


def test_continuous_episode_counts_the_initial_top_face_in_coverage():
  command, _ = _make_command()

  command._resample_command(torch.tensor([0]))

  assert command.face_coverage_count.item() == 1


def test_continuous_flip_requires_ten_valid_contact_steps(monkeypatch):
  command, env = _make_command()
  monkeypatch.setattr(torch, "rand", lambda n, device: torch.ones(n, device=device))
  command._resample_command(torch.tensor([0]))
  env.scene["object"].data.root_link_quat_w[:] = command.goal_quat_w
  env.sim.data.qpos[:, 3:7] = command.goal_quat_w
  env.scene["tip_object_contact"].data.found[:, 0] = 1.0

  for step in range(1, 10):
    env.common_step_counter = step
    command.evaluate_contact_task()
    assert not command.success_achieved.any()

  env.common_step_counter = 11
  command.evaluate_contact_task()
  assert command.success_achieved.all()
  assert command.goal_reach_count.item() == 1


def test_continuous_flip_advances_to_a_new_face_without_completing_episode(monkeypatch):
  command, env = _make_command()
  monkeypatch.setattr(torch, "rand", lambda n, device: torch.ones(n, device=device))
  command._resample_command(torch.tensor([0]))
  first_target = command.goal_quat_w.clone()
  first_face = command.target_face_id.clone()
  env.scene["object"].data.root_link_quat_w[:] = first_target
  env.sim.data.qpos[:, 3:7] = first_target
  env.scene["tip_object_contact"].data.found[:, 0] = 1.0

  for step in range(1, 10):
    env.common_step_counter = step
    command.evaluate_contact_task()
    command._update_command()

  env.common_step_counter = 10
  command.evaluate_contact_task()
  assert command.success_achieved.all()
  assert not command.flip_completed.any()
  command._update_command()
  assert command.goal_switched.all()
  assert command.target_face_id.item() != first_face.item()
  assert not torch.allclose(command.goal_quat_w, first_target)

  second_target = command.goal_quat_w.clone()
  env.scene["object"].data.root_link_quat_w[:] = second_target
  env.sim.data.qpos[:, 3:7] = second_target
  for step in range(11, 20):
    env.common_step_counter = step
    command.evaluate_contact_task()
    command._update_command()
  env.common_step_counter = 20
  command.evaluate_contact_task()
  assert command.goal_reach_count.item() == 2
