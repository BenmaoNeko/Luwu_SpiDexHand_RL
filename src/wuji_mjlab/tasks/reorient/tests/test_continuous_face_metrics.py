# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from wuji_mjlab.tasks.reorient.mdp.curriculums import reorient_success_curriculum
from wuji_mjlab.tasks.reorient.tests.test_continuous_face_command import _make_command


def test_continuous_flip_metrics_count_chain_coverage_and_geometry():
  command, _ = _make_command()

  command._current_face_id[:] = 4  # +Z
  command._target_face_id[:] = 2  # +Y: adjacent
  command._record_continuous_flip(torch.tensor([True]))
  assert command.consecutive_flip_count.item() == 1
  assert command.face_coverage_count.item() == 1
  assert command.adjacent_flip_count.item() == 1
  assert command.opposite_flip_count.item() == 0

  command._current_face_id[:] = 2
  command._target_face_id[:] = 0  # +X: adjacent
  command._record_continuous_flip(torch.tensor([True]))
  assert command.consecutive_flip_count.item() == 2
  assert command.face_coverage_count.item() == 2
  assert command.adjacent_flip_count.item() == 2

  command._current_face_id[:] = 0
  command._target_face_id[:] = 1  # -X: opposite
  command._record_continuous_flip(torch.tensor([True]))
  assert command.consecutive_flip_count.item() == 3
  assert command.face_coverage_count.item() == 3
  assert command.opposite_flip_count.item() == 1


def test_revisiting_a_face_does_not_increase_coverage_but_keeps_chain_count():
  command, _ = _make_command()
  command._current_face_id[:] = 4
  command._target_face_id[:] = 2
  command._record_continuous_flip(torch.tensor([True]))
  command._current_face_id[:] = 0
  command._target_face_id[:] = 2
  command._record_continuous_flip(torch.tensor([True]))

  assert command.consecutive_flip_count.item() == 2
  assert command.face_coverage_count.item() == 1


def test_reset_clears_continuous_flip_metrics_only_for_selected_envs():
  command, _ = _make_command(num_envs=2)
  command._consecutive_flip_count[:] = torch.tensor([3, 5])
  command._face_visited_mask[:] = torch.tensor([0b101, 0b100001])
  command._adjacent_flip_count[:] = torch.tensor([2, 4])
  command._opposite_flip_count[:] = torch.tensor([1, 3])

  command._reset_episode_task_state(torch.tensor([0]))

  assert command.consecutive_flip_count.tolist() == [0, 5]
  assert command.face_coverage_count.tolist() == [0, 2]
  assert command.adjacent_flip_count.tolist() == [0, 4]
  assert command.opposite_flip_count.tolist() == [0, 3]


def test_task_aware_curriculum_can_require_two_flips_for_transfer():
  command, _ = _make_command(num_envs=2)
  command.metrics["goal_reach_count"][:] = torch.tensor([1.0, 2.0])
  command.metrics["rotation_goal_reach_count"][:] = torch.tensor([0.0, 0.0])
  command.metrics["flip_success_count"][:] = torch.tensor([1.0, 2.0])
  env = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    command_manager=SimpleNamespace(get_term=lambda _name: command),
  )

  result = reorient_success_curriculum(
    env,
    torch.tensor([0, 1]),
    task_aware=True,
    rotation_count_threshold=3,
    flip_count_threshold=2,
  )

  assert result["success_env_frac"] == 0.5
  assert result["mean_goal_reach_count"] == 1.0
