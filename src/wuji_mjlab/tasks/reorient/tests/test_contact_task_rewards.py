# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import pytest

from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wuji_mjlab.tasks.reorient.config.luwu_hand.env_cfgs import luwu_hand_reorient_env_cfg

from wuji_mjlab.tasks.reorient.mdp.contact_task_state import (
  box_surface_distance,
  bounded_monotonic_increment,
  contact_milestone_events,
  goal_error_progress,
  mean_budgeted_increment,
  per_finger_surface_proximity_score,
  surface_proximity_score,
)
from wuji_mjlab.tasks.reorient.mdp.rewards import (
  action_target_smoothness_penalty_from_normalized_targets,
  flip_success_reward,
  approach_progress_reward,
)


def _integrated_reward(env, name):
  """Exercise the real manager's dt integration with one production term."""
  manager = RewardManager.__new__(RewardManager)
  manager._env = env
  manager._scale_by_dt = True
  manager._term_names = [name]
  manager._term_cfgs = [luwu_hand_reorient_env_cfg(num_envs=env.num_envs).rewards[name]]
  manager._reward_buf = torch.zeros(env.num_envs)
  manager._step_reward = torch.zeros((env.num_envs, 1))
  manager._episode_sums = {name: torch.zeros(env.num_envs)}
  return manager.compute(env.step_dt)


@pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
def test_drop_event_cost_is_ten_after_manager_integration(dt):
  env = SimpleNamespace(
    num_envs=3, step_dt=dt,
    termination_manager=SimpleNamespace(
      get_term=lambda name: torch.tensor([True, False, False])
    ),
  )
  torch.testing.assert_close(_integrated_reward(env, "cage_drop"), torch.tensor([-10., 0., 0.]))


def test_collision_cost_uses_all_sensor_channels():
  found = torch.zeros((3, 16))
  found[1, 0] = 1
  found[2] = 1
  env = SimpleNamespace(
    num_envs=3, device="cpu", step_dt=0.05,
    scene={"finger_collision": SimpleNamespace(data=SimpleNamespace(found=found))},
  )
  torch.testing.assert_close(
    _integrated_reward(env, "finger_collision"),
    -0.08 * 0.05 * torch.tensor([0., 1. / 16., 1.]),
  )


def test_approach_on_drop_updates_best_without_payout_or_later_catchup():
  command = SimpleNamespace(
    task_reward_active=torch.ones(2, dtype=torch.bool),
    _approach_best_score=torch.zeros((2, 4)),
    _approach_initialized=torch.zeros(2, dtype=torch.bool),
  )
  sites = torch.full((2, 4, 3), 0.1)
  dropped = torch.tensor([True, False])
  env = SimpleNamespace(
    num_envs=2, device="cpu", step_dt=0.05,
    command_manager=SimpleNamespace(get_term=lambda name: command),
    termination_manager=SimpleNamespace(get_term=lambda name: dropped),
    scene={
      "robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=sites)),
      "object": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.zeros((2, 3)),
                             root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)),
        indexing=SimpleNamespace(geom_ids=torch.tensor([0])),
      ),
    },
    sim=SimpleNamespace(model=SimpleNamespace(geom_size=torch.full((2, 1, 3), 0.03))),
  )
  kwargs = dict(robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object"), step_budget=2.)
  assert not approach_progress_reward(env, **kwargs).any()  # baseline
  sites.fill_(0.02)
  reward = approach_progress_reward(env, **kwargs) * env.step_dt
  assert reward[0] == 0
  assert reward[1] > 0
  torch.testing.assert_close(command._approach_best_score, torch.ones((2, 4)))
  dropped.zero_()
  assert not approach_progress_reward(env, **kwargs).any()


def test_box_surface_distance_uses_surface_and_respects_box_rotation():
  site = torch.tensor([[0.08, 0.0, 0.0]])
  center = torch.zeros((1, 3))
  half_size = torch.tensor([[0.05, 0.03, 0.02]])
  identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  quarter_turn_z = torch.tensor(
    [[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]]
  )

  distance_identity = box_surface_distance(site, center, identity, half_size)
  distance_rotated = box_surface_distance(
    torch.tensor([[0.0, 0.08, 0.0]]), center, quarter_turn_z, half_size
  )

  assert torch.allclose(distance_identity, torch.tensor([0.03]))
  assert torch.allclose(distance_rotated, distance_identity, atol=1e-6)
  assert torch.equal(
    box_surface_distance(torch.zeros((1, 3)), center, identity, half_size),
    torch.zeros(1),
  )


def test_bounded_monotonic_increment_cannot_reward_retreat_and_return():
  best = torch.tensor([0.2])
  delta, best = bounded_monotonic_increment(torch.tensor([0.7]), best)
  assert torch.allclose(delta, torch.tensor([0.5]))
  assert torch.allclose(best, torch.tensor([0.7]))

  retreat_delta, best = bounded_monotonic_increment(torch.tensor([0.3]), best)
  return_delta, best = bounded_monotonic_increment(torch.tensor([0.7]), best)
  assert torch.equal(retreat_delta, torch.zeros(1))
  assert torch.equal(return_delta, torch.zeros(1))


def test_contact_milestones_are_one_shot_and_require_stable_contact():
  hold = torch.zeros((1, 2))
  first_awarded = torch.zeros(1, dtype=torch.bool)
  second_awarded = torch.zeros(1, dtype=torch.bool)

  for _ in range(3):
    first, second, hold, first_awarded, second_awarded = contact_milestone_events(
      torch.tensor([1.0]), hold, first_awarded, second_awarded, 0.05, 0.2
    )
    assert torch.equal(first, torch.zeros(1))
    assert torch.equal(second, torch.zeros(1))

  first, second, hold, first_awarded, second_awarded = contact_milestone_events(
    torch.tensor([1.0]), hold, first_awarded, second_awarded, 0.05, 0.2
  )
  assert torch.allclose(first, torch.tensor([1.0]))
  assert torch.equal(second, torch.zeros(1))

  first, second, hold, first_awarded, second_awarded = contact_milestone_events(
    torch.tensor([2.0]), hold, first_awarded, second_awarded, 0.05, 0.2
  )
  assert torch.equal(first, torch.zeros(1))
  assert torch.equal(second, torch.zeros(1))

  for _ in range(3):
    first, second, hold, first_awarded, second_awarded = contact_milestone_events(
      torch.tensor([2.0]), hold, first_awarded, second_awarded, 0.05, 0.2
    )
  assert torch.allclose(second, torch.tensor([1.0]))

  first, second, *_ = contact_milestone_events(
    torch.tensor([2.0]), hold, first_awarded, second_awarded, 0.05, 0.2
  )
  assert torch.equal(first, torch.zeros(1))
  assert torch.equal(second, torch.zeros(1))


def test_goal_error_progress_is_zero_inside_tolerance_and_bounded():
  initial = torch.tensor([math.pi / 2])
  tolerance = math.radians(4.0)
  progress = goal_error_progress(initial, torch.tensor([math.pi / 4]), tolerance)
  expected = (initial - math.pi / 4) / (initial - tolerance)
  assert torch.allclose(progress, expected, atol=1e-6)
  assert torch.equal(
    goal_error_progress(initial, torch.tensor([0.0]), tolerance), torch.ones(1)
  )
  assert torch.equal(
    goal_error_progress(initial, torch.tensor([math.pi]), tolerance), torch.zeros(1)
  )


def test_surface_proximity_score_is_finite_and_monotonic():
  distances = torch.tensor([[0.0, 0.01, 0.02, 0.04]])
  scores = surface_proximity_score(distances, length_scale=0.02)
  assert scores.shape == (1,)
  assert torch.isfinite(scores).all()
  assert 0.0 < scores.item() < 1.0


def test_per_finger_proximity_scores_keep_independent_progress_channels():
  distances = torch.tensor([[0.0, 0.02, 0.04, 0.08]])
  scores = per_finger_surface_proximity_score(distances, length_scale=0.02)

  assert scores.shape == distances.shape
  assert torch.all(scores[:, :-1] > scores[:, 1:])
  assert torch.all((scores >= 0.0) & (scores <= 1.0))


def test_approach_budget_is_shared_across_fingertips():
  increment = torch.ones((2, 4))
  assert torch.equal(mean_budgeted_increment(increment, 2.0), torch.tensor([2.0, 2.0]))


def test_smoothness_penalty_uses_normalized_smoothed_targets_and_mean():
  current = torch.tensor([[0.0, 0.5, -1.0, 1.0]])
  previous = torch.tensor([[0.0, 0.0, -0.5, 0.5]])
  previous_previous = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

  first, second = action_target_smoothness_penalty_from_normalized_targets(
    current, previous, previous_previous
  )

  assert torch.allclose(first, torch.tensor([0.1875]))
  assert torch.allclose(second, torch.tensor([0.0625]))


def test_flip_success_is_masked_when_cage_drop_occurs_on_same_step():
  command = SimpleNamespace(
    success_achieved=torch.tensor([True]),
    rotation_mode=torch.tensor([False]),
  )
  env = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    step_dt=0.05,
    command_manager=SimpleNamespace(get_term=lambda _name: command),
    termination_manager=SimpleNamespace(
      get_term=lambda _name: torch.tensor([True])
    ),
  )

  assert torch.equal(flip_success_reward(env), torch.zeros(1))
