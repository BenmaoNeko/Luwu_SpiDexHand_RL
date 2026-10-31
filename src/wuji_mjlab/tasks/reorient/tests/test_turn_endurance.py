from types import SimpleNamespace

import pytest
import torch


def test_streak_reward_only_pays_completed_independent_turns():
  from wuji_mjlab.tasks.reorient.mdp.rewards import rubik_turn_streak_reward

  command = SimpleNamespace(evaluate_task=lambda: None,
    turn_completed=torch.tensor([True, True, True, False, True, True]),
    turn_success_count=torch.tensor([1., 2., 10., 10., 3., 0.]),
    episode_kind=torch.tensor([1, 1, 1, 1, 2, 1]))
  env = SimpleNamespace(step_dt=.05, command_manager=SimpleNamespace(get_term=lambda _: command))
  # The manager multiplies by dt and configured weight=10.
  result = rubik_turn_streak_reward(env) * env.step_dt * 10
  assert result.tolist() == pytest.approx([10, 20, 100, 0, 0, 0])
  command.turn_success_count[:] = 0
  assert (rubik_turn_streak_reward(env) == 0).all()


def test_positive_residual_requires_new_progress_but_penalty_remains():
  from wuji_mjlab.tasks.reorient.mdp.rewards import rubik_smart_move_residual_reward

  command = SimpleNamespace(evaluate_task=lambda: None, update_smart_move_pending=lambda: None,
    u_angle_error=torch.deg2rad(torch.tensor([2., 2., 15., 2.])),
    turn_progress_delta=torch.tensor([0., .1, 0., .1]),
    smart_move_pending=torch.tensor([True, True, True, False]), phase=torch.ones(4, dtype=torch.long))
  env = SimpleNamespace(step_dt=.05, command_manager=SimpleNamespace(get_term=lambda _: command))
  result = rubik_smart_move_residual_reward(env, specialized=True, positive_progress_only=True) * env.step_dt
  assert result.tolist() == pytest.approx([0, 3.2, -1, 0], abs=1e-5)


def test_endurance_configuration():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_smart_cube_transfer_env_cfg,
  )

  cfg = luwu_hand_rubik_smart_cube_transfer_env_cfg(num_envs=4096)
  cmd = cfg.commands['reorient_command']
  assert cmd.task_fractions == (.1, .8, .1)
  assert cmd.turn_180_probability == .2
  assert cmd.turn_episode_seconds == 120
  assert cfg.rewards['rubik_u_turn_success'].weight == 5
  assert cfg.rewards['rubik_u_turn_progress'].weight == 1
  assert cfg.rewards['rubik_smart_move_residual'].weight == .5
  assert cfg.rewards['rubik_smart_move_residual'].params['positive_progress_only']
  assert cfg.rewards['rubik_turn_streak'].weight == 10


def test_success_advances_to_next_turn_without_episode_end():
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import RubikFullCubeCommand

  command = object.__new__(RubikFullCubeCommand)
  command._pending_goal_advance = torch.tensor([True])
  command.phase = torch.tensor([1])
  command.episode_kind = torch.tensor([1])
  command.current_top_face = torch.tensor([4])
  command.target_face_id = torch.tensor([4])
  command.turn_index = torch.tensor([0])
  command._goal_terminal = torch.tensor([False])
  installed = []
  command._install_basic_turn_goal = lambda ids: installed.extend(ids.tolist())
  command._advance_pending_goals()
  assert installed == [0]
  assert command.turn_index.item() == 1
  assert not command._goal_terminal.item()
  assert not command._pending_goal_advance.item()
