import math

import pytest
import torch


def test_reward_matches_requested_nodes_and_interpolates():
  from wuji_mjlab.tasks.reorient.mdp.turn_specialization import residual_score
  degrees = torch.tensor([0., 5., 7.5, 10., 15., 20., 25., 40.])
  assert residual_score(torch.deg2rad(degrees)).tolist() == pytest.approx(
    [4., 2., .75, -.5, -1., -2.5, -4., -4.], abs=1e-5)


def test_task_allocation_preserves_legacy_and_specializes_4096():
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import assign_full_cube_task_groups
  assert assign_full_cube_task_groups(8).bincount(minlength=3).tolist() == [2, 2, 4]
  groups = assign_full_cube_task_groups(4096, fractions=(.1, .8, .1))
  assert groups.bincount(minlength=3).tolist() == [410, 3277, 409]
  assert assign_full_cube_task_groups(4, fractions=(.1, .8, .1)).numel() == 4
  with pytest.raises(ValueError):
    assign_full_cube_task_groups(8, fractions=(.1, .8, .2))


def test_partial_start_has_remaining_angle_and_signed_quarter_target():
  from wuji_mjlab.tasks.reorient.mdp.turn_specialization import sample_turn_starts
  torch.manual_seed(3)
  angle, goal, partial = sample_turn_starts(2000, 'cpu', .5)
  assert .45 < partial.float().mean() < .55
  assert torch.allclose(goal.abs(), torch.full_like(goal, math.pi / 2))
  assert (angle[~partial] == 0).all()
  assert (angle[partial].sign() == goal[partial].sign()).all()
  assert torch.rad2deg(angle[partial].abs()).min() >= 45
  assert torch.rad2deg(angle[partial].abs()).max() <= 80
  remaining = torch.rad2deg((goal - angle).abs())[partial]
  assert remaining.min() >= 10 and remaining.max() <= 45


def test_metrics_use_finished_attempts_and_count_timeout_without_success():
  from wuji_mjlab.tasks.reorient.mdp.turn_specialization import TurnMoveStats
  stats = TurnMoveStats(3, 'cpu')
  ids = torch.ones(3, dtype=torch.bool)
  stats.observe(10, ids, torch.zeros(3, dtype=torch.bool), torch.tensor([.1, .2, .3]), .05)
  stats.observe(14, ids, torch.tensor([True, False, False]), torch.deg2rad(torch.tensor([3., 8., 30.])), .05)
  stats.finish(torch.tensor([1]), torch.deg2rad(torch.tensor([3., 8., 30.])), .05, 14, timeout=True)
  log = stats.logs()
  assert log['turn_post_move/finished'] == 2
  assert log['turn_post_move/success_rate'] == .5
  assert log['turn_post_move/success_seconds'] == pytest.approx(.2)
  assert log['turn_post_move/timeout_residual_deg'] == pytest.approx(8.)
  assert log['turn_post_move/final_le_5deg'] == .5
  assert log['turn_post_move/final_le_10deg'] == 1.
  stats.finish(torch.tensor([1]), torch.ones(3), .05, 15, timeout=True)
  assert stats.logs()['turn_post_move/finished'] == 2
  assert stats.active.tolist() == [False, False, True]


def test_smart_config_and_discrete_curriculum():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_smart_cube_transfer_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.mdp.smart_cube_curriculum import SmartCubeCurriculum
  cfg = luwu_hand_rubik_smart_cube_transfer_env_cfg(num_envs=4096)
  cmd = cfg.commands['reorient_command']
  assert cmd.task_fractions == (.1, .8, .1)
  assert cmd.partial_turn_start_probability == .5
  assert cmd.turn_180_probability == .2
  assert cfg.rewards['rubik_smart_move_residual'].params['specialized']
  assert SmartCubeCurriculum(cmd.smart_cube_curriculum).beta == 1.


def test_partial_reset_replays_move_crossing_with_delay():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp import smart_cube_observations as m
  p = m.SmartCubeProfile(threshold_degrees=(48., 48.), move_delay_steps=1)
  state = m.create_smart_cube_state(2, 'cpu', p)
  env = SimpleNamespace(scene=SimpleNamespace(device='cpu'), num_envs=2, device='cpu',
                        _smart_cube_state=state, _rubik_partial_turn=torch.tensor([True, False]))
  m.reset_smart_cube_observation_state(env, torch.arange(2), p)
  angles = torch.zeros(2, 6)
  angles[0, 4] = math.radians(60)
  out = m.step_move_model(angles, state, p, step_id=0)
  assert out[0, 4] == 0
  assert not state.move_event[0]
  out = m.step_move_model(angles, state, p, step_id=1)
  assert out[0, 4] == pytest.approx(math.pi / 2)
  assert state.move_event.tolist() == [True, False]


def test_specialized_reward_is_gated_and_dt_integrated():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.rewards import rubik_smart_move_residual_reward
  command = SimpleNamespace(evaluate_task=lambda: None, update_smart_move_pending=lambda: None,
    u_angle_error=torch.deg2rad(torch.tensor([0., 5., 10., 15., 25., 0., 0.])),
    smart_move_pending=torch.tensor([True, True, True, True, True, False, True]),
    phase=torch.tensor([1, 1, 1, 1, 1, 1, 0]))
  env = SimpleNamespace(step_dt=.05, command_manager=SimpleNamespace(get_term=lambda _: command))
  reward = rubik_smart_move_residual_reward(env, specialized=True) * env.step_dt
  assert reward.tolist() == pytest.approx([4., 2., -.5, -1., -4., 0., 0.], abs=1e-5)


def test_partial_turn_environment_reset_and_step(tmp_path):
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_smart_cube_transfer_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool
  pool = tmp_path / 'pool.json'
  generate_solver_plan_pool(pool, count=2, seed=42, min_scramble_moves=2, max_scramble_moves=2)
  cfg = luwu_hand_rubik_smart_cube_transfer_env_cfg(num_envs=10, plan_pool_path=str(pool))
  cfg.commands['reorient_command'].partial_turn_start_probability = 1.
  env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
  try:
    obs, _ = env.reset(seed=42)
    cmd = env.command_manager.get_term('reorient_command')
    turns = cmd.episode_kind == 1
    assert cmd.episode_kind.bincount(minlength=3).tolist() == [1, 8, 1]
    assert (cmd._episode_budget_steps[turns] == round(120 / env.step_dt)).all()
    angle = torch.rad2deg(cmd.u_angle[turns])
    assert angle.abs().min() >= 44.99 and angle.abs().max() <= 80.01
    assert torch.allclose(cmd.u_target_angle[turns].abs(), torch.full_like(angle, math.pi/2), atol=1e-5)
    assert (cmd.u_target_angle[turns].sign() == angle.sign()).all()
    for _ in range(3):
      obs, reward, _, _, _ = env.step(torch.zeros((10, env.action_manager.total_action_dim)))
      assert torch.isfinite(reward).all()
    assert 'turn_post_move/finished' in env.extras['log']
    # Force independent-turn phase timeouts to exercise pre-reset outcome closure.
    cmd._goal_start_step[turns] = -10000
    env.step(torch.zeros((10, env.action_manager.total_action_dim)))
    assert env._turn_move_stats.logs()['turn_post_move/timeouts'] > 0
  finally:
    env.close()


def test_quarter_turn_sampler_disables_half_turns_and_avoids_target_drift():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.rubik_command import RubikUFaceCommand

  count = 200
  command = SimpleNamespace(device='cpu',
    cfg=SimpleNamespace(turn_180_probability=0., turn_specialization=True),
    _env=SimpleNamespace(_rubik_curriculum={'turn_stage': torch.tensor(2), 'turn90_trials': 0, 'turn180_trials': 0}),
    u_angle=torch.full((count,), math.radians(86)), u_target_angle=torch.zeros(count),
    turn_start_error=torch.zeros(count), _turn_goal_180=torch.zeros(count, dtype=torch.bool))
  RubikUFaceCommand._sample_next_turn_goal(command, torch.arange(count))
  assert not command._turn_goal_180.any()
  # From a near-90-degree physical state, the next targets are 0 or 180, not -4/176.
  assert torch.sin(command.u_target_angle).abs().max() < 1e-6
  assert command._env._rubik_curriculum['turn180_trials'] == 0
  command.cfg.turn_180_probability = 1.
  RubikUFaceCommand._sample_next_turn_goal(command, torch.arange(count))
  assert command._turn_goal_180.all()
  assert command._env._rubik_curriculum['turn180_trials'] == count
