import math

import pytest
import torch


def rotations(degrees):
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import _matrix_from_xyz
  angles = torch.zeros(len(degrees), 20, 3)
  angles[:, :8, 2] = torch.deg2rad(torch.tensor(degrees))[:, None]
  return _matrix_from_xyz(angles)


def monitor(count=1):
  from wuji_mjlab.tasks.reorient.mdp.flip_specialization import FlipPreservation
  result = FlipPreservation(count, 'cpu')
  result.reset(torch.arange(count), rotations([0.] * count), torch.zeros(count, 6))
  return result


def step(state, frame, degrees, arrived=False, valid=True, contact=True):
  count = state.error.numel()
  state.update(frame, torch.full((count,), arrived), torch.full((count,), valid),
    rotations([degrees] * count), torch.zeros(count, 6), .05,
    contact=torch.full((count,), contact))


def test_deformation_is_not_checked_or_penalized_during_flip():
  state = monitor()
  state.start_goal(torch.tensor([0]))
  step(state, 1, 20.)
  assert state.error.item() == 0
  assert state.boundary_penalty.item() == 0
  assert not state.settling.item()
  step(state, 2, 20., arrived=True)
  assert state.settling.item()
  assert state.error.item() == pytest.approx(math.radians(20), abs=1e-5)
  assert state.boundary_penalty.item() > 0
  step(state, 3, 20.)
  assert state.boundary_penalty.item() == 0


def test_ninety_degrees_is_aligned_but_move_gets_only_small_event_penalty():
  state = monitor()
  state.start_goal(torch.tensor([0]))
  state.update(1, torch.tensor([True]), torch.tensor([True]), rotations([90.]),
    torch.full((1, 6), math.pi/2), .05, contact=torch.tensor([True]),
    move_event=torch.tensor([True]), move_step=torch.tensor([1]))
  assert state.error.item() < 1e-3
  assert state.boundary_penalty.item() < 1e-3
  assert state.move_penalty.item() == 1
  state.update(2, torch.tensor([False]), torch.tensor([True]), rotations([90.]),
    torch.full((1, 6), math.pi/2), .05, contact=torch.tensor([True]),
    move_event=torch.tensor([True]), move_step=torch.tensor([1]))
  assert state.move_penalty.item() == 0


def test_twenty_degree_error_can_be_repaired_without_oscillation_reward():
  state = monitor()
  state.start_goal(torch.tensor([0]))
  step(state, 1, 20., arrived=True)
  step(state, 2, 15.)
  first = state.improvement.clone()
  assert first.item() > 0
  step(state, 3, 20.)
  assert state.improvement.item() == 0
  step(state, 4, 15.)
  assert state.improvement.item() == pytest.approx(0., abs=1e-5)
  step(state, 5, 3.)
  assert state.improvement.item() > 0
  for frame in range(6, 10):
    step(state, frame, 3.)
  assert state.done.item() and state.aligned_event.item()


def test_bad_alignment_times_out_and_advances_instead_of_blocking():
  state = monitor()
  state.start_goal(torch.tensor([0]))
  step(state, 1, 20., arrived=True)
  for frame in range(2, 22):
    step(state, frame, 20.)
  assert state.done.item() and state.timeout_event.item()
  assert not state.aligned_event.item()
  assert not state.settling.item()


def test_repeated_reads_do_not_duplicate_boundary_or_improvement():
  state = monitor()
  state.start_goal(torch.tensor([0]))
  step(state, 1, 20., arrived=True)
  penalty = state.boundary_penalty.clone()
  step(state, 1, 0., arrived=True)
  assert torch.equal(state.boundary_penalty, penalty)
  assert state.error.item() == pytest.approx(math.radians(20), abs=1e-5)


def test_all_layers_and_cubies_contribute_to_maximum_and_mean():
  state = monitor()
  maximum, mean = state.shape_error(rotations([0.]), torch.deg2rad(torch.tensor([[0., 0., 0., 0., 0., 20.]])))
  assert torch.rad2deg(maximum).item() == pytest.approx(20., abs=1e-4)
  assert torch.rad2deg(mean).item() == pytest.approx(20 / 26, abs=1e-4)
  maximum, mean = state.shape_error(rotations([20.]), torch.zeros(1, 6))
  assert torch.rad2deg(maximum).item() == pytest.approx(20., abs=1e-4)
  assert torch.rad2deg(mean).item() == pytest.approx(8 * 20 / 26, abs=1e-4)


def test_repair_cannot_pay_more_than_creating_the_error_costs():
  state = monitor()
  state.start_goal(torch.tensor([0]))
  step(state, 1, 20., arrived=True)
  penalty = state.boundary_penalty.item()
  step(state, 2, 0.)
  gain = state.improvement.item()
  assert gain > 0
  assert -2 * penalty + gain < 0


def test_alignment_requires_contact_and_requested_face_but_deadline_always_advances():
  state = monitor()
  state.start_goal(torch.tensor([0]))
  step(state, 1, 20., arrived=True)
  for frame in range(2, 22):
    step(state, frame, 0., valid=False, contact=False)
    assert state.improvement.item() == 0
  assert state.done.item() and not state.aligned_event.item()


def test_specialist_is_separate_and_preserves_actor_contract():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_specialist_env_cfg,
    luwu_hand_rubik_smart_cube_transfer_env_cfg,
  )
  cfg = luwu_hand_rubik_flip_specialist_env_cfg(num_envs=4)
  old = luwu_hand_rubik_smart_cube_transfer_env_cfg(num_envs=4)
  cmd = cfg.commands['reorient_command']
  assert cmd.task_fractions == (1., 0., 0.)
  assert cmd.flip_specialization and not cmd.turn_specialization
  assert cmd.flip_scramble_probability == .5
  assert cmd.flip_alignment_seconds == 1.
  assert cmd.flip_alignment_hold_steps == 5
  assert cmd.flip_alignment_tolerance_rad == pytest.approx(math.radians(5))
  assert cfg.rewards['flip_align_boundary'].weight == -2.
  assert cfg.rewards['flip_align_improvement'].weight == 1.
  assert cfg.rewards['flip_align_move'].weight == -2.
  assert not any(name in cfg.rewards for name in
    ('flip_clean_deviation', 'flip_clean_speed', 'flip_clean_damage'))
  assert list(cfg.observations['policy'].terms) == list(old.observations['policy'].terms)
  assert old.commands['reorient_command'].task_fractions == (.1, .8, .1)
  assert 'rubik_turn_streak' not in cfg.rewards


def test_next_goal_preserves_episode_baseline_and_bottom_uses_waypoint():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import RubikFullCubeCommand
  command = object.__new__(RubikFullCubeCommand)
  command._pending_goal_advance = torch.tensor([True])
  command.phase = torch.tensor([0])
  command.episode_kind = torch.tensor([0])
  command.current_top_face = torch.tensor([4])
  command.target_face_id = torch.tensor([0])
  command.requested_face = torch.tensor([0])
  command.bottom_intermediate = torch.tensor([False])
  command._flip_goal_bottom = torch.tensor([False])
  command.cfg = SimpleNamespace(flip_specialization=True)
  def sample(ids):
    command.target_face_id[ids] = 1
    command._flip_goal_bottom[ids] = True
  command._sample_next_flip_goal = sample
  installed = []
  command._install_flip_or_turn_start = lambda ids, face: installed.append((ids.tolist(), face.tolist()))
  command._advance_pending_goals()
  assert installed == [([0], [1])]
  assert not command._pending_goal_advance.any()


def test_specialist_transfer_requires_pinned_continuous_adapter(tmp_path, monkeypatch):
  from wuji_mjlab.rl import transfer
  path = tmp_path / 'model.pt'
  path.write_bytes(b'checkpoint')
  monkeypatch.setattr(transfer, 'sha256_file', lambda _: transfer.RUBIK_CONTINUOUS_TRANSFER_SOURCE_SHA256)
  transfer.validate_transfer_layout_args('LuwuHand_Rubik_FlipSpecialist', path,
                                        'continuous162-to-rubik204')
  with pytest.raises(ValueError):
    transfer.validate_transfer_layout_args('LuwuHand_Rubik_FlipSpecialist', path, None)


def test_stationary_or_unproductive_alignment_cannot_farm_positive_stability():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.rewards import rubik_cube_position_stability_reward
  command = SimpleNamespace(
    evaluate_task=lambda: None, episode_kind=torch.zeros(3), phase=torch.zeros(3),
    tip_contact_count=torch.ones(3), cube_position_delta_valid=torch.ones(3, dtype=torch.bool),
    cube_position_delta=torch.zeros(3), progress_delta=torch.tensor([0., .1, .1]),
    _read_flip_target_dot=lambda: torch.zeros(3),
    flip_preservation=SimpleNamespace(settling=torch.tensor([False, False, True]),
                                    improvement=torch.zeros(3), hold=torch.zeros(3)),
    cfg=SimpleNamespace(min_contact_fingertips=1, flip_specialization=True,
                        continuous_flip_upright_tolerance_rad=.07),
  )
  env = SimpleNamespace(step_dt=.05, num_envs=3, device='cpu',
    command_manager=SimpleNamespace(get_term=lambda _: command),
    termination_manager=SimpleNamespace(get_term=lambda _: torch.zeros(3, dtype=torch.bool)))
  reward = rubik_cube_position_stability_reward(env) * env.step_dt
  assert reward.tolist() == pytest.approx([0., 1., 0.])


def test_physical_flip_specialist_reset_arrival_alignment_and_advance(tmp_path):
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_specialist_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool
  pool = tmp_path / 'pool.json'
  generate_solver_plan_pool(pool, count=2, seed=42, min_scramble_moves=2, max_scramble_moves=2)
  cfg = luwu_hand_rubik_flip_specialist_env_cfg(num_envs=4, plan_pool_path=str(pool))
  cfg.commands['reorient_command'].flip_scramble_probability = 1.
  env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
  try:
    obs, _ = env.reset(seed=42)
    cmd = env.command_manager.get_term('reorient_command')
    assert (cmd.episode_kind == 0).all()
    assert obs['policy'].shape == (4, 204)
    assert cmd.episode_initial_joint_pos.abs().sum() > 0
    baseline = cmd.episode_initial_joint_pos.clone()
    assert torch.allclose(cmd.expected_joint_pos, baseline)
    for _ in range(2):
      _, reward, _, _, _ = env.step(torch.zeros(4, 16))
      assert torch.isfinite(reward).all()
    # Inject a real legal quarter-turn of a whole layer: all layers align,
    # and must no longer cause an immediate internal-error termination.
    from wuji_mjlab.tasks.reorient.mdp.rubik_solver import apply_face_move
    actual = cmd.object.data.joint_pos.clone()
    row = dict(zip(cmd.joint_names, baseline[0].tolist(), strict=True))
    moved = apply_face_move(row, 'U', cmd.joint_names)
    actual[0] = torch.tensor([moved[name] for name in cmd.joint_names])
    cmd.object.write_joint_state_to_sim(actual, torch.zeros_like(actual))
    env.sim.forward()
    env.common_step_counter += 1
    cmd.evaluate_task()
    assert not cmd.failure_event.any()
    assert not cmd.flip_preservation.settling.any()
    assert not cmd.flip_to_top_achieved[0]
    assert torch.allclose(cmd.episode_initial_joint_pos, baseline)
    # Publishing another target must not silently adopt a corrupted state.
    cmd._install_goal(torch.tensor([0]), torch.tensor([2]), 0, actual[:1])
    assert torch.allclose(cmd.expected_joint_pos[0], baseline[0])
    # A20-degree residual on the SELECTED layer must not prevent geometric
    # arrival. Exercise the actual command evaluator, not only the helper.
    actual[0] = baseline[0]
    actual[0, cmd._driver_joint_indices[2]] += math.radians(20)
    cmd.object.write_joint_state_to_sim(actual, torch.zeros_like(actual))
    cmd.object.write_root_link_velocity_to_sim(torch.zeros(4, 6))
    env.sim.forward()
    cmd._read_flip_target_dot = lambda: torch.ones(4)
    cmd._read_tip_contact_count = lambda: torch.ones(4)
    cmd._install_goal(torch.tensor([0]), torch.tensor([2]), 0, baseline[:1])
    # set_solver_goal records the current layer angle as its geometric flip
    # starting angle; force a20-degree difference as seen during a real flip.
    cmd.u_target_angle[0] = cmd.u_angle[0] - math.radians(20)
    for _ in range(10):
      env.common_step_counter += 1
      cmd.evaluate_task()
    state = cmd.flip_preservation
    assert state.settling[0]
    assert torch.rad2deg(state.error[0]) == pytest.approx(20., abs=.1)
    assert not cmd._pending_goal_advance[0]
    assert not cmd.failure_event[0]
    state.arrival_step[0] = env.common_step_counter - 20
    env.common_step_counter += 1
    cmd.evaluate_task()
    assert state.timeout_event[0]
    assert cmd._pending_goal_advance[0]
    cmd._advance_pending_goals()
    assert not state.settling[0]
    assert not cmd._pending_goal_advance[0]
    assert cmd.phase[0] == 0
  finally:
    env.close()


def test_already_up_flip_request_never_installs_turn_phase():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import RubikFullCubeCommand
  command = object.__new__(RubikFullCubeCommand)
  command._env = SimpleNamespace(device='cpu')
  command.cfg = SimpleNamespace(flip_specialization=True)
  command.current_top_face = torch.tensor([4])
  command.requested_face = torch.tensor([4])
  command.bottom_intermediate = torch.tensor([False])
  command.bottom_final_face = torch.tensor([4])
  command._read_top_face_id = lambda: torch.tensor([4])
  command._internal_state_before_current_move = lambda ids: torch.zeros(len(ids), 1)
  installed = []
  command._install_goal = lambda ids, face, phase, state: installed.append((face.clone(), phase))
  command._install_next_flip_turn = lambda ids: pytest.fail('Flip request entered turn')
  command._install_flip_or_turn_start(torch.tensor([0]), torch.tensor([4]))
  assert len(installed) == 1 and installed[0][1] == 0
  assert command.requested_face.item() != 4
