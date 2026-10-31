from __future__ import annotations

import json

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
  ('verify_facelets', 'verify_final_facelets'),
  [(False, False), (True, False), (False, True)],
)
def test_full_cube_accepts_legacy_skill_events_without_internal_veto(
  monkeypatch, verify_facelets, verify_final_facelets
):
  from types import SimpleNamespace

  import wuji_mjlab.tasks.reorient.mdp.rubik_full_cube as full
  from wuji_mjlab.tasks.reorient.mdp.rubik_command import RubikUFaceCommand

  command = object.__new__(full.RubikFullCubeCommand)
  command._env = SimpleNamespace(
    common_step_counter=1,
    termination_manager=SimpleNamespace(get_term=lambda name: torch.zeros(4, dtype=torch.bool)),
  )
  command._last_update_step = torch.zeros(4, dtype=torch.long)
  command.object = SimpleNamespace(data=SimpleNamespace(
    joint_pos=torch.full((4, 6), 0.5), joint_vel=torch.zeros(4, 6),
    body_link_vel_w=torch.zeros(4, 1, 6), root_link_vel_w=torch.zeros(4, 6),
  ))
  command.expected_joint_pos = torch.zeros(4, 6)
  command.phase_start_joint_pos = torch.zeros(4, 6)
  command.joint_names = tuple(full._DRIVER_NAMES)
  command._piece_joint_indices = torch.zeros((1, 3), dtype=torch.long)
  command._piece_home_coords = torch.ones(1, 3)
  command._driver_joint_indices = torch.arange(6)
  command._cubelet_body_indices = torch.tensor([0])
  command._face_driver_ids = torch.arange(6)
  command.target_face_id = torch.tensor([0, 1, 2, 3])
  command.phase = torch.tensor([0, 1, 0, 1])
  command.episode_kind = torch.tensor([0, 1, 2, 2])
  command.cfg = SimpleNamespace(
    solver_mode=True, strict_tolerance_rad=0.087, max_internal_speed=0.45,
    min_contact_fingertips=1, upright_tolerance_rad=0.2,
    max_flip_root_speed=1.0, flip_angle_tolerance_rad=0.1745,
    turn_tolerance_rad=0.087, max_u_face_speed=0.45,
    upright_hold_steps=8, turn_hold_steps=8, wrong_turn_hold_steps=8,
    verify_physical_facelets=verify_facelets,
    verify_final_physical_facelets=verify_final_facelets,
  )
  for name in ('u_target_angle', 'u_initial_angle', '_turn_progress_best', 'turn_progress_delta'):
    setattr(command, name, torch.zeros(4))
  command.tip_contact_count = torch.ones(4)
  command.u_up_dot = torch.ones(4)
  command._read_flip_target_dot = lambda: torch.ones(4)
  command.failure_reason = torch.zeros(4, dtype=torch.long)
  command.facelet_check_mismatch = torch.zeros(4, dtype=torch.bool)
  command._goal_terminal = torch.zeros(4, dtype=torch.bool)
  command._stable_unexpected = torch.zeros(4, dtype=torch.long)
  command._stable_invalid = torch.zeros(4, dtype=torch.long)
  command._full_cube_success_hold = torch.zeros(4, dtype=torch.long)
  command._goal_success_count = torch.zeros(4, 2, 6)
  command._pending_goal_advance = torch.zeros(4, dtype=torch.bool)
  command._solved_event = torch.zeros(4, dtype=torch.bool)
  command._turn_program_complete = torch.zeros(4, dtype=torch.bool)
  command.plan_index = torch.zeros(4, dtype=torch.long)
  command.move_index = torch.zeros(4, dtype=torch.long)
  command.turn_index = torch.zeros(4, dtype=torch.long)
  command._plan_step_count = torch.ones(1, dtype=torch.long)
  command._turn_program_counts = torch.ones(1, 6, dtype=torch.long)
  command.flip_to_top_count = torch.zeros(4)
  command.turn_success_count = torch.zeros(4)
  command.goal_reach_count = torch.zeros(4)

  def legacy_events(self):
    self._flip_event = self.phase == 0
    self._turn_event = self.phase == 1
    self.flip_to_top_count += self._flip_event.float()
    self.turn_success_count += self._turn_event.float()
    self.goal_reach_count += 1
    self.turn_progress_delta[:] = 0.2
    self._last_update_step[:] = 1

  monkeypatch.setattr(RubikUFaceCommand, '_evaluate_current_step', legacy_events)
  errors = torch.full((4, 2), 0.5)
  monkeypatch.setattr(full, '_rotation_errors', lambda *args, **kwargs: (None, errors))
  monkeypatch.setattr(full, '_face_turn_path_error', lambda *args, **kwargs: errors)
  monkeypatch.setattr(full, '_cubie_occupancy_valid', lambda *args: torch.ones(4, dtype=torch.bool))
  # The opt-in physical boundary check must veto a false final solve.
  command._verify_physical_facelets = lambda ids: torch.tensor([3])
  command._evaluate_current_step()

  assert command._flip_event.tolist() == [True, False, True, False]
  blocked = verify_facelets or verify_final_facelets
  assert command._turn_event.tolist() == [False, True, False, not blocked]
  assert command._pending_goal_advance.tolist() == [True, True, True, not blocked]
  assert command.goal_reach_count.tolist() == [1, 1, 1, 0 if blocked else 1]
  torch.testing.assert_close(command.turn_progress_delta, torch.full((4,), 0.2))
  assert command._solved_event.tolist() == [False, False, False, not blocked]
  if blocked:
    assert command.failure_reason[3] == 5
    assert command.failure_event[3]
  command.failure_reason[:] = 3
  assert command.failure_event.tolist() == [False, False, True, True]


def test_full_cube_reset_uses_calibrated_cube_position_in_each_world():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.utils.lab_api.math import quat_apply
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_full_cube_solver_transfer_env_cfg,
  )

  env = ManagerBasedRlEnv(
    cfg=luwu_hand_rubik_full_cube_solver_transfer_env_cfg(num_envs=4),
    device="cpu",
  )
  try:
    asset = env.scene["object"]
    robot = env.scene["robot"]
    palm_ids, _ = robot.find_bodies(("right_palm_link",))
    for seed in range(4):
      ids = torch.arange(4) if seed < 2 else torch.tensor([1, 3])
      env.reset(seed=seed, env_ids=ids)
      basic_ids = ids[env._rubik_full_task_groups[ids] != 2]
      torch.testing.assert_close(
        asset.data.joint_pos[basic_ids], asset.data.default_joint_pos[basic_ids]
      )
      normal = torch.zeros((len(ids), 3))
      normal[:, 2] = 1.0
      normal = quat_apply(
        robot.data.body_link_pose_w[ids, palm_ids[0], 3:7], normal
      )
      expected = (
        asset.data.default_root_state[ids, :3]
        + env.scene.env_origins[ids]
        + 0.006 * normal
      )
      torch.testing.assert_close(asset.data.root_link_pos_w[ids], expected)
      assert torch.isfinite(env.obs_buf["policy"]).all()
      assert torch.isfinite(env.obs_buf["critic"]).all()
      # Exercise resets after physics has updated the cached body transforms.
      _, reward, _, _, _ = env.step(
        torch.zeros((4, env.action_manager.total_action_dim))
      )
      assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_episode_groups_keep_exact_25_25_50_share_after_resets():
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import (
    assign_full_cube_task_groups,
  )

  groups = assign_full_cube_task_groups(8)
  assert groups.tolist() == [0, 1, 2, 2, 0, 1, 2, 2]
  assert torch.bincount(groups, minlength=3).tolist() == [2, 2, 4]
  assert assign_full_cube_task_groups(256).bincount(minlength=3).tolist() == [64, 64, 128]
  with pytest.raises(ValueError, match="multiple of 4"):
    assign_full_cube_task_groups(5)


def test_solver_flip_baseline_tracks_last_committed_submove():
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import (
    FULL_CUBE_SOLVE,
    RubikFullCubeCommand,
  )

  command = object.__new__(RubikFullCubeCommand)
  command.plan_index = torch.zeros(3, dtype=torch.long)
  command.move_index = torch.tensor([0, 1, 2])
  command.episode_kind = torch.full((3,), FULL_CUBE_SOLVE, dtype=torch.long)
  command.episode_initial_joint_pos = torch.full((3, 2), 9.0)
  command._plan_step_state_tensor = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
  )

  state = command._internal_state_before_current_move(torch.arange(3))

  torch.testing.assert_close(
    state, torch.tensor([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]])
  )


def test_plan_pool_generation_and_joint_state_reconstruct_scramble(tmp_path):
  import mujoco
  from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
    build_solver_joint_state,
    generate_solver_plan_pool,
    load_solver_plan_pool,
  )

  model = get_rubik_cfg(full_cube=True, dynamic_layers=False).spec_fn().compile()
  joint_names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  )
  defaults = {name: 0.0 for name in joint_names}
  pool_path = tmp_path / "plan-pool.json"
  generated = generate_solver_plan_pool(
    pool_path,
    count=3,
    seed=17,
    min_scramble_moves=2,
    max_scramble_moves=4,
  )
  pool = load_solver_plan_pool(pool_path)

  assert generated["solver_fingerprint"] == pool.solver_fingerprint
  assert len(pool.plans) == 3
  assert len({plan["initialFacelets"] for plan in pool.plans}) == 3
  for plan in pool.plans:
    state = build_solver_joint_state(plan, defaults, joint_names)
    assert state.shape == (len(joint_names),)
    assert pool.initial_facelets[pool.plans.index(plan)] == plan["initialFacelets"]


def test_plan_pool_rejects_solver_version_drift_and_invalid_action_order(
  monkeypatch, tmp_path
):
  from wuji_mjlab.tasks.reorient.mdp import rubik_solver

  monkeypatch.setattr(rubik_solver, "solver_source_fingerprint", lambda: "current")
  plan = {
    "seed": 1,
    "scramble": "R",
    "initialFacelets": "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9,
    "solution": [],
    "actions": [],
    "stateFormat": "54 facelets, URFDLB, row-major, centers fixed",
  }
  path = tmp_path / "pool.json"
  path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "solver_fingerprint": "old",
        "seed": 1,
        "scramble_length": [1, 1],
        "plans": [plan],
      }
    ),
    encoding="utf-8",
  )
  with pytest.raises(ValueError, match="fingerprint"):
    rubik_solver.load_solver_plan_pool(path)

  plan["actions"] = [{"type": "turn_top", "notation": "R"}]
  path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "solver_fingerprint": "current",
        "seed": 1,
        "scramble_length": [1, 1],
        "plans": [plan],
      }
    ),
    encoding="utf-8",
  )
  with pytest.raises(ValueError, match="reorient"):
    rubik_solver.load_solver_plan_pool(path)


def test_internal_privileged_error_is_132d_and_zero_at_expected_state():
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import (
    internal_state_error_features,
  )

  names = [
    "cubelet:driver:pos_x",
    "cubelet:driver:neg_x",
    "cubelet:driver:pos_y",
    "cubelet:driver:neg_y",
    "cubelet:driver:pos_z",
    "cubelet:driver:neg_z",
  ]
  pieces = (
    "neg_x_neg_y",
    "neg_x_pos_y",
    "pos_x_neg_y",
    "pos_x_pos_y",
    "neg_x_neg_z",
    "neg_x_pos_z",
    "pos_x_neg_z",
    "pos_x_pos_z",
    "neg_y_neg_z",
    "neg_y_pos_z",
    "pos_y_neg_z",
    "pos_y_pos_z",
    "neg_x_neg_y_neg_z",
    "neg_x_neg_y_pos_z",
    "neg_x_pos_y_neg_z",
    "neg_x_pos_y_pos_z",
    "pos_x_neg_y_neg_z",
    "pos_x_neg_y_pos_z",
    "pos_x_pos_y_neg_z",
    "pos_x_pos_y_pos_z",
  )
  names.extend(f"cubelet:rot{axis}:{piece}" for piece in pieces for axis in "xyz")
  expected = torch.zeros((2, len(names)))
  actual = expected.clone()

  errors = internal_state_error_features(actual, expected, tuple(names))
  assert errors.shape == (2, 132)
  torch.testing.assert_close(errors, torch.zeros_like(errors))

  piece_x = names.index("cubelet:rotx:neg_x_neg_y")
  actual[0, piece_x] = np.pi / 2
  errors = internal_state_error_features(actual, expected, tuple(names))
  assert torch.isclose(errors[0, :120].abs().max(), torch.tensor(1.0))
  torch.testing.assert_close(errors[1], torch.zeros_like(errors[1]))
  actual[0, piece_x] = 0

  actual[0, names.index("cubelet:driver:pos_x")] = np.pi / 2
  changed = internal_state_error_features(actual, expected, tuple(names))
  assert changed[0, 120:122].abs().sum() > 0.9
  torch.testing.assert_close(changed[1], torch.zeros_like(changed[1]))


def test_joint_goal_path_interpolates_periodically_without_penalizing_progress():
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import _interpolate_joint_path

  start = torch.tensor([[0.0, 3 * np.pi / 4]])
  goal = torch.tensor([[np.pi / 2, -3 * np.pi / 4]])
  halfway = _interpolate_joint_path(start, goal, torch.tensor([0.5]))
  torch.testing.assert_close(
    halfway,
    torch.tensor([[np.pi / 4, np.pi]]),
    atol=1e-6,
    rtol=0.0,
  )


def test_goal_change_resets_only_face_relative_histories_for_selected_envs():
  from types import SimpleNamespace

  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import RubikFullCubeCommand

  class Buffer:
    def __init__(self):
      self.reset_ids = []

    def reset(self, *, batch_ids):
      self.reset_ids.append(batch_ids.clone())

  face_terms = {
    "rubik_cube_orientation": Buffer(),
    "rubik_compat_cube_ori_error": Buffer(),
    "rubik_u_face_angle": Buffer(),
    "action_history": Buffer(),
  }
  delayed_face_term = Buffer()
  manager = SimpleNamespace(
    _obs_buffer={"stale": True},
    _group_obs_term_history_buffer={"policy": face_terms},
    _group_obs_term_delay_buffer={
      "policy": {"rubik_compat_cube_ori_error": delayed_face_term}
    },
  )
  command = object.__new__(RubikFullCubeCommand)
  command._env = SimpleNamespace(observation_manager=manager)

  command._reset_face_relative_observation_history(torch.tensor([1, 3]))

  assert manager._obs_buffer is None
  expected_ids = torch.tensor([1, 3])
  for name in (
    "rubik_cube_orientation",
    "rubik_compat_cube_ori_error",
    "rubik_u_face_angle",
  ):
    torch.testing.assert_close(face_terms[name].reset_ids[0], expected_ids)
  torch.testing.assert_close(delayed_face_term.reset_ids[0], expected_ids)
  assert not face_terms["action_history"].reset_ids


def test_same_shape_transfer_requires_verified_full_cube_source():
  from wuji_mjlab.rl.transfer import (
    FULL_CUBE_TRANSFER_TASK_ID,
    validate_transfer_layout_args,
  )

  checkpoint = (
    "logs/rsl_rl/wuji_reorient/remote-2026-09-29_18-17-51_"
    "Luwu_Rubik_Continuous_Transfer/model_2999.pt"
  )
  validate_transfer_layout_args(FULL_CUBE_TRANSFER_TASK_ID, checkpoint, None)
  with pytest.raises(ValueError, match="SHA-256"):
    validate_transfer_layout_args(FULL_CUBE_TRANSFER_TASK_ID, __file__, None)


def test_full_cube_task_config_keeps_actor_contract_and_unlocks_every_hinge():
  from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_full_cube_solver_transfer_env_cfg,
  )

  cfg = luwu_hand_rubik_full_cube_solver_transfer_env_cfg(num_envs=8)
  spec = cfg.scene.entities["object"].spec_fn()
  model = spec.compile()
  policy = cfg.observations["policy"].terms
  critic = cfg.observations["critic"].terms

  assert len(spec.equalities) == 0
  assert model.neq == 0
  assert sum(model.jnt_type == 3) == 66  # MuJoCo hinge type.
  assert len(model.actuator_ctrlrange) == 0
  assert "rubik_internal_state_privileged" in critic
  assert "rubik_internal_state_privileged" not in policy
  assert "rubik_compat_cube_ori_error" in policy
  assert "rubik_cube_yaw_success" not in cfg.rewards
  assert {
    "rubik_flip_task_fraction",
    "rubik_turn_task_fraction",
    "rubik_solve_task_fraction",
  } <= cfg.metrics.keys()
  assert "rubik_full_cube_turn_program_complete" not in cfg.terminations
  assert "rubik_internal_offset" not in cfg.rewards
  assert cfg.commands["reorient_command"].flip_angle_tolerance_rad == pytest.approx(0.1745329252)
  assert cfg.scene.num_envs == 8
  assert get_rubik_cfg(full_cube=True, dynamic_layers=False).spec_fn().compile().neq == 0
