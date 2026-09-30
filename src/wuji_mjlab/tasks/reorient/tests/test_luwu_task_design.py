# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from wuji_mjlab.assets.robots.luwu_hand.luwu_hand_cfg import LUWU_HAND_XML
from wuji_mjlab.tasks.reorient.config.luwu_hand.env_cfgs import (
  luwu_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.luwu_hand.luwu_constants import (
  LUWU_CUBE_RESET_QUATERNIONS_WXYZ,
)
from wuji_mjlab.tasks.reorient.mdp.actions import (
  map_normalized_actions_to_joint_limits,
)
from wuji_mjlab.tasks.reorient.mdp.commands import (
  InHandReorientCommand,
  InHandReorientCommandCfg,
)
from wuji_mjlab.utils.play_runner import _PreviewFixedTaskPolicy


def test_full_range_action_mapping_keeps_home_zero_and_reaches_each_limit():
  home = torch.tensor([[0.25, -0.5, 0.0]])
  lower = torch.tensor([[-1.0, -1.5, -2.0]])
  upper = torch.tensor([[1.25, 0.5, 1.0]])

  mapped = map_normalized_actions_to_joint_limits(
    torch.tensor([[-1.0, 0.0, 1.0]]), home, lower, upper
  )

  assert torch.allclose(mapped, torch.tensor([[-1.0, -0.5, 1.0]]))


def test_luwu_action_config_uses_per_joint_full_limits():
  cfg = luwu_hand_reorient_env_cfg(num_envs=1).actions["joint_pos"]

  assert cfg.use_joint_limits is True
  assert cfg.action_scale == 1.0


def test_luwu_reset_starts_with_only_face_a_up():
  cfg = luwu_hand_reorient_env_cfg(num_envs=1)
  reset_params = cfg.events["reset_object_pose"].params

  assert reset_params["orientation_quaternions"] == (
    LUWU_CUBE_RESET_QUATERNIONS_WXYZ
  )
  assert LUWU_CUBE_RESET_QUATERNIONS_WXYZ == ((1.0, 0.0, 0.0, 0.0),)


def test_luwu_goals_are_configured_as_equal_rotation_and_flip_mix():
  cfg = luwu_hand_reorient_env_cfg(num_envs=1)
  command_cfg = cfg.commands["reorient_command"]

  assert command_cfg.goal_sampling_mode == "incremental_yaw_or_flip"
  assert command_cfg.yaw_goal_probability == 0.5
  assert command_cfg.rotation_step_degrees == 15.0
  assert math.isclose(
    sum(x * x for x in command_cfg.yaw_goal_quat_wxyz), 1.0, abs_tol=1e-6
  )
  assert math.isclose(
    sum(x * x for x in command_cfg.flip_goal_quat_wxyz), 1.0, abs_tol=1e-6
  )


def test_luwu_goal_sampler_uses_yaw_then_flip_relative_to_reset(monkeypatch):
  num_envs = 2
  pose_w = torch.zeros((num_envs, 1, 7))
  pose_w[:, :, 3] = 1.0

  class FakeEntity:
    def __init__(self, *, with_root: bool):
      data = SimpleNamespace(body_link_pose_w=pose_w)
      if with_root:
        default_root_state = torch.zeros((num_envs, 13))
        default_root_state[:, 3] = 1.0
        data.default_root_state = default_root_state
        data.root_link_quat_w = default_root_state[:, 3:7].clone()
      self.data = data

    def find_bodies(self, _pattern):
      return [0], ["robot_palm_link"]

  env = SimpleNamespace(
    num_envs=num_envs,
    device=torch.device("cpu"),
    step_dt=0.02,
    scene={"robot": FakeEntity(with_root=False), "object": FakeEntity(with_root=True)},
  )
  cfg = InHandReorientCommandCfg(
    entity_name="object",
    resampling_time_range=(1.0, 1.0),
    goal_sampling_mode="yaw_or_flip",
    robot_entity_name="robot",
  )
  command = InHandReorientCommand(cfg, env)

  monkeypatch.setattr(
    torch,
    "rand",
    lambda n, device: torch.tensor([0.1, 0.9], device=device)[:n],
  )
  command._sample_goal_in_world(torch.arange(num_envs))

  assert torch.allclose(
    command.goal_quat_w[0], torch.tensor(cfg.yaw_goal_quat_wxyz)
  )
  assert torch.allclose(
    command.goal_quat_w[1], torch.tensor(cfg.flip_goal_quat_wxyz)
  )


def test_incremental_rotation_advances_by_fifteen_degrees():
  num_envs = 1
  pose_w = torch.zeros((num_envs, 1, 7))
  pose_w[:, :, 3] = 1.0

  class FakeEntity:
    def __init__(self, *, with_root: bool):
      data = SimpleNamespace(body_link_pose_w=pose_w)
      if with_root:
        default_root_state = torch.zeros((num_envs, 13))
        default_root_state[:, 3] = 1.0
        data.default_root_state = default_root_state
        data.root_link_quat_w = default_root_state[:, 3:7].clone()
      self.data = data

    def find_bodies(self, _pattern):
      return [0], ["robot_palm_link"]

  env = SimpleNamespace(
    num_envs=num_envs,
    device=torch.device("cpu"),
    step_dt=0.02,
    scene={"robot": FakeEntity(with_root=False), "object": FakeEntity(with_root=True)},
  )
  command = InHandReorientCommand(
    InHandReorientCommandCfg(
      entity_name="object",
      resampling_time_range=(1.0, 1.0),
      goal_sampling_mode="incremental_yaw_or_flip",
      yaw_goal_probability=1.0,
      rotation_step_degrees=15.0,
      robot_entity_name="robot",
    ),
    env,
  )

  env_ids = torch.arange(num_envs)
  command.preview_set_rotation_mode(env_ids)
  expected_15 = torch.tensor(
    [math.cos(math.radians(7.5)), 0.0, 0.0, math.sin(math.radians(7.5))]
  )
  assert torch.allclose(command.goal_quat_w[0], expected_15, atol=1e-6)

  command.preview_advance_goal(env_ids)
  expected_30 = torch.tensor(
    [math.cos(math.radians(15.0)), 0.0, 0.0, math.sin(math.radians(15.0))]
  )
  assert torch.allclose(command.goal_quat_w[0], expected_30, atol=1e-6)


def test_training_rotation_advances_only_after_successful_hold():
  pose_w = torch.zeros((1, 1, 7))
  pose_w[:, :, 3] = 1.0

  class FakeEntity:
    def __init__(self, *, with_root: bool):
      data = SimpleNamespace(body_link_pose_w=pose_w)
      if with_root:
        default_root_state = torch.zeros((1, 13))
        default_root_state[:, 3] = 1.0
        data.default_root_state = default_root_state
        data.root_link_quat_w = default_root_state[:, 3:7].clone()
      self.data = data

    def find_bodies(self, _pattern):
      return [0], ["robot_palm_link"]

  object_entity = FakeEntity(with_root=True)
  env = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    step_dt=0.02,
    scene={"robot": FakeEntity(with_root=False), "object": object_entity},
  )
  command = InHandReorientCommand(
    InHandReorientCommandCfg(
      entity_name="object",
      resampling_time_range=(1.0, 1.0),
      goal_sampling_mode="incremental_yaw_or_flip",
      yaw_goal_probability=1.0,
      rotation_step_degrees=15.0,
      success_hold_steps=1,
      goal_switch_delay=1,
      robot_entity_name="robot",
    ),
    env,
  )
  env_ids = torch.zeros(1, dtype=torch.long)
  command._sample_goal_in_world(env_ids)
  first_goal = command.goal_quat_w.clone()

  # The target must remain unchanged before the success state is entered.
  command._update_command()
  assert torch.allclose(command.goal_quat_w, first_goal)

  # Put the cube exactly at the target; one held step then advances by 15°.
  object_entity.data.root_link_quat_w[:] = first_goal
  command._update_command()
  expected_30 = torch.tensor(
    [math.cos(math.radians(15.0)), 0.0, 0.0, math.sin(math.radians(15.0))]
  )
  assert torch.allclose(command.goal_quat_w[0], expected_30, atol=1e-6)
  assert bool(command.goal_switched[0])


def test_preview_goal_resample_clears_progress_state():
  num_envs = 2
  pose_w = torch.zeros((num_envs, 1, 7))
  pose_w[:, :, 3] = 1.0

  class FakeEntity:
    def __init__(self, *, with_root: bool):
      data = SimpleNamespace(body_link_pose_w=pose_w)
      if with_root:
        default_root_state = torch.zeros((num_envs, 13))
        default_root_state[:, 3] = 1.0
        data.default_root_state = default_root_state
        data.root_link_quat_w = default_root_state[:, 3:7].clone()
      self.data = data

    def find_bodies(self, _pattern):
      return [0], ["robot_palm_link"]

  env = SimpleNamespace(
    num_envs=num_envs,
    device=torch.device("cpu"),
    step_dt=0.02,
    scene={"robot": FakeEntity(with_root=False), "object": FakeEntity(with_root=True)},
  )
  command = InHandReorientCommand(
    InHandReorientCommandCfg(
      entity_name="object",
      resampling_time_range=(1.0, 1.0),
      goal_sampling_mode="yaw_or_flip",
      robot_entity_name="robot",
    ),
    env,
  )
  command.goal_timer[:] = 20
  command.hold_counter[:] = 4
  command.in_success_window[:] = True
  command.window_timer[:] = 8
  command.reward_hold_counter[:] = 7

  command.preview_resample_goals(torch.arange(num_envs))

  assert torch.equal(command.goal_timer, torch.zeros(num_envs, dtype=torch.int32))
  assert torch.equal(command.hold_counter, torch.zeros(num_envs, dtype=torch.int32))
  assert not torch.any(command.in_success_window)
  assert torch.equal(command.window_timer, torch.zeros(num_envs, dtype=torch.int32))
  assert torch.equal(
    command.reward_hold_counter, torch.zeros(num_envs, dtype=torch.int32)
  )


def test_preview_fixed_task_groups_assign_rotation_and_flip_goals():
  num_envs = 4
  pose_w = torch.zeros((num_envs, 1, 7))
  pose_w[:, :, 3] = 1.0

  class FakeEntity:
    def __init__(self, *, with_root: bool):
      data = SimpleNamespace(body_link_pose_w=pose_w)
      if with_root:
        default_root_state = torch.zeros((num_envs, 13))
        default_root_state[:, 3] = 1.0
        data.default_root_state = default_root_state
        data.root_link_quat_w = default_root_state[:, 3:7].clone()
      self.data = data

    def find_bodies(self, _pattern):
      return [0], ["robot_palm_link"]

  env = SimpleNamespace(
    num_envs=num_envs,
    device=torch.device("cpu"),
    step_dt=0.02,
    common_step_counter=0,
    scene={"robot": FakeEntity(with_root=False), "object": FakeEntity(with_root=True)},
  )
  command = InHandReorientCommand(
    InHandReorientCommandCfg(
      entity_name="object",
      resampling_time_range=(1.0, 1.0),
      goal_sampling_mode="incremental_yaw_or_flip",
      rotation_step_degrees=15.0,
      robot_entity_name="robot",
    ),
    env,
  )

  command.preview_set_fixed_task_modes(
    torch.tensor([0, 1]), torch.tensor([2, 3])
  )

  assert torch.equal(command.rotation_mode, torch.tensor([True, True, False, False]))
  expected_rotation = torch.tensor(
    [math.cos(math.radians(7.5)), 0.0, 0.0, math.sin(math.radians(7.5))]
  )
  assert torch.allclose(command.goal_quat_w[0], expected_rotation, atol=1e-6)
  assert torch.allclose(command.goal_quat_w[1], expected_rotation, atol=1e-6)
  assert torch.allclose(
    command.goal_quat_w[2],
    torch.tensor(command.cfg.flip_goal_quat_wxyz),
    atol=1e-6,
  )
  assert torch.allclose(
    command.goal_quat_w[3],
    torch.tensor(command.cfg.flip_goal_quat_wxyz),
    atol=1e-6,
  )


def test_preview_fixed_task_policy_reapplies_groups_only_after_reset():
  calls = []
  command = SimpleNamespace(
    flip_completed=torch.zeros(8, dtype=torch.bool),
    evaluate_contact_task=lambda: None,
    preview_set_fixed_task_modes=lambda rotation_ids, flip_ids: calls.append(
      (rotation_ids.clone(), flip_ids.clone())
    ),
  )
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      num_envs=8,
      step_dt=0.05,
      device=torch.device("cpu"),
      episode_length_buf=torch.zeros(8, dtype=torch.long),
      command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
  )
  policy = _PreviewFixedTaskPolicy(
    lambda _obs: torch.ones((8, 16)), env
  )

  policy(None)
  assert len(calls) == 1
  assert torch.equal(calls[-1][0], torch.tensor([0, 1, 2, 3]))
  assert torch.equal(calls[-1][1], torch.tensor([4, 5, 6, 7]))

  env.unwrapped.episode_length_buf[:] = 1
  policy(None)
  assert len(calls) == 1

  env.unwrapped.episode_length_buf[[2, 6]] = 0
  policy(None)
  assert len(calls) == 2
  assert torch.equal(calls[-1][0], torch.tensor([2]))
  assert torch.equal(calls[-1][1], torch.tensor([6]))


def test_preview_fixed_task_policy_does_not_hold_or_resample_after_flip():
  resample_calls = []
  command = SimpleNamespace(
    flip_completed=torch.tensor([True] * 8),
    evaluate_contact_task=lambda: None,
    preview_set_fixed_task_modes=lambda _rotation_ids, _flip_ids: None,
    preview_resample_after_success=lambda env_ids: resample_calls.append(env_ids),
  )
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      num_envs=8,
      step_dt=0.05,
      device=torch.device("cpu"),
      episode_length_buf=torch.ones(8, dtype=torch.long),
      command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
  )
  policy = _PreviewFixedTaskPolicy(
    lambda _obs: torch.ones((8, 16)), env
  )

  actions = policy(None)

  assert torch.equal(actions, torch.ones((8, 16)))
  assert not resample_calls


def test_training_pd_is_nominally_soft_and_narrowly_randomized():
  cfg = luwu_hand_reorient_env_cfg(num_envs=1)
  pd_event = cfg.events["pd_gains"]

  assert pd_event.params["kp_range"] == (0.8, 1.2)
  assert pd_event.params["kd_range"] == (0.8, 1.2)

  xml_text = LUWU_HAND_XML.read_text(encoding="utf-8")
  assert xml_text.count('kp="1"') == 16
  assert 'kp="4"' not in xml_text


def test_luwu_rewards_and_success_are_contact_gated():
  cfg = luwu_hand_reorient_env_cfg(num_envs=1)
  command_cfg = cfg.commands["reorient_command"]

  assert math.isclose(command_cfg.success_threshold, math.radians(4.0))
  assert command_cfg.require_tip_contact is True
  assert command_cfg.min_contact_fingertips == 1

  for old_term in (
    "orientation_alignment",
    "hold_escalation",
    "palm_detach",
    "hand_pose",
    "tip_slide",
  ):
    assert old_term not in cfg.rewards
  for new_term in (
    "approach_progress",
    "contact_milestones",
    "goal_progress",
    "rotation_success",
    "flip_success",
  ):
    assert new_term in cfg.rewards

  assert cfg.rewards["approach_progress"].params["step_budget"] == 2.0
  assert cfg.rewards["contact_milestones"].params["single_reward"] == 1.0
  assert cfg.rewards["contact_milestones"].params["double_reward"] == 1.0
  assert cfg.rewards["goal_progress"].params["step_budget"] == 10.0
  assert cfg.rewards["rotation_success"].weight == 20.0
  assert cfg.rewards["flip_success"].weight == 30.0
  assert cfg.rewards["action_rate"].weight == -0.04
  assert cfg.rewards["torque"].weight == -0.02
  assert cfg.rewards["torque"].params["torque_limit"] == 0.5
  assert cfg.rewards["torque"].params["reduce_mean"] is True
  assert cfg.rewards["finger_collision"].weight == -0.08
  assert cfg.rewards["finger_collision"].params["normalize"] is True
  assert cfg.rewards["cage_escape"].weight == -0.2
  assert cfg.rewards["cage_escape"].params["bounded_escape_penalty"] is True
  assert cfg.rewards["cage_drop"].weight == -10.0

  assert "flip_completed" in cfg.terminations
  assert cfg.terminations["flip_completed"].time_out is False
