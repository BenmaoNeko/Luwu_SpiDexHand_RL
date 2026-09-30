"""Exercise the actual termination -> reward -> reset -> command ordering."""
import math
from types import SimpleNamespace

import torch

from wuji_mjlab.tasks.reorient.tests.test_contact_task_success import _make_command
from wuji_mjlab.tasks.reorient.mdp.rewards import (
  contact_milestone_reward, contact_gated_goal_progress_reward,
  flip_success_reward, rotation_success_reward,
)
from wuji_mjlab.tasks.reorient.mdp.terminations import flip_completed
from wuji_mjlab.tasks.reorient.mdp.curriculums import (
  adaptive_episode_curriculum, reorient_success_curriculum,
)


def make_env(yaw=0.0, num_envs=1):
  command, env = _make_command(yaw_probability=yaw, num_envs=num_envs)
  command.cfg.synchronized_contact_task = True
  env.common_step_counter = 0
  env.command_manager = SimpleNamespace(get_term=lambda _: command)
  env.termination_manager = SimpleNamespace(
    get_term=lambda _: torch.zeros(num_envs, dtype=torch.bool),
    terminated=torch.zeros(num_envs, dtype=torch.bool),
  )
  env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
  command.time_left[:] = 10000
  command.object.indexing = SimpleNamespace(free_joint_q_adr=torch.arange(7))
  env.sim = SimpleNamespace(data=SimpleNamespace(qpos=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).repeat(num_envs, 1)))
  command.object.data.root_link_quat_w = env.sim.data.qpos[:, 3:7]
  return command, env


def test_flip_success_reward_and_curriculum_on_same_terminal_step():
  command, env = make_env()
  env.scene['object'].data.root_link_quat_w[:] = command.goal_quat_w
  env.scene['tip_object_contact'].data.found[:, 0] = 1
  for step in (1, 2):
    env.common_step_counter = step
    terminal = flip_completed(env)
    payout = flip_success_reward(env) * env.step_dt
    if step == 1:
      assert not terminal.any()
      assert payout.item() == 0
      command.compute(env.step_dt)
  assert terminal.all()
  assert payout.item() == 1
  assert command.goal_reach_count.item() == 1
  assert reorient_success_curriculum(env, torch.tensor([0]), task_aware=True)['success_env_frac'] == 1
  assert flip_completed(env).all()
  assert command.goal_reach_count.item() == 1


def test_two_finger_contact_needs_its_own_continuous_timer():
  command, env = make_env()
  env.scene['tip_object_contact'].data.found[:, 0] = 1
  payouts = []
  for step in range(1, 7):
    env.common_step_counter = step
    if step >= 5:
      env.scene['tip_object_contact'].data.found[:, 1] = 1
    flip_completed(env)
    reward = contact_milestone_reward(env) * env.step_dt
    assert torch.equal(contact_milestone_reward(env) * env.step_dt, reward)
    payouts.append(reward.item())
    command.compute(env.step_dt)
  assert payouts == [0, 1.0, 0, 0, 0, 1.0]


def test_reset_baseline_reads_written_state_not_stale_derived_quaternion():
  command, env = make_env(yaw=1)
  command.object.data.root_link_quat_w[:] = torch.tensor([[0., 1., 0., 0.]])
  command.object.indexing = SimpleNamespace(free_joint_q_adr=torch.arange(7))
  env.sim = SimpleNamespace(data=SimpleNamespace(qpos=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]])))
  command._resample_command(torch.tensor([0]))
  assert abs(command.goal_initial_error.item() - math.radians(15)) < 1e-5


def test_normal_flip_completion_is_not_an_early_drop():
  command, env = make_env()
  env.episode_length_buf[:] = 100
  env.termination_manager.terminated[:] = True
  env.termination_manager.get_term = lambda name: torch.tensor([name == 'flip_completed'])
  env._adaptive_episode_curriculum = .5
  result = adaptive_episode_curriculum(env, torch.tensor([0]), success_term='flip_completed')
  assert result['early_drop_frac'] == 0
  assert result['value'] == .5


def test_drop_blocks_progress_even_with_contact():
  command, env = make_env(yaw=1)
  env.scene['object'].data.root_link_quat_w[:] = command.goal_quat_w
  env.scene['tip_object_contact'].data.found[:, 0] = 1
  env.termination_manager.get_term = lambda _: torch.tensor([True])
  env.common_step_counter = 1
  flip_completed(env)
  assert contact_gated_goal_progress_reward(env).item() == 0


def test_rotation_goal_is_switched_only_after_old_goal_reward():
  command, env = make_env(yaw=1)
  old_goal = command.goal_quat.clone()
  env.scene['object'].data.root_link_quat_w[:] = old_goal
  env.scene['tip_object_contact'].data.found[:, 0] = 1
  for step in (1, 2):
    env.common_step_counter = step
    flip_completed(env)
    if step == 1:
      command.compute(env.step_dt)
  assert torch.equal(command.goal_quat, old_goal)
  assert (rotation_success_reward(env) * env.step_dt).item() == 1
  command.compute(env.step_dt)
  assert not torch.equal(command.goal_quat, old_goal)
  advanced = command.goal_quat.clone()
  command.compute(env.step_dt)
  assert torch.equal(command.goal_quat, advanced)


def test_partial_reset_preserves_other_env_clock_and_cached_reward():
  command, env = make_env(num_envs=2)
  env.scene['tip_object_contact'].data.found[:, 0] = 1
  for step in (1, 2):
    env.common_step_counter = step
    flip_completed(env)
    contact_milestone_reward(env)
    command.compute(env.step_dt)
  before = command._contact_hold_time[1].clone()
  command.reset(torch.tensor([0]))
  reward = contact_milestone_reward(env) * env.step_dt
  assert torch.equal(reward, torch.tensor([0., 1.]))
  assert torch.equal(command._contact_hold_time[1], before)
  assert not command._contact_hold_time[0].any()
  flip_completed(env)
  assert not command._contact_hold_time[0].any()


def test_contact_interruption_resets_only_affected_clock():
  command, env = make_env()
  sensor = env.scene['tip_object_contact']
  payouts = []
  for step, count in enumerate((2, 1, 2, 2), start=1):
    env.common_step_counter = step
    sensor.data.found[:] = 0
    sensor.data.found[:, :count] = 1
    payouts.append((contact_milestone_reward(env) * env.step_dt).item())
    command.compute(env.step_dt)
  assert payouts == [0, 1.0, 0, 1.0]


def test_goal_switched_flag_clears_on_next_control_step():
  command, env = make_env(yaw=1)
  env.scene['object'].data.root_link_quat_w[:] = command.goal_quat_w
  env.scene['tip_object_contact'].data.found[:, 0] = 1
  for step in (1, 2):
    env.common_step_counter = step
    flip_completed(env)
    command.compute(env.step_dt)
  assert command.goal_switched.all()
  env.common_step_counter = 3
  flip_completed(env)
  command.compute(env.step_dt)
  assert not command.goal_switched.any()


def test_reading_contact_reward_immediately_after_reset_does_not_tick_time():
  command, env = make_env(num_envs=2)
  env.scene['tip_object_contact'].data.found[:, 0] = 1
  env.common_step_counter = 1
  command.reset(torch.tensor([0]))
  contact_milestone_reward(env)
  assert not command._contact_hold_time[0].any()
  assert command._contact_hold_time[1, 0].item() > 0
