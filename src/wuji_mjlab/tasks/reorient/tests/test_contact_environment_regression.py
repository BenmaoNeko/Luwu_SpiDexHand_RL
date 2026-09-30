"""Real CPU MJWarp stepping; no PPO or synthetic success/contact injection."""
import math

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_error_magnitude

from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
  luwu_hand_from_scratch_curriculum_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.luwu_hand.env_cfgs import luwu_hand_reorient_env_cfg


@pytest.mark.parametrize('num_envs', [2, 4])
def test_batched_reset_baseline_and_finite_steps(num_envs):
  cfg = luwu_hand_reorient_env_cfg(play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = 20260920
  cfg.commands['reorient_command'].yaw_goal_probability = 1
  env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
  try:
    env.reset()
    command = env.command_manager.get_term('reorient_command')
    expected = torch.full((num_envs,), math.radians(15))
    torch.testing.assert_close(command.goal_initial_error, expected)
    actual = quat_error_magnitude(env.scene['object'].data.root_link_quat_w, command.goal_quat)
    torch.testing.assert_close(command.goal_initial_error, actual)
    for _ in range(10):
      _, reward, _, _, _ = env.step(torch.zeros((num_envs, 16)))
      assert torch.isfinite(reward).all()
    other_hold = command._contact_hold_time[1:].clone()
    other_clock = command.goal_timer[1:].clone()
    # Follow the same reset -> forward -> command order as env.step.
    env._reset_idx(torch.tensor([0]))
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.command_manager.compute(env.step_dt)
    torch.testing.assert_close(command.goal_initial_error[0], expected[0])
    torch.testing.assert_close(command._contact_hold_time[1:], other_hold)
    torch.testing.assert_close(command.goal_timer[1:], other_clock)
    assert command.goal_timer[0].item() == 0
    assert not command._contact_hold_time[0].any()
    for _ in range(10):
      _, reward, _, _, _ = env.step(torch.zeros((num_envs, 16)))
      assert torch.isfinite(reward).all()
    assert not command.goal_reach_count.any()
  finally:
    env.close()


def test_from_scratch_play_has_expanded_actor_contract_and_finite_steps():
  cfg = luwu_hand_from_scratch_curriculum_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["policy"].shape == (1, 166)
    assert obs["critic"].shape == (1, 344)
    for _ in range(3):
      obs, reward, _, _, _ = env.step(torch.zeros((1, 16)))
      assert obs["policy"].shape == (1, 166)
      assert obs["critic"].shape == (1, 344)
      assert torch.isfinite(obs["policy"]).all()
      assert torch.isfinite(obs["critic"]).all()
      assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_from_scratch_training_enables_visual_dr_without_nan():
  cfg = luwu_hand_from_scratch_curriculum_env_cfg(play=False, num_envs=1)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["policy"].shape == (1, 166)
    for _ in range(2):
      obs, reward, _, _, _ = env.step(torch.zeros((1, 16)))
      assert torch.isfinite(obs["policy"]).all()
      assert torch.isfinite(reward).all()
  finally:
    env.close()
