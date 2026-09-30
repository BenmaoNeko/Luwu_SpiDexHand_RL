# SPDX-License-Identifier: Apache-2.0
import torch

from wuji_mjlab.tasks.reorient.mdp import rubik_command


def test_turn_progress_pays_only_new_contacted_upright_progress():
  progress = getattr(rubik_command, "bounded_u_turn_progress")
  best = torch.zeros(3)
  phase = torch.ones(3, dtype=torch.long)
  initial_error = torch.full((3,), torch.pi / 2)
  error = torch.tensor((torch.pi / 4, torch.pi / 4, torch.pi / 4))
  contact = torch.tensor((True, False, True))
  upright = torch.tensor((True, True, False))

  reward, best = progress(error, initial_error, best, phase, contact, upright)
  assert torch.allclose(reward, torch.tensor((0.5, 0.0, 0.0)))
  assert torch.allclose(best, torch.full((3,), 0.5))

  reward, best = progress(error, initial_error, best, phase, contact, upright)
  assert reward.count_nonzero() == 0

  error = torch.tensor((torch.pi / 3, torch.pi / 8, torch.pi / 8))
  reward, best = progress(error, initial_error, best, phase, contact, upright)
  assert torch.allclose(reward, torch.zeros(3))
  assert torch.allclose(best, torch.tensor((0.5, 0.75, 0.75)))

  error = torch.tensor((0.0, 0.0, 0.0))
  reward, best = progress(error, initial_error, best, phase, contact, upright)
  assert torch.allclose(reward, torch.tensor((0.5, 0.0, 0.0)))
  assert torch.allclose(best, torch.ones(3))


def test_turn_progress_ignores_flip_and_new_goal_step():
  progress = getattr(rubik_command, "bounded_u_turn_progress")
  reward, best = progress(
    torch.zeros(2),
    torch.full((2,), torch.pi / 2),
    torch.zeros(2),
    torch.tensor((0, 1)),
    torch.ones(2, dtype=torch.bool),
    torch.ones(2, dtype=torch.bool),
    same_goal=torch.tensor((True, False)),
  )
  assert reward.count_nonzero() == 0
  assert torch.allclose(best, torch.tensor((0.0, 1.0)))


def test_continuous_transfer_only_adds_turn_progress_reward():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_u_face_continuous_transfer_env_cfg,
    luwu_hand_rubik_u_face_transfer_env_cfg,
  )

  old = luwu_hand_rubik_u_face_transfer_env_cfg(num_envs=1)
  new = luwu_hand_rubik_u_face_continuous_transfer_env_cfg(num_envs=1)
  assert "rubik_u_turn_progress" not in old.rewards
  assert new.rewards["rubik_u_turn_progress"].weight == 5.0
  assert tuple(new.observations["policy"].terms) == tuple(old.observations["policy"].terms)
