# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from wuji_mjlab.tasks.reorient.scripts.view_task import TimedGoalCyclingPolicy


def test_timed_goal_cycling_policy_advances_in_simulation_time():
  class FakeCommand:
    def __init__(self):
      self.calls: list[torch.Tensor] = []

    def preview_advance_goal(self, env_ids: torch.Tensor) -> None:
      self.calls.append(env_ids.clone())

  command = FakeCommand()
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda _name: command),
      device=torch.device("cpu"),
      num_envs=2,
      step_dt=0.5,
    )
  )
  base_calls = 0

  def base_policy(obs):
    nonlocal base_calls
    base_calls += 1
    return obs + 1

  policy = TimedGoalCyclingPolicy(base_policy, env, cycle_seconds=1.0)
  obs = torch.zeros((2, 16))

  assert torch.equal(policy(obs), torch.ones_like(obs))
  assert command.calls == []
  assert torch.equal(policy(obs), torch.ones_like(obs))
  assert len(command.calls) == 1
  assert torch.equal(command.calls[0], torch.tensor([0, 1]))
  assert torch.equal(policy(obs), torch.ones_like(obs))
  assert len(command.calls) == 1
  assert base_calls == 3
