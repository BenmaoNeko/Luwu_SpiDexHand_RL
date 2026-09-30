# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Behavior contracts for the shared cage counter ownership."""

from types import SimpleNamespace

import torch
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.mdp import terminations
from wuji_mjlab.tasks.reorient.mdp import cage as cage_module
from wuji_mjlab.tasks.reorient.mdp.cage import (
  CageEscapePenalty,
  cage_drop,
  compute_cage_escape_severity,
  update_cage_state,
)


def test_cage_drop_uses_shared_counter_behavior():
  env = SimpleNamespace(
    _cage_penalty_counter=torch.tensor([9.0, 10.0]),
    num_envs=2,
    device="cpu",
  )

  result = cage_drop(env, max_outside_steps=10)

  assert torch.equal(result, torch.tensor([False, True]))


def test_cage_public_compatibility_reexports_remain_available():
  assert mdp.CageEscapePenalty is CageEscapePenalty
  assert mdp.cage_drop is cage_drop
  assert terminations.CageEscapePenalty is CageEscapePenalty
  assert terminations.cage_drop is cage_drop


def test_cage_geometry_and_counter_are_updated_once_per_control_step(monkeypatch):
  calls = []

  def fake_escape(*_args, **_kwargs):
    calls.append(True)
    return torch.tensor([0.01])

  monkeypatch.setattr(cage_module, "_compute_cage_escape_dist", fake_escape)
  env = SimpleNamespace(
    num_envs=1,
    device=torch.device("cpu"),
    common_step_counter=7,
    scene={},
  )

  first = update_cage_state(env)
  second = update_cage_state(env)
  assert len(calls) == 1
  assert torch.equal(first["counter"], torch.tensor([1.0]))
  assert torch.equal(second["counter"], torch.tensor([1.0]))

  cage_drop(env, max_outside_steps=10)
  assert len(calls) == 1

  env.common_step_counter = 8
  update_cage_state(env)
  assert len(calls) == 2
  assert torch.equal(env._cage_penalty_counter, torch.tensor([2.0]))


def test_cage_partial_reset_does_not_recount_unreset_envs(monkeypatch):
  calls = []

  def fake_escape(*_args, **_kwargs):
    calls.append(True)
    return torch.tensor([0.01, 0.01])

  monkeypatch.setattr(cage_module, "_compute_cage_escape_dist", fake_escape)
  env = SimpleNamespace(
    num_envs=2,
    device=torch.device("cpu"),
    common_step_counter=4,
    scene={},
  )
  update_cage_state(env)
  env._cage_state["step"][0] = -1
  update_cage_state(env)

  assert len(calls) == 2
  assert torch.equal(env._cage_penalty_counter, torch.tensor([2.0, 1.0]))


def test_cage_escape_severity_is_bounded_by_two_centimeters():
  severity = compute_cage_escape_severity(torch.tensor([0.0, 0.01, 0.04]))
  assert torch.allclose(severity, torch.tensor([0.0, 0.5, 1.0]))
