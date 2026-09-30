# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wuji_mjlab.rl.transfer import (
  load_actor_transfer_checkpoint,
  validate_transfer_layout_args,
  validate_transfer_resume_args,
)
from wuji_mjlab.rl import transfer as transfer_module


class _Policy(torch.nn.Module):
  def __init__(self, obs_dim: int = 3, action_dim: int = 2):
    super().__init__()
    self.obs_normalizer = torch.nn.BatchNorm1d(obs_dim, affine=False)
    self.mlp = torch.nn.Linear(obs_dim, action_dim)


def _runner():
  return SimpleNamespace(
    device="cpu",
    current_learning_iteration=0,
    alg=SimpleNamespace(
      actor=_Policy(),
      critic=torch.nn.Linear(3, 1),
      optimizer=torch.optim.Adam(
        list(_Policy().parameters()), lr=1.0e-3
      ),
    ),
  )


def _checkpoint(path: Path, policy: torch.nn.Module) -> None:
  torch.save(
    {
      "actor_state_dict": policy.state_dict(),
      "critic_state_dict": {"unexpected": torch.tensor(1.0)},
      "optimizer_state_dict": {"state": {}, "param_groups": []},
      "iter": 2999,
    },
    path,
  )


def test_actor_transfer_loads_policy_and_reports_skipped_components(tmp_path):
  source = _Policy()
  with torch.no_grad():
    source.mlp.weight.fill_(3.0)
    source.mlp.bias.fill_(4.0)
  checkpoint = tmp_path / "model_2999.pt"
  _checkpoint(checkpoint, source)

  runner = _runner()
  critic_before = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
  optimizer_before = runner.alg.optimizer.state_dict()
  metadata = load_actor_transfer_checkpoint(runner, checkpoint)

  assert torch.allclose(runner.alg.actor.mlp.weight, source.mlp.weight)
  assert torch.allclose(runner.alg.actor.mlp.bias, source.mlp.bias)
  for name, value in runner.alg.critic.state_dict().items():
    assert torch.equal(value, critic_before[name])
  assert runner.alg.optimizer.state_dict()["state"] == optimizer_before["state"]
  assert runner.current_learning_iteration == 0
  assert metadata["source_iteration"] == 2999
  assert metadata["loaded_components"] == ["actor", "actor_normalizer"]
  assert metadata["skipped_components"] == ["critic", "optimizer", "iteration"]
  assert len(metadata["checkpoint_sha256"]) == 64


def test_actor_transfer_reports_missing_and_shape_mismatch_keys(tmp_path):
  source = _Policy()
  checkpoint = tmp_path / "bad.pt"
  state = source.state_dict()
  state.pop("mlp.bias")
  state["mlp.weight"] = torch.zeros((4, 3))
  torch.save({"actor_state_dict": state}, checkpoint)

  with pytest.raises(ValueError, match="missing.*mlp.bias|shape.*mlp.weight"):
    load_actor_transfer_checkpoint(_runner(), checkpoint)


def test_transfer_and_full_resume_are_mutually_exclusive():
  with pytest.raises(ValueError, match="cannot be used together"):
    validate_transfer_resume_args(Path("model.pt"), True)
  validate_transfer_resume_args(Path("model.pt"), False)


class _EmpiricalNorm(torch.nn.Module):
  def __init__(self, obs_dim: int):
    super().__init__()
    self.register_buffer("_mean", torch.zeros((1, obs_dim)))
    self.register_buffer("_var", torch.ones((1, obs_dim)))
    self.register_buffer("_std", torch.ones((1, obs_dim)))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  def forward(self, obs):
    return (obs - self._mean) / (self._std + 1e-2)


class _TransferActor(torch.nn.Module):
  def __init__(self, obs_dim: int):
    super().__init__()
    self.obs_normalizer = _EmpiricalNorm(obs_dim)
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(obs_dim, 512),
      torch.nn.ELU(),
      torch.nn.Linear(512, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, 32),
    )

  def forward(self, obs):
    return self.mlp(self.obs_normalizer(obs))[:, :16]


def _rubik_transfer_runner(actor):
  return SimpleNamespace(
    device="cpu",
    current_learning_iteration=0,
    env=SimpleNamespace(num_envs=4),
    cfg={"num_steps_per_env": 2},
    alg=SimpleNamespace(
      actor=actor,
      critic=torch.nn.Linear(3, 1),
      optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
    ),
  )


def test_rubik_transfer_copies_legacy_prefix_and_preserves_initial_actions(
  tmp_path, monkeypatch
):
  torch.manual_seed(19)
  source = _TransferActor(162)
  source.obs_normalizer._mean.copy_(torch.linspace(-0.5, 0.5, 162).view(1, -1))
  source.obs_normalizer._std.copy_(torch.linspace(0.5, 1.5, 162).view(1, -1))
  source.obs_normalizer._var.copy_(source.obs_normalizer._std.square())
  source.obs_normalizer.count.fill_(491_520_000)
  checkpoint = tmp_path / "model_2999.pt"
  _checkpoint(checkpoint, source)
  monkeypatch.setattr(
    transfer_module,
    "RUBIK_CUBE_TRANSFER_SOURCE_SHA256",
    transfer_module.sha256_file(checkpoint),
  )

  target = _TransferActor(204)
  runner = _rubik_transfer_runner(target)
  metadata = load_actor_transfer_checkpoint(
    runner, checkpoint, transfer_layout="cube162-to-rubik204"
  )

  source_state = source.state_dict()
  target_state = target.state_dict()
  assert torch.equal(target_state["mlp.0.weight"][:, :162], source_state["mlp.0.weight"])
  assert torch.count_nonzero(target_state["mlp.0.weight"][:, 162:]) == 0
  for suffix in ("_mean", "_var", "_std"):
    key = f"obs_normalizer.{suffix}"
    assert torch.equal(target_state[key][:, :162], source_state[key])
  assert torch.equal(target.obs_normalizer._mean[:, 162:], torch.zeros((1, 42)))
  assert torch.equal(target.obs_normalizer._var[:, 162:], torch.ones((1, 42)))
  assert torch.equal(target.obs_normalizer._std[:, 162:], torch.ones((1, 42)))
  assert int(target.obs_normalizer.count) == 10 * 4 * 2

  old_obs = torch.randn((8, 162))
  new_obs = torch.randn((8, 204))
  new_obs[:, :162] = old_obs
  source.eval()
  target.eval()
  with torch.no_grad():
    assert torch.allclose(source(old_obs), target(new_obs), atol=1e-5, rtol=1e-5)

  assert metadata["transfer_layout"] == "cube162-to-rubik204"
  assert metadata["source_obs_dim"] == 162
  assert metadata["target_obs_dim"] == 204
  assert metadata["normalizer_prior_count"] == 80
  assert metadata["skipped_components"] == ["critic", "optimizer", "iteration"]


def test_rubik_transfer_rejects_unknown_layout_and_wrong_source_width(
  tmp_path, monkeypatch
):
  source = _TransferActor(162)
  checkpoint = tmp_path / "model_2999.pt"
  _checkpoint(checkpoint, source)

  with pytest.raises(ValueError, match="Unsupported actor transfer layout"):
    load_actor_transfer_checkpoint(
      _rubik_transfer_runner(_TransferActor(204)),
      checkpoint,
      transfer_layout="unknown-layout",
    )

  wrong_source = _TransferActor(161)
  wrong_checkpoint = tmp_path / "wrong_width.pt"
  _checkpoint(wrong_checkpoint, wrong_source)
  monkeypatch.setattr(
    transfer_module,
    "RUBIK_CUBE_TRANSFER_SOURCE_SHA256",
    transfer_module.sha256_file(wrong_checkpoint),
  )
  with pytest.raises(ValueError, match="162.*204|transfer layout.*162"):
    load_actor_transfer_checkpoint(
      _rubik_transfer_runner(_TransferActor(204)),
      wrong_checkpoint,
      transfer_layout="cube162-to-rubik204",
    )


def test_rubik_transfer_layout_is_limited_to_verified_task_and_checkpoint(
  tmp_path, monkeypatch
):
  checkpoint = tmp_path / "model_2999.pt"
  _checkpoint(checkpoint, _TransferActor(162))
  monkeypatch.setattr(
    transfer_module,
    "RUBIK_CUBE_TRANSFER_SOURCE_SHA256",
    transfer_module.sha256_file(checkpoint),
  )

  validate_transfer_layout_args(
    "LuwuHand_Rubik_UFace_2999Transfer",
    checkpoint,
    "cube162-to-rubik204",
  )
  validate_transfer_layout_args("LuwuHand_Rubik_UFace_Stage1", None, None)
  with pytest.raises(ValueError, match="requires --transfer-checkpoint-file"):
    validate_transfer_layout_args(
      "LuwuHand_Rubik_UFace_2999Transfer", None, "cube162-to-rubik204"
    )
  with pytest.raises(ValueError, match="only valid"):
    validate_transfer_layout_args(
      "LuwuHand_Rubik_UFace_Stage1", checkpoint, "cube162-to-rubik204"
    )


def test_rubik_transfer_task_can_resume_without_original_transfer_source():
  validate_transfer_layout_args(
    "LuwuHand_Rubik_UFace_2999Transfer",
    None,
    None,
    resume=True,
  )
  with pytest.raises(ValueError, match="omit.*transfer"):
    validate_transfer_layout_args(
      "LuwuHand_Rubik_UFace_2999Transfer",
      Path("model_2999.pt"),
      "cube162-to-rubik204",
      resume=True,
    )


def test_continuous_rubik_transfer_accepts_only_its_checkpoint(tmp_path, monkeypatch):
  checkpoint = tmp_path / "continuous_model_2999.pt"
  _checkpoint(checkpoint, _TransferActor(162))
  monkeypatch.setattr(
    transfer_module,
    "RUBIK_CONTINUOUS_TRANSFER_SOURCE_SHA256",
    transfer_module.sha256_file(checkpoint),
  )
  task = "LuwuHand_Rubik_UFace_ContinuousTransfer"
  layout = "continuous162-to-rubik204"
  validate_transfer_layout_args(task, checkpoint, layout)
  validate_transfer_layout_args(task, None, None, resume=True)
  with pytest.raises(ValueError, match="continuous162-to-rubik204"):
    validate_transfer_layout_args(task, checkpoint, "cube162-to-rubik204")
  with pytest.raises(ValueError, match="cube162-to-rubik204"):
    validate_transfer_layout_args("LuwuHand_Rubik_UFace_2999Transfer", checkpoint, layout)

  runner = _rubik_transfer_runner(_TransferActor(204))
  metadata = load_actor_transfer_checkpoint(runner, checkpoint, transfer_layout=layout)
  assert metadata["transfer_layout"] == layout
  assert metadata["checkpoint_sha256"] == transfer_module.sha256_file(checkpoint)
  assert runner.alg.actor.mlp[0].weight[:, 162:].count_nonzero() == 0

  with pytest.raises(ValueError, match="archived model_2999"):
    load_actor_transfer_checkpoint(
      _rubik_transfer_runner(_TransferActor(204)),
      checkpoint,
      transfer_layout="cube162-to-rubik204",
    )


def test_actor_transfer_monitor_reports_action_and_normalizer_drift():
  monitor_type = getattr(transfer_module, "ActorTransferMonitor", None)
  assert monitor_type is not None

  class _ProbeNorm(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.register_buffer("_mean", torch.zeros((1, 204)))
      self.register_buffer("_std", torch.ones((1, 204)))
      self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  class _ProbeActor(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.obs_normalizer = _ProbeNorm()
      self.projection = torch.nn.Linear(204, 16, bias=False)

    def forward(self, obs):
      return self.projection((obs - self.obs_normalizer._mean) / self.obs_normalizer._std)

  torch.manual_seed(27)
  actor = _ProbeActor()
  probe = torch.randn((8, 204))
  monitor = monitor_type(actor, probe, legacy_prefix_dim=162)
  with torch.no_grad():
    actor.projection.weight[:, :162].add_(0.01)
    actor.obs_normalizer._mean[:, :162].add_(0.05)
    actor.obs_normalizer._std[:, 162:].mul_(1.1)
    actor.obs_normalizer.count.fill_(100)

  metrics = monitor.snapshot()
  assert metrics["probe_action_delta_rms"] > 0.0
  assert metrics["legacy_prefix_mean_delta_rms"] > 0.0
  assert metrics["rubik_channels_std_rms"] > 0.0
  assert metrics["normalizer_count"] == 100.0
