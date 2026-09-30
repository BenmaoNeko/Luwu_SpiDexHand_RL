# SPDX-License-Identifier: Apache-2.0
"""Safe actor-only checkpoint transfer for fine-tuning a changed task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

RUBIK_CUBE_TRANSFER_LAYOUT = "cube162-to-rubik204"
RUBIK_CONTINUOUS_TRANSFER_LAYOUT = "continuous162-to-rubik204"
TRANSFER_LAYOUTS = (RUBIK_CUBE_TRANSFER_LAYOUT, RUBIK_CONTINUOUS_TRANSFER_LAYOUT)
RUBIK_CUBE_TRANSFER_SOURCE_SHA256 = (
  "17ca59396093228b87b9488b1b6d4ac3576b095df4187d8c4387211de6dc4d3d"
)
RUBIK_CONTINUOUS_TRANSFER_SOURCE_SHA256 = (
  "399b3876a714fc9ed19e9305742942f6c7595a10672d11b9ce46bb955d888313"
)
RUBIK_TRANSFER_TASK_ID = "LuwuHand_Rubik_UFace_2999Transfer"
RUBIK_CONTINUOUS_TRANSFER_TASK_ID = "LuwuHand_Rubik_UFace_ContinuousTransfer"
RUBIK_TRANSFER_TASK_IDS = (RUBIK_TRANSFER_TASK_ID, RUBIK_CONTINUOUS_TRANSFER_TASK_ID)


def _transfer_source_sha256(layout: str) -> str:
  if layout == RUBIK_CUBE_TRANSFER_LAYOUT:
    return RUBIK_CUBE_TRANSFER_SOURCE_SHA256
  if layout == RUBIK_CONTINUOUS_TRANSFER_LAYOUT:
    return RUBIK_CONTINUOUS_TRANSFER_SOURCE_SHA256
  raise ValueError(f"Unsupported actor transfer layout {layout!r}")


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def validate_transfer_resume_args(
  transfer_checkpoint: Path | None, resume: bool
) -> None:
  if transfer_checkpoint is not None and resume:
    raise ValueError(
      "--transfer-checkpoint-file and full resume cannot be used together; "
      "choose actor-only transfer or full runner resume."
    )


def validate_transfer_layout_args(
  task_id: str,
  checkpoint_path: Path | None,
  transfer_layout: str | None,
  *,
  resume: bool = False,
) -> None:
  """Keep the experimental Rubik tensor adapter opt-in and source-specific."""
  if task_id in RUBIK_TRANSFER_TASK_IDS:
    if resume:
      if checkpoint_path is not None or transfer_layout is not None:
        raise ValueError(
          "When resuming a Rubik transfer run, omit actor-transfer checkpoint "
          "and layout flags; the run checkpoint already contains the mapped Actor."
        )
      return
    required_layout = (
      RUBIK_CUBE_TRANSFER_LAYOUT
      if task_id == RUBIK_TRANSFER_TASK_ID
      else RUBIK_CONTINUOUS_TRANSFER_LAYOUT
    )
    if checkpoint_path is None or transfer_layout != required_layout:
      raise ValueError(
        f"{task_id} requires --transfer-checkpoint-file and "
        f"--transfer-layout {required_layout}"
      )
    source_path = Path(checkpoint_path).expanduser().resolve()
    if not source_path.is_file():
      raise FileNotFoundError(f"Transfer checkpoint does not exist: {source_path}")
    source_sha256 = sha256_file(source_path)
    expected_sha256 = _transfer_source_sha256(required_layout)
    if source_sha256 != expected_sha256:
      raise ValueError(
        f"Rubik transfer {required_layout} requires its archived model_2999.pt; "
        f"expected SHA-256 {expected_sha256}, got {source_sha256}"
      )
  elif transfer_layout is not None:
    raise ValueError(
      f"--transfer-layout {transfer_layout} is only valid for {RUBIK_TRANSFER_TASK_IDS}"
    )


class ActorTransferMonitor:
  """Measure first-rollout observation-stat and fixed-probe action drift."""

  def __init__(
    self,
    actor: torch.nn.Module,
    probe_observation: Any,
    *,
    legacy_prefix_dim: int,
  ) -> None:
    normalizer = getattr(actor, "obs_normalizer", None)
    mean = getattr(normalizer, "_mean", None)
    std = getattr(normalizer, "_std", None)
    if mean is None or std is None or mean.shape != std.shape:
      raise ValueError("Transfer monitor requires an empirical actor observation normalizer")
    observation_dim = int(mean.shape[-1])
    if not 0 < legacy_prefix_dim < observation_dim:
      raise ValueError(
        f"legacy_prefix_dim must be inside the observation width {observation_dim}"
      )

    try:
      device = next(actor.parameters()).device
    except StopIteration:
      device = mean.device
    self.actor = actor
    self.probe_observation = probe_observation.detach().clone().to(device)
    self.legacy_prefix_dim = legacy_prefix_dim
    self.initial_mean = mean.detach().clone()
    self.initial_std = std.detach().clone()
    self.initial_action = self._action()

  def _action(self) -> torch.Tensor:
    with torch.inference_mode():
      action = self.actor(self.probe_observation)
    if not isinstance(action, torch.Tensor) or action.ndim != 2:
      raise ValueError("Transfer probe actor must return a batched action tensor")
    return action.detach().clone()

  @staticmethod
  def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(value))).item())

  def snapshot(self) -> dict[str, float]:
    normalizer = self.actor.obs_normalizer
    mean = normalizer._mean
    std = normalizer._std
    action = self._action()
    split = self.legacy_prefix_dim
    return {
      "probe_action_delta_rms": self._rms(action - self.initial_action),
      "legacy_prefix_mean_delta_rms": self._rms(
        mean[..., :split] - self.initial_mean[..., :split]
      ),
      "legacy_prefix_std_delta_rms": self._rms(
        std[..., :split] - self.initial_std[..., :split]
      ),
      "rubik_channels_mean_rms": self._rms(mean[..., split:]),
      "rubik_channels_std_rms": self._rms(std[..., split:]),
      "normalizer_count": float(normalizer.count.item()),
    }


def _policy_from_runner(runner: Any) -> torch.nn.Module:
  algorithm = runner.alg
  get_policy = getattr(algorithm, "get_policy", None)
  if callable(get_policy):
    return get_policy()
  return algorithm.actor


def load_actor_transfer_checkpoint(
  runner: Any,
  checkpoint_path: Path,
  *,
  transfer_layout: str | None = None,
) -> dict[str, object]:
  """Load only actor weights and actor normalization buffers into ``runner``.

  The critic, optimizer, runner iteration, environment state, and curriculum
  are deliberately untouched so the new task learns fresh value targets.
  """
  checkpoint_path = Path(checkpoint_path).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Transfer checkpoint does not exist: {checkpoint_path}")

  checkpoint = torch.load(
    checkpoint_path,
    map_location=getattr(runner, "device", "cpu"),
    weights_only=False,
  )
  actor_state = checkpoint.get("actor_state_dict") if isinstance(checkpoint, dict) else None
  if not isinstance(actor_state, dict):
    raise ValueError(
      f"Transfer checkpoint {checkpoint_path} has no actor_state_dict; "
      "full or legacy checkpoints are not accepted by actor-only transfer."
    )

  if transfer_layout not in (None, *TRANSFER_LAYOUTS):
    raise ValueError(
      f"Unsupported actor transfer layout {transfer_layout!r}; "
      f"available layouts: {list(TRANSFER_LAYOUTS)}"
    )
  checkpoint_sha256 = sha256_file(checkpoint_path)
  if transfer_layout in TRANSFER_LAYOUTS and checkpoint_sha256 != _transfer_source_sha256(transfer_layout):
    raise ValueError(
      f"Rubik transfer {transfer_layout} requires its archived model_2999.pt; "
      f"expected SHA-256 {_transfer_source_sha256(transfer_layout)}, got {checkpoint_sha256}"
    )

  actor = _policy_from_runner(runner)
  destination = actor.state_dict()
  source_keys = set(actor_state)
  destination_keys = set(destination)
  missing = sorted(destination_keys - source_keys)
  unexpected = sorted(source_keys - destination_keys)
  if missing or unexpected:
    details = []
    if missing:
      details.append(f"missing destination keys: {missing}")
    if unexpected:
      details.append(f"unexpected source keys: {unexpected}")
    raise ValueError("Actor transfer state_dict key mismatch; " + "; ".join(details))

  if transfer_layout in TRANSFER_LAYOUTS:
    actor_state, destination, normalizer_prior_count = _map_cube162_to_rubik204(
      runner, actor_state, destination
    )
  else:
    normalizer_prior_count = None

  shape_errors = []
  for name in sorted(destination_keys):
    source_shape = tuple(actor_state[name].shape)
    destination_shape = tuple(destination[name].shape)
    if source_shape != destination_shape:
      if transfer_layout in TRANSFER_LAYOUTS and name in {
        "mlp.0.weight",
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
      }:
        continue
      shape_errors.append(
        f"{name}: source {source_shape} != destination {destination_shape}"
      )
  if shape_errors:
    raise ValueError("Actor transfer state_dict shape mismatch: " + "; ".join(shape_errors))

  actor.load_state_dict(
    destination if transfer_layout in TRANSFER_LAYOUTS else actor_state,
    strict=True,
  )
  loaded_components = ["actor"]
  if any(key.startswith("obs_normalizer.") for key in actor_state):
    loaded_components.append("actor_normalizer")
  metadata: dict[str, object] = {
    "source_path": str(checkpoint_path),
    "source_iteration": checkpoint.get("iter"),
    "checkpoint_sha256": checkpoint_sha256,
    "loaded_components": loaded_components,
    "skipped_components": ["critic", "optimizer", "iteration"],
    "loaded_keys": sorted(actor_state),
  }
  if transfer_layout is not None:
    metadata.update(
      {
        "transfer_layout": transfer_layout,
        "source_obs_dim": 162,
        "target_obs_dim": 204,
        "normalizer_prior_count": normalizer_prior_count,
        "legacy_actor_prefix_columns": [0, 162],
        "zero_initialized_actor_columns": [162, 204],
      }
    )
  return metadata


def _map_cube162_to_rubik204(
  runner: Any,
  source: dict[str, torch.Tensor],
  destination: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
  """Map the successful 162-D Luwu actor into the 204-D Rubik transfer actor."""
  source_mean = source.get("obs_normalizer._mean")
  destination_mean = destination.get("obs_normalizer._mean")
  source_first = source.get("mlp.0.weight")
  destination_first = destination.get("mlp.0.weight")
  if (
    source_mean is None
    or destination_mean is None
    or source_first is None
    or destination_first is None
  ):
    raise ValueError("Rubik actor transfer requires an MLP actor with observation normalization")

  source_dim = source_mean.shape[-1]
  destination_dim = destination_mean.shape[-1]
  if (source_dim, destination_dim) != (162, 204):
    raise ValueError(
      "Rubik transfer layout cube162-to-rubik204 requires observation widths "
      f"162→204, got {source_dim}→{destination_dim}"
    )
  if source_first.shape != (512, 162) or destination_first.shape != (512, 204):
    raise ValueError(
      "Rubik transfer layout requires first-layer weights (512,162)→(512,204), "
      f"got {tuple(source_first.shape)}→{tuple(destination_first.shape)}"
    )

  allowed_shape_mismatches = {
    "mlp.0.weight",
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
  }
  shape_errors = [
    f"{name}: source {tuple(source[name].shape)} != destination {tuple(destination[name].shape)}"
    for name in sorted(source.keys() & destination.keys())
    if source[name].shape != destination[name].shape
    and name not in allowed_shape_mismatches
  ]
  if shape_errors:
    raise ValueError("Rubik actor transfer has incompatible tensors: " + "; ".join(shape_errors))

  try:
    num_envs = int(runner.env.num_envs)
    steps_per_env = int(runner.cfg["num_steps_per_env"])
  except (AttributeError, KeyError, TypeError, ValueError) as exc:
    raise ValueError(
      "Rubik actor transfer needs runner.env.num_envs and cfg['num_steps_per_env']"
    ) from exc
  if num_envs <= 0 or steps_per_env <= 0:
    raise ValueError("Rubik actor transfer rollout dimensions must be positive")
  prior_count = 10 * num_envs * steps_per_env

  mapped = {name: value.clone() for name, value in destination.items()}
  for name, source_value in source.items():
    if name == "mlp.0.weight":
      mapped[name].zero_()
      mapped[name][:, :162].copy_(source_value)
    elif name in {
      "obs_normalizer._mean",
      "obs_normalizer._var",
      "obs_normalizer._std",
    }:
      mapped[name][:, :162].copy_(source_value)
    elif name == "obs_normalizer.count":
      mapped[name].fill_(prior_count)
    elif name in mapped:
      if source_value.shape != mapped[name].shape:
        raise ValueError(
          f"Rubik actor transfer tensor {name} has shape "
          f"{tuple(source_value.shape)} != {tuple(mapped[name].shape)}"
        )
      mapped[name].copy_(source_value)

  return source, mapped, prior_count


def write_transfer_provenance(log_dir: Path, metadata: dict[str, object]) -> Path:
  """Persist the actor-transfer source and intentionally skipped state."""
  path = Path(log_dir) / "transfer_source.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return path
