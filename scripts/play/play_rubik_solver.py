# SPDX-License-Identifier: Apache-2.0
"""Run the continuous-transfer actor against one local Rubik solver sequence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import wuji_mjlab.tasks  # noqa: F401
from mjlab.scripts.play import PlayConfig
from wuji_mjlab.assets.objects.rubik.rubik_cfg import (
  RUBIK_INIT_STATE,
  get_rubik_cfg,
  get_rubik_target_cfg,
)
from wuji_mjlab.utils.play_runner import run_play_with_cfg
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

TASK_ID = "LuwuHand_Rubik_UFace_ContinuousTransfer"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / (
  "logs/rsl_rl/wuji_reorient/remote-2026-09-29_18-17-51_Luwu_Rubik_Continuous_Transfer/model_2999.pt"
)
CHECKPOINT_SHA256 = "eb858b5639320365d186c1f0891e1791e3f17bc66d33f36b3510c63079b94249"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--scramble", type=str, help='Singmaster sequence, e.g. "R U F2"')
  source.add_argument(
    "--random", action="store_true", help="Generate a legal random cube"
  )
  parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
  parser.add_argument("--device", type=str, default=None)
  args = parser.parse_args()

  checkpoint = (
    args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
  )
  checkpoint = checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
  digest = hashlib.sha256()
  with checkpoint.open("rb") as checkpoint_file:
    for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
      digest.update(chunk)
  checkpoint_hash = digest.hexdigest()
  if checkpoint_hash != CHECKPOINT_SHA256:
    raise ValueError(
      "This preview is pinned to the documented model_2999 checkpoint; "
      f"expected SHA256 {CHECKPOINT_SHA256}, got {checkpoint_hash}."
    )

  env_cfg, agent_cfg = prepare_task_cfgs(TASK_ID, [], play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.scene.entities["object"] = get_rubik_cfg(full_cube=True, dynamic_layers=True)
  env_cfg.scene.entities["object"].init_state = RUBIK_INIT_STATE
  env_cfg.scene.entities["target"] = get_rubik_target_cfg(full_cube=True)
  # The articulated full cube generates more contacts and constraint rows than
  # the simplified training object. Keep the preview capacities aligned with
  # the full-cube Shadow Hand setup used by the bundled Robogym reference.
  env_cfg.sim.nconmax = max(env_cfg.sim.nconmax, 600)
  env_cfg.sim.njmax = max(env_cfg.sim.njmax, 6000)
  env_cfg.commands["reorient_command"].solver_mode = True
  # The preview pauses on the first drop/timeout instead of auto-resetting into
  # a stale solver sequence. This long horizon only prevents ordinary timeouts.
  env_cfg.episode_length_s = 100_000.0

  play_cfg = PlayConfig(
    checkpoint_file=str(checkpoint),
    num_envs=1,
    device=args.device,
    viewer="native",
  )
  run_play_with_cfg(
    TASK_ID,
    play_cfg,
    env_cfg,
    agent_cfg,
    rubik_solver_options={
      "scramble": args.scramble,
      "random_state": args.random,
    },
  )


if __name__ == "__main__":
  main()
