# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Visualize a task with a dummy policy for model/collision inspection.

Examples:
  pixi run python src/wuji_mjlab/tasks/reorient/scripts/view_task.py WujiHand_Reorient --num-envs 1 --viewer viser
  pixi run python src/wuji_mjlab/tasks/reorient/scripts/view_task.py WujiHand_Reorient --num-envs 1 --viewer native --play
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import tyro
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.viewer import NativeMujocoViewer, VerbosityLevel, ViserPlayViewer
from mjlab.viewer.native.keys import KEY_R


class ResettableNativeMujocoViewer(NativeMujocoViewer):
  """Native preview viewer with an explicit R-key full environment reset.

  The stock viewer maps Enter to reset and R to toggle debug visualization.
  For this task the reset path is the important behavior to inspect, so R is
  intentionally reserved for the same queued ``env.reset()`` action used by
  the training reset manager. The callback only queues an action; the actual
  reset runs on the simulation thread in ``BaseViewer._process_actions``.
  """

  def reset_environment(self) -> None:
    super().reset_environment()
    self.log(
      "[RESET] full env.reset() completed; all parallel environments use "
      "the training reset events",
      VerbosityLevel.INFO,
    )

  def _safe_key_callback(self, key: int) -> None:
    if key == KEY_R:
      self.request_reset()
      return
    super()._safe_key_callback(key)


class TimedGoalCyclingPolicy:
  """Wrap a preview policy and cycle reorientation goals in sim time."""

  def __init__(self, policy, env, cycle_seconds: float):
    self._policy = policy
    self._env = env.unwrapped
    self._cycle_seconds = max(float(cycle_seconds), 0.0)
    self._elapsed_seconds = 0.0
    self._command = None
    if self._cycle_seconds > 0.0:
      try:
        self._command = self._env.command_manager.get_term("reorient_command")
      except (KeyError, ValueError):
        # Keep this generic preview script usable for tasks without this term.
        self._command = None
    self.reset()

  def reset(self) -> None:
    """Reapply the deterministic rotation preview after a viewer reset."""
    self._elapsed_seconds = 0.0
    if self._command is None:
      return
    set_rotation_mode = getattr(self._command, "preview_set_rotation_mode", None)
    if set_rotation_mode is not None:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
      set_rotation_mode(env_ids)

  def __call__(self, obs):
    actions = self._policy(obs)
    if self._command is None:
      return actions

    self._elapsed_seconds += float(self._env.step_dt)
    if self._elapsed_seconds >= self._cycle_seconds:
      env_ids = torch.arange(self._env.num_envs, device=self._env.device)
      advance_goal = getattr(self._command, "preview_advance_goal", None)
      if advance_goal is not None:
        advance_goal(env_ids)
      else:
        # Backward-compatible fallback for non-reorient command terms.
        self._command.preview_resample_goals(env_ids)
      self._elapsed_seconds = 0.0
    return actions


@dataclass(frozen=True)
class ViewTaskConfig:
  agent: Literal["zero", "random"] = "zero"
  play: bool = False
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  preview_goal_cycle_seconds: float = 2.0


def run_view(task_id: str, cfg: ViewTaskConfig) -> None:
  import mjlab.tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=cfg.play)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=None)

  action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
  if cfg.agent == "zero":

    class PolicyZero:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    policy = PolicyZero()
  else:

    class PolicyRandom:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

    policy = PolicyRandom()

  policy = TimedGoalCyclingPolicy(
    policy, env, cycle_seconds=cfg.preview_goal_cycle_seconds
  )

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  else:
    viewer = cfg.viewer

  if viewer == "native":
    print(
      "[INFO] Native preview reset key: R (Enter also resets). "
      "Each reset calls the same full env.reset() path as training."
    )
    ResettableNativeMujocoViewer(
      env, policy, verbosity=VerbosityLevel.INFO
    ).run()
  elif viewer == "viser":
    print(
      "[INFO] In Viser, open the 'Geoms' tab and enable Group 3 to inspect collision geometry."
    )
    if cfg.preview_goal_cycle_seconds > 0.0:
      print(
        "[INFO] Preview target goal cycles every "
        f"{cfg.preview_goal_cycle_seconds:g}s of simulation time."
      )
    else:
      print("[INFO] Preview target goal cycling disabled.")
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {viewer}")

  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()

  # Handle --help / -h before the first tyro pass, since the positional
  # task-id parser uses `add_help=False` and would reject `--help` as an
  # invalid choice otherwise.
  if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print("Usage: view_task.py TASK_ID [OPTIONS]\n")
    print("Available TASK_ID choices:")
    for t in all_tasks:
      print(f"  {t}")
    print("\nOptions (after TASK_ID):")
    tyro.cli(
      ViewTaskConfig,
      args=["--help"],
      default=ViewTaskConfig(),
      prog=sys.argv[0] + " TASK_ID",
      config=(
        tyro.conf.AvoidSubcommands,
        tyro.conf.FlagConversionOff,
      ),
    )
    sys.exit(0)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    ViewTaskConfig,
    args=remaining_args,
    default=ViewTaskConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  run_view(chosen_task, args)
