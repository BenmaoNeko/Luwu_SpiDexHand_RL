# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Training wrapper with IsaacLab-style env/agent overrides."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import mjlab
import tyro
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import list_tasks, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder
from wuji_mjlab.rl.transfer import (
  BOTTOM_TIP_TASK_ID,
  FLIP_LOW_FRICTION_TASK_ID,
  FLIP_SPECIALIST_TASK_ID,
  FULL_CUBE_TRANSFER_TASK_IDS,
  GENTLE_FLIP_TASK_ID,
  RUBIK_TRANSFER_TASK_IDS,
  SMART_CUBE_TRANSFER_TASK_ID,
  TRANSFER_LAYOUTS,
  load_actor_transfer_checkpoint,
  validate_transfer_layout_args,
  validate_transfer_resume_args,
  write_transfer_provenance,
)
from wuji_mjlab.utils.cli_override_utils import (
  DEFAULT_TASK_ALIASES,
  resolve_task,
  split_special_args,
)
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs


def _cuda_device_count() -> int:
  import torch

  try:
    return int(torch.cuda.device_count())
  except Exception:
    return 0


def _maybe_force_cpu_when_cuda_unavailable(
  cfg: TrainConfig,
) -> tuple[bool, TrainConfig, str | None]:
  if cfg.gpu_ids is None:
    return False, cfg, None

  if _cuda_device_count() > 0:
    return False, cfg, None

  updated_cfg = replace(cfg, gpu_ids=None)
  return (
    True,
    updated_cfg,
    "[WARN] CUDA requested but no CUDA devices are available; falling back to CPU.",
  )


def _run_train(
  task_id: str,
  cfg: TrainConfig,
  log_dir: Path,
  transfer_checkpoint: Path | None = None,
  transfer_layout: str | None = None,
  bottom_tip_teacher_checkpoint: Path | None = None,
  bottom_tip_resume_dr_alpha: float | None = None,
  bottom_tip_resume_skill_sr_ema: float | None = None,
) -> None:
  validate_transfer_resume_args(transfer_checkpoint, cfg.agent.resume)
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  if task_id in (BOTTOM_TIP_TASK_ID, FLIP_LOW_FRICTION_TASK_ID, GENTLE_FLIP_TASK_ID):
    from wuji_mjlab.rl.physics_failure_guard import install_physics_failure_guard
    install_physics_failure_guard(env, log_dir / 'physics_failures',
      preserve_failure_rewards=('cage_drop',) if task_id != BOTTOM_TIP_TASK_ID else ())

  log_root_path = log_dir.parent

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  transfer_metadata = None
  if transfer_checkpoint is not None:
    transfer_metadata = load_actor_transfer_checkpoint(
      runner, transfer_checkpoint, transfer_layout=transfer_layout,
      normalizer_prior_rollouts=(
        10 if task_id in (SMART_CUBE_TRANSFER_TASK_ID, FLIP_SPECIALIST_TASK_ID)
        else None
      ),
    )
    print(
      "[INFO] Actor-only transfer loaded: "
      f"{transfer_metadata['source_path']} "
      "(critic, optimizer, iteration skipped)"
    )
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    resume_infos = runner.load(str(resume_path)) or {}
    if task_id == FLIP_LOW_FRICTION_TASK_ID:
      controller = env.unwrapped._smart_cube_curriculum
      # The initial reset preceded checkpoint loading. Resample physical DR
      # using restored progress without counting startup as a skill failure.
      state = {k: v for k, v in vars(controller).items() if k != 'cfg'}
      env.reset()
      for name, value in state.items():
        setattr(controller, name, value)
      print(f'[INFO] LowFriction FULL RESUME: next_iteration={runner.current_learning_iteration}; '
        f'low_friction_progress={controller.low_friction_progress}; alpha={controller.alpha}; '
        'actor, critic, optimizer and curriculum restored', flush=True)

  if task_id == BOTTOM_TIP_TASK_ID:
    if resume_path is not None:
      if runner.alg.teacher is None and bottom_tip_teacher_checkpoint is None:
        raise ValueError('Legacy BottomTip resume requires --bottom-tip-teacher-checkpoint-file')
      from wuji_mjlab.tasks.reorient.mdp.smart_cube_curriculum import (
        get_smart_cube_curriculum,
      )
      controller = get_smart_cube_curriculum(env.unwrapped)
      if 'smart_cube_curriculum' not in resume_infos:
        if bottom_tip_resume_dr_alpha is None or not 0 <= bottom_tip_resume_dr_alpha <= 1:
          raise ValueError('Legacy BottomTip resume requires --bottom-tip-resume-dr-alpha in [0,1]')
        controller.alpha = bottom_tip_resume_dr_alpha
        controller.beta = 1.
        controller.last_check_step = env.unwrapped.common_step_counter
        if bottom_tip_resume_skill_sr_ema is None or not 0 <= bottom_tip_resume_skill_sr_ema <= 1:
          raise ValueError('Legacy BottomTip resume requires --bottom-tip-resume-skill-sr-ema in [0,1]')
        controller.skill_sr_ema = bottom_tip_resume_skill_sr_ema
        controller._has_outcomes = True
        controller.pending_successes = controller.pending_attempts = 0
      runner.alg.prepare_transfer(teacher_checkpoint=bottom_tip_teacher_checkpoint)
      # Apply restored reset-time DR to fresh episodes, not the startup alpha=0.
      curriculum_state = {k:v for k,v in vars(controller).items() if k != 'cfg'}
      env.reset()
      # Startup reset is not a failed attempt from the previous training run.
      for name, value in curriculum_state.items():
        setattr(controller, name, value)
      print(f'[INFO] BottomTip FULL RESUME: next_iteration={runner.current_learning_iteration}; '
        f'updates={runner.alg.updates}; alpha={controller.alpha}; beta={controller.beta}; '
        'actor, critic, optimizer restored; teacher retained; no repeated warmup', flush=True)
    else:
      runner.alg.prepare_transfer()
    command_cfg = cfg.env.commands['reorient_command']
    unsupported, supported = command_cfg.bottom_tip_success_rewards
    print(f'[INFO] BottomTip transfer: first{command_cfg.bottom_tip_contact_window_deg:g} '
      'contact window, independently reset/rewarded per90; '
      f'supported success={supported:g}, unsupported={unsupported:g}; '
      'actor normalization frozen; critic warmup=10; actor_lr=5e-6 critic_lr=1e-4; '
      'KL limit=.01; on-policy teacher retention=.1 (gradient <=20% PPO); '
      'fixed reward schedule; no automatic promotion', flush=True)

  if task_id in (*RUBIK_TRANSFER_TASK_IDS, *FULL_CUBE_TRANSFER_TASK_IDS):
    monitor_metadata = runner.enable_transfer_monitor(
      env.get_observations(), legacy_prefix_dim=162, window_iterations=100
    )
    if transfer_metadata is None:
      transfer_metadata = {
        "source_path": str(resume_path),
        "source_iteration": runner.current_learning_iteration,
        "loaded_components": ["actor", "critic", "optimizer", "iteration"],
        "skipped_components": [],
      }
    transfer_metadata["normalization_action_monitor"] = monitor_metadata

  if transfer_metadata is not None and rank == 0:
    write_transfer_provenance(log_dir, transfer_metadata)

  if task_id == BOTTOM_TIP_TASK_ID:
    from wuji_mjlab.rl.nonfinite_capture import install_failure_capture
    install_failure_capture(env, log_dir / 'nonfinite')
  try:
    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations,
      init_at_random_ep_len=task_id not in (*FULL_CUBE_TRANSFER_TASK_IDS, FLIP_SPECIALIST_TASK_ID, BOTTOM_TIP_TASK_ID),
    )
  except (FloatingPointError, ValueError, RuntimeError):
    if task_id == BOTTOM_TIP_TASK_ID:
      # Diagnostic only: never automatically resume from this possibly bad state.
      failure_path = log_dir / 'failure_diagnostic.pt'
      runner.save(str(failure_path), infos={'diagnostic_only': True})
      print(f'[ERROR] Failure state saved (NOT a resume checkpoint): {failure_path}', flush=True)
    raise

  env.close()


def _launch_training(
  task_id: str,
  cfg: TrainConfig,
  transfer_checkpoint: Path | None = None,
  transfer_layout: str | None = None,
  bottom_tip_teacher_checkpoint: Path | None = None,
  bottom_tip_resume_dr_alpha: float | None = None,
  bottom_tip_resume_skill_sr_ema: float | None = None,
) -> None:
  validate_transfer_resume_args(transfer_checkpoint, cfg.agent.resume)
  validate_transfer_layout_args(
    task_id,
    transfer_checkpoint,
    transfer_layout,
    resume=cfg.agent.resume,
  )
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir_name += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)

  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    _run_train(task_id, cfg, log_dir, transfer_checkpoint, transfer_layout,
      bottom_tip_teacher_checkpoint, bottom_tip_resume_dr_alpha, bottom_tip_resume_skill_sr_ema)
  else:
    import torchrunx

    logging.basicConfig(level=logging.INFO)

    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if cfg.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = cfg.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(_run_train, task_id, cfg, log_dir, transfer_checkpoint, transfer_layout,
      bottom_tip_teacher_checkpoint, bottom_tip_resume_dr_alpha, bottom_tip_resume_skill_sr_ema)


def main() -> None:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--task", type=str, default=None)
  parser.add_argument("--transfer-checkpoint-file", type=Path, default=None)
  parser.add_argument("--transfer-layout", choices=TRANSFER_LAYOUTS)
  gentle = parser.add_mutually_exclusive_group()
  gentle.add_argument('--gentle-calibration', type=Path)
  gentle.add_argument('--gentle-ema-only', action='store_true')
  gentle.add_argument('--gentle-runtime-checkpoint', type=Path)
  parser.add_argument('--bottom-tip-teacher-checkpoint-file', type=Path)
  parser.add_argument('--bottom-tip-resume-dr-alpha', type=float)
  parser.add_argument('--bottom-tip-resume-skill-sr-ema', type=float)
  known, remaining = parser.parse_known_args()

  task_from_equals, tyro_args, overrides = split_special_args(remaining)
  chosen_task = known.task or task_from_equals or "reorient"
  task_id = resolve_task(
    chosen_task,
    task_aliases=DEFAULT_TASK_ALIASES,
    all_tasks=list_tasks(),
  )
  prepared_env_cfg, prepared_agent_cfg = prepare_task_cfgs(task_id, overrides, play=False)
  if task_id == GENTLE_FLIP_TASK_ID:
    if known.gentle_runtime_checkpoint is not None:
      import torch
      from wuji_mjlab.tasks.reorient.mdp.gentle_flip import GentleFlipSchedule
      runtime_infos = torch.load(known.gentle_runtime_checkpoint,
        map_location='cpu', weights_only=False)['infos']
      runtime = runtime_infos['gentle_flip_runtime']
      prepared_env_cfg.actions['joint_pos'].ema_alpha = runtime_infos['gentle_flip_action_ema']
      prepared_env_cfg.rewards['gentle_joint_velocity'].params = dict(runtime['reward'])
      for name, weight in runtime.get('reward_weights', {}).items():
        prepared_env_cfg.rewards[name].weight = weight
      prepared_env_cfg.events['gentle_flip_schedule'].params.update(
        schedule=GentleFlipSchedule(**runtime['schedule']),
        coefficient=runtime['reward']['coefficient'])
    elif known.gentle_calibration is None and not known.gentle_ema_only:
      parser.error('Gentle flip requires --gentle-calibration PATH or explicit --gentle-ema-only')
    if known.gentle_calibration is not None:
      from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
        configure_gentle_calibration,
      )
      configure_gentle_calibration(prepared_env_cfg, known.gentle_calibration)
  elif (known.gentle_calibration is not None or known.gentle_ema_only
        or known.gentle_runtime_checkpoint is not None):
    parser.error('Gentle options require LuwuHand_Rubik_FlipGentleTransfer')

  cfg = tyro.cli(
    TrainConfig,
    args=tyro_args,
    default=replace(
      TrainConfig.from_task(task_id),
      env=prepared_env_cfg,
      agent=prepared_agent_cfg,
    ),
    prog=sys.argv[0] + f" --task {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  if known.gentle_runtime_checkpoint is not None:
    if not cfg.agent.resume or known.transfer_checkpoint_file is not None:
      parser.error('--gentle-runtime-checkpoint requires full resume, not actor transfer')
    selected_resume = get_checkpoint_path(
      Path('logs/rsl_rl') / cfg.agent.experiment_name,
      cfg.agent.load_run, cfg.agent.load_checkpoint)
    if selected_resume.resolve() != known.gentle_runtime_checkpoint.resolve():
      parser.error('Runtime checkpoint must match the full-resume checkpoint')
  changed, cfg, message = _maybe_force_cpu_when_cuda_unavailable(cfg)
  if changed and message is not None:
    print(message)
  _launch_training(
    task_id=task_id,
    cfg=cfg,
    transfer_checkpoint=known.transfer_checkpoint_file,
    transfer_layout=known.transfer_layout,
    bottom_tip_teacher_checkpoint=known.bottom_tip_teacher_checkpoint_file,
    bottom_tip_resume_dr_alpha=known.bottom_tip_resume_dr_alpha,
    bottom_tip_resume_skill_sr_ema=known.bottom_tip_resume_skill_sr_ema,
  )


if __name__ == "__main__":
  main()
