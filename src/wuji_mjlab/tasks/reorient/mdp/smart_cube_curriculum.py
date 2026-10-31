# SPDX-License-Identifier: Apache-2.0
"""Opt-in completed-skill curriculum for smart-cube transfer (20 Hz policy steps)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .event_impl.randomization import (
  _get_default_field_values,
  _write_randomized_model_field,
)
from .event_utils import resolve_env_ids
from .smart_cube_observations import SmartCubeProfile


@dataclass(frozen=True)
class SmartCubeCurriculumCfg:
  cooldown_steps: int = 25
  beta_transition_steps: int = 5000
  beta_fast_increment: float = 0.1
  ema_smoothing: float = 0.1
  play: bool = False
  play_alpha: float = 0.7
  force_discrete: bool = False
  imu_noise_max_degrees: float = 1.0
  irregular_imu_latency: bool = False
  low_friction_transition_steps: int = 0
  low_friction_require_bottom_contact: bool = True
  joint_dr_transition_steps: int = 0
  dr_success_gated: bool = True
  realistic_dr: bool = False
  realistic_success_gated: bool = False
  realistic_initial_progress: float = 0.0
  realistic_success_threshold: float = 0.78
  realistic_progress_increment: float = 0.05

  def __post_init__(self):
    if self.joint_dr_transition_steps < 0:
      raise ValueError('Joint DR transition steps must be nonnegative')
    if self.low_friction_transition_steps < 0:
      raise ValueError('Low friction transition steps must be nonnegative')
    if self.cooldown_steps < 25 or self.beta_transition_steps < 1:
      raise ValueError(
        "Cooldown must be >=25 and beta transition steps must be positive"
      )
    if not 0 < self.ema_smoothing <= 1 or not 0 < self.beta_fast_increment <= 1:
      raise ValueError("EMA smoothing and beta increment must be in (0, 1]")
    if not 0 <= self.play_alpha <= 1:
      raise ValueError("play_alpha must be in [0, 1]")
    for name in ('realistic_initial_progress', 'realistic_success_threshold',
                 'realistic_progress_increment'):
      value = getattr(self, name)
      if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(self.imu_noise_max_degrees) or self.imu_noise_max_degrees < 0.1:
      raise ValueError("IMU noise maximum must be finite and >=0.1 degrees")


class SmartCubeCurriculum:
  def __init__(self, cfg: SmartCubeCurriculumCfg):
    self.cfg = cfg
    self.beta = 1.0 if cfg.play or cfg.force_discrete else 0.3
    self.alpha = cfg.play_alpha if cfg.play else 0.0
    self.skill_sr_ema = 0.0
    self.pending_successes = 0
    self.pending_attempts = 0
    self.last_check_step = 0
    self._has_outcomes = False
    self.low_friction_progress = 0.
    self.low_friction_last_step = 0
    self.low_friction_last_totals = (0, 0, 0, 0)
    self.low_friction_skill_attempts = 0
    self.low_friction_skill_successes = 0
    self.joint_dr_progress = 0.
    self.joint_dr_last_step = 0
    self.joint_dr_last_totals = (0, 0)
    self.realistic_progress = cfg.realistic_initial_progress
    self.realistic_steps = int(round(self.realistic_progress * 40000))
    self.realistic_last_step = 0
    self.realistic_promotion_step = 0
    self.realistic_promotion_totals = (0, 0)

  def record_outcomes(self, successes: int, attempts: int):
    if not 0 <= successes <= attempts:
      raise ValueError("Skill outcomes must satisfy 0 <= successes <= attempts")
    self.pending_successes += successes
    self.pending_attempts += attempts
    self.low_friction_skill_attempts += attempts
    self.low_friction_skill_successes += successes

  def update(self, step: int):
    if step - self.last_check_step < self.cfg.cooldown_steps:
      return
    self.last_check_step = step
    has_new = self.pending_attempts > 0
    if has_new:
      sr = self.pending_successes / self.pending_attempts
      self.skill_sr_ema = (
        (1 - self.cfg.ema_smoothing) * self.skill_sr_ema + self.cfg.ema_smoothing * sr
        if self._has_outcomes
        else sr
      )
      self._has_outcomes = True
    self.pending_successes = self.pending_attempts = 0
    if self.cfg.play:
      return
    unlocked = self.beta == 1.0
    scheduled_beta = 0.3 + 0.7 * min(1.0, step / self.cfg.beta_transition_steps)
    fast_beta = (
      self.beta + self.cfg.beta_fast_increment
      if has_new and self.skill_sr_ema >= 0.70
      else self.beta
    )
    self.beta = min(1.0, max(scheduled_beta, fast_beta))
    if self.beta >= 1 - 1e-9:
      self.beta = 1.0
    if unlocked and has_new:
      if self.cfg.realistic_dr:
        # Allow 50 PPO iterations at each level before deciding; require fresh
        # outcomes so old easy-level successes cannot immediately promote DR.
        if self.cfg.realistic_success_gated and step - self.realistic_promotion_step >= 2000:
          totals = (self.low_friction_skill_attempts, self.low_friction_skill_successes)
          attempts, successes = (now - old for now, old in
            zip(totals, self.realistic_promotion_totals, strict=True))
          if attempts >= 256:
            if (self.skill_sr_ema >= self.cfg.realistic_success_threshold
                and successes / attempts >= self.cfg.realistic_success_threshold):
              self.realistic_progress = min(
                1.0, self.realistic_progress + self.cfg.realistic_progress_increment)
              self.realistic_steps = int(round(self.realistic_progress * 40000))
            self.realistic_promotion_step = step
            self.realistic_promotion_totals = totals
        return
      if self.skill_sr_ema >= 0.78:
        self.alpha = min(1.0, self.alpha + 0.05)
      elif self.skill_sr_ema < 0.45:
        self.alpha = max(0.0, self.alpha - 0.025)


def interpolate_smart_cube_profile(
  base: SmartCubeProfile, alpha: float, imu_noise_max_degrees: float = 1.0,
  irregular_imu_latency: bool = False,
):
  if not 0 <= alpha <= 1:
    raise ValueError("DR alpha must be in [0, 1]")
  return replace(
    base,
    threshold_degrees=(43.0, 53.0),
    move_delay_range=(alpha, 1 + alpha),
    imu_noise_rad=math.radians(0.1 + (imu_noise_max_degrees - 0.1) * alpha),
    calibration_noise_rad=math.radians(0.2 + 2.8 * alpha),
    yaw_drift_rad_per_s=math.radians(0.05 * alpha),
    imu_dropout_probability=0.03 * alpha,
    imu_latency_range_seconds=(.05-.02*alpha, .05+.07*alpha) if irregular_imu_latency else base.imu_latency_range_seconds,
    imu_latency_mode_seconds=.05+.04*alpha if irregular_imu_latency else base.imu_latency_mode_seconds,
  )


def get_smart_cube_curriculum(env, cfg: SmartCubeCurriculumCfg | None = None):
  controller = getattr(env, "_smart_cube_curriculum", None)
  if controller is None:
    if cfg is None:
      cfg = env.command_manager.get_term("reorient_command").cfg.smart_cube_curriculum
    controller = SmartCubeCurriculum(cfg)
    env._smart_cube_curriculum = controller
  return controller


def realistic_stage_progress(step: int) -> tuple[float, float]:
  """Fixed 100/600/1000-iteration schedule at 40 control steps per iteration."""
  return (min(1., max(0., (step - 4000) / 20000)),
          min(1., max(0., (step - 24000) / 16000)))


def realistic_friction_bounds(step: int) -> tuple[float, float]:
  first, second = realistic_stage_progress(step)
  return (.20 - .05 * first - .05 * second,
          .30 + .05 * first + .15 * second)


def realistic_palm_sliding() -> float:
  return .30


def realistic_parameter_progress(progress: float) -> float:
  """Global DR progress used by the success-gated transfer curriculum."""
  return min(1., max(0., float(progress)))


def realistic_profile_friction_bounds(progress: float) -> tuple[float, float]:
  progress = realistic_parameter_progress(progress)
  return (.20 - .10 * progress, .30 + .20 * progress)


@requires_model_fields('geom_friction')
def randomize_realistic_friction_surfaces(
  env, env_ids,
  finger_cfg: SceneEntityCfg,
  cube_cfg: SceneEntityCfg,
  palm_cfg: SceneEntityCfg,
):
  """Set absolute sliding friction once per episode, shared across cube and fingers."""
  ids = resolve_env_ids(env, env_ids)
  if not ids.numel():
    return
  nominal = getattr(env, '_realistic_nominal_mask', None)
  if nominal is None:
    nominal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._realistic_nominal_mask = nominal
  nominal[ids] = torch.rand(ids.numel(), device=env.device) < .30
  state = getattr(env, '_smart_cube_state', None)
  if state is not None:
    state.yaw_rate[ids[nominal[ids]]] = 0
  controller = get_smart_cube_curriculum(env)
  if controller.cfg.realistic_success_gated:
    low, high = realistic_profile_friction_bounds(controller.realistic_progress)
  else:
    low, high = realistic_friction_bounds(controller.realistic_steps)
  sliding = torch.empty(ids.numel(), 1, device=env.device).uniform_(low, high)
  sliding[nominal[ids]] = .25
  for cfg, values in ((finger_cfg, sliding), (cube_cfg, sliding),
                      (palm_cfg, torch.full_like(sliding, realistic_palm_sliding()))):
    geom_ids = env.scene[cfg.name].indexing.geom_ids[cfg.geom_ids].long()
    env.sim.model.geom_friction[ids[:, None], geom_ids[None, :], 0] = values
  env._realistic_friction_bounds = (low, high)


def record_smart_cube_skill_successes(env, command, successful_ids, failed_mask):
  """Count once; a simultaneous drop/timeout overrides a geometric success."""
  controller = getattr(env, "_smart_cube_curriculum", None)
  if controller is None:
    return
  ids = successful_ids[
    ~command._smart_skill_recorded[successful_ids] & ~failed_mask[successful_ids]
  ]
  controller.record_outcomes(int(ids.numel()), int(ids.numel()))
  command._smart_skill_recorded[ids] = True


def record_reset_skill_failures(env, env_ids):
  """Curriculum reset callback runs before scene/command buffers are cleared."""
  command = env.command_manager.get_term("reorient_command")
  preservation = getattr(command, 'flip_preservation', None)
  if preservation is not None:
    ids = resolve_env_ids(env, env_ids)
    timeout = command.failure_reason[ids] == 3
    preservation.finish(ids[timeout], int(env.common_step_counter), env.step_dt, timeout=True)
    preservation.finish(ids[~timeout], int(env.common_step_counter), env.step_dt)
  stats = getattr(command, '_turn_move_stats', None)
  if stats is not None:
    ids = resolve_env_ids(env, env_ids)
    timeout = env.termination_manager.time_outs[ids]
    # Stage timeout is a non-bootstrap termination, but still a timeout outcome.
    timeout |= (command.failure_reason[ids] == 3)
    stats.finish(ids[timeout], command.u_angle_error, env.step_dt, int(env.common_step_counter), timeout=True)
    stats.finish(ids[~timeout], command.u_angle_error, env.step_dt, int(env.common_step_counter))
  controller = get_smart_cube_curriculum(env)
  if int(env.common_step_counter) == 0:
    return
  ids = resolve_env_ids(env, env_ids)
  failed = ids[~command._smart_skill_recorded[ids]]
  controller.record_outcomes(0, int(failed.numel()))
  command._smart_skill_recorded[failed] = True


def update_smart_cube_curriculum(env, env_ids=None):
  """Step event; log after auto-reset so exact tags survive extras replacement."""
  controller = get_smart_cube_curriculum(env)
  controller.update(int(env.common_step_counter))
  if controller.cfg.realistic_dr and not controller.cfg.play:
    if controller.cfg.realistic_success_gated:
      progress = controller.realistic_progress
      first = second = realistic_parameter_progress(progress)
      low, high = realistic_profile_friction_bounds(progress)
    else:
      step = int(env.common_step_counter)
      if step < controller.realistic_last_step:
        controller.realistic_last_step = step
      else:
        controller.realistic_steps += step - controller.realistic_last_step
        controller.realistic_last_step = step
      first, second = realistic_stage_progress(controller.realistic_steps)
      low, high = realistic_friction_bounds(controller.realistic_steps)
    env.extras.setdefault('log', {}).update({
      'curriculum/realistic_steps': controller.realistic_steps,
      'curriculum/realistic_stage1': first,
      'curriculum/realistic_stage2': second,
      'curriculum/realistic_progress': controller.realistic_progress,
      'curriculum/finger_sliding_min': low,
      'curriculum/finger_sliding_max': high,
      'curriculum/palm_sliding': realistic_palm_sliding(),
      'curriculum/action_bias_max_deg': 3 * first,
      'curriculum/joint_observation_error_max_deg': .2 + .3 * first,
      'curriculum/imu_yaw_drift_max_deg_s': .05 * second,
      'curriculum/nominal_fraction': .30,
      'curriculum/action_ema_alpha': .75,
    })
  if controller.cfg.joint_dr_transition_steps:
    update_joint_dr_curriculum(controller, int(env.common_step_counter))
    p = controller.joint_dr_progress
    env.extras.setdefault('log', {}).update({
      'curriculum/joint_dr_progress': p,
      'curriculum/joint_execution_error_min_deg': 5 * p,
      'curriculum/joint_execution_error_max_deg': math.degrees(.02 * controller.alpha) * (1-p) + 15 * p,
      'curriculum/joint_observation_error_max_deg': 1 + 2 * p,
      'curriculum/dr_success_gated': float(controller.cfg.dr_success_gated),
    })
  if controller.cfg.low_friction_transition_steps:
    command = env.command_manager.get_term('reorient_command')
    tracker = getattr(command, 'bottom_tip_runtime', None)
    elapsed = int(env.common_step_counter) - controller.low_friction_last_step
    if not controller.cfg.play and (elapsed < 0 or elapsed >= controller.cfg.cooldown_steps):
      if not controller.cfg.low_friction_require_bottom_contact:
        totals = (controller.low_friction_skill_attempts,
                  controller.low_friction_skill_successes, 0, 0)
        update_low_friction_curriculum(controller, int(env.common_step_counter), totals)
      elif tracker is not None:
        update_low_friction_curriculum(controller, int(env.common_step_counter),
          tuple(tracker.requests.totals.tolist()))
    env.extras.setdefault('log', {})['curriculum/low_friction_progress'] = controller.low_friction_progress
    env.extras['log'].update(getattr(env, '_low_friction_bounds', {}))
  preservation = getattr(env, '_flip_preservation', None)
  if preservation is not None:
    env.extras.setdefault('log', {}).update(preservation.logs())
  stats = getattr(env, '_turn_move_stats', None)
  if stats is not None:
    env.extras.setdefault("log", {}).update(stats.logs())
  env.extras.setdefault("log", {}).update(
    {
      "curriculum/beta": controller.beta,
      "curriculum/dr_alpha": controller.alpha,
      "curriculum/skill_sr_ema": controller.skill_sr_ema,
    }
  )


def update_joint_dr_curriculum(controller, step):
  """Advance by control steps, optionally gated by fresh skill outcomes."""
  if controller.cfg.play or not controller.cfg.joint_dr_transition_steps:
    return
  elapsed = step - controller.joint_dr_last_step
  if 0 <= elapsed < controller.cfg.cooldown_steps:
    return
  totals = (controller.low_friction_skill_attempts, controller.low_friction_skill_successes)
  da, ds = (now - old for now, old in zip(totals, controller.joint_dr_last_totals, strict=True))
  controller.joint_dr_last_step, controller.joint_dr_last_totals = step, totals
  if not controller.cfg.dr_success_gated:
    if elapsed > 0:
      controller.joint_dr_progress = min(1., controller.joint_dr_progress
        + elapsed / controller.cfg.joint_dr_transition_steps)
    return
  if elapsed < 0 or da <= 0 or ds < 0 or ds > da:
    return
  if ds / da >= .85:
    controller.joint_dr_progress = min(1., controller.joint_dr_progress
      + min(elapsed, controller.cfg.cooldown_steps) / controller.cfg.joint_dr_transition_steps)


def update_low_friction_curriculum(controller, step, totals):
  """Advance sliding DR by steps, with optional legacy success/contact gates."""
  if controller.cfg.play or not controller.cfg.low_friction_transition_steps:
    return
  elapsed = step - controller.low_friction_last_step
  if 0 <= elapsed < controller.cfg.cooldown_steps:
    return
  previous = controller.low_friction_last_totals
  controller.low_friction_last_totals = tuple(totals)
  controller.low_friction_last_step = step
  if not controller.cfg.dr_success_gated:
    if elapsed > 0:
      controller.low_friction_progress = min(1., controller.low_friction_progress
        + elapsed / controller.cfg.low_friction_transition_steps)
    return
  deltas = tuple(current - old for current, old in zip(totals, previous, strict=True))
  # New simulator counters may restart on resume; never treat that as evidence.
  if elapsed < 0 or any(value < 0 for value in deltas) or deltas[0] <= 0:
    return
  contact_ok = (not controller.cfg.low_friction_require_bottom_contact
                or deltas[3] / deltas[0] >= .60)
  if deltas[1] / deltas[0] >= .85 and contact_ok:
    eligible = min(elapsed, controller.cfg.cooldown_steps)
    controller.low_friction_progress = min(1., controller.low_friction_progress
      + eligible / controller.cfg.low_friction_transition_steps)


def low_friction_bounds(default_sliding, progress, *, finger):
  """Geometric transition to absolute coefficients, not default multipliers."""
  if not 0 <= progress <= 1:
    raise ValueError('Low friction progress must be in [0,1]')
  if not torch.isfinite(default_sliding).all() or (default_sliding <= 0).any():
    raise ValueError('Default sliding friction must be finite and positive')
  lower = default_sliding * (.05 if finger else 1.)
  upper = default_sliding * (.1 if finger else 1.)
  return (lower * (.001 / lower).pow(progress),
          upper * (.005 / upper).pow(progress))


@requires_model_fields('geom_friction')
def randomize_low_friction_surfaces(
  env, env_ids,
  finger_cfg: SceneEntityCfg = SceneEntityCfg('robot', geom_names=r'.*finger.*_col'),
  cube_cfg: SceneEntityCfg = SceneEntityCfg('object', geom_names=r'cubelet:.*'),
  palm_cfg: SceneEntityCfg | None = None,
):
  """Reset-only sliding DR; optional half-strength legacy palm friction."""
  ids = resolve_env_ids(env, env_ids)
  if not ids.numel():
    return
  progress = get_smart_cube_curriculum(env).low_friction_progress
  bounds = {}
  if palm_cfg is not None:
    # Default-relative every time: repeated resets must not halve it again.
    alpha = get_smart_cube_curriculum(env).alpha
    palm_ids = env.scene[palm_cfg.name].indexing.geom_ids[palm_cfg.geom_ids].long()
    defaults = _get_default_field_values(env, 'geom_friction', ids, palm_ids)[..., 0]
    low, high = defaults * (.7 - .65 * alpha) * .5, defaults * (1.1 - alpha) * .5
    world, geom = torch.meshgrid(ids.long(), palm_ids, indexing='ij')
    env.sim.model.geom_friction[world, geom, 0] = low + torch.rand_like(low) * (high - low)
    bounds['curriculum/palm_sliding_min'] = low.amin()
    bounds['curriculum/palm_sliding_max'] = high.amax()
  for cfg, finger, name in ((finger_cfg, True, 'finger'), (cube_cfg, False, 'cube')):
    geom_ids = env.scene[cfg.name].indexing.geom_ids[cfg.geom_ids].long()
    defaults = _get_default_field_values(env, 'geom_friction', ids, geom_ids)[..., 0]
    low, high = low_friction_bounds(defaults, progress, finger=finger)
    values = low + torch.rand_like(low) * (high - low)
    world, geom = torch.meshgrid(ids.long(), geom_ids, indexing='ij')
    env.sim.model.geom_friction[world, geom, 0] = values
    bounds[f'curriculum/{name}_sliding_min'] = low.amin()
    bounds[f'curriculum/{name}_sliding_max'] = high.amax()
  env._low_friction_bounds = bounds


@requires_model_fields("dof_damping", "dof_frictionloss")
def randomize_smart_cube_face_resistance(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "object", joint_names=r"cubelet:driver:.*"
  ),
  scale_range: tuple[float, float] = (0.5, 1.8),
):
  """Reset only; default-relative scales never compound between episodes."""
  alpha = get_smart_cube_curriculum(env).alpha
  ranges = tuple(1 + (endpoint - 1) * alpha for endpoint in scale_range)
  for field in ("dof_damping", "dof_frictionloss"):
    _write_randomized_model_field(
      env, env_ids, field, ranges, "uniform", "scale", asset_cfg
    )


@requires_model_fields("geom_friction")
def randomize_smart_cube_hand_friction(
  env, env_ids,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", geom_names=(".*palm_.*", ".*finger.*_col"),
  ),
):
  """Anneal sliding friction to 0.05–0.1 of default without compounding."""
  from mjlab.envs.mdp import dr

  alpha = get_smart_cube_curriculum(env).alpha
  dr.geom_friction(
    env, env_ids, ranges=(0.7 - 0.65 * alpha, 1.1 - alpha),
    asset_cfg=asset_cfg, operation="scale", axes=[0],
  )
