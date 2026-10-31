"""Opt-in gentle-flip schedule and vectorized, physical joint-speed cost."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch


def speed_parameter(value, joints: int, *, positive: bool, name: str):
  values = (float(value),) if isinstance(value, (int, float)) else tuple(value)
  if len(values) not in (1, joints) or any(
    not math.isfinite(v) or (v <= 0 if positive else v < 0) for v in values
  ):
    raise ValueError(f'{name} must have 1 or {joints} finite, valid values')
  return values


def _velocity_cost(velocity, reference, soft):
  # Faulty worlds are terminated by PhysicsFailureGuard. Keep this term finite
  # even before the guard masks rewards; never interpret NaN as zero movement.
  finite = torch.isfinite(velocity).all(-1)
  safe = torch.nan_to_num(velocity, nan=200., posinf=200., neginf=-200.).clamp(-200., 200.)
  value = (.1 * (safe / reference).square()
    + (torch.relu(safe.abs() - soft) / reference).square()).mean(-1)
  return torch.where(finite, value, torch.full_like(value, 1.e6))


def gentle_velocity_cost(velocity, v_ref, v_soft):
  """Nonnegative reward *rate*; RewardManager integrates dt exactly once."""
  ref = speed_parameter(v_ref, velocity.shape[-1], positive=True, name='v_ref')
  soft = speed_parameter(v_soft, velocity.shape[-1], positive=False, name='v_soft')
  return _velocity_cost(velocity, velocity.new_tensor(ref), velocity.new_tensor(soft))


@dataclass(frozen=True)
class GentleFlipSchedule:
  ema_start: float = .75
  ema_end: float = .50
  ema_steps: int = 4000
  settle_steps: int = 4000
  check_steps: int = 2000
  min_attempts: int = 256
  success_threshold: float = .95
  aligned_threshold: float = .90
  require_alignment: bool = True
  speed_increment: float = .25
  max_speed_scale: float = 1.
  frozen: bool = False

  def __post_init__(self):
    if not all(math.isfinite(a) and 0 < a <= 1 for a in (self.ema_start, self.ema_end)):
      raise ValueError('EMA coefficients must be in (0, 1]')
    if self.ema_steps < 1 or self.settle_steps < 0 or self.check_steps < 1 or self.min_attempts < 1:
      raise ValueError('Invalid gentle-flip schedule step counts')
    if not 0 < self.speed_increment <= 1 or not 0 <= self.max_speed_scale <= 1:
      raise ValueError('Invalid speed penalty scale')
    if not math.isfinite(self.success_threshold) or not 0 <= self.success_threshold <= 1:
      raise ValueError('Invalid success threshold')
    if not math.isfinite(self.aligned_threshold) or not 0 <= self.aligned_threshold <= 1:
      raise ValueError('Invalid alignment threshold')


@dataclass
class GentleFlipState:
  elapsed_steps: int = 0
  last_step: int = 0
  last_check: int = 0
  last_attempts: int = 0
  last_successes: int = 0
  last_aligned: int = 0
  last_alignment_attempts: int = 0
  last_success_rate: float = 0.
  last_aligned_rate: float = 0.
  speed_scale: float = 0.

  def alpha(self, cfg):
    if cfg.frozen:
      return cfg.ema_end
    progress = min(1., self.elapsed_steps / cfg.ema_steps)
    return cfg.ema_start + (cfg.ema_end - cfg.ema_start) * progress

  def advance(self, step, attempts, successes, cfg, *, aligned=None, alignment_attempts=None):
    if cfg.frozen:
      self.speed_scale = cfg.max_speed_scale
      return
    self.elapsed_steps += max(0, step - self.last_step)
    self.last_step = step
    ready = cfg.ema_steps + cfg.settle_steps
    if self.elapsed_steps <= ready:
      self.last_check = self.elapsed_steps
      self.last_attempts, self.last_successes = attempts, successes
      self.last_aligned = aligned or 0
      self.last_alignment_attempts = alignment_attempts or 0
      return
    if self.elapsed_steps - self.last_check < cfg.check_steps:
      return
    fresh = attempts - self.last_attempts
    if fresh < cfg.min_attempts:
      return
    rate = (successes - self.last_successes) / fresh
    self.last_success_rate = rate
    alignment_ok = True
    if aligned is not None and alignment_attempts is not None:
      fresh_alignment = alignment_attempts - self.last_alignment_attempts
      self.last_aligned_rate = ((aligned - self.last_aligned) / fresh_alignment
        if fresh_alignment > 0 else 0.)
      alignment_ok = (not cfg.require_alignment or (fresh_alignment > 0
        and self.last_aligned_rate >= cfg.aligned_threshold))
    if rate >= cfg.success_threshold and alignment_ok:
      self.speed_scale = min(cfg.max_speed_scale, self.speed_scale + cfg.speed_increment)
    self.last_check = self.elapsed_steps
    self.last_attempts, self.last_successes = attempts, successes
    self.last_aligned = aligned or 0
    self.last_alignment_attempts = alignment_attempts or 0


def gentle_state(env):
  if not hasattr(env, '_gentle_flip_state'):
    env._gentle_flip_state = GentleFlipState()
  return env._gentle_flip_state


def apply_gentle_ema(env, schedule):
  """Update both the action cache and its export-visible config."""
  alpha = gentle_state(env).alpha(schedule)
  action = env.action_manager.get_term('joint_pos')
  action._ema_alpha = action.cfg.ema_alpha = alpha
  env.cfg.actions['joint_pos'].ema_alpha = alpha
  return alpha


def gentle_checkpoint_metadata(env):
  event = getattr(getattr(env, 'cfg', None), 'events', {}).get('gentle_flip_schedule')
  if event is None:
    return {}
  return {
    'gentle_flip': asdict(gentle_state(env)),
    'gentle_flip_action_ema': env.action_manager.get_term('joint_pos')._ema_alpha,
    'gentle_flip_runtime': {
      'schedule': asdict(event.params['schedule']),
      'reward': dict(env.cfg.rewards['gentle_joint_velocity'].params),
      'reward_weights': {name: env.cfg.rewards[name].weight for name in
        ('flip_align_streak', 'rubik_cube_position_stability')},
      'fraction_calibration': getattr(env, '_gentle_fraction_calibration', None),
    },
  }


def restore_gentle_checkpoint(env, infos, *, preview=False):
  """Restore full resumes; callers must not use this for actor-only transfer."""
  event = getattr(getattr(env, 'cfg', None), 'events', {}).get('gentle_flip_schedule')
  if event is None or not infos or 'gentle_flip' not in infos:
    return
  runtime = infos.get('gentle_flip_runtime')
  if runtime is not None:
    env._gentle_fraction_calibration = runtime.get('fraction_calibration')
    schedule = GentleFlipSchedule(**runtime['schedule'])
    reward = dict(runtime['reward'])
    event.params.update(schedule=schedule, coefficient=reward['coefficient'])
    env.cfg.rewards['gentle_joint_velocity'].params = reward
    env.reward_manager.get_term_cfg('gentle_joint_velocity').params = dict(reward)
    for name, weight in runtime.get('reward_weights', {}).items():
      env.cfg.rewards[name].weight = weight
      env.reward_manager.get_term_cfg(name).weight = weight
    if hasattr(env, '_gentle_speed_tensors'):
      del env._gentle_speed_tensors
  else:
    schedule = event.params['schedule']
  env._gentle_flip_state = GentleFlipState(**infos['gentle_flip'])
  if preview:
    schedule = replace(schedule, frozen=True, ema_end=infos['gentle_flip_action_ema'],
      max_speed_scale=env._gentle_flip_state.speed_scale)
    event.params['schedule'] = schedule
  env.event_manager.get_term_cfg('gentle_flip_schedule').params = dict(event.params)
  apply_gentle_ema(env, schedule)


def update_gentle_flip(env, env_ids=None, *, schedule, coefficient=0.):
  del env_ids
  from .smart_cube_curriculum import get_smart_cube_curriculum

  controller = get_smart_cube_curriculum(env)
  totals = env.command_manager.get_term('reorient_command').flip_preservation._totals
  state = gentle_state(env)
  state.advance(int(env.common_step_counter), controller.low_friction_skill_attempts,
    controller.low_friction_skill_successes, schedule,
    aligned=totals['aligned'],
    alignment_attempts=totals['aligned'] + totals['alignment_timeouts'])
  alpha = apply_gentle_ema(env, schedule)
  env.extras.setdefault('log', {}).update({
    'curriculum/action_ema_alpha': alpha,
    'gentle_flip/ema_alpha': alpha,
    'gentle_flip/elapsed_steps': state.elapsed_steps,
    'gentle_flip/velocity_scale': state.speed_scale,
    'gentle_flip/velocity_coefficient': coefficient * state.speed_scale,
    'gentle_flip/window_success_rate': state.last_success_rate,
    'gentle_flip/window_aligned_rate': state.last_aligned_rate,
  })
  if hasattr(env, '_gentle_velocity_cost'):
    env.extras['log']['gentle_flip/velocity_cost'] = env._gentle_velocity_cost.mean().detach()
