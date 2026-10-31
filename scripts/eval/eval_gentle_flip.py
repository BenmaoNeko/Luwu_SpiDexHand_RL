"""Frozen, request-level flip evaluation and joint-speed calibration.

Run with PYTHONPATH=src and the same MJLab Python used for training. All
outcomes count, including requests unfinished at the finite evaluation horizon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


class RequestRecorder:
  def __init__(self, count, joint_names, dt):
    self.active = [None] * count
    self._step_rows = [None] * count
    self.rows = []
    self.joint_names, self.dt = list(joint_names), dt

  def begin(self, env_id, *, face, half_turn):
    if self.active[env_id] is not None:
      self.finish(env_id, 'unfinished')
    self.active[env_id] = dict(env_id=env_id, face=face, half_turn=half_turn,
      nominal=None, legs=0, aligned=True, duration_s=0., arrivals=0,
      contact_steps=0, samples=0, warmup_samples=0, invalid_samples=0,
      positive_reward=0., positive_reward_terms={}, reward_samples=0,
      _velocities=[], _targets=[], exit_internal_error_deg=None)

  def sample(self, velocities, targets, nominal, contact, *, warmup=None):
    self._step_rows = list(self.active)
    velocities = np.asarray(velocities, dtype=float)
    targets = np.asarray(targets, dtype=float)
    for index, row in enumerate(self.active):
      if row is None:
        continue
      if row['nominal'] is None:
        row['nominal'] = bool(nominal[index])
      row['duration_s'] += self.dt
      row['samples'] += 1
      row['contact_steps'] += int(contact[index] > 0)
      if warmup is not None and warmup[index]:
        row['warmup_samples'] += 1
        continue
      if not np.isfinite(velocities[index]).all() or not np.isfinite(targets[index]).all():
        row['invalid_samples'] += 1
        continue
      row['_velocities'].append(velocities[index].tolist())
      row['_targets'].append(targets[index].tolist())

  def add_rewards(self, weighted_rates, names):
    """Attribute reward to the pre-reset request; manager exposes rates, not sums."""
    for row, values in zip(self._step_rows, weighted_rates, strict=True):
      if row is None:
        continue
      row['reward_samples'] += 1
      for name, value in zip(names, values, strict=True):
        value = float(value) * self.dt
        if not math.isfinite(value):
          row['invalid_samples'] += 1
        elif value > 0:
          row['positive_reward'] += value
          terms = row['positive_reward_terms']
          terms[name] = terms.get(name, 0.) + value

  def finish(self, env_id, outcome, *, intermediate=False, aligned=False,
             arrived=False, internal_error_deg=None):
    row = self.active[env_id]
    if row is None:
      return
    row['aligned'] &= bool(aligned)
    row['arrivals'] += int(arrived)
    row['legs'] += int(outcome == 'success')
    if intermediate and outcome == 'success':
      return
    row['outcome'] = outcome
    row['exit_internal_error_deg'] = internal_error_deg
    self.rows.append(row)
    self.active[env_id] = None

  def flush(self):
    for index in range(len(self.active)):
      if self.active[index] is not None:
        self.finish(index, 'unfinished')


def _summary(rows):
  count = len(rows)
  result = {name: sum(row['outcome'] == name for row in rows)
    for name in ('success', 'drop', 'timeout', 'physics_failure', 'unfinished', 'failure')}
  result.update(attempts=count, success_rate=result['success'] / count if count else None,
    strict_aligned_successes=sum(r['outcome'] == 'success' and r['aligned'] for r in rows),
    geometric_arrival_requests=sum(r['arrivals'] > 0 for r in rows),
    invalid_samples=sum(r['invalid_samples'] for r in rows))
  for outcome, selected in (('all', rows), ('success', [r for r in rows if r['outcome'] == 'success']),
                            ('not_success', [r for r in rows if r['outcome'] != 'success'])):
    velocities = [r['_velocities'] for r in selected if r['_velocities']]
    if velocities:
      values = np.concatenate(velocities)
      diffs = [np.diff(np.asarray(r['_targets']), axis=0) for r in selected if len(r['_targets']) > 1]
      second = [np.diff(np.asarray(r['_targets']), n=2, axis=0) for r in selected if len(r['_targets']) > 2]
      reversal = [((np.asarray(r['_velocities'])[1:] * np.asarray(r['_velocities'])[:-1] < 0)
        & (np.abs(np.asarray(r['_velocities'])[1:]) > .05)
        & (np.abs(np.asarray(r['_velocities'])[:-1]) > .05))
        for r in selected if len(r['_velocities']) > 1]
      result[outcome + '_motion'] = dict(
        speed_p95=np.quantile(np.abs(values), .95, axis=0).tolist(),
        speed_peak=np.abs(values).max(axis=0).tolist(),
        mean_joint_rms=float(np.sqrt(np.mean(values ** 2, axis=1)).mean()),
        target_delta_rms=float(np.sqrt(np.mean(np.concatenate(diffs) ** 2))) if diffs else None,
        target_second_delta_rms=float(np.sqrt(np.mean(np.concatenate(second) ** 2))) if second else None,
        reversal_fraction=float(np.concatenate(reversal).mean()) if reversal else None)
  result['success_duration_median_s'] = (float(np.median([
    r['duration_s'] for r in rows if r['outcome'] == 'success'])) if result['success'] else None)
  result['no_contact_fraction'] = (sum(r['samples'] - r['contact_steps'] for r in rows)
    / sum(r['samples'] for r in rows)) if sum(r['samples'] for r in rows) else None
  errors = [r['exit_internal_error_deg'] for r in rows
    if r['exit_internal_error_deg'] is not None and math.isfinite(r['exit_internal_error_deg'])]
  result['exit_internal_error_median_deg'] = float(np.median(errors)) if errors else None
  return result


def summarize_requests(rows):
  return {**_summary(rows),
    'nominal': _summary([r for r in rows if r['nominal'] is True]),
    'non_nominal': _summary([r for r in rows if r['nominal'] is False]),
    'by_face': {str(face): _summary([r for r in rows if r['face'] == face]) for face in range(6)},
    'half_turn': _summary([r for r in rows if r['half_turn']])}


def calibrate(rows, joint_names):
  successes = [r for r in rows if r['outcome'] == 'success'
    and r['_velocities'] and not r['invalid_samples']]
  if not successes:
    raise ValueError('No complete, finite successful requests for calibration')
  speed = np.abs(np.concatenate([r['_velocities'] for r in successes]))
  if float(np.max(speed)) < 1.e-4:
    raise ValueError('Zero-motion calibration is not meaningful')
  # Threshold at baseline P75; reference at P95. A floor protects nearly static
  # joints from amplifying numerical noise into an enormous penalty.
  reference = np.maximum(.1, np.quantile(speed, .95, axis=0))
  soft = np.quantile(speed, .75, axis=0)
  integrals = []
  for row in successes:
    velocity = np.abs(np.asarray(row['_velocities']))
    cost = (.1 * (velocity / reference) ** 2
      + (np.maximum(velocity - soft, 0) / reference) ** 2).mean(axis=1)
    dt = row['duration_s'] / row['samples']
    integrals.append(float(cost.sum() * dt))
  median = float(np.median(integrals))
  if not math.isfinite(median) or median < 1.e-5:
    raise ValueError('Successful integrated cost too small for a reliable calibration')
  return dict(schema_version=1, joint_names=list(joint_names),
    v_ref=reference.tolist(), v_soft=soft.tolist(), median_success_cost=median,
    lambda_initial=1. / median, lambda_target=2.5 / median,
    successful_requests=len(successes), reference_floor_rad_s=.1,
    reference_quantile=.95, soft_quantile=.75,
    integrated_cost_p95=float(np.quantile(integrals, .95)))


def calibrate_reward_fraction(rows, fixed, joint_names, fraction, *, min_requests=256):
  """Freeze lambda using mean positive reward / mean cost on successful requests."""
  from wuji_mjlab.tasks.reorient.mdp.gentle_flip import speed_parameter
  if not math.isfinite(fraction) or not 0 < fraction < 1:
    raise ValueError('Reward fraction must be in (0, 1)')
  if list(fixed['joint_names']) != list(joint_names):
    raise ValueError('Frozen calibration joint order differs')
  ref = np.asarray(speed_parameter(fixed['v_ref'], len(joint_names), positive=True, name='v_ref'))
  soft = np.asarray(speed_parameter(fixed['v_soft'], len(joint_names), positive=False, name='v_soft'))
  selected = [r for r in rows if r['outcome'] == 'success' and r['_velocities']
    and not r['invalid_samples'] and r['reward_samples'] == r['samples']]
  if len(selected) < min_requests:
    raise ValueError('Insufficient complete successful reward samples')
  costs, positives = [], []
  for row in selected:
    velocity = np.abs(np.asarray(row['_velocities']))
    integral = float((.1 * (velocity / ref) ** 2
      + (np.maximum(velocity - soft, 0) / ref) ** 2).mean(-1).sum()
      * row['duration_s'] / row['samples'])
    costs.append(integral)
    positives.append(row['positive_reward'])
  if not np.isfinite(costs).all() or not np.isfinite(positives).all() or min(costs) <= 1.e-8 or min(positives) <= 0:
    raise ValueError('Invalid successful cost or positive reward')
  coefficient = float(fraction * np.sum(positives) / np.sum(costs))
  ratios = coefficient * np.asarray(costs) / np.asarray(positives)
  return dict(schema_version=1, target_fraction=fraction, coefficient=coefficient,
    joint_names=list(joint_names), v_ref=ref.tolist(), v_soft=soft.tolist(),
    reward_weights={'flip_align_streak': 1., 'rubik_cube_position_stability': .2},
    successful_requests=len(selected), mean_success_positive_reward=float(np.mean(positives)),
    mean_success_velocity_cost=float(np.mean(costs)),
    achieved_fraction=float(coefficient * np.sum(costs) / np.sum(positives)),
    per_request_fraction_quantiles=np.quantile(ratios, [.1, .5, .9]).tolist())


def _install_recorder(env, command, recorder):
  """Observe BEFORE reset/goal advance; never infer success from post-reset obs."""
  original_start = command._install_flip_or_turn_start

  def start(ids, requested):
    original_start(ids, requested)
    for index in ids.cpu().tolist():
      recorder.begin(index, face=int(command.requested_face[index]),
        half_turn=bool(command.bottom_intermediate[index]))

  command._install_flip_or_turn_start = start
  state = command.flip_preservation
  original_finish = state.finish

  def finish(ids, step, dt, success=False, timeout=False):
    selected = ids[state.active[ids]]
    physics = getattr(env, '_physics_failure_mask', None)
    for index in selected.cpu().tolist():
      failed_physics = physics is not None and bool(physics[index])
      dropped = bool(env.termination_manager.get_term('cage_drop')[index])
      # reason=3 is the existing stage/episode timeout; both remain failures.
      expired = timeout or int(command.failure_reason[index]) == 3
      outcome = ('physics_failure' if failed_physics else 'drop' if dropped else
        'success' if success else 'timeout' if expired else 'failure')
      recorder.finish(index, outcome, intermediate=bool(command.bottom_intermediate[index]),
        aligned=bool(state.aligned_event[index]), arrived=bool(state.arrival_seen[index]),
        internal_error_deg=float(torch.rad2deg(state.error[index])))
    original_finish(ids, step, dt, success=success, timeout=timeout)

  state.finish = finish
  original_compute = env.termination_manager.compute

  def compute():
    action = env.action_manager.get_term('joint_pos')
    recorder.sample(env.scene['robot'].data.joint_vel.detach().cpu().numpy(),
      action.processed_action.detach().cpu().numpy(),
      env._realistic_nominal_mask.cpu().tolist(), command._read_tip_contact_count().cpu().tolist(),
      warmup=(env.episode_length_buf * env.step_dt < action.cfg.warmup_time_s).cpu().tolist())
    return original_compute()

  env.termination_manager.compute = compute
  original_reward = env.reward_manager.compute

  def reward_compute(dt):
    result = original_reward(dt)
    if not env.reward_manager._scale_by_dt:
      raise ValueError('Evaluator requires dt-scaled rewards')
    recorder.add_rewards(env.reward_manager._step_reward.detach().cpu().tolist(),
      env.reward_manager.active_terms)
    return result

  env.reward_manager.compute = reward_compute


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--calibration-output', type=Path)
  parser.add_argument('--calibration', type=Path, help='Use frozen thresholds from an earlier baseline')
  parser.add_argument('--ema', type=float, default=.75)
  parser.add_argument('--num-envs', type=int, default=64)
  parser.add_argument('--steps', type=int, default=2400)
  parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--plan-pool', type=Path)
  parser.add_argument('--save-traces', action='store_true')
  parser.add_argument('--reward-fraction', type=float)
  parser.add_argument('--reward-fraction-output', type=Path)
  args = parser.parse_args()
  if not 0 < args.ema <= 1 or args.steps < 1 or args.num_envs < 1:
    parser.error('Invalid EMA, steps, or environment count')
  if args.output.exists() or (args.calibration_output and args.calibration_output.exists()):
    parser.error('Output already exists; choose a new output path')
  if (args.reward_fraction is None) != (args.reward_fraction_output is None):
    parser.error('--reward-fraction and --reward-fraction-output must be provided together')
  if args.reward_fraction_output and args.reward_fraction_output.exists():
    parser.error('Reward fraction output already exists')
  if args.device.startswith('cuda') and not torch.cuda.is_available():
    parser.error('CUDA unavailable; select --device cpu explicitly for a CPU smoke test')
  source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(Path('src/wuji_mjlab').rglob('*.py'))}
  evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

  from mjlab.envs import ManagerBasedRlEnv
  from replay_flip_bottom_tip_real import load_actor
  from tensordict import TensorDict
  from wuji_mjlab.rl.physics_failure_guard import install_physics_failure_guard
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_gentle_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
    generate_solver_plan_pool,
    load_solver_plan_pool,
  )

  # Never overwrite the project's pool or bypass its source fingerprint check.
  if args.plan_pool is None:
    args.plan_pool = Path(tempfile.mkdtemp(prefix='gentle-flip-pool-')) / 'plans.json'
    generate_solver_plan_pool(args.plan_pool, count=256, seed=20261008)
  load_solver_plan_pool(args.plan_pool)

  actor = load_actor(args.checkpoint).to(args.device)
  fixed_reward = None
  if args.reward_fraction is not None:
    fixed_reward = torch.load(args.checkpoint, map_location='cpu', weights_only=False)[
      'infos']['gentle_flip_runtime']['reward']
  all_rows, seeds, names = [], [], None
  for seed in args.seeds:
    torch.manual_seed(seed)
    cfg = luwu_hand_rubik_flip_gentle_env_cfg(num_envs=args.num_envs,
      plan_pool_path=str(args.plan_pool) if args.plan_pool else None)
    cfg.seed = seed
    command_cfg = cfg.commands['reorient_command']
    command_cfg.smart_cube_curriculum = replace(command_cfg.smart_cube_curriculum, play=True, play_alpha=0.)
    cfg.actions['joint_pos'].ema_alpha = args.ema
    if fixed_reward is not None:
      cfg.rewards['flip_align_streak'].weight = 1.
      cfg.rewards['rubik_cube_position_stability'].weight = .2
    params = cfg.events['gentle_flip_schedule'].params
    params['schedule'] = replace(params['schedule'], frozen=True, ema_end=args.ema, max_speed_scale=0.)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
      install_physics_failure_guard(env, args.output.parent / 'physics_failures',
        preserve_failure_rewards=('cage_drop',))
      names = list(env.scene['robot'].joint_names)
      recorder = RequestRecorder(args.num_envs, names, env.step_dt)
      command = env.command_manager.get_term('reorient_command')
      _install_recorder(env, command, recorder)
      obs, _ = env.reset(seed=seed)
      if obs['policy'].shape[-1] != 204 or len(names) != 16:
        raise ValueError('Checkpoint requires exactly 204 policy observations and 16 joint actions')
      started = time.perf_counter()
      with torch.inference_mode():
        for step in range(args.steps):
          actions = actor(TensorDict({'policy': obs['policy']}, batch_size=[args.num_envs]))
          obs, *_ = env.step(actions)
          if (step + 1) % 200 == 0:
            print(f'GENTLE_EVAL seed={seed} steps={step + 1} requests={len(recorder.rows)}', flush=True)
      recorder.flush()
      for row in recorder.rows:
        row['seed'] = seed
      all_rows.extend(recorder.rows)
      seeds.append(dict(seed=seed, elapsed_s=time.perf_counter() - started,
        physics_failures=env._physics_failure_guard.total_failures,
        summary=summarize_requests(recorder.rows)))
    finally:
      env.close()
  try:
    calibration = calibrate(all_rows, names)
    calibration_error = None
  except ValueError as exc:
    calibration, calibration_error = None, str(exc)
  try:
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
      stderr=subprocess.DEVNULL).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    revision = None  # Source-file hashes remain available in deployed code snapshots.
  result = dict(schema_version=1, checkpoint=str(args.checkpoint.resolve()),
    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    code_revision=revision,
    ema=args.ema, realistic_progress=1., nominal_fraction=.3, device=args.device,
    num_envs=args.num_envs, steps=args.steps, joint_names=names, seeds=seeds,
    summary=summarize_requests(all_rows), calibration=calibration,
    calibration_error=calibration_error,
    plan_pool=str(args.plan_pool.resolve()),
    plan_pool_sha256=hashlib.sha256(args.plan_pool.read_bytes()).hexdigest(),
    evaluator_sha256=evaluator_hash, source_sha256=source_hashes,
    limitations=['Control-step samples may miss substep impact peaks.',
      'Finite-horizon unfinished requests count in the success-rate denominator.',
      'Simulation evaluation does not demonstrate real hardware performance.'])
  if args.calibration:
    fixed = json.loads(args.calibration.read_text())
    if fixed['joint_names'] != names:
      raise ValueError('Frozen calibration joint order differs')
    from wuji_mjlab.tasks.reorient.mdp.gentle_flip import speed_parameter
    speed_parameter(fixed['v_ref'], len(names), positive=True, name='v_ref')
    speed_parameter(fixed['v_soft'], len(names), positive=False, name='v_soft')
    for row in all_rows:
      values = np.abs(np.asarray(row['_velocities']))
      ref, soft = np.asarray(fixed['v_ref']), np.asarray(fixed['v_soft'])
      integral = float((.1 * (values / ref) ** 2
        + (np.maximum(values - soft, 0) / ref) ** 2).mean(-1).sum()
        * row['duration_s'] / row['samples']) if values.size else 0.
      row['velocity_cost_integral'] = integral
      row['initial_penalty'] = fixed['lambda_initial'] * integral
      row['target_penalty'] = fixed['lambda_target'] * integral
    result['frozen_calibration'] = fixed
  result['requests'] = [{k: v for k, v in r.items() if args.save_traces or not k.startswith('_')}
    for r in all_rows]
  if fixed_reward is not None:
    fraction_calibration = calibrate_reward_fraction(all_rows, fixed_reward, names, args.reward_fraction)
    fraction_calibration.update(checkpoint_sha256=result['checkpoint_sha256'], ema=args.ema)
    result['reward_fraction_calibration'] = fraction_calibration
    args.reward_fraction_output.parent.mkdir(parents=True, exist_ok=True)
    args.reward_fraction_output.write_text(json.dumps(fraction_calibration, indent=2, allow_nan=False) + '\n')
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
  if args.calibration_output:
    if calibration is None:
      raise ValueError(f'Evaluation saved, but calibration failed: {calibration_error}')
    calibration['checkpoint_sha256'] = result['checkpoint_sha256']
    calibration['ema'] = args.ema
    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_output.write_text(json.dumps(calibration, indent=2, allow_nan=False) + '\n')
  print('GENTLE_RESULT ' + json.dumps({key: result['summary'][key] for key in (
    'attempts', 'success', 'success_rate', 'drop', 'timeout', 'physics_failure', 'unfinished')}), flush=True)


if __name__ == '__main__':
  main()
