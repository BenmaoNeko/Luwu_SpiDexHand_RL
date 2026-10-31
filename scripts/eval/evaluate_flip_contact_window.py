"""Paired frozen-policy probe: full versus first-45-degree contact per 90-degree leg.

Evaluation-only hooks; no optimizer, training-task changes or remote writes.
"""
import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from replay_flip_bottom_tip_real import load_actor
from tensordict import TensorDict
from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
  luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip_runtime import BottomTipRuntime
from wuji_mjlab.tasks.reorient.mdp.flip_contact_window import EarlyContactWindow
from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool


class LegProbe(BottomTipRuntime):
  def __init__(self, command):
    super().__init__(command)
    env = command._env
    self.early = EarlyContactWindow(env.num_envs, env.device)
    self.initial_angle = torch.zeros(env.num_envs, device=env.device)
    self.second_pending = torch.zeros_like(self.initialized)
    self.kind = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.rows = []
    original_contact = self.requests.contact

    def contact(valid, dt):
      original_contact(valid, dt)
      current_angle = torch.acos(command._read_flip_target_dot().clamp(-1, 1))
      self.early.sample(self.initial_angle - current_angle, valid, dt)

    self.requests.contact = contact

  def start_leg(self, ids):
    self.finish_leg(ids, torch.zeros_like(ids, dtype=torch.bool), reason='replaced')
    super().start(ids)
    self.early.start(ids)
    self.initial_angle[ids] = torch.acos(self.command.flip_start_dot[ids].clamp(-1, 1))
    self.kind[ids] = torch.where(self.command.bottom_intermediate[ids], 1,
      torch.where(self.second_pending[ids], 2, 0))
    self.second_pending[ids] = False

  def finish_leg(self, ids, success, *, reason, aligned=None):
    valid = self.requests.active[ids]
    ids, success = ids[valid], success[valid]
    aligned = torch.zeros_like(success) if aligned is None else aligned[valid]
    values = torch.stack((success, self.requests.supported[ids],
      self.early.evidence.supported[ids], self.geometry_valid[ids], self.overflow[ids],
      self.kind[ids], self.early.closed[ids], aligned), dim=-1).cpu().tolist()
    angles = torch.rad2deg(self.initial_angle[ids]).cpu().tolist()
    for row, angle in zip(values, angles, strict=True):
      self.rows.append(dict(zip(('success','full_contact','early_contact','geometry_valid',
        'overflow','kind','crossed_45','aligned'), row, strict=True),
        initial_angle_deg=angle, reason=reason))
    self.requests.finish(ids, success)
    self.early.evidence.finish(ids, success)

  def install_hooks(self):
    c = self.command
    state = c.flip_preservation
    install, advance, update, finish = c._install_goal, c._advance_pending_goals, state.update, state.finish

    def install_goal(ids, faces, phase, targets):
      install(ids, faces, phase, targets)
      if phase == 0:
        self.start_leg(ids)

    def advance_goals():
      self.second_pending |= c._pending_goal_advance & c.bottom_intermediate
      advance()

    def update_shape(*args, **kwargs):
      update(*args, **kwargs)
      ids = state.arrival_event.nonzero().flatten()
      self.requests.freeze(ids)
      self.early.evidence.freeze(ids)

    def finish_shape(ids, step, dt, success=False, timeout=False):
      face_valid = c._read_flip_target_dot()[ids] >= math.cos(c.cfg.continuous_flip_upright_tolerance_rad)
      dropped = c._env.termination_manager.get_term('cage_drop')[ids]
      outcome = face_valid & ~dropped & success
      self.finish_leg(ids, outcome, reason='completed' if success else 'reset',
        aligned=state.aligned_event[ids] & outcome)
      finish(ids, step, dt, success=success, timeout=timeout)

    c._install_goal, c._advance_pending_goals = install_goal, advance_goals
    state.update, state.finish = update_shape, finish_shape


def summarize(rows):
  success = sum(bool(r['success']) for r in rows)
  def ratio(a, b):
    return a/b if b else None
  result = {'legs':len(rows), 'successes':success, 'success_rate':ratio(success,len(rows)),
    'geometry_invalid':sum(not r['geometry_valid'] for r in rows),
    'overflow':sum(bool(r['overflow']) for r in rows),
    'kind_counts':{name:sum(r['kind']==kind for r in rows)
      for kind,name in enumerate(('single_90','180_first_90','180_second_90'))}}
  for name in ('full_contact','early_contact'):
    h = sum(bool(r[name]) for r in rows)
    sh = sum(bool(r[name] and r['success']) for r in rows)
    strict = sum(bool(r[name] and r['aligned']) for r in rows)
    result[name] = dict(contacted=h, contacted_successes=sh,
      success_after_contact=ratio(sh,h), contact_given_success=ratio(sh,success),
      aligned_success_after_contact=ratio(strict,h))
  result['successful_late_only'] = sum(bool(r['success'] and r['full_contact'] and not r['early_contact']) for r in rows)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--num-envs', type=int, default=32)
  parser.add_argument('--steps', type=int, default=800)
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  if args.num_envs < 1 or args.steps < 1:
    parser.error('positive environments and steps required')
  torch.manual_seed(args.seed)
  actor = load_actor(args.checkpoint).to(args.device)
  with tempfile.TemporaryDirectory(prefix='flip-window-probe-') as directory:
    pool = Path(directory)/'pool.json'
    generate_solver_plan_pool(pool, count=2, seed=args.seed, min_scramble_moves=2, max_scramble_moves=2)
    cfg = luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(num_envs=args.num_envs, plan_pool_path=str(pool))
    holder = {}
    cfg.metrics['paired_leg_probe'] = MetricsTermCfg(func=lambda env: holder['probe'].substep(), per_substep=True)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
      probe = LegProbe(env.command_manager.get_term('reorient_command'))
      holder['probe'] = probe
      probe.install_hooks()
      obs, _ = env.reset(seed=args.seed)
      for step in range(args.steps):
        with torch.inference_mode():
          action = actor(TensorDict({'policy':obs['policy']},batch_size=[args.num_envs]))
        obs, _, _, _, _ = env.step(action)
        if (step+1) % 200 == 0:
          print(f'WINDOW_PROGRESS steps={step+1} settled_legs={len(probe.rows)}', flush=True)
      probe.finish_leg(torch.arange(args.num_envs,device=args.device),
        torch.zeros(args.num_envs,dtype=torch.bool,device=args.device), reason='horizon')
      settled = [r for r in probe.rows if r['reason'] != 'horizon']
      output = dict(seed=args.seed, checkpoint=str(args.checkpoint), steps=args.steps,
        num_envs=args.num_envs, unit='90-degree leg; each leg relocks lower-side patches',
        angle='initial target-face tilt minus current tilt; first >=45deg closes permanently',
        settled=summarize(settled), horizon_censored=summarize([r for r in probe.rows if r['reason']=='horizon']),
        horizon_as_failure_sensitivity=summarize(probe.rows),
        by_kind={name:summarize([r for r in settled if r['kind']==i])
          for i,name in enumerate(('single_90','180_first_90','180_second_90'))},
        limitation='Frozen-policy simulation probe, not real-robot support/causality validation', rows=probe.rows)
      print('CONTACT_WINDOW_JSON '+json.dumps(output,allow_nan=False),flush=True)
    finally:
      env.close()


if __name__ == '__main__':
  main()
