"""Copy a checkpoint, preserving learning state and replacing only the gentle course."""
import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from wuji_mjlab.tasks.reorient.mdp.gentle_flip import (
  GentleFlipSchedule,
  GentleFlipState,
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source', type=Path)
  parser.add_argument('output', type=Path)
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument('--strong-speed', action='store_true',
    help='Reduce auxiliary rewards and use lambda16.12 through64.48')
  mode.add_argument('--reward-fraction-calibration', type=Path,
    help='Start immediately at a fixed, evaluated positive-reward fraction')
  mode.add_argument('--speed-coefficient', type=float,
    help='Start immediately at an explicitly chosen fixed velocity coefficient')
  args = parser.parse_args()
  if args.speed_coefficient is not None and (
      not math.isfinite(args.speed_coefficient) or args.speed_coefficient <= 0):
    parser.error('--speed-coefficient must be finite and positive')
  if args.output.exists():
    parser.error('Output exists; refusing overwrite')
  checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
  infos = checkpoint['infos']
  runtime = infos['gentle_flip_runtime']
  previous = infos['gentle_flip']
  controller = infos['smart_cube_curriculum']
  increment = .25 if args.strong_speed else .1
  cap = 1. if args.strong_speed else .4
  if args.strong_speed:
    # Absolute, not multiplied: repeated preparation must not amplify lambda.
    runtime['reward']['coefficient'] = 64.48486930321931
    runtime['reward_weights'] = dict(runtime.get('reward_weights', {}),
      flip_align_streak=1., rubik_cube_position_stability=.2)
  if args.reward_fraction_calibration:
    calibration = json.loads(args.reward_fraction_calibration.read_text())
    coefficient = float(calibration['coefficient'])
    positive = float(calibration['mean_success_positive_reward'])
    cost = float(calibration['mean_success_velocity_cost'])
    if (calibration['checkpoint_sha256'] != hashlib.sha256(args.source.read_bytes()).hexdigest()
        or calibration['schema_version'] != 1 or calibration['ema'] != .5
        or calibration['successful_requests'] < 256
        or calibration['target_fraction'] != .4
        or not all(math.isfinite(x) and x > 0 for x in (coefficient, positive, cost))
        or not math.isclose(coefficient * cost / positive, .4, rel_tol=1.e-6)
        or calibration['reward_weights'] != {'flip_align_streak': 1., 'rubik_cube_position_stability': .2}
        or any(list(calibration[key]) != list(runtime['reward'][key])
          for key in ('joint_names', 'v_ref', 'v_soft'))):
      parser.error('Invalid reward-fraction calibration or mismatched checkpoint/thresholds')
    runtime['reward']['coefficient'] = coefficient
    runtime['reward_weights'] = calibration['reward_weights']
    runtime['fraction_calibration'] = calibration
    increment = cap = 1.
  if args.speed_coefficient is not None:
    runtime['reward']['coefficient'] = args.speed_coefficient
    runtime['reward_weights'] = dict(runtime.get('reward_weights', {}),
      flip_align_streak=1., rubik_cube_position_stability=.2)
    # A manual coefficient has no newly measured reward-fraction calibration.
    runtime.pop('fraction_calibration', None)
    increment = cap = 1.
  schedule = GentleFlipSchedule(ema_start=.5, ema_end=.5, ema_steps=1,
    settle_steps=0, check_steps=2000, min_attempts=256,
    success_threshold=.95, require_alignment=False,
    speed_increment=increment, max_speed_scale=cap)
  runtime['schedule'] = asdict(schedule)
  infos['gentle_flip_action_ema'] = .5
  infos['gentle_flip'] = asdict(GentleFlipState(
    elapsed_steps=previous['elapsed_steps'], last_step=previous['last_step'],
    last_check=previous['elapsed_steps'],
    last_attempts=controller['low_friction_skill_attempts'],
    last_successes=controller['low_friction_skill_successes'], speed_scale=increment))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, args.output)
  print(f'Prepared full resume from iteration {checkpoint["iter"]}; '
    f'initial coefficient={runtime["reward"]["coefficient"] * increment}; '
    f'EMA=.5, success-only>=.95, scale +{increment} per2000 fresh steps, cap={cap}; '
    f'reward weights={runtime.get("reward_weights", "legacy")}')


if __name__ == '__main__':
  main()
