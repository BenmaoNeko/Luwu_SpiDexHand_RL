"""Frozen 900 instrumentation probe; no optimizer or training updates."""
import argparse
import json
import math
import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv

from replay_flip_bottom_tip_real import load_actor
from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg
from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--device',default='cpu')
  parser.add_argument('--num-envs',type=int,default=4)
  parser.add_argument('--steps',type=int,default=400)
  parser.add_argument('--seed',type=int,default=42)
  args=parser.parse_args()
  if args.num_envs < 1 or args.steps < 1: parser.error('positive envs/steps required')
  actor=load_actor(args.checkpoint).to(args.device)
  with tempfile.TemporaryDirectory(prefix='bottom-tip-eval-') as directory:
    pool=Path(directory)/'pool.json'
    generate_solver_plan_pool(pool,count=2,seed=args.seed,min_scramble_moves=2,max_scramble_moves=2)
    cfg=luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(num_envs=args.num_envs,plan_pool_path=str(pool))
    env=ManagerBasedRlEnv(cfg=cfg,device=args.device)
    try:
      obs,_=env.reset(seed=args.seed)
      geometry_frames=overflow_frames=0
      for _ in range(args.steps):
        with torch.inference_mode():
          action=actor(TensorDict({'policy':obs['policy']},batch_size=[args.num_envs]))
        obs,_,_,_,_=env.step(action)
        tracker=env.command_manager.get_term('reorient_command').bottom_tip_runtime
        geometry_frames += int(tracker.geometry_valid.sum())
        overflow_frames += int(tracker.overflow.sum())
      tracker.requests.finish(torch.arange(args.num_envs,device=args.device),torch.zeros(args.num_envs,dtype=torch.bool,device=args.device))
      metrics=tracker.requests.metrics()
      metrics={k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in metrics.items()}
      print('BOTTOM_TIP_PROBE_JSON '+json.dumps(dict(metrics=metrics,
        contact_filter_counts=dict(zip(('found','force','distal','patch','patch_and_normal'),tracker.diagnostics.tolist())),
        valid_geometry_env_frames=geometry_frames,overflow_env_frames=overflow_frames,
        frames=args.num_envs*args.steps,terminal_censoring='active requests counted as horizon failures',
        status='instrumentation probe; not the balanced 2048-request acceptance evaluation')))
    finally: env.close()


if __name__=='__main__': main()
