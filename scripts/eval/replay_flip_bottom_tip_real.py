"""Replay recorded actor inputs; report unknown contact instead of inventing H."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel


def load_actor(checkpoint):
  actor = MLPModel(TensorDict({'policy':torch.zeros(1,204)},batch_size=[1]),
    {'actor':['policy']},'actor',16,hidden_dims=[512,256,128],activation='elu',
    obs_normalization=True,distribution_cfg={
      'class_name':'rsl_rl.modules.distribution:SoftplusGaussianDistribution',
      'init_std':.5,'min_std':.2,'max_std':.8})
  actor.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['actor_state_dict'],strict=True)
  return actor.eval()


def audit(checkpoint, data):
  actor = load_actor(checkpoint)
  episodes = []
  total_squared = total_values = 0
  max_error = 0.
  for label in sorted(data.glob('episode_*.label.json')):
    annotation = json.loads(label.read_text())
    if annotation['result'] != 'success': continue
    path = data / annotation['episode']
    frames = [r for line in path.open() if (r:=json.loads(line)).get('event')=='frame']
    if not frames: continue
    obs = torch.tensor([f['observation'] for f in frames],dtype=torch.float32)
    recorded = torch.tensor([f['raw_action'] for f in frames],dtype=torch.float32)
    if obs.shape[1] != 204 or recorded.shape[1] != 16:
      raise ValueError(f'{path}: incompatible observation/action layout')
    with torch.inference_mode():
      inferred = actor(TensorDict({'policy':obs},batch_size=[len(frames)]))
    diff = inferred - recorded
    total_squared += diff.square().sum().item(); total_values += diff.numel()
    max_error = max(max_error,diff.abs().max().item())
    episodes.append(dict(episode=path.name,frames=len(frames),
      action_mse=diff.square().mean().item(),manual_useful_flip=True,
      requested_face_success=None,bottom_tip_contact=None,highest_reward_tier=None,
      goal_reward_D_if_target_success={'without_contact':4,'with_contact':24},
      reason='No measured object translation/contact points/contact forces or per-request completion labels'))
  return dict(checkpoint=str(checkpoint.resolve()),
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    source_data=str(data.resolve()),episodes=len(episodes),frames=sum(e['frames'] for e in episodes),
    action_mse=total_squared/total_values if total_values else None,
    max_action_error=max_error if total_values else None,
    highest_reward_tier_verdict='unknown',
    conclusion='Actor inference is reproducible; successful physical flips cannot be assigned the highest reward tier from these observations alone.',
    details=episodes)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--data',type=Path,required=True)
  args=parser.parse_args()
  print(json.dumps(audit(args.checkpoint,args.data),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__': main()
