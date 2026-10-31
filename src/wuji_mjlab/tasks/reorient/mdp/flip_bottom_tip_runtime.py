"""Physics-frequency instrumentation; experimental geometry needs probe validation."""
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .flip_bottom_tip import (
  BottomTipRequests,
  BottomTipRewardLedger,
  lower_side_patch_mask,
  on_face_patch,
)
from .flip_contact_window import EarlyContactWindow


class BottomTipRuntime:
  def __init__(self, command):
    self.command = command
    env = command._env
    self.window = (EarlyContactWindow(env.num_envs, env.device,
      max_progress_deg=command.cfg.bottom_tip_contact_window_deg,
      success_rewards=command.cfg.bottom_tip_success_rewards)
      if command.cfg.bottom_tip_early_per_leg else None)
    self.requests = (self.window.evidence if self.window is not None
      else BottomTipRequests(env.num_envs, env.device))
    self.initial_angle = torch.zeros(env.num_envs, device=env.device)
    self.ledger = BottomTipRewardLedger(env.num_envs, env.device)
    self.local_normal = torch.zeros((env.num_envs,26,4,3), device=env.device)
    self.selected = torch.zeros((env.num_envs,26,4), dtype=torch.bool, device=env.device)
    self.compact_contact = command.cfg.bottom_tip_compact_contact
    self.debug_diagnostics = command.cfg.bottom_tip_debug_diagnostics
    self.patch_cubelets = torch.zeros((env.num_envs,12), dtype=torch.long, device=env.device)
    self.patch_normals = torch.zeros((env.num_envs,12,3), device=env.device)
    self.patch_valid = torch.zeros((env.num_envs,12), dtype=torch.bool, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.overflow = torch.zeros_like(self.initialized)
    self.geometry_valid = torch.zeros_like(self.initialized)
    self.diagnostics = torch.zeros(5,dtype=torch.long,device=env.device)
    robot = env.scene['robot']
    palm, _ = robot.find_bodies('right_palm_link')
    tips, names = robot.find_bodies(tuple(f'right_finger{i}_link4' for i in range(1,5)))
    by_name = dict(zip(names,tips,strict=True))
    self.tip_ids = [by_name[f'right_finger{i}_link4'] for i in range(1,5)]
    self.palm_id = palm[0]

  def start(self, ids):
    # Aborted previous requests count as failures, including contacted failures.
    self.requests.finish(ids, torch.zeros_like(ids,dtype=torch.bool))
    if self.window is None:
      self.requests.start(ids)
    else:
      self.window.start(ids)
      self.initial_angle[ids] = torch.acos(self.command.flip_start_dot[ids].clamp(-1,1))
    self.ledger.start(ids)
    self.initialized[ids] = False
    self.overflow[ids] = False
    self.geometry_valid[ids] = False
    self.patch_valid[ids] = False

  def _cache_patches(self, ids):
    # Cache identities, not world poses: each patch still follows its cubelet.
    selected = self.selected[ids].flatten(1)
    indices = selected.long().topk(12,dim=-1).indices
    self.patch_cubelets[ids] = indices // 4
    self.patch_normals[ids] = self.local_normal[ids].flatten(1,2).gather(
      1,indices[:,:,None].expand(-1,-1,3))
    self.patch_valid[ids] = selected.gather(1,indices) & self.geometry_valid[ids,None]

  def _record_contact(self, valid):
    env = self.command._env
    if self.window is None:
      self.requests.contact(valid,env.physics_dt)
    else:
      angle = torch.acos(self.command._read_flip_target_dot().clamp(-1,1))
      self.window.sample(self.initial_angle-angle,valid,env.physics_dt)

  def _lock(self, ids):
    c = self.command
    env = c._env
    poses = c.object.data.body_link_pose_w[ids][:,c._cubelet_body_indices]
    root = c.object.data.root_link_pose_w[ids]
    palm = env.scene['robot'].data.body_link_pose_w[ids,self.palm_id]
    up = torch.zeros_like(palm[:,:3])
    up[:,2] = 1
    up = quat_apply(palm[:,3:],up)
    axes = torch.eye(3,device=env.device).repeat_interleave(2,0)
    axes[1::2] *= -1
    normals = quat_apply(root[:,None,3:].expand(-1,6,-1),axes[None].expand(len(ids),-1,-1))
    face = (normals*up[:,None]).sum(-1).argmin(-1)
    normal = normals[torch.arange(len(ids),device=env.device),face]
    face_ids = torch.arange(6,device=env.device)[None].expand(len(ids),-1)
    side_ids = face_ids[(face_ids != face[:,None]) & (face_ids != (face ^ 1)[:,None])].reshape(-1,4)
    sides = normals.gather(1,side_ids[:,:,None].expand(-1,-1,3))
    local = quat_apply(quat_inv(poses[:,:,None,3:]).expand(-1,-1,4,-1),sides[:,None].expand(-1,26,-1,-1))
    axis = local.abs().argmax(-1)
    snapped = torch.zeros_like(local).scatter_(-1,axis[...,None],local.gather(-1,axis[...,None]).sign())
    self.local_normal[ids] = snapped
    self.selected[ids] = lower_side_patch_mask(poses[:,:,:3]-root[:,None,:3],sides,normal) & (local.abs().amax(-1) > .99)
    self.initialized[ids] = True
    self.geometry_valid[ids] = self.selected[ids].sum((-1,-2)) == 12
    if self.compact_contact:
      self._cache_patches(ids)

  def substep(self):
    c = self.command
    env = c._env
    # Match the actual action term's warmup rather than guessing a duration.
    warmup = max((getattr(t,'_warmup_time_s',0.) for t in env.action_manager._terms.values()),default=0.)
    enabled = env.episode_length_buf * env.step_dt >= warmup
    ids = (self.requests.active & ~self.initialized & enabled).nonzero().flatten()
    if ids.numel():
      self._lock(ids)
    poses = c.object.data.body_link_pose_w[:,c._cubelet_body_indices]
    if self.compact_contact:
      poses = poses.gather(1,self.patch_cubelets[:,:,None].expand(-1,-1,7))
    inverse_rotation = quat_inv(poses[:,None,:,3:])
    world_normal = (quat_apply(poses[:,:,3:],self.patch_normals) if self.compact_contact
      else quat_apply(poses[:,:,None,3:].expand(-1,-1,4,-1),self.local_normal))
    robot = env.scene['robot']
    valid = torch.zeros((env.num_envs,4),dtype=torch.bool,device=env.device)
    for finger in range(4):
      data = env.scene.sensors[f'bottom_tip_{finger+1}'].data
      points = data.pos
      local = quat_apply(inverse_rotation.expand(-1,points.shape[1],-1,-1),points[:,:,None,:]-poses[:,None,:,:3])
      if self.compact_contact:
        patch = on_face_patch(local,self.patch_normals[:,None]) & self.patch_valid[:,None]
        aligned = (data.normal[:,:,None,:]*world_normal[:,None]).sum(-1).abs() > .85
        aligned_patch = (patch & aligned).any(-1)
        if self.debug_diagnostics:
          on_patch = patch.any(-1)
      else:
        patch = on_face_patch(local[:,:,:,None,:],self.local_normal[:,None]) & self.selected[:,None]
        aligned = (data.normal[:,:,None,None,:]*world_normal[:,None]).sum(-1).abs() > .85
        aligned_patch = (patch & aligned).any(dim=(-1,-2))
        if self.debug_diagnostics:
          on_patch = patch.any(dim=(-1,-2))
      tip_pose = robot.data.body_link_pose_w[:,self.tip_ids[finger]]
      tip_local = quat_apply(quat_inv(tip_pose[:,None,3:]).expand(-1,points.shape[1],-1),points-tip_pose[:,None,:3])
      # Distal cap region excludes the large link4 boxes and proximal pad.
      distal = tip_local[:,:,0] <= -.049
      force = data.force.norm(dim=-1) > .01
      found = data.found > 0
      if self.debug_diagnostics:
        self.diagnostics += torch.stack((found.sum(),(found & force).sum(),
          (found & distal).sum(),(found & on_patch).sum(),
          (found & aligned_patch).sum()))
      self.overflow |= data.found.amax(-1) > 16
      valid[:,finger] = (aligned_patch & distal & force & found).any(-1)
    valid &= (enabled & self.initialized & self.geometry_valid & ~self.overflow)[:,None]
    self._record_contact(valid)
    return self.requests.supported.float()


def runtime(command):
  if not hasattr(command,'bottom_tip_runtime'):
    command.bottom_tip_runtime = BottomTipRuntime(command)
  return command.bottom_tip_runtime


def sample_bottom_tip(env):
  return runtime(env.command_manager.get_term('reorient_command')).substep()


def bottom_tip_metric(env, key):
  observer = runtime(env.command_manager.get_term('reorient_command'))
  if key == 'geometry_valid':
    return observer.geometry_valid.float()
  if key == 'contact_overflow':
    return observer.overflow.float()
  totals = observer.requests.totals
  counts = {'requests': 0, 'successes': 1, 'contacted_requests': 2, 'supported_successes': 3}
  if key in counts:
    value = totals[counts[key]].float()
  else:
    numerator, denominator = {
      'success_rate': (1, 0), 'success_after_bottom_contact': (3, 2),
      'bottom_contact_given_success': (3, 1), 'supported_success_rate': (3, 0),
    }[key]
    # All zero-denominator cases are0/0: keep the existing NaN metric semantics.
    value = totals[numerator] / totals[denominator]
  return value.expand(env.num_envs)


def bottom_tip_goal_reward(env):
  command = env.command_manager.get_term('reorient_command')
  command.evaluate_task()
  return runtime(command).requests.goal_event / env.step_dt


def bottom_tip_budget_reward(env, original, budget_name, limit, original_weight=1., **params):
  command = env.command_manager.get_term('reorient_command')
  score = original(env, **params) * original_weight * env.step_dt
  return runtime(command).ledger.pay(budget_name, score, limit) / env.step_dt


def bottom_tip_stability_reward(env, **params):
  from .rewards import rubik_cube_position_stability_reward
  command = env.command_manager.get_term('reorient_command')
  score = rubik_cube_position_stability_reward(env, **params) * env.step_dt
  observer = runtime(command)
  observer.ledger.stability(score)
  success = observer.requests.success_event
  return (score.clamp_max(0) + observer.ledger.stability_payout(success)) / env.step_dt


def bottom_tip_privileged(env):
  observer = runtime(env.command_manager.get_term('reorient_command'))
  state = observer.requests
  return torch.cat((state.supported[:,None].float(), state.frozen[:,None].float(),
    state.duration.clamp(max=.03) / .03, observer.geometry_valid[:,None].float()), dim=-1)
