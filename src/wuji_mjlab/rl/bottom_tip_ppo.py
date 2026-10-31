"""Conservative, single-GPU PPO for the contact-reward transfer stage."""
import copy

import torch
from rsl_rl.algorithms import PPO


def retention_gradient_scale(ppo_grad, retained_grad):
  # Float32 sum-of-squares can overflow even when every gradient is finite.
  # Accumulate in float64 so the cap cannot become inf / inf -> NaN.
  def norm(grads):
    return torch.stack([g.double().square().sum() for g in grads if g is not None]).sum().sqrt()
  ppo_norm, retained_norm = norm(ppo_grad), norm(retained_grad)
  if not torch.isfinite(ppo_norm) or not torch.isfinite(retained_norm):
    raise FloatingPointError('Non-finite BottomTip gradient before optimizer step')
  return (.2 * ppo_norm / retained_norm.clamp_min(1e-12)).clamp(max=1.)


class BottomTipPPO(PPO):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.is_multi_gpu or self.rnd or self.symmetry or self.actor.is_recurrent:
      raise ValueError('BottomTipPPO supports feed-forward single-GPU PPO only')
    self.optimizer = torch.optim.Adam([
      {'params': self.actor.parameters(), 'lr': 5e-6},
      {'params': self.critic.parameters(), 'lr': 1e-4},
    ])
    self.updates = 0
    self.teacher = None

  def prepare_transfer(self, teacher_checkpoint=None):
    # Called AFTER checkpoint loading: preserve the exact source normalization.
    self.actor.obs_normalizer.until = 0
    if self.teacher is None or teacher_checkpoint is not None:
      self.teacher = copy.deepcopy(self.actor).eval().requires_grad_(False)
      if teacher_checkpoint is not None:
        saved = torch.load(teacher_checkpoint, map_location=self.device, weights_only=False)
        self.teacher.load_state_dict(saved['actor_state_dict'], strict=True)

  def save(self):
    saved = super().save()
    saved['bottom_tip_state'] = {
      'updates': self.updates,
      'teacher_state_dict': None if self.teacher is None else self.teacher.state_dict(),
    }
    return saved

  def load(self, loaded_dict, load_cfg=None, strict=True):
    restore_iteration = super().load(loaded_dict, load_cfg, strict)
    if restore_iteration:
      state = loaded_dict.get('bottom_tip_state', {})
      self.updates = state.get('updates', int(loaded_dict.get('iter', -1)) + 1)
      teacher = state.get('teacher_state_dict')
      if teacher is not None:
        self.prepare_transfer()
        self.teacher.load_state_dict(teacher, strict=True)
    return restore_iteration

  def update(self):
    if self.teacher is None:
      self.prepare_transfer()
    actor_enabled = self.updates >= 10
    actor_stopped = not actor_enabled
    totals = dict(value=0., surrogate=0., entropy=0., kl=0., retention=0.)
    batches = 0
    for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      entropy = self.actor.output_entropy.mean()
      kl = self.actor.get_kl_divergence(batch.old_distribution_params,
        self.actor.output_distribution_params).mean()
      actor_stopped |= float(kl.detach()) > .01
      ratio = (log_prob - batch.old_actions_log_prob.squeeze(-1)).exp()
      advantage = batch.advantages.squeeze(-1)
      surrogate = torch.maximum(-advantage * ratio,
        -advantage * ratio.clamp(1-self.clip_param, 1+self.clip_param)).mean()
      values = self.critic(batch.observations)
      value_loss = (values - batch.returns).square().mean()
      if not torch.isfinite(torch.stack((surrogate, value_loss, entropy, kl))).all():
        raise FloatingPointError(f'Non-finite BottomTip loss at update {self.updates}')
      self.optimizer.zero_grad(set_to_none=True)
      retention = torch.zeros((), device=self.device)
      if not actor_stopped:
        ppo_loss = surrogate - self.entropy_coef * entropy
        ppo_loss.backward()
        parameters = list(self.actor.parameters())
        ppo_grad = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        with torch.no_grad():
          teacher_action = self.teacher(batch.observations)
        retention = .1 * (self.actor(batch.observations) - teacher_action).square().mean()
        retained_grad = torch.autograd.grad(retention, parameters, allow_unused=True)
        scale = retention_gradient_scale(ppo_grad, retained_grad)
        for parameter, pg, rg in zip(parameters, ppo_grad, retained_grad, strict=True):
          parameter.grad = pg
          if rg is not None:
            parameter.grad = scale * rg if pg is None else pg + scale * rg
      (self.value_loss_coef * value_loss).backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
      self.optimizer.step()
      for key, value in zip(totals, (value_loss, surrogate, entropy, kl, retention), strict=True):
        totals[key] += float(value.detach())
      batches += 1
    self.storage.clear()
    self.updates += 1
    return {**{k:v/max(batches,1) for k,v in totals.items()},
      'actor_enabled': float(actor_enabled), 'actor_kl_stopped': float(actor_stopped and actor_enabled)}
