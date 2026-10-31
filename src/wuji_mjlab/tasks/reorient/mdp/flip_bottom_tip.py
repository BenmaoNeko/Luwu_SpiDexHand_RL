"""Request-level bottom fingertip evidence and bounded goal accounting.

Contact inputs must already be geometrically classified at physics frequency.
Unknown contact evidence must never be passed as a measured negative.
"""
import math

import torch


def lower_side_patch_mask(relative_centers, normals, bottom_normal):
  """Initial lower cubelet row, lateral outward faces only (not its underside)."""
  lower = (relative_centers * bottom_normal[:,None]).sum(-1) > .01
  lateral = (normals * bottom_normal[:,None]).sum(-1).abs() < .5
  outward = torch.einsum('nbi,nfi->nbf',relative_centers,normals) > .01
  return lower[:,:,None] & lateral[:,None] & outward


def on_face_patch(local_points, local_normals, half_extent=.0095, tolerance=.001):
  distance = (local_points * local_normals).sum(-1)
  tangent = local_points - distance[..., None] * local_normals
  return ((distance - half_extent).abs() <= tolerance) & (tangent.abs().amax(-1) <= half_extent + tolerance)


class RequestBudget:
  def __init__(self, count, device, limit):
    self.limit = float(limit)
    self.remaining = torch.zeros(count,device=device)

  def reset(self, ids):
    self.remaining[ids] = self.limit

  def pay(self, positive_score):
    score = torch.minimum(positive_score.clamp_min(0),self.remaining)
    self.remaining -= score
    return score


class BottomTipRewardLedger:
  """Positive auxiliary budgets reset with the configured reward unit's start."""
  def __init__(self, count, device):
    self.count, self.device = count, device
    self.budgets = {}
    self.stability_sum = torch.zeros(count, device=device)
    self.stability_count = torch.zeros(count, device=device)

  def start(self, ids):
    for budget in self.budgets.values():
      budget.reset(ids)
    self.stability_sum[ids] = 0
    self.stability_count[ids] = 0

  def pay(self, name, score, limit):
    if name not in self.budgets:
      self.budgets[name] = RequestBudget(self.count, self.device, limit)
      self.budgets[name].reset(torch.arange(self.count, device=self.device))
    return self.budgets[name].pay(score)

  def stability(self, score):
    positive = score > 0
    self.stability_sum += score.clamp(0, 1)
    self.stability_count += positive

  def stability_payout(self, success):
    return success * self.stability_sum / self.stability_count.clamp_min(1)


def request_goal_score(success, supported, strength):
  if not math.isfinite(strength) or not 0 <= strength <= 1:
    raise ValueError('contact strength must be in [0, 1]')
  return success * ((20 - 16 * strength) + (4 + 16 * strength) * supported)


class BottomTipRequests:
  def __init__(self, count, device, *, success_rewards=None):
    # Optional fixed (unsupported, supported) scores; None preserves legacy stages.
    if success_rewards is not None:
      if len(success_rewards) != 2 or not all(math.isfinite(v) for v in success_rewards):
        raise ValueError('success rewards must be two finite scores')
    self.success_rewards = success_rewards
    self.active = torch.zeros(count, dtype=torch.bool, device=device)
    self.frozen = torch.zeros_like(self.active)
    self.supported = torch.zeros_like(self.active)
    self.duration = torch.zeros((count, 4), device=device)
    self.credited = torch.full((count,), -1, dtype=torch.long, device=device)
    self.totals = torch.zeros(4, dtype=torch.long, device=device)
    self.goal_event = torch.zeros(count, device=device)
    self.success_event = torch.zeros_like(self.active)
    self.strength = torch.zeros(count, device=device)

  def start(self, ids, strength=0.):
    request_goal_score(0, 0, strength)
    self.active[ids] = True
    self.frozen[ids] = False
    self.supported[ids] = False
    self.duration[ids] = 0
    self.credited[ids] = -1
    self.strength[ids] = strength
    self.goal_event[ids] = 0
    self.success_event[ids] = False

  def contact(self, valid, dt):
    if dt <= 0 or not math.isfinite(dt):
      raise ValueError('physics dt must be positive')
    enabled = self.active & ~self.frozen
    self.duration = torch.where(valid & enabled[:,None], self.duration + dt, 0.)
    qualified = self.duration >= .03 - 1e-7
    first = enabled & ~self.supported & qualified.any(-1)
    self.credited[first] = qualified[first].long().argmax(-1)
    self.supported |= enabled & qualified.any(-1)

  def freeze(self, ids):
    self.frozen[ids] = True

  def finish(self, ids, success):
    valid = self.active[ids]
    ids, success = ids[valid], success[valid].bool()
    h = self.supported[ids]
    self.totals += torch.stack((torch.tensor(ids.numel(),device=ids.device),
      success.sum(), h.sum(), (success & h).sum()))
    if self.success_rewards is None:
      a = self.strength[ids]
      score = (20-16*a)+(4+16*a)*h
    else:
      unsupported, supported = self.success_rewards
      score = torch.where(h, supported, unsupported)
    self.goal_event[ids] = success * score
    self.success_event[ids] = success
    self.active[ids] = False

  def metrics(self):
    n, s, h, sh = self.totals.tolist()
    def ratio(x, y):
      return x/y if y else float('nan')
    return dict(requests=n, successes=s, contacted_requests=h,
      supported_successes=sh, success_rate=ratio(s,n),
      success_after_bottom_contact=ratio(sh,h),
      bottom_contact_given_success=ratio(sh,s), supported_success_rate=ratio(sh,n))
