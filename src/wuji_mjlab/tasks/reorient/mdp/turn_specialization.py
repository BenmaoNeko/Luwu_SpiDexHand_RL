"""Turn-only reset sampling, residual shaping and post-MOVE outcome counters."""
from __future__ import annotations

import math

import torch


def residual_score(residual: torch.Tensor) -> torch.Tensor:
  degrees = torch.rad2deg(residual.abs())
  score = torch.full_like(degrees, -4.)
  for lo, hi, a, b in ((15., 25., -1., -4.), (10., 15., -.5, -1.),
                        (5., 10., 2., -.5), (0., 5., 4., 2.)):
    value = a + (degrees - lo) * ((b - a) / (hi - lo))
    score = torch.where((degrees >= lo) & (degrees <= hi), value, score)
  return score


def sample_turn_starts(count: int, device, partial_probability: float):
  if not 0 <= partial_probability <= 1:
    raise ValueError('partial_probability must be in [0, 1]')
  partial = torch.rand(count, device=device) < partial_probability
  sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
  angle = torch.deg2rad(torch.empty(count, device=device).uniform_(45., 80.)) * sign
  angle = torch.where(partial, angle, torch.zeros_like(angle))
  return angle, sign * (math.pi / 2), partial


class TurnMoveStats:
  """Cumulative closed-attempt statistics; unfinished attempts are excluded."""
  def __init__(self, count, device):
    self.active = torch.zeros(count, device=device, dtype=torch.bool)
    self.started = torch.zeros(count, device=device, dtype=torch.long)
    self.last_step = torch.full_like(self.started, -1)
    self.totals = torch.zeros(8, device=device, dtype=torch.float64)

  def observe(self, step, pending, success, residual, dt):
    fresh = self.last_step != step
    new = pending & ~self.active & fresh
    self.active |= new
    self.started[new] = step
    ids = (self.active & success & fresh).nonzero().flatten()
    self.finish(ids, residual, dt, step, success=True)
    self.last_step[:] = step

  def finish(self, ids, residual, dt, step, *, success=False, timeout=False):
    ids = ids[self.active[ids]]
    errors = residual[ids].abs()
    self.totals[0] += ids.numel()
    if success:
      self.totals[1] += ids.numel()
      self.totals[2] += ((step - self.started[ids]) * dt).sum()
    if timeout:
      self.totals[3] += ids.numel()
      self.totals[4] += torch.rad2deg(errors).sum()
    self.totals[5] += (errors <= math.radians(5) + 1e-7).sum()
    self.totals[6] += (errors <= math.radians(10) + 1e-7).sum()
    if not success and not timeout:
      self.totals[7] += ids.numel()
    self.active[ids] = False

  def logs(self):
    ended, won, seconds, timed_out, error, le5, le10, other = self.totals.tolist()
    return {
      'turn_post_move/finished': ended,
      'turn_post_move/succeeded': won,
      'turn_post_move/timeouts': timed_out,
      'turn_post_move/other_failures': other,
      'turn_post_move/success_rate': won / max(ended, 1),
      'turn_post_move/success_seconds': seconds / max(won, 1),
      'turn_post_move/timeout_residual_deg': error / max(timed_out, 1),
      'turn_post_move/final_le_5deg': le5 / max(ended, 1),
      'turn_post_move/final_le_10deg': le10 / max(ended, 1),
    }
