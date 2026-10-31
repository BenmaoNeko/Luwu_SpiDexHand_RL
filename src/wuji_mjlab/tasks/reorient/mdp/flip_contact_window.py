"""Monotone early-contact gate, reset independently per 90-degree leg."""
import math

import torch

from .flip_bottom_tip import BottomTipRequests


class EarlyContactWindow:
  def __init__(self, count, device, *, max_progress_deg=45., success_rewards=None):
    if not math.isfinite(max_progress_deg) or not 0 < max_progress_deg <= 90:
      raise ValueError('contact window must be in (0,90] degrees')
    self.max_progress_rad = math.radians(max_progress_deg)
    self.evidence = BottomTipRequests(count, device, success_rewards=success_rewards)
    self.closed = torch.zeros(count, dtype=torch.bool, device=device)

  def start(self, ids):
    self.evidence.start(ids)
    self.closed[ids] = False

  def sample(self, progress_rad, valid_contact, dt):
    # Close before crediting the crossing sample; retreat cannot reopen it.
    self.closed |= progress_rad >= self.max_progress_rad
    self.evidence.frozen |= self.closed
    self.evidence.contact(valid_contact, dt)
