# SPDX-License-Identifier: Apache-2.0
"""Post-flip shape assessment and a bounded alignment window (critic only)."""

import itertools
import math

import torch


class FlipPreservation:
  """Flip freely, assess at arrival, then reward new-best shape alignment.

  Shape is measured against the nearest valid cube geometry, independently of
  sticker arrangement. A completed 90-degree layer turn has zero residual;
  unintended turns are handled separately through small MOVE event penalties.
  No continuous internal-error penalty or internal-error termination is used.
  """

  def __init__(self, count, device):
    self.error = torch.zeros(count, device=device)
    self.mean_error = torch.zeros_like(self.error)
    self.entry_error = torch.zeros_like(self.error)
    self.entry_mean_error = torch.zeros_like(self.error)
    self.best_score = torch.zeros_like(self.error)
    self.boundary_penalty = torch.zeros_like(self.error)
    self.improvement = torch.zeros_like(self.error)
    self.move_penalty = torch.zeros_like(self.error)
    self.request_count = torch.zeros_like(self.error)
    self.settling = torch.zeros(count, device=device, dtype=torch.bool)
    self.active = torch.zeros_like(self.settling)
    self.arrival_seen = torch.zeros_like(self.settling)
    self.arrival_event = torch.zeros_like(self.settling)
    self.done = torch.zeros_like(self.settling)
    self.aligned_event = torch.zeros_like(self.settling)
    self.timeout_event = torch.zeros_like(self.settling)
    self.hold = torch.zeros(count, device=device, dtype=torch.long)
    self.start_step = torch.zeros_like(self.hold)
    self.arrival_step = torch.zeros_like(self.hold)
    self.last_step = torch.full_like(self.hold, -1)
    self.seen_move_step = torch.full_like(self.hold, -1)
    self._totals = dict(attempts=0, finished=0, geometric_arrivals=0,
      aligned=0, alignment_timeouts=0, task_timeouts=0, completed_requests=0,
      move_events=0, entry_max_deg=0., exit_max_deg=0., entry_mean_deg=0.,
      exit_mean_deg=0., alignment_seconds=0.)
    symmetries = []
    for permutation in itertools.permutations(range(3)):
      for signs in itertools.product((-1., 1.), repeat=3):
        matrix = torch.zeros(3, 3)
        matrix[torch.arange(3), torch.tensor(permutation)] = torch.tensor(signs)
        if torch.linalg.det(matrix) > 0:
          symmetries.append(matrix)
    self.symmetries = torch.stack(symmetries).to(device)

  def shape_error(self, rotations, drivers):
    # Trace(A.T @ B) gives the SO(3) distance to each of the24 proper cube
    # rotations, covering every cubie's surface normals, not just6 drivers.
    trace = torch.einsum('nbij,kij->nbk', rotations, self.symmetries).amax(-1)
    pieces = torch.acos(((trace - 1) / 2).clamp(-1, 1))
    drivers = torch.atan2(torch.sin(4 * drivers), torch.cos(4 * drivers)).abs() / 4
    errors = torch.cat((pieces, drivers), dim=-1)
    return errors.amax(-1), errors.mean(-1)

  def reset(self, ids, rotations=None, drivers=None):
    # Initial state may be any legal scramble. Shape does not depend on the
    # original sticker permutation, so no fixed arrangement baseline is needed.
    for buffer in (self.error, self.mean_error, self.entry_error,
      self.entry_mean_error, self.best_score, self.boundary_penalty,
      self.improvement, self.move_penalty, self.request_count, self.hold):
      buffer[ids] = 0
    for buffer in (self.settling, self.active, self.arrival_seen, self.arrival_event,
      self.done, self.aligned_event, self.timeout_event):
      buffer[ids] = False
    self.last_step[ids] = -1
    self.seen_move_step[ids] = -1

  def start_goal(self, ids, step=0):
    self.active[ids] = True
    self.settling[ids] = False
    self.arrival_seen[ids] = False
    self.hold[ids] = 0
    self.start_step[ids] = step
    self._totals['attempts'] += ids.numel()

  def update(self, step, arrived, geometry_valid, rotations, drivers, dt, *,
             contact, move_event=None, move_step=None, window_seconds=1.,
             tolerance_rad=math.radians(5), hold_steps=5):
    fresh = self.last_step != step
    if not fresh.any():
      return
    for buffer in (self.boundary_penalty, self.improvement, self.move_penalty):
      buffer[fresh] = 0
    for buffer in (self.arrival_event, self.done, self.aligned_event, self.timeout_event):
      buffer[fresh] = False
    if move_event is not None:
      new_move = fresh & self.active & move_event & (move_step > self.seen_move_step)
      self.move_penalty[new_move] = 1
      self._totals['move_events'] += int(new_move.sum())
      self.seen_move_step[new_move] = move_step[new_move]
    entering = fresh & self.active & arrived & ~self.arrival_seen
    self.arrival_event[entering] = True
    self.arrival_seen |= entering
    self.settling |= entering
    self.arrival_step[entering] = step
    self._totals['geometric_arrivals'] += int(entering.sum())
    ids = (fresh & self.settling).nonzero().flatten()
    # Crucially, no shape assessment is run on environments still flipping.
    if ids.numel():
      maximum, mean = self.shape_error(rotations[ids], drivers[ids])
      self.error[ids], self.mean_error[ids] = maximum, mean
      score = ((maximum + mean) / (2 * math.radians(20))).clamp(max=4)
      first = entering[ids]
      new_ids = ids[first]
      self.entry_error[new_ids] = maximum[first]
      self.entry_mean_error[new_ids] = mean[first]
      self.boundary_penalty[new_ids] = score[first]
      self.best_score[new_ids] = score[first]
      valid = contact[ids] & geometry_valid[ids]
      gain = (self.best_score[ids] - score).clamp_min(0)
      self.improvement[ids] = torch.where(valid & ~first, gain, 0.)
      self.best_score[ids] = torch.where(valid,
        torch.minimum(self.best_score[ids], score), self.best_score[ids])
      aligned = valid & (maximum <= tolerance_rad)
      self.hold[ids] = torch.where(aligned, self.hold[ids] + 1, 0)
      stable = self.hold[ids] >= hold_steps
      expired = step - self.arrival_step[ids] >= math.ceil(window_seconds / dt)
      done = stable | expired
      self.done[ids] = done
      self.aligned_event[ids] = stable
      self.timeout_event[ids] = expired & ~stable
      completed = ids[done]
      self.settling[completed] = False
      self._totals['aligned'] += int(stable.sum())
      self._totals['alignment_timeouts'] += int((expired & ~stable).sum())
      self._totals['entry_max_deg'] += float(torch.rad2deg(self.entry_error[completed]).sum())
      self._totals['exit_max_deg'] += float(torch.rad2deg(self.error[completed]).sum())
      self._totals['entry_mean_deg'] += float(torch.rad2deg(self.entry_mean_error[completed]).sum())
      self._totals['exit_mean_deg'] += float(torch.rad2deg(self.mean_error[completed]).sum())
      self._totals['alignment_seconds'] += float(((step - self.arrival_step[completed]) * dt).sum())
    self.last_step[fresh] = step

  def finish(self, ids, step, dt, success=False, timeout=False):
    ids = ids[self.active[ids]]
    self._totals['finished'] += ids.numel()
    if timeout:
      self._totals['task_timeouts'] += ids.numel()
    self.active[ids] = False
    self.settling[ids] = False

  def logs(self):
    totals = self._totals
    completed = max(totals['aligned'] + totals['alignment_timeouts'], 1)
    means = ('entry_max_deg', 'exit_max_deg', 'entry_mean_deg',
             'exit_mean_deg', 'alignment_seconds')
    result = {f'flip_align/{key}': value / completed if key in means else value
              for key, value in totals.items()}
    result['flip_align/aligned_rate'] = totals['aligned'] / completed
    return result


def flip_preservation_privileged(env, command_name='reorient_command'):
  state = env.command_manager.get_term(command_name).flip_preservation
  return torch.stack((state.error / math.pi, state.mean_error / math.pi,
    state.best_score / 4, state.settling.float(), state.hold.float() / 5), dim=-1)


def flip_preservation_reward(env, component, command_name='reorient_command'):
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  state = command.flip_preservation
  if component == 'boundary':
    score = state.boundary_penalty
  elif component == 'improvement':
    score = state.improvement
  elif component == 'move':
    score = state.move_penalty
  elif component == 'arrival':
    score = state.arrival_event.float()
  elif component == 'streak':
    score = command._flip_request_event.float() * state.request_count.clamp(max=10)
  else:
    raise ValueError(f'Unknown flip reward component {component}')
  return score / env.step_dt
