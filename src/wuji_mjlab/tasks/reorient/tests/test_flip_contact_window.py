import math

import torch


def test_forty_degree_window_closes_before_crossing_sample_and_never_reopens():
  from wuji_mjlab.tasks.reorient.mdp.flip_contact_window import EarlyContactWindow
  window = EarlyContactWindow(2, 'cpu', max_progress_deg=40.)
  window.start(torch.arange(2))
  valid = torch.ones(2,4,dtype=torch.bool)
  for _ in range(2):
    window.sample(torch.deg2rad(torch.tensor([39.,39.])),valid,.01)
  window.sample(torch.deg2rad(torch.tensor([40.,39.])),valid,.01)
  assert window.closed.tolist() == [True,False]
  assert window.evidence.supported.tolist() == [False,True]
  for _ in range(3):
    window.sample(torch.deg2rad(torch.tensor([35.,35.])),valid,.01)
  assert window.evidence.supported.tolist() == [False,True]


def test_window_closes_at_first_crossing_and_does_not_reopen():
  from wuji_mjlab.tasks.reorient.mdp.flip_contact_window import EarlyContactWindow
  window = EarlyContactWindow(1, 'cpu')
  ids = torch.tensor([0])
  window.start(ids)
  valid = torch.ones(1,4,dtype=torch.bool)
  window.sample(torch.tensor([math.radians(40)]), valid, .01)
  window.sample(torch.tensor([math.radians(46)]), valid, .01)
  for _ in range(5):
    window.sample(torch.tensor([math.radians(20)]), valid, .01)
  assert window.closed.item()
  assert not window.evidence.supported.item()


def test_early_contact_is_retained_and_next_ninety_degree_leg_starts_fresh():
  from wuji_mjlab.tasks.reorient.mdp.flip_contact_window import EarlyContactWindow
  window = EarlyContactWindow(1, 'cpu')
  ids = torch.tensor([0])
  window.start(ids)
  valid = torch.ones(1,4,dtype=torch.bool)
  for angle in (0.,10.,20.):
    window.sample(torch.tensor([math.radians(angle)]), valid, .01)
  assert window.evidence.supported.item()
  window.sample(torch.tensor([math.radians(60.)]), torch.zeros_like(valid), .01)
  assert window.evidence.supported.item()
  window.start(ids)
  assert not window.closed.item()
  assert not window.evidence.supported.item()
  for _ in range(3):
    window.sample(torch.tensor([math.radians(10.)]), valid, .01)
  assert window.evidence.supported.item()


def test_late_contact_qualifies_for_full_window_but_not_early_window():
  from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import BottomTipRequests
  from wuji_mjlab.tasks.reorient.mdp.flip_contact_window import EarlyContactWindow
  full = BottomTipRequests(1, 'cpu')
  early = EarlyContactWindow(1, 'cpu')
  ids = torch.tensor([0])
  full.start(ids)
  early.start(ids)
  valid = torch.ones(1,4,dtype=torch.bool)
  for _ in range(3):
    full.contact(valid,.01)
    early.sample(torch.tensor([math.radians(50)]),valid,.01)
  assert full.supported.item()
  assert not early.evidence.supported.item()
