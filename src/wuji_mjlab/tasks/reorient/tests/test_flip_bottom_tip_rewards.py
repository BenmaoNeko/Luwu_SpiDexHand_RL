import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import (
  BottomTipRequests,
  request_goal_score,
)


@pytest.mark.parametrize('a,score', [(0.,20),(.375,14),(.75,8),(1.,4)])
def test_goal(a, score):
  assert request_goal_score(True, False, a) == score
  assert request_goal_score(True, True, a) == 24
  assert request_goal_score(False, True, a) == 0


def test_contact_conditional_metric_counts_failures_and_no_contact_separately():
  state = BottomTipRequests(3, 'cpu')
  ids = torch.arange(3)
  state.start(ids)
  contact = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,0]], dtype=torch.bool)
  for _ in range(3):
    state.contact(contact, .01)
  state.finish(ids, torch.tensor([True,False,True]))
  assert state.totals.tolist() == [3,2,2,1]
  assert state.metrics()['success_after_bottom_contact'] == .5
  assert state.metrics()['bottom_contact_given_success'] == .5
  state.finish(ids, torch.ones(3,dtype=torch.bool))
  assert state.totals.tolist() == [3,2,2,1]


def test_fixed_negative_success_reward_preserves_success_and_contact_metrics():
  state = BottomTipRequests(4,'cpu',success_rewards=(-1.,30.))
  ids = torch.arange(4)
  state.start(ids)
  valid = torch.tensor([[1,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0]],dtype=torch.bool)
  for _ in range(3):
    state.contact(valid,.01)
  state.finish(ids,torch.tensor([True,True,False,False]))
  assert state.goal_event.tolist() == [30.,-1.,0.,0.]
  assert state.success_event.tolist() == [True,True,False,False]
  assert state.totals.tolist() == [4,2,2,1]
  assert state.metrics()['success_rate'] == .5
  assert state.metrics()['success_after_bottom_contact'] == .5
  state.start(ids)
  assert not state.success_event.any()
  assert not state.goal_event.any()


def test_short_contact_and_frozen_late_contact_do_not_qualify():
  state = BottomTipRequests(1, 'cpu')
  state.start(torch.tensor([0]))
  contact = torch.ones((1,4), dtype=torch.bool)
  state.contact(contact, .02)
  state.freeze(torch.tensor([0]))
  state.contact(contact, .02)
  assert not state.supported.item()


def test_no_contact_metric_is_nan_instead_of_fake_zero():
  state = BottomTipRequests(1, 'cpu')
  assert torch.isnan(torch.tensor(state.metrics()['success_after_bottom_contact']))


def test_budget_cannot_be_reearned_after_partial_spending():
  from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import RequestBudget
  budget = RequestBudget(2,'cpu',.25)
  ids = torch.arange(2)
  budget.reset(ids)
  assert budget.pay(torch.tensor([.2,.3])).tolist() == pytest.approx([.2,.25])
  assert budget.pay(torch.tensor([.2,.3])).tolist() == pytest.approx([.05,0])
  assert budget.pay(torch.tensor([-.1,-.1])).tolist() == [0,0]
  budget.reset(torch.tensor([0]))
  assert budget.pay(torch.tensor([1.,1.])).tolist() == [.25,0]
