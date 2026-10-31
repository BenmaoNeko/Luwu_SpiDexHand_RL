"""Small scheduling regressions: no actor rollout needed."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg
from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
  apply_face_move,
  decode_actual_facelets,
)
from wuji_mjlab.utils.play_runner import _RubikSolverPolicy


def test_failed_layer_unlock_does_not_expose_turn_phase(monkeypatch):
  from wuji_mjlab.utils import play_runner

  policy = object.__new__(_RubikSolverPolicy)
  policy._phase = "flip"
  policy._joint_names = ()
  policy._command = SimpleNamespace(u_angle=[0.0])
  monkeypatch.setattr(play_runner, "aligned_target_state", lambda *args: {})
  monkeypatch.setattr(play_runner, "apply_face_move", lambda *args: {})

  def fail(*args):
    raise ValueError("unlock failed")

  policy._install_goal = fail
  with pytest.raises(ValueError, match="unlock failed"):
    policy._start_turn("U'", {}, "original")
  assert policy._phase == "flip"


def test_geometry_only_completion_without_contact_velocity_or_hold_inputs():
  policy = object.__new__(_RubikSolverPolicy)
  policy._command = SimpleNamespace(
    cfg=SimpleNamespace(upright_tolerance_rad=0.2, turn_tolerance_rad=np.deg2rad(5)),
    _read_flip_target_dot=lambda: [1.0],
    u_angle=torch.tensor([1.0]),
    u_target_angle=torch.tensor([0.0]),
  )
  assert policy._geometric_goal_reached(turn=False)
  assert not policy._geometric_goal_reached(turn=True)
  policy._command.u_angle[:] = 2 * np.pi + 0.04
  assert policy._geometric_goal_reached(turn=True)
  policy._command._read_flip_target_dot = lambda: [np.cos(np.deg2rad(17))]
  assert policy._geometric_goal_reached(turn=False)
  assert not policy._geometric_goal_reached(turn=True)
  policy._command._read_flip_target_dot = lambda: [0.0]
  assert not policy._geometric_goal_reached(turn=False)
  assert not policy._geometric_goal_reached(turn=True)


def test_second_quarter_waits_for_completion_and_checks_actual_state(monkeypatch):
  from wuji_mjlab.utils import play_runner

  policy = object.__new__(_RubikSolverPolicy)
  policy._status = "running"
  policy._phase = "turn"
  policy._phase_steps = 0
  policy._turn_boundary_pending = False
  policy._pending_turn = ("R'", "expected")
  policy._joint_names = ()
  policy._raw_env = SimpleNamespace(
    step_dt=0.05,
    termination_manager=SimpleNamespace(get_term=lambda _: torch.tensor([False])),
  )
  policy._command = SimpleNamespace(turn_completed=torch.tensor([False]))
  policy._geometric_goal_reached = lambda **_: bool(policy._command.turn_completed[0])
  policy._monitor_actual_state = lambda _: None
  policy._joint_state = lambda: {}
  # A completed legal turn must advance even while the cube is still moving.
  policy._stationary = lambda: False
  policy._replan_count = 0
  calls = []
  policy._start_turn = lambda *args: calls.append(("turn", args))
  policy._accept_actual_state = lambda *args, **kwargs: calls.append(("replan", args))
  monkeypatch.setattr(play_runner, "decode_actual_facelets", lambda *_: "expected")
  policy.after_step(None)
  assert not calls
  policy._command.turn_completed[:] = True
  policy.after_step(None)
  assert calls[-1] == ("turn", ("R'", {}, "expected"))
  monkeypatch.setattr(play_runner, "decode_actual_facelets", lambda *_: "unexpected")
  policy.after_step(None)
  assert calls[-1] == ("replan", ({}, "unexpected"))


def test_bottom_flip_keeps_layers_locked_through_both_waypoints():
  policy = object.__new__(_RubikSolverPolicy)
  goals = []
  policy._install_goal = lambda *args: goals.append(args)
  for face in range(6):
    policy._command = SimpleNamespace(
      u_angle=[0.0], _read_top_face_id=lambda face=face: [face ^ 1]
    )
    policy._start_flip(face, {})
    assert goals[-1][0] // 2 != face // 2
    assert goals[-1][1] == 0
    assert policy._advance_flip_waypoint({})
    assert goals[-1][:2] == (face, 0)
    assert not policy._advance_flip_waypoint({})
    policy._command._read_top_face_id = lambda face=face: [face]
    policy._start_flip(face, {})
    assert goals[-1][:2] == (face, 0)
    assert policy._pending_flip_face is None


def test_completed_flip_advances_without_extra_rest_gate(monkeypatch):
  from wuji_mjlab.utils import play_runner

  policy = object.__new__(_RubikSolverPolicy)
  policy._status = "running"
  policy._phase = "flip"
  policy._phase_steps = 0
  policy._flip_boundary_pending = False
  policy._pending_flip_face = None
  policy._phase_start_facelets = "same"
  policy._solution = ("U",)
  policy._joint_names = ()
  policy._raw_env = SimpleNamespace(
    step_dt=0.05,
    termination_manager=SimpleNamespace(get_term=lambda _: torch.tensor([False])),
  )
  policy._command = SimpleNamespace(flip_to_top_achieved=torch.tensor([False]))
  policy._geometric_goal_reached = lambda **_: bool(
    policy._command.flip_to_top_achieved[0]
  )
  policy._monitor_actual_state = lambda _: None
  policy._joint_state = lambda: {}
  policy._stationary = lambda: False
  calls = []
  policy._start_turn = lambda *args: calls.append(args)
  monkeypatch.setattr(play_runner, "decode_actual_facelets", lambda *_: "same")
  policy.after_step(None)
  assert not calls
  policy._command.flip_to_top_achieved[:] = True
  policy.after_step(None)
  assert calls == [("U", {}, "same")]


def test_half_turns_use_two_legal_quarter_turn_targets():
  model = get_rubik_cfg(full_cube=True).spec_fn().compile()
  names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  )
  state = dict.fromkeys(names, 0.0)
  policy = object.__new__(_RubikSolverPolicy)
  policy._joint_names = names
  goals = []
  policy._install_goal = lambda *args: goals.append(args)
  for face in "RLBFUD":
    policy._command = SimpleNamespace(u_angle=[0.0])
    policy._start_turn(face + "2", state, decode_actual_facelets(state, names))
    first = goals[-1][3]
    move, expected = policy._pending_turn
    assert np.isclose(goals[-1][2], np.pi / 2)
    assert expected == decode_actual_facelets(first, names)
    assert expected != decode_actual_facelets(state, names)
    policy._command.u_angle = [np.pi / 2]
    policy._start_turn(move, first, expected)
    assert np.isclose(goals[-1][2], np.pi)
    assert policy._pending_turn is None
    assert decode_actual_facelets(goals[-1][3], names) == decode_actual_facelets(
      apply_face_move(state, face + "2", names), names
    )
