import subprocess

import pytest
from wuji_mjlab.tasks.reorient.mdp import rubik_solver

SOLVED = "".join(f * 9 for f in "URFDLB")


def api():
  from wuji_mjlab.tasks.reorient.mdp import smart_cube_facelets

  return smart_cube_facelets


def test_facelets_requires_fresh_settled_matching_counter():
  m = api()
  snapshot = m.FaceletSnapshot(SOLVED, move_counter=8, received_at=10)
  assert (
    m.check_facelets(
      snapshot, SOLVED, now=10.1, expected_move_counter=8, settled=True
    ).status
    == "solved"
  )
  assert (
    m.check_facelets(
      snapshot, SOLVED, now=11, expected_move_counter=8, settled=True
    ).status
    == "wait"
  )
  assert (
    m.check_facelets(
      snapshot, SOLVED, now=10.1, expected_move_counter=9, settled=True
    ).status
    == "wait"
  )
  assert (
    m.check_facelets(
      snapshot, SOLVED, now=10.1, expected_move_counter=8, settled=False
    ).status
    == "wait"
  )
  assert (
    m.check_facelets(
      snapshot, SOLVED, now=9, expected_move_counter=8, settled=True
    ).status
    == "wait"
  )


def test_mismatch_replans_from_received_state_with_real_node():
  m = api()
  plan = rubik_solver.make_solver_plan(scramble="R U")
  snapshot = m.FaceletSnapshot(plan["initialFacelets"], 2, 10)
  result = m.check_facelets(
    snapshot, SOLVED, now=10.1, expected_move_counter=2, settled=True
  )
  assert result.status == "replan"
  new_plan = m.replan_from_facelets(result)
  assert new_plan["initialFacelets"] == snapshot.facelets
  assert new_plan["solution"]


def test_invalid_state_never_replans_and_solved_wrong_expected_requests_replan():
  m = api()
  bad = m.FaceletSnapshot("U" * 54, 0, 10)
  result = m.check_facelets(bad, SOLVED, now=10, expected_move_counter=0, settled=True)
  assert result.status == "invalid"
  with pytest.raises(ValueError):
    m.replan_from_facelets(result)
  scrambled = rubik_solver.make_solver_plan(scramble="R")["initialFacelets"]
  result = m.check_facelets(
    m.FaceletSnapshot(SOLVED, 1, 10),
    scrambled,
    now=10,
    expected_move_counter=1,
    settled=True,
  )
  assert result.status == "replan"


def test_pre_boundary_snapshot_is_not_accepted():
  m = api()
  snapshot = m.FaceletSnapshot(SOLVED, 1, 10)
  assert (
    m.check_facelets(
      snapshot,
      SOLVED,
      now=10.1,
      expected_move_counter=1,
      settled=True,
      not_before=10.05,
    ).status
    == "wait"
  )


def test_node_rejects_count_valid_but_impossible_single_flipped_edge():
  m = api()
  # U right-middle / R top-middle are the two stickers of the UR edge.
  invalid = list(SOLVED)
  invalid[5], invalid[10] = invalid[10], invalid[5]
  snapshot = m.FaceletSnapshot("".join(invalid), 1, 10)
  result = m.check_facelets(
    snapshot, SOLVED, now=10, expected_move_counter=1, settled=True
  )
  assert result.status == "replan"
  with pytest.raises(subprocess.CalledProcessError):
    m.replan_from_facelets(result)
