# SPDX-License-Identifier: Apache-2.0
"""Transport-independent checks at settled smart-cube skill boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .rubik_solver import make_solver_plan

SOLVED_FACELETS = "".join(face * 9 for face in "URFDLB")


@dataclass(frozen=True)
class FaceletSnapshot:
  facelets: str
  move_counter: int  # Unwrapped by transport, and tied to this state snapshot.
  received_at: float  # Host monotonic clock; request after the skill boundary.


@dataclass(frozen=True)
class FaceletCheck:
  status: str  # wait / invalid / verified / solved / replan
  reason: str
  facelets: str | None = None


def _valid_facelet_format(facelets):
  return (
    isinstance(facelets, str)
    and len(facelets) == 54
    and all(facelets.count(face) == 9 for face in "URFDLB")
    and "".join(facelets[i * 9 + 4] for i in range(6)) == "URFDLB"
  )


def check_facelets(
  snapshot: FaceletSnapshot,
  expected_facelets: str,
  *,
  now: float,
  expected_move_counter: int,
  settled: bool,
  max_age_seconds: float = 0.25,
  not_before: float | None = None,
) -> FaceletCheck:
  """Only compare normalized URFDLB states from a fresh settled snapshot.

  The BLE bridge must normalize face order, centers and MOVE signs, unwrap the
  counter and request FACELETS after the boundary. This checks logical state;
  settled motion must be supplied by the controller, not inferred from FACELETS.
  """
  if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
    raise ValueError("max_age_seconds must be finite and positive")
  if not _valid_facelet_format(expected_facelets):
    raise ValueError("Expected state must be 54 URFDLB facelets with fixed centers")
  if not _valid_facelet_format(snapshot.facelets):
    return FaceletCheck("invalid", "Malformed facelets or unmapped centers")
  age = now - snapshot.received_at
  if (
    not math.isfinite(age)
    or not 0 <= age <= max_age_seconds
    or (not_before is not None and snapshot.received_at < not_before)
  ):
    return FaceletCheck("wait", "Request a fresh post-boundary FACELETS snapshot")
  if not settled or snapshot.move_counter != expected_move_counter:
    return FaceletCheck("wait", "Motion unsettled or MOVE counter not synchronized")
  if snapshot.facelets != expected_facelets:
    return FaceletCheck(
      "replan", "Physical logical state differs from expected state", snapshot.facelets
    )
  status = "solved" if snapshot.facelets == SOLVED_FACELETS else "verified"
  return FaceletCheck(status, "Logical state verified", snapshot.facelets)


def replan_from_facelets(check: FaceletCheck) -> dict:
  """Blocking host-side recovery, outside PPO/BLE callbacks and physical stepping.

  Return a new Node plan; the controller must pause, invalidate the old plan and
  install the returned goals. Illegal cubie parity/orientation is rejected by
  Node. Exceptions propagate so callers keep motion paused on solver failure.
  """
  if check.status != "replan" or not _valid_facelet_format(check.facelets):
    raise ValueError("Replanning requires a fresh mismatch check")
  return make_solver_plan(facelets=check.facelets)
