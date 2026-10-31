# SPDX-License-Identifier: Apache-2.0
"""Small bridge from the local facelet solver to Rubik MJLab previews."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_SCRIPT = (
  _PROJECT_ROOT / "Vision/xgo_spidex_cube-master/robot_solver/scripts/plan.mjs"
)
_ROBOGYM_ROOT = _PROJECT_ROOT / "URDF/robogym-master"

_FACE = {
  "R": (0, 1, 0),
  "L": (0, -1, 1),
  "B": (1, 1, 2),
  "F": (1, -1, 3),
  "U": (2, 1, 4),
  "D": (2, -1, 5),
}
_AXIS_NAME = "xyz"
_FACE_ORDER = "URFDLB"
_FACE_LAYOUT = {
  "U": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
  "R": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
  "F": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
  "D": ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
  "L": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
  "B": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
}
_FACE_FROM_NORMAL = {layout[0]: face for face, layout in _FACE_LAYOUT.items()}
# Allow residual physical misalignment when decoding a discrete solver state.
# A corner displaced by a 15-degree layer turn moves ~0.37 normalized units.
# These tolerances only affect readout; they never snap the live joint positions.
_FACELET_MAX_ANGLE_RAD = math.radians(15.0)
_FACELET_MAX_POSITION_ERROR = 0.40
_FACE_BY_ID = {
  value[2]: (
    f"{'pos' if value[1] > 0 else 'neg'}_{_AXIS_NAME[value[0]]}",
    value[0],
    value[1],
  )
  for value in _FACE.values()
}
_COORD_TOKEN = re.compile(r"(pos|neg)_([xyz])")
_FACE_FRAME_WXYZ = (
  (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
  (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0),
  (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
  (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
  (1.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0),
)
_SOLVER_FINGERPRINT_FILES = (
  "scripts/plan.mjs",
  "src/cube-model.js",
  "src/solver-core.js",
  "src/planner.js",
)


@dataclass(frozen=True)
class SolverPlanPool:
  path: Path
  solver_fingerprint: str
  plans: tuple[dict, ...]

  @property
  def initial_facelets(self) -> tuple[str, ...]:
    return tuple(plan["initialFacelets"] for plan in self.plans)


def solver_source_fingerprint() -> str:
  """Hash the exact Node planner sources used to build a plan pool."""
  digest = hashlib.sha256()
  solver_root = _SOLVER_SCRIPT.parent.parent
  for relative_path in _SOLVER_FINGERPRINT_FILES:
    digest.update(relative_path.encode("utf-8"))
    digest.update(b"\0")
    digest.update((solver_root / relative_path).read_bytes())
    digest.update(b"\0")
  return digest.hexdigest()


def _validate_plan_record(plan: object, index: int) -> dict:
  if not isinstance(plan, dict):
    raise ValueError(f"Plan pool entry {index} must be an object")
  facelets = plan.get("initialFacelets")
  scramble = plan.get("scramble")
  seed = plan.get("seed")
  solution = plan.get("solution")
  actions = plan.get("actions")
  if (
    not isinstance(facelets, str)
    or len(facelets) != 54
    or set(facelets) - set("URFDLB")
  ):
    raise ValueError(f"Plan pool entry {index} has invalid initialFacelets")
  if not isinstance(scramble, str) or not scramble or not isinstance(seed, int):
    raise ValueError(f"Plan pool entry {index} must include its seed and scramble")
  for token in scramble.split():
    face_turn_parameters(token)
  for face in "URFDLB":
    if facelets.count(face) != 9:
      raise ValueError(f"Plan pool entry {index} has an invalid {face} face count")
  if not isinstance(solution, list) or not isinstance(actions, list):
    raise ValueError(f"Plan pool entry {index} must include solution and actions arrays")
  for token in solution:
    face_turn_parameters(token)
  if len(actions) != 2 * len(solution):
    raise ValueError(f"Plan pool entry {index} must contain reorient/turn_top pairs")
  for move_index, token in enumerate(solution):
    reorient, turn = actions[move_index * 2 : move_index * 2 + 2]
    if (
      not isinstance(reorient, dict)
      or reorient.get("type") != "reorient"
      or reorient.get("notation") != token
      or not isinstance(turn, dict)
      or turn.get("type") != "turn_top"
      or turn.get("notation") != token
      or turn.get("targetFace") != token[0]
    ):
      raise ValueError(
        f"Plan pool entry {index} action {move_index} must preserve "
        "Node reorient -> turn_top semantics"
      )
  if plan.get("stateFormat") != "54 facelets, URFDLB, row-major, centers fixed":
    raise ValueError(f"Plan pool entry {index} has an unsupported state format")
  return plan


def load_solver_plan_pool(path: str | Path) -> SolverPlanPool:
  """Read a precomputed plan pool and reject stale/invalid Node plans."""
  pool_path = Path(path).expanduser().resolve()
  if not pool_path.is_file():
    raise FileNotFoundError(f"Rubik solver plan pool does not exist: {pool_path}")
  try:
    document = json.loads(pool_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"Could not read Rubik plan pool {pool_path}: {error}") from error
  if not isinstance(document, dict) or document.get("schema_version") != 1:
    raise ValueError("Rubik solver plan pool schema_version must be 1")
  bounds = document.get("scramble_length")
  if (
    not isinstance(document.get("seed"), int)
    or not isinstance(bounds, list)
    or len(bounds) != 2
    or not all(isinstance(value, int) for value in bounds)
    or bounds[0] < 1
    or bounds[1] < bounds[0]
  ):
    raise ValueError("Rubik solver plan pool must declare its seed and scramble bounds")
  fingerprint = document.get("solver_fingerprint")
  if fingerprint != solver_source_fingerprint():
    raise ValueError(
      "Rubik solver plan pool fingerprint does not match the current Node solver; "
      "regenerate the pool before training"
    )
  plans_raw = document.get("plans")
  if not isinstance(plans_raw, list) or not plans_raw:
    raise ValueError("Rubik solver plan pool must contain at least one plan")
  plans = tuple(_validate_plan_record(plan, i) for i, plan in enumerate(plans_raw))
  facelets = [plan["initialFacelets"] for plan in plans]
  if len(facelets) != len(set(facelets)):
    raise ValueError("Rubik solver plan pool contains duplicate initial states")
  return SolverPlanPool(pool_path, fingerprint, plans)


def generate_solver_plan_pool(
  path: str | Path,
  *,
  count: int = 256,
  seed: int = 20261008,
  min_scramble_moves: int = 2,
  max_scramble_moves: int = 6,
) -> dict:
  """Generate and validate a deterministic pool in one Node process."""
  if count < 1:
    raise ValueError("Plan pool count must be positive")
  if min_scramble_moves < 1 or max_scramble_moves < min_scramble_moves:
    raise ValueError("Scramble move bounds must be positive and ordered")
  pool_path = Path(path).expanduser().resolve()
  pool_path.parent.mkdir(parents=True, exist_ok=True)
  command = [
    "node",
    str(_SOLVER_SCRIPT),
    "--pool-count",
    str(count),
    "--seed",
    str(seed),
    "--scramble-min",
    str(min_scramble_moves),
    "--scramble-max",
    str(max_scramble_moves),
    "--output",
    str(pool_path),
  ]
  subprocess.run(
    command,
    cwd=_SOLVER_SCRIPT.parent.parent,
    check=True,
    capture_output=True,
    text=True,
    timeout=600,
  )
  load_solver_plan_pool(pool_path)
  return json.loads(pool_path.read_text(encoding="utf-8"))


def build_solver_joint_state(
  plan: dict,
  default_joint_positions: Mapping[str, float],
  joint_names: Sequence[str],
) -> np.ndarray:
  """Build the physical reset hinges by reversing the Node solution."""
  state = {name: float(default_joint_positions[name]) for name in joint_names}
  for token in invert_moves(tuple(plan["solution"])):
    state = apply_face_move(state, token, joint_names)
  decoded = decode_actual_facelets(state, joint_names)
  if decoded != plan["initialFacelets"]:
    raise ValueError(
      "Node plan solution does not reconstruct initialFacelets from the solved state"
    )
  return np.asarray([state[name] for name in joint_names], dtype=np.float64)


def face_turn_parameters(token: str) -> tuple[int, int, int]:
  """Return (axis, layer sign, right-handed degrees about +axis)."""
  if not isinstance(token, str) or not re.fullmatch(r"[URFDLB](?:2|')?", token):
    raise ValueError(f"Unsupported Rubik move: {token!r}")
  axis, side, _face_id = _FACE[token[0]]
  outward_sign = side
  suffix = token[1:]
  degrees = 180 if suffix == "2" else 90 if suffix == "'" else -90
  return axis, side, degrees * outward_sign


def face_id_for_move(token: str) -> int:
  """Return the local face id used by ``cube_faces.FACE_NORMALS``."""
  face_turn_parameters(token)
  return _FACE[token[0]][2]


def face_id_for_driver_joint(name: str) -> int:
  """Return the physical face represented by one of the six layer drivers."""
  for face_id, (face_name, _axis, _side) in _FACE_BY_ID.items():
    if name == f"cubelet:driver:{face_name}":
      return face_id
  raise ValueError(f"Not a Rubik face driver joint: {name!r}")


def solver_face_frame_quaternion(
  face_id: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
  """Quaternion mapping actor-local +Z to the selected physical cube face."""
  table = torch.tensor(_FACE_FRAME_WXYZ, dtype=dtype, device=face_id.device)
  return table[face_id.long()]


def invert_moves(moves: Sequence[str]) -> tuple[str, ...]:
  """Invert a validated Singmaster outer-face move sequence."""
  result = []
  for token in reversed(moves):
    face_turn_parameters(token)
    suffix = token[1:]
    result.append(
      token if suffix == "2" else token[0] if suffix == "'" else token + "'"
    )
  return tuple(result)


def make_solver_plan(
  *,
  scramble: str | None = None,
  random_state: bool = False,
  facelets: str | None = None,
) -> dict:
  """Call the existing Node solver once and validate its JSON response."""
  if sum((scramble is not None, random_state, facelets is not None)) != 1:
    raise ValueError("Choose exactly one of scramble, random_state, or facelets")
  if not _SOLVER_SCRIPT.is_file():
    raise FileNotFoundError(f"Local solver script not found: {_SOLVER_SCRIPT}")
  requested_facelets = facelets
  command = ["node", str(_SOLVER_SCRIPT)]
  if random_state:
    command.append("--random")
  elif facelets is not None:
    command.extend(("--facelets", facelets))
  else:
    command.extend(("--scramble", scramble or ""))
  completed = subprocess.run(
    command,
    cwd=_SOLVER_SCRIPT.parent.parent,
    check=True,
    capture_output=True,
    text=True,
    timeout=120,
  )
  plan = json.loads(completed.stdout)
  facelets = plan.get("initialFacelets")
  solution = plan.get("solution")
  if (
    not isinstance(facelets, str)
    or len(facelets) != 54
    or set(facelets) - set("URFDLB")
    or not isinstance(solution, list)
  ):
    raise ValueError("Solver returned an invalid facelet plan")
  if requested_facelets is not None and facelets != requested_facelets:
    raise ValueError("Solver returned a plan for a different actual cube state")
  for token in solution:
    face_turn_parameters(token)
  return plan


def _piece_home_coords(piece_name: str) -> np.ndarray:
  tokens = _COORD_TOKEN.findall(piece_name)
  if not tokens or "_".join(f"{sign}_{axis}" for sign, axis in tokens) != piece_name:
    raise ValueError(f"Invalid Rubik cubelet name: {piece_name}")
  coords = np.zeros(3, dtype=np.float64)
  for sign, axis_name in tokens:
    coords[_AXIS_NAME.index(axis_name)] = 1.0 if sign == "pos" else -1.0
  return coords


def _axis_rotation(axis: int, angle_rad: float) -> np.ndarray:
  cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
  if axis == 0:
    return np.asarray(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))
  if axis == 1:
    return np.asarray(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))
  return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))


def face_layer_joint_names(
  face_id: int,
  joint_positions: Mapping[str, float],
  joint_names: Sequence[str],
) -> tuple[str, ...]:
  """Return the face driver and piece hinges that form a layer in this state."""
  if face_id not in _FACE_BY_ID:
    raise ValueError(f"Unsupported Rubik face id: {face_id!r}")
  face_name, axis, side = _FACE_BY_ID[face_id]
  axis_name = _AXIS_NAME[axis]
  driver_name = f"cubelet:driver:{face_name}"
  if driver_name not in joint_positions or driver_name not in joint_names:
    raise ValueError(f"Cube is missing face driver joint {driver_name!r}")

  euler2mat, _ = _rotation_conversions()
  pieces = sorted(
    name.removeprefix("cubelet:rotx:")
    for name in joint_names
    if name.startswith("cubelet:rotx:")
  )
  layer_joints = [driver_name]
  for piece in pieces:
    home = _piece_home_coords(piece)
    axes = [f"cubelet:rot{letter}:{piece}" for letter in _AXIS_NAME]
    if not all(name in joint_positions for name in axes):
      raise ValueError(f"Cubelet {piece} is missing a rotation joint")
    rotation = euler2mat([joint_positions[name] for name in axes])
    current_position = rotation @ home
    if current_position[axis] * side >= 0.5:
      layer_joint = f"cubelet:rot{axis_name}:{piece}"
      if layer_joint not in joint_names:
        raise ValueError(f"Cube is missing layer hinge joint {layer_joint!r}")
      layer_joints.append(layer_joint)

  if len(layer_joints) != 9:
    raise ValueError(
      f"Rubik face {face_name} should contain one driver and eight pieces; "
      f"found {len(layer_joints)} joints"
    )
  return tuple(layer_joints)


def _rotation_conversions():
  if str(_ROBOGYM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOGYM_ROOT))
  from robogym.utils.rotation import euler2mat, mat2euler

  return euler2mat, mat2euler


def decode_actual_facelets(
  joint_positions: Mapping[str, float], joint_names: Sequence[str]
) -> str | None:
  """Decode a stable, aligned cube state without changing its joint snapshot."""
  name_set = set(joint_names)
  piece_names = sorted(
    name.removeprefix("cubelet:rotx:")
    for name in name_set
    if name.startswith("cubelet:rotx:")
  )
  if len(piece_names) != 20:
    return None
  for axis in _AXIS_NAME:
    if (
      f"cubelet:driver:pos_{axis}" not in joint_positions
      or f"cubelet:driver:neg_{axis}" not in joint_positions
    ):
      return None

  euler2mat, _ = _rotation_conversions()
  facelets = ["?"] * 54
  for face_index, face in enumerate(_FACE_ORDER):
    facelets[face_index * 9 + 4] = face

  axis_vectors = np.eye(3, dtype=np.float64)
  angular_threshold = math.cos(_FACELET_MAX_ANGLE_RAD)
  for piece_name in piece_names:
    home = _piece_home_coords(piece_name)
    rotation_names = [f"cubelet:rot{axis}:{piece_name}" for axis in _AXIS_NAME]
    if not all(name in joint_positions for name in rotation_names):
      return None
    rotation = euler2mat([joint_positions[name] for name in rotation_names])
    current_position = rotation @ home
    rounded_position = np.rint(current_position).astype(np.int8)
    if (
      np.linalg.norm(current_position - rounded_position) > _FACELET_MAX_POSITION_ERROR
      or np.any(np.abs(rounded_position) > 1)
      or np.count_nonzero(rounded_position) != np.count_nonzero(home)
    ):
      return None

    for axis in np.flatnonzero(home):
      sticker_normal = rotation @ (axis_vectors[:, axis] * np.sign(home[axis]))
      normal_axis = int(np.argmax(np.abs(sticker_normal)))
      face_normal = axis_vectors[:, normal_axis] * np.sign(sticker_normal[normal_axis])
      if float(np.dot(sticker_normal, face_normal)) < angular_threshold:
        return None
      face = _FACE_FROM_NORMAL.get(tuple(int(value) for value in face_normal))
      if face is None:
        return None
      _normal, right, up = _FACE_LAYOUT[face]
      column = int(round(float(np.dot(rounded_position, right)) + 1.0))
      row = int(round(1.0 - float(np.dot(rounded_position, up))))
      if not (0 <= row < 3 and 0 <= column < 3):
        return None
      slot = _FACE_ORDER.index(face) * 9 + row * 3 + column
      if facelets[slot] != "?":
        return None
      home_normal = axis_vectors[:, axis] * np.sign(home[axis])
      home_face_normal = tuple(int(value) for value in home_normal)
      home_face = _FACE_FROM_NORMAL.get(home_face_normal)
      if home_face is None:
        return None
      facelets[slot] = home_face

  result = "".join(facelets)
  if "?" in result or any(result.count(face) != 9 for face in _FACE_ORDER):
    return None
  return result


def aligned_target_state(
  joint_positions: Mapping[str, float], joint_names: Sequence[str]
) -> dict[str, float]:
  """Build an aligned visual goal from a decodable snapshot; never write physics."""
  facelets = decode_actual_facelets(joint_positions, joint_names)
  if facelets is None:
    raise ValueError("Cannot construct an aligned target from an off-grid cube")
  euler2mat, mat2euler = _rotation_conversions()
  result = dict(joint_positions)
  for name in joint_names:
    if name.startswith("cubelet:driver:"):
      result[name] = round(result[name] / (math.pi / 2)) * (math.pi / 2)
    elif name.startswith("cubelet:rotx:"):
      piece = name.removeprefix("cubelet:rotx:")
      names = [f"cubelet:rot{axis}:{piece}" for axis in _AXIS_NAME]
      rotation = np.rint(euler2mat([joint_positions[key] for key in names]))
      if not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(
        np.linalg.det(rotation), 1.0
      ):
        raise ValueError("Cubelet does not identify a unique aligned orientation")
      for key, angle in zip(names, mat2euler(rotation), strict=True):
        result[key] = float(angle)
  if decode_actual_facelets(result, joint_names) != facelets:
    raise ValueError("Target alignment changed the observed sticker arrangement")
  return result


def apply_face_move(
  joint_positions: Mapping[str, float], token: str, joint_names: Sequence[str]
) -> dict[str, float]:
  """Apply one legal layer turn to a copied, reset-time cube joint state.

  This is only for constructing the reset state or the visual target. It must
  never be called on the live object during policy execution.
  """
  _axis, _side, signed_degrees = face_turn_parameters(token)
  return apply_face_turn_angle(
    joint_positions,
    face_id_for_move(token),
    math.radians(signed_degrees),
    joint_names,
  )


def apply_face_turn_angle(
  joint_positions: Mapping[str, float],
  face_id: int,
  angle_rad: float,
  joint_names: Sequence[str],
) -> dict[str, float]:
  """Construct a target state for one face rotation, without touching the sim."""
  if face_id not in _FACE_BY_ID:
    raise ValueError(f"Unsupported Rubik face id: {face_id!r}")
  _face_name, axis, side = _FACE_BY_ID[face_id]
  euler2mat, mat2euler = _rotation_conversions()
  result = dict(joint_positions)
  name_set = set(joint_names)
  center_names = {
    "x": ("cubelet:driver:neg_x", "cubelet:driver:pos_x"),
    "y": ("cubelet:driver:neg_y", "cubelet:driver:pos_y"),
    "z": ("cubelet:driver:neg_z", "cubelet:driver:pos_z"),
  }
  center_name = center_names[_AXIS_NAME[axis]][0 if side < 0 else 1]
  if center_name in result and center_name in name_set:
    result[center_name] = float(result[center_name]) + angle_rad

  pieces = sorted(
    name.removeprefix("cubelet:rotx:")
    for name in name_set
    if name.startswith("cubelet:rotx:")
  )
  turn = _axis_rotation(axis, angle_rad)
  for piece in pieces:
    home = _piece_home_coords(piece)
    rot_names = [f"cubelet:rot{axis_name}:{piece}" for axis_name in _AXIS_NAME]
    if not all(name in result for name in rot_names):
      raise ValueError(f"Cubelet {piece} is missing a rotation joint")
    angles = np.asarray([result[name] for name in rot_names], dtype=np.float64)
    current_rotation = euler2mat(angles)
    current_position = current_rotation @ home
    if current_position[axis] * side < 0.5:
      continue
    updated_angles = mat2euler(turn @ current_rotation)
    for name, value in zip(rot_names, updated_angles, strict=True):
      result[name] = float(value)
  return result


def joint_state_max_error(
  expected: Mapping[str, float], actual: Mapping[str, float], joint_names: Sequence[str]
) -> float:
  """Maximum SO(3)/hinge error in radians, without modifying either state."""
  euler2mat, _ = _rotation_conversions()
  max_error = 0.0
  for name in joint_names:
    if name.startswith("cubelet:driver:"):
      difference = float(actual[name]) - float(expected[name])
      error = abs(math.atan2(math.sin(difference), math.cos(difference)))
    elif name.startswith("cubelet:rotx:"):
      piece = name.removeprefix("cubelet:rotx:")
      axis_names = [f"cubelet:rot{axis}:{piece}" for axis in _AXIS_NAME]
      expected_rotation = euler2mat([expected[joint] for joint in axis_names])
      actual_rotation = euler2mat([actual[joint] for joint in axis_names])
      relative = actual_rotation @ expected_rotation.T
      cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
      error = math.acos(cosine)
    else:
      continue
    max_error = max(max_error, error)
  return max_error
