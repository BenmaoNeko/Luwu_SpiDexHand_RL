# SPDX-License-Identifier: Apache-2.0
"""Discrete cube-face orientation helpers for continuous reorientation tasks.

The cube is treated as a rigid body with six labelled local face normals.  The
24 proper signed-permutation rotations are the complete orientation group of a
cube.  Grouping those rotations by the local face that maps to world ``+Z``
gives four equivalent orientations for each possible top face.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_error_magnitude, quat_from_matrix


__all__ = [
  "FACE_NORMALS",
  "canonical_orientations_by_top_face",
  "sample_other_face",
  "sample_adjacent_face",
  "sample_curriculum_face",
  "opposite_face_ids",
  "select_min_rotation_target",
  "face_id_from_quat",
]


FACE_NORMALS = torch.tensor(
  (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
  ),
  dtype=torch.float32,
)

# Face IDs are ``+X, -X, +Y, -Y, +Z, -Z``.  The four entries in each row
# are the faces that share an edge with the row's face.  Keeping this table
# explicit makes the easy curriculum deterministic and avoids accidentally
# treating the opposite face as an adjacent target.
_ADJACENT_FACE_IDS = torch.tensor(
  (
    (2, 3, 4, 5),
    (2, 3, 4, 5),
    (0, 1, 4, 5),
    (0, 1, 4, 5),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
  ),
  dtype=torch.long,
)


def _canonical_matrices() -> list[tuple[int, torch.Tensor]]:
  matrices: list[tuple[int, torch.Tensor]] = []
  for permutation in permutations(range(3)):
    for signs in product((-1.0, 1.0), repeat=3):
      matrix = torch.zeros((3, 3), dtype=torch.float32)
      for row, column in enumerate(permutation):
        matrix[row, column] = signs[row]
      if not torch.isclose(torch.linalg.det(matrix), torch.tensor(1.0)):
        continue
      top_normal = matrix @ FACE_NORMALS.T
      top_face = torch.nonzero(
        torch.all(top_normal.T == torch.tensor((0.0, 0.0, 1.0)), dim=-1),
        as_tuple=False,
      ).squeeze(-1)
      if top_face.numel() != 1:
        raise RuntimeError("A proper cube rotation must have exactly one top face.")
      matrices.append((int(top_face.item()), matrix))
  return matrices


@lru_cache(maxsize=1)
def _canonical_groups_cpu() -> tuple[torch.Tensor, ...]:
  """Build the 24 cube orientations once in CPU float32 storage."""
  groups: dict[int, list[torch.Tensor]] = {face_id: [] for face_id in range(6)}
  for face_id, matrix in _canonical_matrices():
    groups[face_id].append(quat_from_matrix(matrix.unsqueeze(0))[0])
  if any(len(group) != 4 for group in groups.values()):
    raise RuntimeError("Cube symmetry generation did not produce four rotations per face.")
  return tuple(torch.stack(groups[face_id]).contiguous() for face_id in range(6))


@lru_cache(maxsize=None)
def _canonical_groups_for_device(
  device_key: str, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
  """Cache one converted copy of the static table per device and dtype."""
  device = torch.device(device_key)
  return tuple(group.to(device=device, dtype=dtype) for group in _canonical_groups_cpu())


@lru_cache(maxsize=None)
def _canonical_candidates_for_device(
  device_key: str, dtype: torch.dtype
) -> torch.Tensor:
  return torch.stack(_canonical_groups_for_device(device_key, dtype))


def canonical_orientations_by_top_face(
  *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> dict[int, torch.Tensor]:
  """Return the 24 canonical cube rotations grouped by top face.

  Quaternions use MuJoCo/MJLab's ``(w, x, y, z)`` convention.  Face IDs are
  ``+X, -X, +Y, -Y, +Z, -Z`` in that order.
  """
  device_key = str(torch.device("cpu" if device is None else device))
  groups = _canonical_groups_for_device(device_key, dtype)
  return {face_id: groups[face_id] for face_id in range(6)}


def _random_tensor(
  shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator | None
) -> torch.Tensor:
  if generator is None:
    return torch.rand(shape, device=device)
  try:
    return torch.rand(shape, device=device, generator=generator)
  except RuntimeError:
    # A CPU generator is convenient for deterministic unit tests, while a
    # command normally samples on CUDA without supplying a generator.
    return torch.rand(shape, generator=generator).to(device=device)


def sample_other_face(
  current_face_ids: torch.Tensor, *, generator: torch.Generator | None = None
) -> torch.Tensor:
  """Uniformly sample one of the five faces different from each current face."""
  current = current_face_ids.to(dtype=torch.long)
  if torch.any((current < 0) | (current >= 6)):
    raise ValueError("current_face_ids must contain values in [0, 5].")
  raw = torch.randint(
    0,
    5,
    current.shape,
    device=current.device,
    generator=generator,
  )
  return raw + (raw >= current).to(dtype=torch.long)


def opposite_face_ids(current_face_ids: torch.Tensor) -> torch.Tensor:
  """Return the opposite face for each face ID."""
  current = current_face_ids.to(dtype=torch.long)
  if torch.any((current < 0) | (current >= 6)):
    raise ValueError("current_face_ids must contain values in [0, 5].")
  return current ^ 1


def sample_adjacent_face(
  current_face_ids: torch.Tensor, *, generator: torch.Generator | None = None
) -> torch.Tensor:
  """Uniformly sample one of the four edge-adjacent faces."""
  current = current_face_ids.to(dtype=torch.long)
  if torch.any((current < 0) | (current >= 6)):
    raise ValueError("current_face_ids must contain values in [0, 5].")
  table = _ADJACENT_FACE_IDS.to(device=current.device)
  raw = torch.randint(0, 4, current.shape, device=current.device, generator=generator)
  return table[current, raw]


def sample_curriculum_face(
  current_face_ids: torch.Tensor,
  *,
  stage: int,
  adjacent_probability: float = 0.8,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """Sample a flip target according to the three-stage curriculum.

  Stage 0 uses adjacent faces only, stage 1 mixes adjacent/opposite faces,
  and stage 2 samples uniformly from all five non-current faces.
  """
  if stage not in (0, 1, 2):
    raise ValueError("stage must be one of {0, 1, 2}.")
  if not 0.0 <= adjacent_probability <= 1.0:
    raise ValueError("adjacent_probability must be in [0, 1].")
  if stage == 0:
    return sample_adjacent_face(current_face_ids, generator=generator)
  if stage == 2:
    return sample_other_face(current_face_ids, generator=generator)
  adjacent = sample_adjacent_face(current_face_ids, generator=generator)
  opposite = opposite_face_ids(current_face_ids)
  choose_adjacent = _random_tensor(
    tuple(current_face_ids.shape),
    device=current_face_ids.device,
    generator=generator,
  ) < adjacent_probability
  return torch.where(choose_adjacent, adjacent, opposite)


def select_min_rotation_target(
  current_quat: torch.Tensor,
  target_face_ids: torch.Tensor,
  *,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """Select the closest canonical orientation for each requested top face.

  If multiple in-plane orientations have the same minimum geodesic distance,
  one of the tied candidates is sampled uniformly instead of always choosing
  the first canonical representative.
  """
  if current_quat.ndim != 2 or current_quat.shape[-1] != 4:
    raise ValueError("current_quat must have shape (N, 4).")
  face_ids = target_face_ids.to(device=current_quat.device, dtype=torch.long).reshape(-1)
  if face_ids.shape[0] != current_quat.shape[0] or torch.any((face_ids < 0) | (face_ids >= 6)):
    raise ValueError("target_face_ids must have shape (N,) and values in [0, 5].")

  all_candidates = _canonical_candidates_for_device(
    str(current_quat.device), current_quat.dtype
  )
  candidates = all_candidates[face_ids]
  errors = quat_error_magnitude(current_quat.unsqueeze(1).expand_as(candidates), candidates)
  minimum = errors.min(dim=-1, keepdim=True).values
  tied = errors <= minimum + 1e-6
  random_scores = _random_tensor(
    tuple(errors.shape), device=current_quat.device, generator=generator
  )
  random_scores = torch.where(tied, random_scores, torch.full_like(random_scores, float("inf")))
  selected = random_scores.argmin(dim=-1)
  return candidates[torch.arange(current_quat.shape[0], device=current_quat.device), selected]


def face_id_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Return the local face normal currently mapped closest to world ``+Z``."""
  if quat.ndim != 2 or quat.shape[-1] != 4:
    raise ValueError("quat must have shape (N, 4).")
  rotation = matrix_from_quat(quat)
  normals = FACE_NORMALS.to(device=quat.device, dtype=quat.dtype)
  world_normals = torch.einsum("nij,fj->nfi", rotation, normals)
  return world_normals[..., 2].argmax(dim=-1)
