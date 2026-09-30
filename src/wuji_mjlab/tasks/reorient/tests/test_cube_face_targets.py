# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_error_magnitude,
  quat_from_angle_axis,
)
from wuji_mjlab.tasks.reorient.mdp.cube_faces import (
  sample_curriculum_face,
  sample_adjacent_face,
  canonical_orientations_by_top_face,
  sample_other_face,
  select_min_rotation_target,
)


def test_adjacent_face_sampler_excludes_current_and_opposite_faces():
  current = torch.tensor([0, 1, 2, 3, 4, 5])
  sampled = sample_adjacent_face(current, generator=torch.Generator().manual_seed(4))
  opposite = torch.tensor([1, 0, 3, 2, 5, 4])

  assert torch.all(sampled != current)
  assert torch.all(sampled != opposite)


def test_curriculum_face_sampler_uses_adjacent_only_stage():
  current = torch.full((128,), 4, dtype=torch.long)
  sampled = sample_curriculum_face(
    current, stage=0, generator=torch.Generator().manual_seed(5)
  )

  assert set(sampled.tolist()).issubset({0, 1, 2, 3})


def test_curriculum_face_sampler_can_use_opposite_face_in_stage_one():
  current = torch.full((512,), 4, dtype=torch.long)
  sampled = sample_curriculum_face(
    current,
    stage=1,
    adjacent_probability=0.2,
    generator=torch.Generator().manual_seed(6),
  )

  assert (sampled == 5).any()
  assert (sampled != 5).any()


def test_cube_symmetry_has_six_faces_and_four_orientations_per_face():
  groups = canonical_orientations_by_top_face()

  assert set(groups) == set(range(6))
  assert all(group.shape == (4, 4) for group in groups.values())
  all_quats = torch.cat(list(groups.values()))
  assert torch.allclose(
    torch.linalg.vector_norm(all_quats, dim=-1), torch.ones(24), atol=1e-6
  )

  matrices = torch.cat([matrix_from_quat(group) for group in groups.values()])
  assert torch.allclose(
    torch.det(matrices), torch.ones(24), atol=1e-6
  )
  rounded = torch.round(matrices.reshape(24, -1) * 1_000_000).to(torch.int64)
  assert torch.unique(rounded, dim=0).shape[0] == 24


def test_cube_symmetry_tables_are_cached_between_target_selections():
  first = canonical_orientations_by_top_face()
  second = canonical_orientations_by_top_face()

  assert first[0].data_ptr() == second[0].data_ptr()


def test_sample_other_face_never_returns_current_face_and_stays_in_range():
  current = torch.tensor([0, 1, 2, 3, 4, 5, 0, 5])
  sampled = sample_other_face(current, generator=torch.Generator().manual_seed(7))

  assert torch.all((sampled >= 0) & (sampled < 6))
  assert torch.all(sampled != current)


def test_sample_other_face_can_reach_all_five_alternatives():
  current = torch.tensor([4])
  generator = torch.Generator().manual_seed(11)
  sampled = torch.cat(
    [sample_other_face(current, generator=generator) for _ in range(256)]
  )

  assert set(sampled.tolist()) == {0, 1, 2, 3, 5}


def test_select_min_rotation_target_uses_requested_face_and_minimizes_error():
  groups = canonical_orientations_by_top_face()
  current = groups[2][1]
  target = select_min_rotation_target(
    current.unsqueeze(0), torch.tensor([4]), generator=torch.Generator().manual_seed(3)
  )

  candidates = groups[4]
  errors = quat_error_magnitude(current.expand(4, -1), candidates)
  selected_error = quat_error_magnitude(current.unsqueeze(0), target)[0]

  assert torch.allclose(selected_error, errors.min(), atol=1e-6)
  assert any(torch.allclose(target[0], candidate, atol=1e-6) for candidate in candidates)


def test_select_min_rotation_target_samples_tied_candidates():
  # At 45 degrees of in-plane yaw, two canonical orientations of the current
  # top face are equally close.
  current = quat_from_angle_axis(
    torch.tensor([math.pi / 4]), torch.tensor([[0.0, 0.0, 1.0]])
  )
  face = torch.tensor([4])
  seen = set()
  generator = torch.Generator().manual_seed(13)
  for _ in range(128):
    target = select_min_rotation_target(current, face, generator=generator)
    seen.add(tuple(torch.round(target[0] * 1_000_000).to(torch.int64).tolist()))

  assert len(seen) > 1


def test_existing_a_to_b_calibration_anchor_maps_to_expected_faces():
  groups = canonical_orientations_by_top_face()
  identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
  plus_90_x = torch.tensor(
    [[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]]
  )

  assert any(torch.allclose(identity[0], q, atol=1e-6) for q in groups[4])
  assert any(torch.allclose(plus_90_x[0], q, atol=1e-6) for q in groups[2])
