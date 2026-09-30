# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

from wuji_mjlab.tasks.reorient.mdp.flip_curriculum import (
  promote_flip_curriculum_stage,
  stage_parameters,
)


def test_stage_parameters_progress_from_easy_to_final_flip_task():
  stage0 = stage_parameters(0)
  stage1 = stage_parameters(1)
  stage2 = stage_parameters(2)
  stage3 = stage_parameters(3)

  assert stage0.task_mode == "adjacent_continuous"
  assert stage1.task_mode == "adjacent_continuous"
  assert stage2.task_mode == "all_faces_continuous"
  assert stage3.task_mode == "all_faces_continuous"
  assert stage0.target_sampling_mode == "adjacent_only"
  assert stage1.target_sampling_mode == "adjacent_only"
  assert stage2.target_sampling_mode == "all_other"
  assert stage3.target_sampling_mode == "all_other"
  assert stage0.continuous is True
  assert stage1.continuous is True
  assert stage2.continuous is True
  assert stage3.continuous is True
  assert stage0.flip_terminates is False
  assert stage1.flip_terminates is False
  assert stage2.flip_terminates is False
  assert stage3.flip_terminates is False
  assert (
    stage0.yaw_goal_probability,
    stage1.yaw_goal_probability,
    stage2.yaw_goal_probability,
    stage3.yaw_goal_probability,
  ) == (0.30, 0.30, 0.30, 0.30)
  assert stage0.success_threshold_rad == math.radians(8.0)
  assert stage1.success_threshold_rad == math.radians(6.0)
  assert stage2.success_threshold_rad == math.radians(6.0)
  assert stage3.success_threshold_rad == math.radians(4.0)
  assert (stage0.hold_steps, stage1.hold_steps, stage2.hold_steps, stage3.hold_steps) == (5, 8, 8, 10)
  assert stage0.object_mass_range == (1.0, 1.0)
  assert stage1.object_mass_range == (1.0, 1.0)
  assert stage2.object_mass_range == (1.0, 1.0)
  assert stage3.object_mass_range == (0.85, 1.15)


def test_stage_promotion_uses_flip_metrics_not_rotation_metrics():
  assert promote_flip_curriculum_stage(
    0,
    observations=256,
    flip_success_rate=0.26,
    mean_flip_chain=0.0,
    mean_face_coverage=1.5,
    cage_drop_rate=0.20,
  ) == 1
  assert promote_flip_curriculum_stage(
    0,
    observations=256,
    flip_success_rate=0.24,
    mean_flip_chain=0.0,
    mean_face_coverage=1.0,
    cage_drop_rate=0.0,
  ) == 0


def test_adjacent_continuous_promotion_does_not_require_a_chain():
  assert promote_flip_curriculum_stage(
    1,
    observations=256,
    flip_success_rate=0.26,
    mean_flip_chain=0.0,
    mean_face_coverage=1.0,
    cage_drop_rate=0.20,
  ) == 2


def test_continuous_adjacent_promotion_requires_coverage_and_chain():
  assert promote_flip_curriculum_stage(
    2,
    observations=256,
    flip_success_rate=0.21,
    mean_flip_chain=1.5,
    mean_face_coverage=2.5,
    cage_drop_rate=0.20,
  ) == 3
  assert promote_flip_curriculum_stage(
    2,
    observations=256,
    flip_success_rate=0.21,
    mean_flip_chain=1.5,
    mean_face_coverage=2.4,
    cage_drop_rate=0.20,
  ) == 2


def test_final_stage_is_sticky():
  assert promote_flip_curriculum_stage(
    3,
    observations=10000,
    flip_success_rate=1.0,
    mean_flip_chain=10.0,
    mean_face_coverage=6.0,
    cage_drop_rate=0.0,
  ) == 3
