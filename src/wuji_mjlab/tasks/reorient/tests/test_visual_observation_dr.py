# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_from_angle_axis
from wuji_mjlab.tasks.reorient.mdp.vision_observations import (
  create_visual_pose_state,
  reset_visual_pose_state,
  step_visual_pose_model,
  vision_stage_parameters,
)


def _yaw_quat(angle: float, *, device: torch.device) -> torch.Tensor:
  return quat_from_angle_axis(
    torch.tensor([angle], device=device),
    torch.tensor([[0.0, 0.0, 1.0]], device=device),
  )


def test_vision_stage_parameters_match_approved_ranges():
  stage0 = vision_stage_parameters(0)
  stage1 = vision_stage_parameters(1)
  stage2 = vision_stage_parameters(2)
  stage3 = vision_stage_parameters(3)

  assert math.isclose(stage0.noise_range_rad, math.radians(1.0))
  assert math.isclose(stage1.noise_range_rad, math.radians(2.0))
  assert math.isclose(stage2.noise_range_rad, math.radians(3.0))
  assert math.isclose(stage0.bias_limit_rad, math.radians(0.5))
  assert math.isclose(stage1.bias_limit_rad, math.radians(1.5))
  assert math.isclose(stage2.bias_limit_rad, math.radians(3.0))
  assert math.isclose(stage3.noise_range_rad, math.radians(5.0))
  assert math.isclose(stage3.bias_limit_rad, math.radians(5.0))
  assert stage0.delay_steps == (0, 1)
  assert stage1.delay_steps == (0, 2)
  assert stage2.delay_steps == (1, 4)
  assert stage0.dropout_probability == 0.0
  assert stage1.dropout_probability == 0.01
  assert stage2.dropout_probability == 0.03
  assert stage0.max_hold_steps == 0
  assert stage1.max_hold_steps == 1
  assert stage2.max_hold_steps == 2
  assert stage3.delay_steps == (1, 6)
  assert stage3.dropout_probability == 0.05
  assert stage3.max_hold_steps == 4


def test_visual_pose_model_returns_previous_pose_for_one_step_delay():
  device = torch.device("cpu")
  state = create_visual_pose_state(num_envs=1, device=device, max_delay_steps=4)
  profile = vision_stage_parameters(2)
  state.initialized[:] = True
  state.pose_history[:] = _yaw_quat(0.0, device=device).unsqueeze(1)
  state.last_output[:] = _yaw_quat(0.0, device=device)
  state.delay_steps[:] = 1
  zero_noise = torch.zeros((1, 3), device=device)

  first = step_visual_pose_model(
    _yaw_quat(0.0, device=device),
    state,
    profile,
    noise_rotvec=zero_noise,
    update_bias=False,
  )
  second = step_visual_pose_model(
    _yaw_quat(math.pi / 2.0, device=device),
    state,
    profile,
    noise_rotvec=zero_noise,
    update_bias=False,
  )

  assert torch.allclose(first, _yaw_quat(0.0, device=device), atol=1e-6)
  assert torch.allclose(second, _yaw_quat(0.0, device=device), atol=1e-6)


def test_visual_pose_bias_is_bounded_and_state_reset_is_per_environment():
  device = torch.device("cpu")
  state = create_visual_pose_state(num_envs=4, device=device, max_delay_steps=4)
  profile = vision_stage_parameters(2)
  state.bias_rotvec[:] = 100.0
  current = torch.zeros((4, 4), device=device)
  current[:, 0] = 1.0

  step_visual_pose_model(current, state, profile, update_bias=False)
  assert float(state.bias_rotvec.abs().max()) <= profile.bias_limit_rad + 1e-6

  state.pose_history[:] = 1.0
  state.last_output[:] = 1.0
  state.initialized[:] = True
  state.hold_remaining[:] = 2
  reset_visual_pose_state(state, torch.tensor([1, 3], device=device))

  assert torch.all(state.pose_history[[0, 2]] == 1.0)
  assert torch.all(state.last_output[[0, 2]] == 1.0)
  assert torch.all(state.initialized[[0, 2]])
  assert torch.all(state.pose_history[[1, 3]] == 0.0)
  assert torch.all(state.last_output[[1, 3]] == 0.0)
  assert torch.all(~state.initialized[[1, 3]])
  assert torch.all(state.hold_remaining[[1, 3]] == 0)


def test_visual_pose_model_preserves_quaternion_shape_and_finite_values():
  device = torch.device("cpu")
  state = create_visual_pose_state(num_envs=3, device=device, max_delay_steps=4)
  profile = vision_stage_parameters(1)
  current = torch.zeros((3, 4), device=device)
  current[:, 0] = 1.0

  measured = step_visual_pose_model(current, state, profile)

  assert measured.shape == (3, 4)
  assert torch.isfinite(measured).all()
  assert torch.allclose(
    torch.linalg.vector_norm(measured, dim=-1),
    torch.ones(3, device=device),
    atol=1e-5,
  )


def test_visual_pose_model_tracks_validity_age_confidence_and_innovation():
  device = torch.device("cpu")
  state = create_visual_pose_state(num_envs=1, device=device, max_delay_steps=6)
  profile = vision_stage_parameters(3)
  current = _yaw_quat(0.0, device=device)
  zero_noise = torch.zeros((1, 3), device=device)

  step_visual_pose_model(
    current,
    state,
    profile,
    noise_rotvec=zero_noise,
    dropout_mask=torch.tensor([False]),
    update_bias=False,
  )
  assert state.vision_valid.item() == 1.0
  assert state.vision_age.item() == 0.0
  assert state.vision_confidence.item() == 1.0
  assert state.vision_innovation.item() == 0.0

  step_visual_pose_model(
    _yaw_quat(math.pi / 2.0, device=device),
    state,
    profile,
    noise_rotvec=zero_noise,
    dropout_mask=torch.tensor([True]),
    update_bias=False,
  )
  assert state.vision_valid.item() == 0.0
  assert state.vision_age.item() > 0.0
  assert state.vision_confidence.item() < 1.0

  state.hold_remaining[:] = 0
  step_visual_pose_model(
    _yaw_quat(math.pi, device=device),
    state,
    profile,
    noise_rotvec=zero_noise,
    dropout_mask=torch.tensor([False]),
    update_bias=False,
  )
  assert state.vision_valid.item() == 1.0
  assert state.vision_age.item() == 0.0
  assert state.vision_innovation.item() >= 0.0
