# SPDX-License-Identifier: Apache-2.0
"""Actor-only smart-cube measurement model; parameters require device calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from mjlab.utils.lab_api.math import quat_from_angle_axis, quat_mul


@dataclass(frozen=True)
class SmartCubeProfile:
  threshold_degrees: tuple[float, float] = (43.0, 53.0)
  move_delay_steps: int = 1
  move_delay_range: tuple[float, float] | None = None
  imu_delay_steps: int = 1
  imu_latency_range_seconds: tuple[float, float] | None = None
  imu_latency_mode_seconds: float = .09
  imu_noise_rad: float = math.radians(0.5)
  calibration_noise_rad: float = math.radians(1.0)
  yaw_drift_rad_per_s: float = math.radians(0.02)
  imu_dropout_probability: float = 0.01
  max_age_seconds: float = 0.25
  age_scale_seconds: float = 1.0
  confidence_tau_seconds: float = 0.25
  innovation_scale_rad: float = math.pi

  def __post_init__(self):
    if self.imu_latency_range_seconds is not None:
      low_latency, high_latency = self.imu_latency_range_seconds
      if not 0 <= low_latency <= self.imu_latency_mode_seconds <= high_latency <= .35:
        raise ValueError('IMU latency must satisfy 0 <= low <= mode <= high <= .35 seconds')
    low, high = self.threshold_degrees
    if not 40 <= low <= high < 90:
      raise ValueError("MOVE thresholds must satisfy 40 <= low <= high < 90 degrees")
    if self.move_delay_range is not None:
      low_delay, high_delay = self.move_delay_range
      if not 0 <= low_delay <= 7 or not math.isclose(high_delay - low_delay, 1):
        raise ValueError("MOVE jitter range must span one step within [0, 8]")
    for delay in (self.move_delay_steps, self.imu_delay_steps):
      if not isinstance(delay, int) or not 0 <= delay <= 8:
        raise ValueError("Sensor delay must be an integer in [0, 8] policy steps")
    if not 0 <= self.imu_dropout_probability <= 1:
      raise ValueError("IMU dropout probability must be in [0, 1]")
    if any(
      not math.isfinite(v) or v < 0
      for v in (
        self.imu_noise_rad,
        self.calibration_noise_rad,
        self.yaw_drift_rad_per_s,
      )
    ):
      raise ValueError("Noise and drift bounds must be finite and non-negative")
    if any(
      not math.isfinite(v) or v <= 0
      for v in (
        self.max_age_seconds,
        self.age_scale_seconds,
        self.confidence_tau_seconds,
        self.innovation_scale_rad,
      )
    ):
      raise ValueError("Metadata scales must be finite and positive")


@dataclass
class SmartCubeState:
  thresholds: torch.Tensor
  detent_angles: torch.Tensor
  previous_angles: torch.Tensor
  delivered_move_step: torch.Tensor
  move_event: torch.Tensor
  move_queue: torch.Tensor
  delivered_angles: torch.Tensor
  move_initialized: torch.Tensor
  move_step: torch.Tensor
  imu_queue: torch.Tensor
  orientation: torch.Tensor
  calibration: torch.Tensor
  yaw_rate: torch.Tensor
  yaw_bias: torch.Tensor
  age_seconds: torch.Tensor
  innovation: torch.Tensor
  imu_initialized: torch.Tensor
  imu_step: torch.Tensor
  imu_arrival_times: torch.Tensor
  imu_sample_steps: torch.Tensor
  imu_delivered_sample_step: torch.Tensor


def create_smart_cube_state(num_envs, device, profile, *, dtype=torch.float32):
  def zeros(*shape):
    return torch.zeros((num_envs, *shape), device=device, dtype=dtype)

  state = SmartCubeState(
    thresholds=zeros(6),
    detent_angles=zeros(6),
    previous_angles=zeros(6),
    delivered_move_step=torch.full((num_envs,), -1, device=device, dtype=torch.long),
    move_event=torch.zeros(num_envs, device=device, dtype=torch.bool),
    move_queue=zeros(9, 6),
    delivered_angles=zeros(6),
    move_initialized=torch.zeros(num_envs, device=device, dtype=torch.bool),
    move_step=torch.full((num_envs,), -1, device=device, dtype=torch.long),
    imu_queue=zeros(9, 4),
    imu_arrival_times=torch.full((num_envs, 9), float('inf'), device=device),
    imu_sample_steps=torch.full((num_envs, 9), -1, device=device, dtype=torch.long),
    imu_delivered_sample_step=torch.full((num_envs,), -1, device=device, dtype=torch.long),
    orientation=zeros(4),
    calibration=zeros(3),
    yaw_rate=zeros(),
    yaw_bias=zeros(),
    age_seconds=zeros(),
    innovation=zeros(),
    imu_initialized=torch.zeros(num_envs, device=device, dtype=torch.bool),
    imu_step=torch.full((num_envs,), -1, device=device, dtype=torch.long),
  )
  reset_smart_cube_state(state, torch.arange(num_envs, device=device), profile)
  return state


def reset_smart_cube_state(state, env_ids, profile):
  ids = env_ids.to(device=state.thresholds.device, dtype=torch.long)
  low, high = map(math.radians, profile.threshold_degrees)
  state.thresholds[ids] = torch.empty_like(state.thresholds[ids]).uniform_(low, high)
  for name in (
    "detent_angles",
    "previous_angles",
    "move_queue",
    "delivered_angles",
    "imu_queue",
    "orientation",
    "yaw_bias",
    "age_seconds",
    "innovation",
  ):
    getattr(state, name)[ids] = 0
  state.orientation[ids, 0] = 1
  state.imu_queue[ids, :, 0] = 1
  state.calibration[ids] = torch.empty_like(state.calibration[ids]).uniform_(
    -profile.calibration_noise_rad, profile.calibration_noise_rad
  )
  state.yaw_rate[ids] = torch.empty_like(state.yaw_rate[ids]).uniform_(
    -profile.yaw_drift_rad_per_s, profile.yaw_drift_rad_per_s
  )
  state.move_initialized[ids] = False
  state.imu_initialized[ids] = False
  state.move_step[ids] = -1
  state.delivered_move_step[ids] = -1
  state.move_event[ids] = False
  state.imu_step[ids] = -1
  state.imu_arrival_times[ids] = float('inf')
  state.imu_sample_steps[ids] = -1
  state.imu_delivered_sample_step[ids] = -1


def reset_smart_cube_observation_state(env, env_ids, profile: SmartCubeProfile):
  """Reset event; persist the opt-in profile before observations are computed."""
  env._smart_cube_profile = profile
  state = get_smart_cube_state(env)
  if str(env.scene.device) != "meta":
    reset_smart_cube_state(state, env_ids, effective_smart_cube_profile(env, profile))
    controller = getattr(env, '_smart_cube_curriculum', None)
    if controller is not None and getattr(controller.cfg, 'realistic_dr', False):
      nominal = getattr(env, '_realistic_nominal_mask', None)
      if nominal is not None:
        state.yaw_rate[env_ids[nominal[env_ids]]] = 0
    partial = getattr(env, '_rubik_partial_turn', None)
    if partial is not None:
      ids = env_ids[partial[env_ids]]
      # Replay crossing from the preceding aligned detent, respecting MOVE delay.
      # Do not round a 60-degree partial start directly to a 90-degree seed.
      state.move_initialized[ids] = True


def effective_smart_cube_profile(env, profile):
  controller = getattr(env, '_smart_cube_curriculum', None)
  if controller is None:
    return profile
  if getattr(controller.cfg, 'realistic_dr', False):
    from .smart_cube_curriculum import realistic_parameter_progress, realistic_stage_progress
    late = (realistic_parameter_progress(controller.realistic_progress)
      if controller.cfg.realistic_success_gated else
      realistic_stage_progress(controller.realistic_steps)[1])
    return replace(
      profile, move_delay_range=None, imu_latency_range_seconds=None,
      imu_noise_rad=math.radians(.1),
      calibration_noise_rad=math.radians(.2),
      yaw_drift_rad_per_s=math.radians(.05 * late),
      imu_dropout_probability=0.,
    )
  from .smart_cube_curriculum import interpolate_smart_cube_profile
  return interpolate_smart_cube_profile(
    profile, controller.alpha, controller.cfg.imu_noise_max_degrees,
    controller.cfg.irregular_imu_latency,
  )


def get_smart_cube_state(env):
  state = getattr(env, "_smart_cube_state", None)
  if state is None:
    state = create_smart_cube_state(env.num_envs, env.device, effective_smart_cube_profile(env, env._smart_cube_profile))
    env._smart_cube_state = state
  return state


def step_move_model(angles, state, profile, *, step_id):
  """Six signed, unwrapped driver angles -> delayed reversible 90-degree events."""
  fresh = state.move_step != step_id
  state.move_event[fresh] = False
  initial = fresh & ~state.move_initialized
  quarter = math.pi / 2
  # At reset only, initialize from a discrete aligned state. Never seed from
  # a new goal: all faces continue accumulating regardless of the selected face.
  seed = torch.round(angles / quarter) * quarter
  state.detent_angles[initial] = seed[initial]
  state.move_queue[initial] = seed[initial, None, :]
  state.previous_angles[initial] = angles[initial]
  residual = angles - state.detent_angles
  direction = angles - state.previous_angles
  crossing = ((residual > state.thresholds) & (direction > 0)) | (
    (residual < -state.thresholds) & (direction < 0)
  )
  count = torch.floor((residual.abs() - state.thresholds) / quarter + 1).clamp_min(0)
  count = torch.where(crossing, count, 0.0)
  updated = state.detent_angles + residual.sign() * count * quarter
  state.detent_angles[fresh] = updated[fresh]
  state.previous_angles[fresh] = angles[fresh]
  if state.move_queue.shape[1] > 1:
    state.move_queue[fresh, 1:] = state.move_queue[fresh, :-1].clone()
  state.move_queue[fresh, 0] = state.detent_angles[fresh]
  if profile.move_delay_range is None:
    delays = torch.full_like(state.move_step, profile.move_delay_steps)
  else:
    low = profile.move_delay_range[0]
    delays = (torch.randint(0, 2, state.move_step.shape, device=angles.device)
      + math.floor(low)
      + (torch.rand_like(state.delivered_move_step, dtype=angles.dtype) < low % 1).long())
  sample_step = step_id - delays
  receive = fresh & ((sample_step >= state.delivered_move_step) | initial)
  candidate = state.move_queue.gather(1, delays[:, None, None].expand(-1, 1, 6)).squeeze(1)
  previous_delivered = state.delivered_angles.clone()
  state.delivered_angles[receive] = candidate[receive]
  state.delivered_move_step[receive] = sample_step[receive]
  state.move_event[receive & ~initial] = (
    (candidate[receive & ~initial] - previous_delivered[receive & ~initial]).abs().amax(dim=-1)
    > 1.0e-6
  )
  state.move_initialized[fresh] = True
  state.move_step[fresh] = step_id
  return state.delivered_angles.clone()


def _rotvec_quat(rotvec):
  angle = torch.linalg.vector_norm(rotvec, dim=-1)
  axis = rotvec / angle.unsqueeze(-1).clamp_min(1e-8)
  return quat_from_angle_axis(angle, axis)


def step_imu_model(quat_tag, state, profile, *, step_id, step_dt):
  """One noisy delayed IMU sample shared by both actor attitude terms."""
  fresh = state.imu_step != step_id
  initial = fresh & ~state.imu_initialized
  elapsed = (step_id - state.imu_step).clamp_min(0).to(quat_tag.dtype) * step_dt
  elapsed = torch.where(initial, 0.0, elapsed)
  state.yaw_bias[fresh] += state.yaw_rate[fresh] * elapsed[fresh]
  rotvec = (
    torch.empty_like(state.calibration).uniform_(
      -profile.imu_noise_rad, profile.imu_noise_rad
    )
    + state.calibration
  )
  rotvec[:, 2] += state.yaw_bias
  measured = quat_mul(_rotvec_quat(rotvec), quat_tag)
  measured = measured / torch.linalg.vector_norm(
    measured, dim=-1, keepdim=True
  ).clamp_min(1e-8)
  state.imu_queue[initial] = measured[initial, None, :]
  irregular = profile.imu_latency_range_seconds is not None
  if profile.imu_delay_steps or irregular:
    state.imu_queue[fresh, 1:] = state.imu_queue[fresh, :-1].clone()
  state.imu_queue[fresh, 0] = measured[fresh]
  if irregular:
    # Sample transport once per emitted packet; stale packets cannot rewind IMU.
    low, high = profile.imu_latency_range_seconds
    mode = profile.imu_latency_mode_seconds
    u = torch.rand_like(state.age_seconds)
    if high > low:
      latency = torch.where(u < (mode-low)/(high-low),
        low + torch.sqrt(u*(high-low)*(mode-low)),
        high - torch.sqrt((1-u)*(high-low)*(high-mode)))
    else:
      latency = torch.full_like(u, low)
    state.imu_arrival_times[fresh, 1:] = state.imu_arrival_times[fresh, :-1].clone()
    state.imu_sample_steps[fresh, 1:] = state.imu_sample_steps[fresh, :-1].clone()
    state.imu_arrival_times[fresh, 0] = step_id * step_dt + latency[fresh]
    lost = torch.rand_like(u) < profile.imu_dropout_probability
    state.imu_arrival_times[fresh & lost, 0] = float('inf')
    state.imu_sample_steps[fresh, 0] = step_id
    eligible = ((state.imu_arrival_times <= step_id * step_dt + 1e-7)
      & (state.imu_sample_steps > state.imu_delivered_sample_step[:, None]))
    latest, slot = torch.where(eligible, state.imu_sample_steps, -1).max(dim=1)
    receive = fresh & ((latest >= 0) | initial)
    candidate = state.imu_queue.gather(1, slot[:, None, None].expand(-1, 1, 4)).squeeze(1)
    candidate = torch.where(initial[:, None], measured, candidate)
    latest = torch.where(initial, step_id, latest)
    state.imu_delivered_sample_step[receive] = latest[receive]
    sample_age = (step_id - latest).to(measured.dtype) * step_dt
  else:
    candidate = state.imu_queue[:, profile.imu_delay_steps]
    receive = fresh & (
      (torch.rand_like(state.age_seconds) >= profile.imu_dropout_probability) | initial
    )
    sample_age = torch.full_like(state.age_seconds, profile.imu_delay_steps * step_dt)
  dot = (candidate * state.orientation).sum(-1).abs().clamp(0, 1)
  innovation = 2 * torch.acos(dot)
  state.innovation[fresh] = torch.where(receive & ~initial, innovation, 0.0)[fresh]
  state.orientation[receive] = candidate[receive]
  state.age_seconds[fresh] += elapsed[fresh]
  state.age_seconds[receive] = sample_age[receive]
  state.imu_initialized[fresh] = True
  state.imu_step[fresh] = step_id
  return state.orientation.clone()


def ble_metadata(age_seconds, innovation_rad, connected, profile):
  """Use the same host-derived semantics for simulated and real BLE packets.

  Age is since the attitude sample (including known transport latency), not
  since policy evaluation. Confidence is a freshness heuristic, not an IMU field.
  """
  valid = connected & (age_seconds >= 0) & (age_seconds <= profile.max_age_seconds)
  confidence = torch.exp(-age_seconds.clamp_min(0) / profile.confidence_tau_seconds)
  return torch.stack(
    (
      valid.to(age_seconds.dtype),
      (age_seconds / profile.age_scale_seconds).clamp(0, 1),
      torch.where(valid, confidence, 0.0),
      (innovation_rad / profile.innovation_scale_rad).clamp(0, 1),
    ),
    dim=-1,
  )


def smart_cube_metadata(state, profile):
  return ble_metadata(
    state.age_seconds, state.innovation, state.imu_initialized, profile
  )
