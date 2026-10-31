# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math
from functools import wraps

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from wuji_mjlab.tasks.reorient.mdp.contact_task_state import (
  bounded_monotonic_increment,
  box_surface_distance,
  contact_milestone_events,
  goal_error_progress,
  mean_budgeted_increment,
  per_finger_surface_proximity_score,
)
from wuji_mjlab.tasks.reorient.mdp.event_impl.state import get_reorient_event_state
from wuji_mjlab.utils.reward_decorators import curriculum_scaled

# ---------------------------------------------------------------------------
# Finger self-collision
# ---------------------------------------------------------------------------


def finger_self_collision_penalty(
  env,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("finger_collision"),
  valid_channel_count: int | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """Count active collisions, or return their configured channel fraction."""
  sensor = env.scene[sensor_cfg.name]
  found = sensor.data.found
  if found is None:
    return torch.zeros(env.num_envs, device=env.device)
  if found.ndim == 1:
    found = found.unsqueeze(-1)
  channels = found.shape[-1] if valid_channel_count is None else valid_channel_count
  if channels <= 0:
    raise ValueError("valid_channel_count must be positive")
  active = (found > 0).float().sum(dim=-1)
  return active / float(channels) if normalize else active


# ---------------------------------------------------------------------------
# Tolerance kernels (reorient-specific)
# ---------------------------------------------------------------------------


def tolerance(
  value: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
  value_at_margin: float = 0.1,
) -> torch.Tensor:
  """Gaussian tolerance kernel.

  Returns 1.0 inside *bounds*, decays as a Gaussian outside.
  *sigma* is derived so that the return value equals *value_at_margin*
  at exactly *margin* distance from the nearest bound.
  """
  lower, upper = bounds
  in_bounds = (value >= lower) & (value <= upper)
  if margin <= 0.0:
    return in_bounds.float()

  below = torch.clamp(lower - value, min=0.0)
  above = torch.clamp(value - upper, min=0.0)
  d = torch.maximum(below, above)

  sigma = margin / math.sqrt(-2.0 * math.log(value_at_margin))
  return torch.where(in_bounds, torch.ones_like(d), torch.exp(-0.5 * (d / sigma) ** 2))


def tolerance_linear(
  value: torch.Tensor,
  bounds: tuple[float, float],
  margin: float,
) -> torch.Tensor:
  """Linear tolerance kernel (MJX-aligned).

  Returns 1.0 inside *bounds*, linearly decays to 0 at *margin* distance.
  """
  lower, upper = bounds
  in_bounds = (value >= lower) & (value <= upper)
  if margin <= 0.0:
    return in_bounds.float()

  below = torch.clamp(lower - value, min=0.0)
  above = torch.clamp(value - upper, min=0.0)
  d = torch.maximum(below, above)

  return torch.where(
    in_bounds, torch.ones_like(d), torch.clamp(1.0 - d / margin, min=0.0)
  )


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


# --- Dense rewards (× step_dt) ---


@curriculum_scaled
def orientation_alignment(
  env,
  command_name: str = "reorient_command",
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  margin: float = 3.14159,
  bound: float = 0.2,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """MJX-aligned orientation reward using a linear tolerance kernel."""
  del curriculum_term, curriculum_min
  obj: Entity = env.scene[object_cfg.name]
  goal_quat = env.command_manager.get_term(command_name).goal_quat
  ori_err = quat_error_magnitude(obj.data.root_link_quat_w, goal_quat)
  return tolerance_linear(ori_err, bounds=(0.0, bound), margin=margin)


def _tip_contact_count(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_cfg.name]
  found = sensor.data.found
  if found is None:
    return torch.zeros(env.num_envs, device=env.device)
  if found.ndim == 1:
    found = found.unsqueeze(-1)
  return (found > 0).float().sum(dim=-1)


def _cage_drop_mask(env, like: torch.Tensor) -> torch.Tensor:
  try:
    return env.termination_manager.get_term("cage_drop").to(dtype=torch.bool)
  except (AttributeError, KeyError, ValueError):
    return torch.zeros_like(like, dtype=torch.bool)


def _cube_half_size(env, object_cfg: SceneEntityCfg) -> torch.Tensor:
  obj: Entity = env.scene[object_cfg.name]
  geom_ids = obj.indexing.geom_ids[object_cfg.geom_ids].long()
  if geom_ids.numel() != 1:
    raise ValueError(
      "contact shaping requires exactly one cube geom; "
      f"matched {geom_ids.numel()} geoms"
    )
  env_ids = torch.arange(env.num_envs, device=env.device)
  return env.sim.model.geom_size[env_ids, geom_ids[0], :3]


def _once_per_task_step(func):
  """Luwu reward terms are read-only on repeat evaluation of the same step."""
  @wraps(func)
  def cached(env, command_name="reorient_command", **kwargs):
    command = env.command_manager.get_term(command_name)
    if not getattr(getattr(command, "cfg", None), "synchronized_contact_task", False):
      return func(env, command_name=command_name, **kwargs)
    command.evaluate_contact_task()
    step = env.common_step_counter
    entry = command._task_reward_cache.get(func.__name__)
    if entry is None or entry[0] != step:
      value = func(env, command_name=command_name, **kwargs)
      command._task_reward_cache[func.__name__] = (step, value)
    return command._task_reward_cache[func.__name__][1].clone()
  return cached


@_once_per_task_step
def approach_progress_reward(
  env,
  command_name: str = "reorient_command",
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  length_scale: float = 0.02,
  step_budget: float = 0.25,
  box_half_size_m: float | None = None,
) -> torch.Tensor:
  """Bounded first-approach progress toward the cube surface.

  The manager multiplies this rate by ``step_dt``. The command stores the
  episode-wide best score, so backing away and returning cannot pay twice.
  """
  command = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  site_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids, :]
  half_size = (
    _cube_half_size(env, object_cfg)
    if box_half_size_m is None
    else torch.full_like(site_pos_w[:, 0, :], box_half_size_m)
  )
  distance = box_surface_distance(
    site_pos_w,
    obj.data.root_link_pos_w,
    obj.data.root_link_quat_w,
    half_size,
  )
  score = per_finger_surface_proximity_score(distance, length_scale=length_scale)
  if command._approach_best_score.shape != score.shape:
    command._approach_best_score = torch.zeros_like(score)
  score = torch.where(
    command.task_reward_active.unsqueeze(-1), score, command._approach_best_score
  )
  increment, new_best = bounded_monotonic_increment(
    score, command._approach_best_score
  )
  uninitialized = ~command._approach_initialized
  increment = torch.where(
    uninitialized.unsqueeze(-1), torch.zeros_like(increment), increment
  )
  command._approach_best_score = new_best
  command._approach_initialized |= command.task_reward_active
  # Consume progress even on a drop, but never pay for it or allow catch-up.
  increment = torch.where(
    _cage_drop_mask(env, command.task_reward_active).unsqueeze(-1),
    torch.zeros_like(increment), increment,
  )
  return (
    mean_budgeted_increment(increment, step_budget)
    / max(float(env.step_dt), 1e-6)
  )


@_once_per_task_step
def contact_milestone_reward(
  env,
  command_name: str = "reorient_command",
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("tip_object_contact"),
  stable_time: float = 0.1,
  single_reward: float = 1.0,
  double_reward: float = 1.0,
) -> torch.Tensor:
  """One-shot rewards for stable one-finger and two-finger contact."""
  command = env.command_manager.get_term(command_name)
  contact_count = _tip_contact_count(env, sensor_cfg)
  contact_count = torch.where(
    _cage_drop_mask(env, contact_count) | ~command.task_reward_active, 0, contact_count
  )
  first, second, hold, first_awarded, second_awarded = contact_milestone_events(
    contact_count,
    command._contact_hold_time,
    command._contact_first_awarded,
    command._contact_second_awarded,
    float(env.step_dt),
    stable_time,
    single_reward,
    double_reward,
  )
  command._contact_hold_time = hold
  command._contact_first_awarded = first_awarded
  command._contact_second_awarded = second_awarded
  return (first + second) / max(float(env.step_dt), 1e-6)


@_once_per_task_step
def contact_gated_goal_progress_reward(
  env,
  command_name: str = "reorient_command",
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("tip_object_contact"),
  step_budget: float = 10.0,
) -> torch.Tensor:
  """Reward new orientation progress only while a fingertip is in contact."""
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_cfg.name]
  ori_error = quat_error_magnitude(obj.data.root_link_quat_w, command.goal_quat)
  progress = goal_error_progress(
    command.goal_initial_error, ori_error, command.success_threshold
  )
  progress = torch.where(command.task_reward_active, progress, command._goal_progress_best)
  increment, new_best = bounded_monotonic_increment(
    progress, command._goal_progress_best
  )
  command._goal_progress_best = new_best
  contact_valid = _tip_contact_count(env, sensor_cfg) >= command.cfg.min_contact_fingertips
  contact_valid &= ~_cage_drop_mask(env, contact_valid)
  return (
    step_budget
    * increment
    * contact_valid.float()
    / max(float(env.step_dt), 1e-6)
  )


@_once_per_task_step
def rotation_success_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """One-shot bounded reward for a completed incremental rotation goal."""
  command = env.command_manager.get_term(command_name)
  event = command.success_achieved & command.rotation_mode & ~_cage_drop_mask(
    env, command.success_achieved
  )
  return event.float() / max(float(env.step_dt), 1e-6)


@_once_per_task_step
def flip_success_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """One-shot bounded reward for the completed flip goal."""
  command = env.command_manager.get_term(command_name)
  # A cube that has already escaped the cage must not receive a flip success
  # payout on the same transition. The termination manager evaluates the
  # current step's terms before the reward is consumed, so mask the drop event
  # here instead of relying on a later reset.
  event = command.success_achieved & ~command.rotation_mode & ~_cage_drop_mask(
    env, command.success_achieved
  )
  return event.float() / max(float(env.step_dt), 1e-6)


def rubik_task_progress_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.progress_delta / max(float(env.step_dt), 1e-6)


def rubik_u_turn_progress_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.turn_progress_delta / max(float(env.step_dt), 1e-6)


def rubik_flip_to_top_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.flip_to_top_achieved.float() / max(float(env.step_dt), 1e-6)


def rubik_cube_position_stability_reward(
  env,
  command_name: str = "reorient_command",
  task_kind: int = 0,
  stable_delta_m: float = 5.0e-3,
  unstable_delta_m: float = 1.0e-2,
) -> torch.Tensor:
  """Reward a contacted flip for keeping cube translation small."""
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  episode_kind = getattr(command, "episode_kind", None)
  if episode_kind is None:
    return torch.zeros(env.num_envs, device=env.device)
  active = (
    (episode_kind == task_kind)
    & (command.phase == 0)
    & (command.tip_contact_count >= command.cfg.min_contact_fingertips)
    & command.cube_position_delta_valid
    & ~env.termination_manager.get_term("cage_drop")
  )
  delta = command.cube_position_delta
  score = ((unstable_delta_m - delta) / (unstable_delta_m - stable_delta_m)).clamp(-1.0, 1.0)
  if getattr(command.cfg, 'flip_specialization', False):
    state = command.flip_preservation
    # Positive stability is useful during new flip progress or productive
    # alignment/valid hold. Being still with misaligned layers does not pay.
    productive = ((~state.settling & (command.progress_delta > 0))
      | (state.settling & ((state.improvement > 0) | (state.hold > 0))))
    score = torch.where(productive, score, score.clamp(max=0))
  return torch.where(active, score, torch.zeros_like(score)) / max(float(env.step_dt), 1e-6)


def rubik_smart_move_residual_reward(
  env,
  command_name: str = "reorient_command",
  threshold_rad: float = 0.0872664626,
  penalty_window_rad: float = 0.3490658504,
  specialized: bool = False,
  positive_progress_only: bool = False,
) -> torch.Tensor:
  """Shape the physical post-MOVE turn residual for SmartCube transfer."""
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  command.update_smart_move_pending()
  residual = command.u_angle_error
  positive = (1.0 - 0.5 * residual / max(threshold_rad, 1e-6)).clamp(0.5, 1.0)
  negative = -((residual - threshold_rad) / max(penalty_window_rad, 1e-6)).clamp(0.0, 1.0)
  score = torch.where(residual <= threshold_rad, positive, negative)
  if specialized:
    from .turn_specialization import residual_score
    score = residual_score(residual)
  if positive_progress_only:
    score = torch.where(
      (score > 0) & (command.turn_progress_delta <= 0),
      torch.zeros_like(score), score,
    )
  active = command.smart_move_pending & (command.phase == 1)
  return torch.where(active, score, torch.zeros_like(score)) / max(float(env.step_dt), 1e-6)


def rubik_u_turn_success_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.turn_completed.float() / max(float(env.step_dt), 1e-6)


def rubik_turn_streak_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Pay the episode's Nth independent turn N units, only on completion."""
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  active = command.turn_completed & (command.episode_kind == 1)
  score = torch.where(
    active, command.turn_success_count, torch.zeros_like(command.turn_success_count)
  )
  return score / max(float(env.step_dt), 1e-6)


def rubik_cube_yaw_success_reward(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.yaw_completed.float() / max(float(env.step_dt), 1e-6)


@curriculum_scaled
def hand_pose_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """L2 deviation from default joint positions."""
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  default_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(joint_pos - default_pos), dim=-1)


@curriculum_scaled
def action_rate_combined(
  env,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """First-order + second-order penalty on raw policy actions."""
  del curriculum_term, curriculum_min
  action = env.action_manager.action
  prev = env.action_manager.prev_action
  prev_prev = env.action_manager.prev_prev_action

  first_order = torch.sum(torch.square(action - prev), dim=-1)
  second_order = torch.sum(torch.square(action - 2.0 * prev + prev_prev), dim=-1)

  return first_order + second_order


def action_target_smoothness_penalty_from_normalized_targets(
  current: torch.Tensor,
  previous: torch.Tensor,
  previous_previous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return mean first/second differences for normalized joint targets."""
  first = torch.square(current - previous).mean(dim=-1)
  second = torch.square(
    current - 2.0 * previous + previous_previous
  ).mean(dim=-1)
  return first, second


@curriculum_scaled
def action_target_smoothness_penalty(
  env,
  action_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """Smooth the actual EMA target, independent of raw PPO exploration noise."""
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  action_term = env.action_manager.get_term(action_name)
  current = action_term.processed_action
  previous = action_term.previous_processed_action
  previous_previous = action_term.previous_previous_processed_action
  joint_ids = asset_cfg.joint_ids
  lower = robot.data.soft_joint_pos_limits[:, joint_ids, 0]
  upper = robot.data.soft_joint_pos_limits[:, joint_ids, 1]

  def normalize(target: torch.Tensor) -> torch.Tensor:
    return 2.0 * (target - lower) / (upper - lower).clamp_min(1e-6) - 1.0

  first, second = action_target_smoothness_penalty_from_normalized_targets(
    normalize(current), normalize(previous), normalize(previous_previous)
  )
  # The Luwu configuration applies the outer -0.02 weight. The 0.25 factor
  # keeps the second-order contribution at -0.005.
  return first + 0.25 * second


@curriculum_scaled
def joint_vel_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  max_velocity: float = 5.0,
  vel_tolerance: float = 1.0,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """Normalized squared joint velocity penalty.

  Matches MJX: sum((vel / (max_velocity - tolerance))^2).
  """
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  denom = max(max_velocity - vel_tolerance, 1e-6)
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(vel / denom), dim=-1)


def gentle_joint_velocity_penalty(
  env, v_ref=1., v_soft=1., coefficient: float = 0.,
  joint_names: tuple[str, ...] = (),
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Calibrated physical speed cost; tensors are cached, with no host sync."""
  from .gentle_flip import _velocity_cost, gentle_state, speed_parameter

  robot = env.scene[asset_cfg.name]
  velocity = robot.data.joint_vel[:, asset_cfg.joint_ids]
  if not hasattr(env, '_gentle_speed_tensors'):
    if not math.isfinite(coefficient) or coefficient < 0:
      raise ValueError('Invalid gentle velocity coefficient')
    names = tuple(robot.joint_names)
    if joint_names and names != tuple(joint_names):
      raise ValueError('Gentle speed calibration joint order does not match robot')
    ref = speed_parameter(v_ref, velocity.shape[-1], positive=True, name='v_ref')
    soft = speed_parameter(v_soft, velocity.shape[-1], positive=False, name='v_soft')
    env._gentle_speed_tensors = velocity.new_tensor(ref), velocity.new_tensor(soft)
  cost = _velocity_cost(velocity, *env._gentle_speed_tensors)
  warmup = env.action_manager.get_term('joint_pos').cfg.warmup_time_s
  cost = torch.where(env.episode_length_buf * env.step_dt >= warmup, cost, 0.)
  env._gentle_velocity_cost = cost
  return cost * coefficient * gentle_state(env).speed_scale


@curriculum_scaled
def energy_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """Energy: sum(|vel| * |torque|)."""
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  torques = robot.data.actuator_force[:, asset_cfg.actuator_ids]
  velocities = robot.data.joint_vel[:, asset_cfg.joint_ids]
  n = min(torques.shape[1], velocities.shape[1])
  return torch.sum(torch.abs(velocities[:, :n]) * torch.abs(torques[:, :n]), dim=-1)


@curriculum_scaled
def torque_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  torque_limit: float | None = None,
  reduce_mean: bool = False,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """Normalized squared torque: sum((torque / torque_max)^2)."""
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  torques = robot.data.actuator_force[:, asset_cfg.actuator_ids]
  torque_limits = getattr(robot.data, "actuator_effort_limit", None)
  if torque_limits is not None:
    limits = torque_limits[:, asset_cfg.actuator_ids].clamp_min(1e-3)
  elif torque_limit is not None:
    limits = torch.full_like(torques, torque_limit)
  else:
    normed = torques
    return torch.mean(torch.square(normed), dim=-1) if reduce_mean else torch.sum(
      torch.square(normed), dim=-1
    )
  if limits.shape != torques.shape:
    limits = limits.expand_as(torques)
  normed = torques / limits.clamp_min(1e-6)
  return torch.mean(torch.square(normed), dim=-1) if reduce_mean else torch.sum(
    torch.square(normed), dim=-1
  )


@curriculum_scaled
def joint_acc_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  denom: float = 4.0,
  curriculum_term: str = "",
  curriculum_min: float = 0.0,
) -> torch.Tensor:
  """Joint acceleration penalty: Σ((vel_t - vel_{t-1}) / denom)².

  Penalizes joint jerk/vibration. Requires reset_joint_acc_cache event
  to clear shared previous-joint-velocity state on episode reset.
  """
  del curriculum_term, curriculum_min
  robot: Entity = env.scene[asset_cfg.name]
  vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  state = get_reorient_event_state(env)

  if state.prev_joint_vel is None:
    state.prev_joint_vel = vel.clone()
    return torch.zeros(env.num_envs, device=env.device)

  acc = (vel - state.prev_joint_vel) / max(denom, 1e-6)
  state.prev_joint_vel = vel.clone()
  return torch.sum(torch.square(acc), dim=-1)


def palm_detach_reward(
  env,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("palm_object_found"),
  distal_sensor_cfg: SceneEntityCfg = SceneEntityCfg("distal_finger_object_found"),
) -> torch.Tensor:
  """Dense reward for the cube being held by the distal fingers only.

  Returns 1.0 per env iff BOTH:
    - palm_object_found.found == 0 (cube is not touching palm or proximal
      finger links link1/link2), AND
    - distal_finger_object_found.found > 0 (cube is in contact with the
      distal phalanges link3/link4 of at least one finger).

  Returns 0.0 if either condition fails — i.e., the cube is rolling on the
  palm/proximal links, or the cube is floating away with no fingertip
  contact at all. Encourages the policy to lift the cube fully into the
  fingertips.

  Uses the binary ``found`` field rather than a force threshold so it
  remains correct for very light cubes (the 54 mm cube weighs only
  ~1.18 N, below typical force thresholds).
  """
  palm_sensor: ContactSensor = env.scene[sensor_cfg.name]
  distal_sensor: ContactSensor = env.scene[distal_sensor_cfg.name]
  if palm_sensor.data.found is None or distal_sensor.data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  palm_found = palm_sensor.data.found[:, 0]
  distal_found = distal_sensor.data.found[:, 0]
  return ((palm_found == 0) & (distal_found > 0)).float()


def tip_slide_penalty(
  env,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  sensor_cfg: SceneEntityCfg = SceneEntityCfg("tip_object_contact"),
  contact_threshold: float = 0.0,
) -> torch.Tensor:
  """MJX-aligned tip slide penalty without curriculum scaling.

  Prioritizes binary contact detection via sensor ``found`` when available,
  matching the MJX task's gating behavior more closely.
  """
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  sensor: ContactSensor = env.scene[sensor_cfg.name]

  tip_lin_vel = robot.data.site_lin_vel_w[:, robot_cfg.site_ids, :]
  obj_lin_vel = obj.data.root_link_lin_vel_w.unsqueeze(1)
  rel_vel = torch.norm(tip_lin_vel - obj_lin_vel, dim=-1)

  if sensor.data.found is not None:
    in_contact = (sensor.data.found > contact_threshold).float()
  elif sensor.data.force is not None:
    in_contact = (torch.norm(sensor.data.force, dim=-1) > contact_threshold).float()
  else:
    return torch.zeros(env.num_envs, device=env.device)

  if in_contact.shape[1] != rel_vel.shape[1]:
    n = min(in_contact.shape[1], rel_vel.shape[1])
    in_contact = in_contact[:, :n]
    rel_vel = rel_vel[:, :n]

  return torch.sum(rel_vel * in_contact, dim=-1)


def hold_escalation_value(
  in_window: torch.Tensor,
  within_threshold: torch.Tensor,
  timer: torch.Tensor,
) -> torch.Tensor:
  """Pure function: timer-scaled reward, active only in window & threshold."""
  return (in_window & within_threshold).float() * timer.float()


def hold_escalation(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Time-escalating dense reward during SUCCESS_WINDOW.

  Replaces sparse success (+100) and flat hold_bonus (+1.0/step).
  Returns timer value (1..20) when in SUCCESS_WINDOW and within
  threshold, zero otherwise. With weight=11.4 and dt=0.05,
  total per cycle ≈ 120 (matching old sparse + hold_bonus).

  Uses reward_window_timer (pre-reset snapshot) so the final step
  before goal switch is not lost.
  """
  command = env.command_manager.get_term(command_name)
  return hold_escalation_value(
    command.in_success_window,
    command.within_threshold,
    command.reward_hold_counter_snapshot,
  )


def drop_penalty_sparse(
  env,
  term_name: str = "cube_drop",
) -> torch.Tensor:
  """One event per terminal transition, converted to a rate for dt integration."""
  return env.termination_manager.get_term(term_name).float() / env.step_dt
