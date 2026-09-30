# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from wuji_mjlab.tasks.reorient.mdp.curriculums import get_curriculum_value
from wuji_mjlab.tasks.reorient.mdp.event_impl.curriculum import (
  _get_total_training_steps,
)
from wuji_mjlab.tasks.reorient.mdp.event_impl.state import (
  get_reorient_event_state,
)
from wuji_mjlab.tasks.reorient.mdp.event_utils import (
  get_linear_progress,
  resolve_env_ids,
)
from wuji_mjlab.utils.math import random_quat_uniform

_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def reset_rubik_u_face(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", body_names=("right_palm_link",)
  ),
  u_face_joint_name: str = "cubelet:driver:pos_z",
  palm_clearance_m: float = 0.006,
) -> None:
  """Sample 45% flip, 45% U-turn, 10% root-yaw starts at current difficulty."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  palm_ids, _ = robot.find_bodies(robot_cfg.body_names)
  joint_ids, _ = asset.find_joints(u_face_joint_name)
  linked_joint_ids, _ = asset.find_joints(r"cubelet:rotz:.*_pos_z")
  if len(palm_ids) != 1 or len(joint_ids) != 1 or len(linked_joint_ids) != 8:
    raise ValueError(
      "Rubik reset requires one palm body, one U driver and eight linked U "
      f"piece hinges; matched {len(palm_ids)}, {len(joint_ids)}, "
      f"and {len(linked_joint_ids)}."
    )

  n = env_ids.numel()
  default_state = asset.data.default_root_state[env_ids].clone()
  positions = default_state[:, :3] + env.scene.env_origins[env_ids]
  palm_quat_w = robot.data.body_link_pose_w[env_ids, int(palm_ids[0]), 3:7]
  up_local = torch.zeros((n, 3), dtype=default_state.dtype, device=env.device)
  up_local[:, 2] = 1.0
  palm_normal_w = quat_apply(palm_quat_w, up_local)
  positions += palm_clearance_m * palm_normal_w
  curriculum = getattr(env, "_rubik_curriculum", None)
  if curriculum is None:
    names = (
      "flip_stage", "turn_stage", "side_trials", "side_success",
      "bottom_trials", "bottom_success", "turn90_trials", "turn90_success",
      "turn180_trials", "turn180_success", "yaw_trials", "yaw_success",
    )
    curriculum = {
      name: torch.zeros((), dtype=torch.long, device=env.device) for name in names
    }
    env._rubik_curriculum = curriculum

  half_sqrt = 2.0**-0.5
  tilt_quats = torch.tensor(
    (
      (half_sqrt, 0.0, half_sqrt, 0.0),
      (half_sqrt, 0.0, -half_sqrt, 0.0),
      (half_sqrt, -half_sqrt, 0.0, 0.0),
      (half_sqrt, half_sqrt, 0.0, 0.0),
      (0.0, 1.0, 0.0, 0.0),
    ),
    device=env.device,
    dtype=default_state.dtype,
  )
  mode_sample = torch.rand(n, device=env.device)
  task_mode = torch.where(
    mode_sample < 0.45, 0, torch.where(mode_sample < 0.90, 1, 2)
  )
  face_ids = torch.randint(4, (n,), device=env.device)
  bottom_probability = torch.where(
    curriculum["flip_stage"] == 0,
    0.0,
    torch.where(curriculum["flip_stage"] == 1, 0.20, 0.40),
  )
  bottom_start = (task_mode == 0) & (
    torch.rand(n, device=env.device) < bottom_probability
  )
  yaw_steps = torch.randint(4, (env_ids.numel(),), device=env.device)
  yaw = yaw_steps.to(default_state.dtype) * (torch.pi / 2.0)
  yaw_quat = torch.stack(
    (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)),
    dim=-1,
  )
  identity_quat = torch.zeros_like(yaw_quat)
  identity_quat[:, 0] = 1.0
  tilt = torch.where(
    (task_mode == 0).unsqueeze(-1),
    torch.where(bottom_start.unsqueeze(-1), tilt_quats[4], tilt_quats[face_ids]),
    identity_quat,
  )
  relative_quat = quat_mul(tilt, yaw_quat)
  cube_quat_w = quat_mul(palm_quat_w, relative_quat)

  asset.write_root_link_pose_to_sim(
    torch.cat((positions, cube_quat_w), dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(
    torch.zeros((env_ids.numel(), 6), device=env.device), env_ids=env_ids
  )

  joint_pos = asset.data.default_joint_pos[env_ids].clone()
  joint_vel = torch.zeros_like(joint_pos)
  quarter_turns = torch.tensor(
    (0.0, torch.pi / 2.0, torch.pi, -torch.pi / 2.0),
    device=env.device,
    dtype=joint_pos.dtype,
  )
  initial_angle = quarter_turns[
    torch.randint(4, (env_ids.numel(),), device=env.device)
  ]
  turn_direction = torch.where(
    torch.rand(env_ids.numel(), device=env.device) < 0.5, -1.0, 1.0
  )
  turn180_probability = torch.where(
    curriculum["turn_stage"] == 0,
    0.0,
    torch.where(curriculum["turn_stage"] == 1, 0.15, 0.40),
  )
  turn_is_180 = (task_mode == 1) & (
    torch.rand(n, device=env.device) < turn180_probability
  )
  turn_step = torch.where(turn_is_180, torch.pi, torch.pi / 2.0)
  turn_target = initial_angle + turn_direction * turn_step
  turn_target = torch.atan2(torch.sin(turn_target), torch.cos(turn_target))
  goal_angle = torch.where(task_mode == 1, turn_target, initial_angle)

  yaw_step = turn_direction * (torch.pi / 12.0)
  root_delta = torch.cat(
    (torch.cos(yaw_step / 2).unsqueeze(-1),
     palm_normal_w * torch.sin(yaw_step / 2).unsqueeze(-1)),
    dim=-1,
  )
  yaw_goal_quat_w = quat_mul(root_delta, cube_quat_w)

  joint_pos[:, int(joint_ids[0])] = initial_angle
  joint_pos[:, linked_joint_ids] = initial_angle.unsqueeze(-1)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

  if not hasattr(env, "_rubik_reset_mode"):
    env._rubik_reset_mode = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    env._rubik_reset_initial_angle = torch.zeros(env.num_envs, device=env.device)
    env._rubik_reset_goal_angle = torch.zeros(env.num_envs, device=env.device)
    env._rubik_reset_yaw_goal_quat = torch.zeros(
      (env.num_envs, 4), device=env.device
    )
    env._rubik_reset_flip_start_dot = torch.zeros(env.num_envs, device=env.device)
    env._rubik_reset_bottom = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    env._rubik_reset_turn180 = torch.zeros_like(env._rubik_reset_bottom)
  env._rubik_reset_mode[env_ids] = task_mode
  env._rubik_reset_initial_angle[env_ids] = initial_angle
  env._rubik_reset_goal_angle[env_ids] = goal_angle
  env._rubik_reset_yaw_goal_quat[env_ids] = yaw_goal_quat_w
  env._rubik_reset_flip_start_dot[env_ids] = torch.where(
    bottom_start, -1.0, 0.0
  )
  env._rubik_reset_bottom[env_ids] = bottom_start
  env._rubik_reset_turn180[env_ids] = turn_is_180
  curriculum["side_trials"] += ((task_mode == 0) & ~bottom_start).sum()
  curriculum["bottom_trials"] += bottom_start.sum()
  curriculum["turn90_trials"] += ((task_mode == 1) & ~turn_is_180).sum()
  curriculum["turn180_trials"] += turn_is_180.sum()
  curriculum["yaw_trials"] += (task_mode == 2).sum()


def reset_object_orientation(
  env,
  env_ids,
  pos_noise: float = 0.01,
  orientation_quaternions: tuple[tuple[float, float, float, float], ...] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> None:
  """Reset object pose with optional discrete orientations and position noise."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  default_state = asset.data.default_root_state[env_ids].clone()

  if orientation_quaternions is None:
    default_state[:, 3:7] = random_quat_uniform(env_ids.numel(), env.device)
  else:
    if not orientation_quaternions:
      raise ValueError("orientation_quaternions must contain at least one quaternion")
    configured_quaternions = torch.as_tensor(
      orientation_quaternions,
      dtype=default_state.dtype,
      device=env.device,
    )
    if configured_quaternions.ndim != 2 or configured_quaternions.shape[1] != 4:
      raise ValueError("orientation_quaternions must have shape (N, 4)")
    choices = torch.randint(
      configured_quaternions.shape[0],
      (env_ids.numel(),),
      device=env.device,
    )
    default_state[:, 3:7] = configured_quaternions[choices]

  positions = default_state[:, :3] + env.scene.env_origins[env_ids]
  if pos_noise > 0.0:
    positions += (
      torch.rand(env_ids.numel(), 3, device=env.device) * 2.0 - 1.0
    ) * pos_noise

  default_state[:, 7:] = 0.0

  pose = torch.cat([positions, default_state[:, 3:7]], dim=-1)
  asset.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)


def reset_disturbance_caches(env, env_ids) -> None:
  """Reset disturbance observation caches to zero."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_reorient_event_state(env)
  assert state.pert_force_dir is not None
  assert state.pert_velocity_cache is not None
  state.pert_force_dir[env_ids] = 0.0
  state.pert_velocity_cache[env_ids] = 0.0


def reset_joint_acc_cache(
  env,
  env_ids,
) -> None:
  """Reset joint acceleration cache on episode reset."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  state = get_reorient_event_state(env)
  if state.prev_joint_vel is None:
    return

  state.prev_joint_vel[env_ids] = 0.0


def apply_velocity_disturbance(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  min_speed: float = 0.05,
  max_speed: float = 0.15,
  warmup_time_s: float = 3.0,
  warmup_frac: float = 0.05,
  rampup_frac: float = 0.80,
  adaptive_curriculum_term: str = "",
) -> None:
  """Apply an instantaneous random velocity to the object on interval trigger."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  linear_progress = get_linear_progress(
    env,
    warmup_frac=warmup_frac,
    rampup_frac=rampup_frac,
    total_steps=_get_total_training_steps(),
  )
  adaptive_scale = 1.0
  if adaptive_curriculum_term:
    adaptive_scale = get_curriculum_value(env, adaptive_curriculum_term, 1.0)

  curr_max_speed = (
    float(min_speed)
    + (float(max_speed) - float(min_speed)) * linear_progress * adaptive_scale
  )
  state = get_reorient_event_state(env)
  assert state.pert_force_dir is not None
  assert state.pert_velocity_cache is not None

  episode_time = env.episode_length_buf[env_ids].float() * env.step_dt
  active_mask = episode_time >= float(warmup_time_s)
  active_local = active_mask.nonzero(as_tuple=False).reshape(-1)

  state.pert_force_dir[env_ids] = 0.0
  state.pert_velocity_cache[env_ids] = 0.0

  if active_local.numel() == 0:
    return

  active_env_ids = env_ids[active_local]

  vel_dir = torch.randn((active_env_ids.numel(), 3), device=env.device)
  vel_dir = vel_dir / vel_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  speed = torch.empty(active_env_ids.numel(), 1, device=env.device).uniform_(
    float(min_speed),
    max(float(min_speed), curr_max_speed),
  )
  velocity = vel_dir * speed

  state.pert_force_dir[active_env_ids] = vel_dir
  state.pert_velocity_cache[active_env_ids] = torch.cat(
    [velocity, torch.zeros_like(velocity)],
    dim=-1,
  )

  asset: Entity = env.scene[asset_cfg.name]
  root_vel = asset.data.root_link_vel_w[active_env_ids].clone()
  root_vel[:, :3] += velocity
  asset.write_root_link_velocity_to_sim(root_vel, env_ids=active_env_ids)
