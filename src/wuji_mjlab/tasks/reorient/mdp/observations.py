# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_from_angle_axis,
  quat_inv,
  quat_mul,
)

from wuji_mjlab.tasks.reorient.reorient_constants import (
  TAG_IN_PALM_POS as _TAG_IN_PALM_POS,
)
from wuji_mjlab.tasks.reorient.reorient_constants import (
  TAG_IN_PALM_QUAT_WXYZ as _TAG_IN_PALM_QUAT_WXYZ,
)
from wuji_mjlab.utils.math import random_quat_uniform

from .cage import read_cage_penalty_counter
from .cube_faces import FACE_NORMALS
from .event_impl.state import get_reorient_event_state
from .rubik_solver import solver_face_frame_quaternion
from .smart_cube_observations import (
  SmartCubeProfile,
  get_smart_cube_state,
  step_imu_model,
  step_move_model,
)
from .vision_observations import (
  get_visual_pose_state,
  step_visual_pose_model,
  vision_stage_parameters,
)

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def _rubik_solver_mode(command) -> bool:
  return bool(getattr(getattr(command, "cfg", None), "solver_mode", False))


def _tag_in_palm_pose(device: torch.device | str, dtype: torch.dtype = torch.float32):
  pos = torch.tensor(_TAG_IN_PALM_POS, device=device, dtype=dtype)
  quat = torch.tensor(_TAG_IN_PALM_QUAT_WXYZ, device=device, dtype=dtype)
  return pos, quat


def _palm_pose_to_tag_pose(palm_pos_w: torch.Tensor, palm_quat_w: torch.Tensor):
  """Compose palm world pose with hardcoded tag-in-palm offset to get tag world pose.

  Args:
    palm_pos_w: (..., 3) palm_link position in world frame.
    palm_quat_w: (..., 4) palm_link orientation in world frame (wxyz).

  Returns:
    tag_pos_w (..., 3), tag_quat_w (..., 4) in world frame.
  """
  tag_in_palm_pos, tag_in_palm_quat = _tag_in_palm_pose(
    palm_pos_w.device, palm_pos_w.dtype
  )
  shape = palm_pos_w.shape[:-1]
  tag_in_palm_pos_b = tag_in_palm_pos.expand(*shape, 3)
  tag_in_palm_quat_b = tag_in_palm_quat.expand(*shape, 4)
  tag_pos_w = palm_pos_w + quat_apply(palm_quat_w, tag_in_palm_pos_b)
  tag_quat_w = quat_mul(palm_quat_w, tag_in_palm_quat_b)
  return tag_pos_w, tag_quat_w


def joint_pos_target_error(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Normalized joint position error: current_normalized - target_normalized."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  center = 0.5 * (soft_limits[..., 0] + soft_limits[..., 1])
  half_range = 0.5 * (soft_limits[..., 1] - soft_limits[..., 0])
  normalized_pos = ((joint_pos - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)

  # Get processed action targets from the joint_pos action term
  processed = env.action_manager.get_term("joint_pos").processed_action
  target = processed[:, asset_cfg.joint_ids]
  normalized_target = ((target - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)

  return normalized_pos - normalized_target


_DEFAULT_PALM_CFG = SceneEntityCfg("robot", body_names=(".*_palm_link",))


def _resolve_palm_body_id(robot: Entity, cfg: SceneEntityCfg) -> int:
  """Resolve palm body index from cfg.

  Manager resolves SceneEntityCfg objects passed via term ``params``, but default
  kwarg values stay unresolved (``body_ids`` remains ``slice(None)``). Fall back to
  ``find_bodies`` so the obs functions work whether or not the caller supplied a
  resolved cfg via params.
  """
  body_ids = cfg.body_ids
  if isinstance(body_ids, slice):
    resolved, _ = robot.find_bodies(cfg.body_names)
    return int(resolved[0])
  return int(body_ids[0])


def cube_pos_in_tag(
  env,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_PALM_CFG,
  injection_prob: float = 0.0,
) -> torch.Tensor:
  """Cube root position expressed in the tag frame.

  Returned shape ``(num_envs, 3)``. The tag frame is reconstructed from
  palm world pose via the hardcoded rigid transform (see ``_TAG_IN_PALM_*``);
  in real deploy this corresponds to the AprilTag mounted on the wrist.
  Compared to the previous ``cube_pos_ref_error`` formulation, no constant
  reference offset is subtracted — the policy sees the cube's absolute
  position in tag frame, which generalizes to randomized cube spawn locations.

  Args:
    injection_prob: probability of replacing per env with a random vector in
      [-0.5, 0.5]^3 (prevents over-reliance on the cube tracking signal).
  """
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  palm_id = _resolve_palm_body_id(robot, robot_cfg)
  palm_pos_w = robot.data.body_link_pose_w[:, palm_id, :3]
  palm_quat_w = robot.data.body_link_pose_w[:, palm_id, 3:7]
  tag_pos_w, tag_quat_w = _palm_pose_to_tag_pose(palm_pos_w, palm_quat_w)

  cube_pos_w = obj.data.root_link_pos_w
  cube_pos_tag = quat_apply_inverse(tag_quat_w, cube_pos_w - tag_pos_w)

  if injection_prob > 0 and env.scene.device != "meta":
    n = cube_pos_tag.shape[0]
    mask = torch.rand(n, 1, device=cube_pos_tag.device) < injection_prob
    rand_pos = torch.empty_like(cube_pos_tag).uniform_(-0.5, 0.5)
    cube_pos_tag = torch.where(mask, rand_pos, cube_pos_tag)

  return cube_pos_tag


def _rubik_cube_quat_in_tag(
  env, *, vision_dr: bool = False, smart_cube_profile: SmartCubeProfile | None = None,
) -> torch.Tensor:
  """Estimated cube root orientation in the wrist-tag frame."""
  command = env.command_manager.get_term("reorient_command")
  obj = command.object
  robot = command.robot
  palm_pos_w = robot.data.body_link_pose_w[:, command.palm_body_id, :3]
  palm_quat_w = robot.data.body_link_pose_w[:, command.palm_body_id, 3:7]
  _, tag_quat_w = _palm_pose_to_tag_pose(palm_pos_w, palm_quat_w)
  cube_quat_tag = quat_mul(quat_inv(tag_quat_w), obj.data.root_link_quat_w)
  if smart_cube_profile is not None:
    env._smart_cube_profile = smart_cube_profile
  if vision_dr and str(env.scene.device) != "meta" and smart_cube_profile is not None:
    from .smart_cube_observations import effective_smart_cube_profile
    cube_quat_tag = step_imu_model(
      cube_quat_tag, get_smart_cube_state(env), effective_smart_cube_profile(env, smart_cube_profile),
      step_id=int(env.common_step_counter), step_dt=env.step_dt,
    )
  elif vision_dr and env.scene.device != "meta":
    cube_quat_tag = step_visual_pose_model(
      cube_quat_tag,
      get_visual_pose_state(env),
      vision_stage_parameters(3),
      step_id=int(env.common_step_counter),
    )
  if _rubik_solver_mode(command):
    cube_quat_tag = quat_mul(
      cube_quat_tag,
      solver_face_frame_quaternion(command.target_face_id, dtype=cube_quat_tag.dtype),
    )
  return cube_quat_tag


def rubik_cube_orientation_6d_in_tag(
  env, vision_dr: bool = False, smart_cube_profile: SmartCubeProfile | None = None,
) -> torch.Tensor:
  quat_tag = _rubik_cube_quat_in_tag(
    env, vision_dr=vision_dr, smart_cube_profile=smart_cube_profile,
  )
  rot = matrix_from_quat(quat_tag)
  return rot.reshape(*rot.shape[:-2], 9)[..., 3:]


def rubik_u_face_angle_sincos(
  env,
  command_name: str = "reorient_command",
  observation_noise_rad: float = 0.05,
) -> torch.Tensor:
  """Face-turn estimate proxy; real deployment supplies the sticker estimate."""
  angle = env.command_manager.get_term(command_name).u_angle
  if observation_noise_rad > 0.0 and env.scene.device != "meta":
    angle = angle + torch.empty_like(angle).uniform_(
      -observation_noise_rad, observation_noise_rad
    )
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def rubik_u_face_goal_sincos(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  angle = env.command_manager.get_term(command_name).u_target_angle
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def rubik_smart_cube_angle_sincos(
  env, command_name: str = "reorient_command", smart_cube_profile: SmartCubeProfile | None = None,
) -> torch.Tensor:
  """Actor-only MOVE estimate; command.u_angle and all critic paths stay continuous."""
  command = env.command_manager.get_term(command_name)
  if smart_cube_profile is not None:
    env._smart_cube_profile = smart_cube_profile
  if str(env.scene.device) == 'meta':
    return torch.zeros((env.num_envs, 2), device=env.device)
  from .smart_cube_observations import effective_smart_cube_profile
  profile = effective_smart_cube_profile(env, env._smart_cube_profile)
  raw = command.object.data.joint_pos[:, command._face_driver_ids]
  signs = torch.tensor((1, -1, 1, -1, 1, -1), device=raw.device, dtype=raw.dtype)
  angles = step_move_model(raw * signs, get_smart_cube_state(env), profile,
    step_id=int(env.common_step_counter))
  face = command.target_face_id if command.cfg.solver_mode else torch.full_like(command.target_face_id, 4)
  angle = angles.gather(1, face.unsqueeze(-1)).squeeze(-1)
  controller = getattr(env, '_smart_cube_curriculum', None)
  if controller is not None and controller.beta < 1:
    continuous = (raw * signs).gather(1, face.unsqueeze(-1)).squeeze(-1)
    angle = (1 - controller.beta) * continuous + controller.beta * angle
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def rubik_face_goal_local(env, command_name: str = "reorient_command") -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  normals = FACE_NORMALS.to(device=env.device, dtype=command.u_target_angle.dtype)
  face_id = (
    torch.full_like(command.target_face_id, 4)
    if _rubik_solver_mode(command)
    else command.target_face_id
  )
  selected = normals[face_id]
  return torch.where((command.phase == 0).unsqueeze(-1), selected, 0.0)


def rubik_root_goal_6d_in_tag(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  palm_pose = command.robot.data.body_link_pose_w[:, command.palm_body_id, :]
  _, tag_quat = _palm_pose_to_tag_pose(palm_pose[:, :3], palm_pose[:, 3:7])
  goal_tag = quat_mul(quat_inv(tag_quat), command.root_target_quat_w)
  rot = matrix_from_quat(goal_tag)
  goal_6d = rot.reshape(*rot.shape[:-2], 9)[..., 3:]
  return torch.where((command.phase == 2).unsqueeze(-1), goal_6d, 0.0)


def _quat_align_vectors(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """Return the shortest quaternion rotating each source vector onto target."""
  source = source / torch.linalg.vector_norm(
    source, dim=-1, keepdim=True
  ).clamp_min(1e-8)
  target = target / torch.linalg.vector_norm(
    target, dim=-1, keepdim=True
  ).clamp_min(1e-8)
  cross = torch.cross(source, target, dim=-1)
  cross_norm = torch.linalg.vector_norm(cross, dim=-1)
  dot = (source * target).sum(dim=-1).clamp(-1.0, 1.0)
  regular_axis = cross / cross_norm.unsqueeze(-1).clamp_min(1e-8)

  least_aligned = source.abs().argmin(dim=-1)
  basis = torch.nn.functional.one_hot(least_aligned, num_classes=3).to(source.dtype)
  opposite_axis = torch.cross(source, basis, dim=-1)
  opposite_axis = opposite_axis / torch.linalg.vector_norm(
    opposite_axis, dim=-1, keepdim=True
  ).clamp_min(1e-8)
  opposite = dot < -1.0 + 1e-6
  axis = torch.where(opposite.unsqueeze(-1), opposite_axis, regular_axis)
  angle = torch.atan2(cross_norm, dot)
  angle = torch.where(opposite, torch.full_like(angle, torch.pi), angle)
  aligned = quat_from_angle_axis(angle, axis)
  identity = torch.zeros_like(aligned)
  identity[..., 0] = 1.0
  return torch.where((dot > 1.0 - 1e-6).unsqueeze(-1), identity, aligned)


def rubik_compat_cube_orientation_error_6d(
  env,
  command_name: str = "reorient_command",
  vision_dr: bool = True,
  smart_cube_profile: SmartCubeProfile | None = None,
) -> torch.Tensor:
  """Source-compatible 6D body-goal error used only by the Rubik transfer actor."""
  command = env.command_manager.get_term(command_name)
  robot = command.robot
  palm_pose_w = robot.data.body_link_pose_w[:, command.palm_body_id, :]
  palm_quat_w = palm_pose_w[:, 3:7]
  _, tag_quat_w = _palm_pose_to_tag_pose(palm_pose_w[:, :3], palm_quat_w)
  cube_quat_tag = _rubik_cube_quat_in_tag(
    env, vision_dr=vision_dr, smart_cube_profile=smart_cube_profile,
  )

  palm_up_local = torch.zeros_like(palm_pose_w[:, :3])
  palm_up_local[:, 2] = 1.0
  palm_normal_tag = quat_apply_inverse(
    tag_quat_w, quat_apply(palm_quat_w, palm_up_local)
  )
  face_normals = FACE_NORMALS.to(
    device=env.device, dtype=cube_quat_tag.dtype
  )
  requested_face = (
    torch.full_like(command.target_face_id, 4)
    if _rubik_solver_mode(command)
    else torch.where(
      command.phase == 0,
      command.target_face_id,
      torch.full_like(command.target_face_id, 4),
    )
  )
  local_normal = face_normals[requested_face]
  current_normal_tag = quat_apply(cube_quat_tag, local_normal)
  align = _quat_align_vectors(current_normal_tag, palm_normal_tag)
  upright_goal_tag = quat_mul(align, cube_quat_tag)
  yaw_goal_tag = quat_mul(quat_inv(tag_quat_w), command.root_target_quat_w)
  goal_quat_tag = torch.where(
    (command.phase == 2).unsqueeze(-1), yaw_goal_tag, upright_goal_tag
  )
  error_quat_tag = quat_mul(cube_quat_tag, quat_inv(goal_quat_tag))
  rotation = matrix_from_quat(error_quat_tag)
  return rotation.reshape(*rotation.shape[:-2], 9)[..., 3:]


def rubik_task_phase(env, command_name: str = "reorient_command") -> torch.Tensor:
  phase = env.command_manager.get_term(command_name).phase
  return torch.nn.functional.one_hot(phase, num_classes=3).float()


def rubik_state_privileged(env, command_name: str = "reorient_command") -> torch.Tensor:
  orientation = rubik_cube_orientation_6d_in_tag(env, vision_dr=False)
  command = env.command_manager.get_term(command_name)
  angle = command.u_angle
  phase = rubik_task_phase(env, command_name)
  angle_features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
  goal_features = torch.stack(
    (torch.sin(command.u_target_angle), torch.cos(command.u_target_angle)), dim=-1
  )
  face_goal = rubik_face_goal_local(env, command_name)
  root_goal = rubik_root_goal_6d_in_tag(env, command_name)
  return torch.cat(
    (orientation, angle_features, goal_features, face_goal, root_goal, phase), dim=-1
  )


def rubik_cube_position_stability_privileged(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  """Critic-only cube position and frame-to-frame translation change."""
  command = env.command_manager.get_term(command_name)
  palm = command.robot.data.body_link_pose_w[:, command.palm_body_id]
  cube_pos_palm = quat_apply_inverse(
    palm[:, 3:7], command.object.data.root_link_pos_w - palm[:, :3]
  )
  return torch.cat((cube_pos_palm, command.cube_position_delta.unsqueeze(-1)), dim=-1)


def rubik_smart_move_residual_privileged(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  """Critic-only post-MOVE residual and pending flag."""
  command = env.command_manager.get_term(command_name)
  command.update_smart_move_pending()
  residual = (command.u_angle_error / torch.pi).unsqueeze(-1)
  pending = command.smart_move_pending.to(residual.dtype).unsqueeze(-1)
  return torch.cat((residual, pending), dim=-1)


def goal_rot_err_6d(
  env,
  command_name: str,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_PALM_CFG,
  injection_prob: float = 0.0,
  vision_dr: bool = False,
  vision_stage_override: int | None = None,
) -> torch.Tensor:
  """6D rotation error in tag frame: ``mat(cube_tag * goal_tag^-1)[3:9]``.

  Matches the deploy obs where both cube and
  goal quaternions are received in tag frame and the residual rotation is taken
  as ``cube * goal^-1``. In sim, both quats are first transformed from world to
  tag frame using the hardcoded palm→tag rigid transform.

  Args:
    injection_prob: probability of replacing with random orientation error per env
      (prevents policy from over-relying on cube tracking signal).
  """
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  palm_id = _resolve_palm_body_id(robot, robot_cfg)
  palm_pos_w = robot.data.body_link_pose_w[:, palm_id, :3]
  palm_quat_w = robot.data.body_link_pose_w[:, palm_id, 3:7]
  _, tag_quat_w = _palm_pose_to_tag_pose(palm_pos_w, palm_quat_w)

  tag_quat_inv = quat_inv(tag_quat_w)
  cube_in_tag = quat_mul(tag_quat_inv, obj.data.root_link_quat_w)
  goal_in_tag = quat_mul(tag_quat_inv, command.goal_quat)

  if vision_dr and env.scene.device != "meta":
    stage = (
      int(vision_stage_override)
      if vision_stage_override is not None
      else int(getattr(env, "_flip_curriculum_stage", 0))
    )
    cube_in_tag = step_visual_pose_model(
      cube_in_tag,
      get_visual_pose_state(env),
      vision_stage_parameters(stage),
      step_id=int(env.common_step_counter),
    )

  q_err_tag = quat_mul(cube_in_tag, quat_inv(goal_in_tag))

  rot = matrix_from_quat(q_err_tag)
  ori_error = rot.reshape(*rot.shape[:-2], 9)[..., 3:]

  if injection_prob > 0 and env.scene.device != "meta":
    n = ori_error.shape[0]
    mask = torch.rand(n, 1, device=ori_error.device) < injection_prob
    rand_quat = random_quat_uniform(n, device=ori_error.device)
    rand_rot = matrix_from_quat(rand_quat)
    rand_error = rand_rot.reshape(*rand_rot.shape[:-2], 9)[..., 3:]
    ori_error = torch.where(mask, rand_error, ori_error)

  return ori_error


def palm_rot_6d_w(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Palm (robot root) orientation as 6D rotation in world frame.

  Privileged critic obs giving full 3-DOF wrist orientation awareness under
  arbitrary wrist DR — disambiguates rotation around the gravity axis, which
  ``projected_gravity_b`` (only 2 DOF) cannot. 6D representation is the last
  2 rows of the rotation matrix (matches ``goal_rot_err_6d`` convention) and
  is more numerically stable for NN inputs than raw quaternions.
  """
  asset: Entity = env.scene[asset_cfg.name]
  palm_quat_w = asset.data.root_link_quat_w
  rot = matrix_from_quat(palm_quat_w)
  return rot.reshape(*rot.shape[:-2], 9)[..., 3:]


def previous_raw_action(env) -> torch.Tensor:
  """Previous raw policy action before any environment-side processing."""
  return env.action_manager.prev_action


def command_state_progress(
  env,
  command_name: str = "reorient_command",
) -> torch.Tensor:
  """Critic-only: state machine progress and episode context (4D).

  Returns [hold_progress, window_progress, episode_progress, goal_reach_count].
  Not manually normalized — critic obs_normalization handles scaling.
  """
  command = env.command_manager.get_term(command_name)
  hold_progress = command.hold_counter.float() / max(command.cfg.success_hold_steps, 1)
  window_progress = command.reward_window_timer.float() / max(
    command.cfg.goal_switch_delay, 1
  )
  episode_progress = env.episode_length_buf.float() / env.max_episode_length
  goal_count = command.goal_reach_count.float()
  return torch.stack(
    [hold_progress, window_progress, episode_progress, goal_count], dim=-1
  )


def cage_counter_progress(env, max_outside_steps: int = 10) -> torch.Tensor:
  """Critic-only: soft cage penalty counter progress (N, 1).

  Reads env._cage_penalty_counter set by CageEscapePenalty each step.
  Gradual decay when inside (not instant reset), so counter > 0 can
  mean "recently outside" even if currently inside.
  """
  counter = read_cage_penalty_counter(env)
  if counter is None:
    return torch.zeros((env.num_envs, 1), device=env.device)
  return (counter.float() / max(max_outside_steps, 1)).clamp(max=1.0).unsqueeze(-1)


def perturbation_direction(env) -> torch.Tensor:
  """Force perturbation direction (3D, MJX-aligned).

  Returns (num_envs, 3). Zeros if disturbance wrench has not been initialized.
  """
  return get_reorient_event_state(env).pert_force_dir


def perturbation_velocity(env) -> torch.Tensor:
  """Velocity disturbance impulse: concat([linvel(3), zeros(3)]).

  Returns (num_envs, 6). Zeros if disturbance has not been initialized.
  """
  return get_reorient_event_state(env).pert_velocity_cache


def joint_pos_limit_normalized(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Joint positions normalized by soft joint limits to [-1, 1]."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  center = 0.5 * (soft_limits[..., 0] + soft_limits[..., 1])
  half_range = 0.5 * (soft_limits[..., 1] - soft_limits[..., 0])
  return ((joint_pos - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)


def joint_pos_limit_normalized_with_dr(
  env, asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Joint observations with uniform physical-angle noise, from +/-1 to +/-3deg."""
  from .smart_cube_curriculum import get_smart_cube_curriculum
  normalized = joint_pos_limit_normalized(env, asset_cfg)
  limits = env.scene[asset_cfg.name].data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  half_range = .5 * (limits[..., 1] - limits[..., 0])
  controller = get_smart_cube_curriculum(env)
  if getattr(getattr(controller, 'cfg', None), 'realistic_dr', False):
    from .smart_cube_curriculum import realistic_parameter_progress, realistic_stage_progress
    progress = (realistic_parameter_progress(controller.realistic_progress)
      if getattr(controller.cfg, 'realistic_success_gated', False) else
      realistic_stage_progress(controller.realistic_steps)[0])
    bound = math.radians(.2 + .3 * progress)
  else:
    bound = math.radians(1 + 2 * controller.joint_dr_progress)
  noise = torch.empty_like(normalized).uniform_(-bound, bound)
  env._joint_dr_observation_noise = noise / (half_range + 1e-6)
  return normalized + env._joint_dr_observation_noise


def joint_pos_target_error_with_dr(env, asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG):
  """Reuse the measurement sampled by the preceding noisy_joint_angles term."""
  return joint_pos_target_error(env, asset_cfg) + env._joint_dr_observation_noise


def root_lin_vel_w(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w


def root_ang_vel_w(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_w


def body_pos_rel(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  offset: tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  offset_tensor = torch.tensor(offset, device=asset.data.body_link_pos_w.device)
  return (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    - env.scene.env_origins.unsqueeze(1)
    - offset_tensor
  ).flatten(start_dim=1)


def dr_params_privileged(
  env,
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
  """Critic-only: DR parameter scale factors (6D, no history).

  Returns [friction_scale, object_mass_scale, kp_scale, kd_scale,
           dof_damping_scale, object_size_scale] as ratios to defaults.
  """
  if env.scene.device == "meta":
    return torch.ones((env.num_envs, 6), device=env.device)

  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]
  n = env.num_envs
  env_ids = torch.arange(n, device=env.device, dtype=torch.long)

  # --- friction: geom_friction[env, geom, 0] (tangential) ---
  robot_geom_ids = robot.indexing.geom_ids[robot_cfg.geom_ids].long()
  env_g, geom_g = torch.meshgrid(env_ids, robot_geom_ids, indexing="ij")
  cur_friction = env.sim.model.geom_friction[env_g, geom_g, 0]
  def_friction = env.sim.get_default_field("geom_friction")[robot_geom_ids, 0]
  friction_ratio = (cur_friction / def_friction.clamp_min(1e-8)).mean(
    dim=-1, keepdim=True
  )

  # --- object_mass: body_mass[env, body] ---
  obj_body_ids = obj.indexing.body_ids[object_cfg.body_ids].long()
  env_b, body_b = torch.meshgrid(env_ids, obj_body_ids, indexing="ij")
  cur_mass = env.sim.model.body_mass[env_b, body_b]
  def_mass = env.sim.get_default_field("body_mass")[obj_body_ids]
  mass_ratio = (cur_mass / def_mass.clamp_min(1e-8)).mean(dim=-1, keepdim=True)

  # --- kp: actuator_gainprm[env, actuator, 0] ---
  robot_act_ids = robot.indexing.ctrl_ids[robot_cfg.actuator_ids].long()
  env_a, act_a = torch.meshgrid(env_ids, robot_act_ids, indexing="ij")
  cur_kp = env.sim.model.actuator_gainprm[env_a, act_a, 0]
  def_kp = env.sim.get_default_field("actuator_gainprm")[robot_act_ids, 0]
  kp_ratio = (cur_kp / def_kp.clamp_min(1e-8)).mean(dim=-1, keepdim=True)

  # --- kd: actuator_biasprm[env, actuator, 2] ---
  # biasprm[:,2] is negative (e.g. -0.1); clamp(max=-1e-8) keeps it negative
  # and prevents divide-by-zero without flipping sign.
  cur_kd = env.sim.model.actuator_biasprm[env_a, act_a, 2]
  def_kd = env.sim.get_default_field("actuator_biasprm")[robot_act_ids, 2]
  kd_ratio = (cur_kd / def_kd.clamp(max=-1e-8)).mean(dim=-1, keepdim=True)

  # --- dof_damping: dof_damping[env, joint_v_adr] ---
  robot_jnt_ids = robot.indexing.joint_v_adr[robot_cfg.joint_ids].long()
  env_j, jnt_j = torch.meshgrid(env_ids, robot_jnt_ids, indexing="ij")
  cur_damp = env.sim.model.dof_damping[env_j, jnt_j]
  def_damp = env.sim.get_default_field("dof_damping")[robot_jnt_ids]
  damp_ratio = (cur_damp / def_damp.clamp_min(1e-8)).mean(dim=-1, keepdim=True)

  # --- object_size: geom_size[env, geom] ---
  obj_geom_ids = obj.indexing.geom_ids[object_cfg.geom_ids].long()
  env_og, geom_og = torch.meshgrid(env_ids, obj_geom_ids, indexing="ij")
  cur_size = env.sim.model.geom_size[env_og, geom_og]
  def_size = env.sim.get_default_field("geom_size")[obj_geom_ids]
  # Flatten all geom×size dims, mean to scalar per env → always (N, 1)
  size_ratio = cur_size / def_size.clamp_min(1e-8)
  size_ratio = size_ratio.reshape(n, -1).mean(dim=-1, keepdim=True)

  return torch.cat(
    [friction_ratio, mass_ratio, kp_ratio, kd_ratio, damp_ratio, size_ratio], dim=-1
  )
