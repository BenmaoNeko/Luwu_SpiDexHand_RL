# SPDX-License-Identifier: Apache-2.0
"""Full-unlocked Rubik transfer command and its simulation-only checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

from .event_utils import resolve_env_ids
from .rubik_command import RubikUFaceCommand, RubikUFaceCommandCfg
from .rubik_solver import (
  apply_face_move,
  apply_face_turn_angle,
  build_solver_joint_state,
  face_id_for_move,
  load_solver_plan_pool,
)
from .smart_cube_curriculum import SmartCubeCurriculumCfg

FULL_CUBE_FLIP = 0
FULL_CUBE_TURN = 1
FULL_CUBE_SOLVE = 2
_FACE_TOKENS = ("R", "L", "B", "F", "U", "D")
_BLOCK_NAMES = (
  "neg_x_neg_y",
  "neg_x_pos_y",
  "pos_x_neg_y",
  "pos_x_pos_y",
  "neg_x_neg_z",
  "neg_x_pos_z",
  "pos_x_neg_z",
  "pos_x_pos_z",
  "neg_y_neg_z",
  "neg_y_pos_z",
  "pos_y_neg_z",
  "pos_y_pos_z",
  "neg_x_neg_y_neg_z",
  "neg_x_neg_y_pos_z",
  "neg_x_pos_y_neg_z",
  "neg_x_pos_y_pos_z",
  "pos_x_neg_y_neg_z",
  "pos_x_neg_y_pos_z",
  "pos_x_pos_y_neg_z",
  "pos_x_pos_y_pos_z",
)
_DRIVER_NAMES = (
  "cubelet:driver:pos_x",
  "cubelet:driver:neg_x",
  "cubelet:driver:pos_y",
  "cubelet:driver:neg_y",
  "cubelet:driver:pos_z",
  "cubelet:driver:neg_z",
)
_CENTER_BODY_NAMES = tuple(
  name.replace("cubelet:driver:", "cubelet:") for name in _DRIVER_NAMES
)
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot", body_names=("right_palm_link",))


def assign_full_cube_task_groups(
  num_envs: int, *, device: torch.device | str = "cpu",
  fractions: tuple[float, float, float] | None = None
) -> torch.Tensor:
  """Assign stable per-env slots in exact 25/25/50 proportions."""
  if num_envs < 1:
    raise ValueError("Full-cube solver transfer requires at least one environment")
  if fractions is not None:
    if len(fractions) != 3 or any(not math.isfinite(x) or x < 0 for x in fractions) or not math.isclose(sum(fractions), 1.):
      raise ValueError("Task fractions must be three nonnegative values summing to one")
    counts = [math.floor(num_envs * x) for x in fractions]
    order = sorted(range(3), key=lambda i: num_envs * fractions[i] - counts[i], reverse=True)
    for i in order[:num_envs - sum(counts)]:
      counts[i] += 1
    return torch.repeat_interleave(torch.arange(3, device=device), torch.tensor(counts, device=device))
  if num_envs % 4:
    raise ValueError("Full-cube solver transfer requires a multiple of 4 environments")
  slot = torch.arange(num_envs, device=device) % 4
  return torch.where(slot < 2, slot, FULL_CUBE_SOLVE)


def _matrix_from_xyz(angles: torch.Tensor) -> torch.Tensor:
  """Robogym's XYZ Euler convention, expressed as batched torch matrices."""
  x, y, z = angles.unbind(dim=-1)
  zero = torch.zeros_like(x)
  one = torch.ones_like(x)
  cx, sx = torch.cos(x), torch.sin(x)
  cy, sy = torch.cos(y), torch.sin(y)
  cz, sz = torch.cos(z), torch.sin(z)
  rx = torch.stack((one, zero, zero, zero, cx, -sx, zero, sx, cx), dim=-1).reshape(
    *angles.shape[:-1], 3, 3
  )
  ry = torch.stack((cy, zero, sy, zero, one, zero, -sy, zero, cy), dim=-1).reshape(
    *angles.shape[:-1], 3, 3
  )
  rz = torch.stack((cz, -sz, zero, sz, cz, zero, zero, zero, one), dim=-1).reshape(
    *angles.shape[:-1], 3, 3
  )
  return rx @ ry @ rz


def _interpolate_joint_path(
  start: torch.Tensor, goal: torch.Tensor, fraction: torch.Tensor
) -> torch.Tensor:
  delta = goal - start
  delta = torch.atan2(torch.sin(delta), torch.cos(delta))
  return start + delta * fraction.unsqueeze(-1)


def _face_turn_path_error(
  actual: torch.Tensor,
  start: torch.Tensor,
  goal: torch.Tensor,
  face_ids: torch.Tensor,
  fraction: torch.Tensor,
  *,
  piece_joint_indices: torch.Tensor,
  driver_joint_indices: torch.Tensor,
  piece_home_coords: torch.Tensor,
) -> torch.Tensor:
  """Compare cube state with the physical SO(3) path for the selected layer."""
  batch = torch.arange(actual.shape[0], device=actual.device)
  start_rotation = _matrix_from_xyz(start[:, piece_joint_indices].reshape(-1, 20, 3))
  actual_rotation = _matrix_from_xyz(actual[:, piece_joint_indices].reshape(-1, 20, 3))
  home = piece_home_coords.to(device=actual.device, dtype=actual.dtype)
  positions = torch.einsum("nbij,bj->nbi", start_rotation, home)
  face_ids = face_ids.long()
  axes = face_ids // 2
  sides = torch.where(face_ids % 2 == 0, 1.0, -1.0).to(actual.dtype)
  in_layer = positions.gather(2, axes[:, None, None].expand(-1, 20, 1)).squeeze(-1)
  moving = in_layer * sides[:, None] >= 0.5

  driver_delta = (
    goal[batch, driver_joint_indices[face_ids]]
    - start[batch, driver_joint_indices[face_ids]]
  )
  driver_delta = torch.atan2(torch.sin(driver_delta), torch.cos(driver_delta))
  angle = driver_delta * fraction
  axis = torch.nn.functional.one_hot(axes, num_classes=3).to(actual.dtype)
  x, y, z = axis.unbind(dim=-1)
  zero = torch.zeros_like(x)
  skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)
  identity = torch.eye(3, device=actual.device, dtype=actual.dtype).expand(
    actual.shape[0], -1, -1
  )
  turn = (
    identity
    + torch.sin(angle)[:, None, None] * skew
    + (1.0 - torch.cos(angle))[:, None, None] * (skew @ skew)
  )
  turned = turn[:, None] @ start_rotation
  expected_rotation = torch.where(moving[:, :, None, None], turned, start_rotation)
  relative = expected_rotation.transpose(-1, -2) @ actual_rotation
  piece_angles = torch.acos(
    ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
  )

  expected_driver = start[:, driver_joint_indices].clone()
  expected_driver += (
    torch.nn.functional.one_hot(face_ids, num_classes=6).to(actual.dtype)
    * angle[:, None]
  )
  driver_delta = actual[:, driver_joint_indices] - expected_driver
  driver_angles = torch.atan2(torch.sin(driver_delta), torch.cos(driver_delta)).abs()
  return torch.stack((piece_angles.amax(dim=-1), driver_angles.amax(dim=-1)), dim=-1)


def _cubie_occupancy_valid(
  joint_pos: torch.Tensor,
  piece_joint_indices: torch.Tensor,
  piece_home_coords: torch.Tensor,
) -> torch.Tensor:
  rotations = _matrix_from_xyz(joint_pos[:, piece_joint_indices].reshape(-1, 20, 3))
  home = piece_home_coords.to(device=joint_pos.device, dtype=joint_pos.dtype)
  positions = torch.einsum("nbij,bj->nbi", rotations, home)
  rounded = positions.round()
  category_matches = rounded.ne(0).sum(dim=-1) == home.ne(0).sum(dim=-1).unsqueeze(0)
  on_grid = (positions - rounded).norm(dim=-1) <= 0.40
  in_cube = rounded.abs().amax(dim=-1) <= 1.0
  edge_matches = (rounded[:, :12, None] == home[None, None, :12]).all(dim=-1)
  corner_matches = (rounded[:, 12:, None] == home[None, None, 12:]).all(dim=-1)
  edge_perm = edge_matches.to(torch.int64).argmax(dim=-1)
  corner_perm = corner_matches.to(torch.int64).argmax(dim=-1)
  edge_order = torch.triu(
    torch.ones((12, 12), dtype=torch.bool, device=joint_pos.device), 1
  )
  corner_order = torch.triu(
    torch.ones((8, 8), dtype=torch.bool, device=joint_pos.device), 1
  )
  edge_parity = ((edge_perm[:, :, None] > edge_perm[:, None, :]) & edge_order).sum(
    (1, 2)
  ) % 2
  corner_parity = (
    (corner_perm[:, :, None] > corner_perm[:, None, :]) & corner_order
  ).sum((1, 2)) % 2
  edge_occupancy = edge_matches.sum(dim=-1).eq(1).all(dim=-1) & edge_matches.sum(
    dim=1
  ).eq(1).all(dim=-1)
  corner_occupancy = corner_matches.sum(dim=-1).eq(1).all(dim=-1) & corner_matches.sum(
    dim=1
  ).eq(1).all(dim=-1)
  return (
    category_matches.all(dim=-1)
    & on_grid.all(dim=-1)
    & in_cube.all(dim=-1)
    & edge_occupancy
    & corner_occupancy
    & (edge_parity == corner_parity)
  )


def _rotation_errors(
  actual_joint_pos: torch.Tensor,
  expected_joint_pos: torch.Tensor,
  joint_names: tuple[str, ...],
  *,
  piece_joint_indices: torch.Tensor | None = None,
  driver_joint_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if actual_joint_pos.ndim == 1:
    actual_joint_pos = actual_joint_pos.unsqueeze(0)
  if expected_joint_pos.ndim == 1:
    expected_joint_pos = expected_joint_pos.unsqueeze(0)
  if actual_joint_pos.shape != expected_joint_pos.shape:
    raise ValueError("Actual and expected Rubik joint state shapes must match")
  if actual_joint_pos.shape[-1] != len(joint_names):
    raise ValueError("Rubik joint_names do not match the joint state width")
  if piece_joint_indices is None or driver_joint_indices is None:
    index = {name: i for i, name in enumerate(joint_names)}
    try:
      piece_joint_indices = torch.tensor(
        [
          [index[f"cubelet:rot{axis}:{piece}"] for axis in "xyz"]
          for piece in _BLOCK_NAMES
        ],
        device=actual_joint_pos.device,
      )
      driver_joint_indices = torch.tensor(
        [index[name] for name in _DRIVER_NAMES], device=actual_joint_pos.device
      )
    except KeyError as error:
      raise ValueError(f"Full-cube state is missing {error.args[0]}") from error
  actual_rot = _matrix_from_xyz(actual_joint_pos[:, piece_joint_indices])
  expected_rot = _matrix_from_xyz(expected_joint_pos[:, piece_joint_indices])
  rotations = expected_rot.transpose(-1, -2) @ actual_rot
  cosine = ((rotations.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
  max_piece_error = torch.acos(cosine).amax(dim=1)
  delta = (
    actual_joint_pos[:, driver_joint_indices]
    - expected_joint_pos[:, driver_joint_indices]
  )
  driver_error = torch.atan2(torch.sin(delta), torch.cos(delta)).abs().amax(dim=1)
  return rotations, torch.stack((max_piece_error, driver_error), dim=-1)


def internal_state_error_features(
  actual_joint_pos: torch.Tensor,
  expected_joint_pos: torch.Tensor,
  joint_names: tuple[str, ...],
  *,
  piece_joint_indices: torch.Tensor | None = None,
  driver_joint_indices: torch.Tensor | None = None,
) -> torch.Tensor:
  """Return 120 piece SO(3) six-d errors and 12 periodic driver errors."""
  if actual_joint_pos.ndim == 1:
    actual_joint_pos = actual_joint_pos.unsqueeze(0)
  if expected_joint_pos.ndim == 1:
    expected_joint_pos = expected_joint_pos.unsqueeze(0)
  rotations, _ = _rotation_errors(
    actual_joint_pos,
    expected_joint_pos,
    joint_names,
    piece_joint_indices=piece_joint_indices,
    driver_joint_indices=driver_joint_indices,
  )
  identity6 = torch.tensor(
    (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    device=actual_joint_pos.device,
    dtype=actual_joint_pos.dtype,
  )
  piece_error = rotations[..., :, :2].reshape(actual_joint_pos.shape[0], 20, 6)
  piece_error = piece_error - identity6
  if driver_joint_indices is None:
    index = {name: i for i, name in enumerate(joint_names)}
    try:
      driver_joint_indices = torch.tensor(
        [index[name] for name in _DRIVER_NAMES], device=actual_joint_pos.device
      )
    except KeyError as error:
      raise ValueError(f"Full-cube state is missing {error.args[0]}") from error
  driver_delta = (
    actual_joint_pos[:, driver_joint_indices]
    - expected_joint_pos[:, driver_joint_indices]
  )
  driver_error = torch.stack(
    (torch.sin(driver_delta), torch.cos(driver_delta) - 1.0), dim=-1
  ).reshape(actual_joint_pos.shape[0], 12)
  return torch.cat(
    (piece_error.reshape(actual_joint_pos.shape[0], 120), driver_error), dim=-1
  )


def palm_centered_cube_position(
  palm_pose: torch.Tensor,
  palm_clearance_m: float = 0.0,
) -> torch.Tensor:
  """Place this asset's cube bottom on the fixed palm support pad.

  The cylinder top is at local Z=96 mm, with a 0.5 mm support pad;
  the current cube mesh half extent is 28.55 mm. Face-aligned reset
  orientations all have the same support extent.
  """
  offset = torch.zeros_like(palm_pose[:, :3])
  offset[:, 2] = 0.068 + 0.028 + 0.0005 + 0.02855 + palm_clearance_m
  return palm_pose[:, :3] + quat_apply(palm_pose[:, 3:7], offset)


def reset_rubik_full_cube(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  palm_clearance_m: float = 0.0,
) -> None:
  """Reset to one verified Node state without locking any internal hinge."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return
  if not hasattr(env, "_rubik_full_pool_joint_states"):
    raise RuntimeError("Full-cube command must load its plan pool before reset")
  asset: Entity = env.scene[asset_cfg.name]
  robot: Entity = env.scene[robot_cfg.name]
  palm_ids, _ = robot.find_bodies(robot_cfg.body_names)
  if len(palm_ids) != 1:
    raise ValueError("Full-cube Rubik reset requires exactly one palm body")

  group = env._rubik_full_task_groups[env_ids]
  command = env.command_manager.get_term('reorient_command')
  plan_count = env._rubik_full_pool_joint_states.shape[0]
  plan_ids = torch.randint(plan_count, (env_ids.numel(),), device=env.device)
  initial_top = torch.randint(6, (env_ids.numel(),), device=env.device)
  initial_top[group == FULL_CUBE_TURN] = 4
  flip_ids = group == FULL_CUBE_FLIP
  initial_top[flip_ids] = torch.randint(4, (int(flip_ids.sum()),), device=env.device)
  bottom_probability = (0.0, 0.20, 0.40)[int(env._rubik_curriculum["flip_stage"])]
  bottom = flip_ids & (torch.rand(env_ids.numel(), device=env.device) < bottom_probability)
  initial_top[bottom] = 5
  if getattr(command.cfg, 'flip_specialization', False):
    initial_top = torch.randint(6, (env_ids.numel(),), device=env.device)
  flip_offset = torch.randint(1, 6, (env_ids.numel(),), device=env.device)
  target_faces = (initial_top + flip_offset) % 6
  target_faces[group == FULL_CUBE_TURN] = 4
  solve_ids = group == FULL_CUBE_SOLVE
  if solve_ids.any():
    plan_face_ids = env._rubik_full_plan_first_face
    target_faces[solve_ids] = plan_face_ids[plan_ids[solve_ids]]

  # Root reset writes mocap poses; refresh derived body poses before reading.
  # Otherwise first-reset replicas still share the template palm position.
  env.sim.forward()
  palm_pose = robot.data.body_link_pose_w[env_ids, int(palm_ids[0])]
  # Follow the current palm frame instead of the old arm-calibrated world pose.
  position = palm_centered_cube_position(palm_pose, palm_clearance_m)
  face_quats = env._rubik_full_face_quats[initial_top]
  yaw_index = torch.randint(4, (env_ids.numel(),), device=env.device)
  yaw_half = yaw_index.to(palm_pose.dtype) * (math.pi / 4.0)
  yaw_quat = torch.stack(
    (
      torch.cos(yaw_half),
      torch.zeros_like(yaw_half),
      torch.zeros_like(yaw_half),
      torch.sin(yaw_half),
    ),
    dim=-1,
  )
  root_quat = quat_mul(palm_pose[:, 3:7], quat_mul(yaw_quat, quat_inv(face_quats)))
  asset.write_root_link_pose_to_sim(
    torch.cat((position, root_quat), dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(
    torch.zeros((env_ids.numel(), 6), device=env.device), env_ids
  )
  joint_pos = asset.data.default_joint_pos[env_ids].clone()
  joint_pos[solve_ids] = env._rubik_full_pool_joint_states[plan_ids[solve_ids]]
  command = env.command_manager.get_term("reorient_command")
  if getattr(command.cfg, 'flip_specialization', False):
    scrambled = flip_ids & (torch.rand(env_ids.numel(), device=env.device)
                            < command.cfg.flip_scramble_probability)
    joint_pos[scrambled] = env._rubik_full_pool_joint_states[plan_ids[scrambled]]
  if command.cfg.partial_turn_start_probability > 0:
    from .turn_specialization import sample_turn_starts
    if not hasattr(env, '_rubik_partial_turn'):
      env._rubik_partial_turn = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
      env._rubik_partial_goal_pending = torch.zeros_like(env._rubik_partial_turn)
      env._rubik_partial_goal = torch.zeros(env.num_envs, device=env.device)
    env._rubik_partial_turn[env_ids] = False
    env._rubik_partial_goal_pending[env_ids] = False
    rows = (group == FULL_CUBE_TURN).nonzero().flatten()
    start, goal, partial = sample_turn_starts(rows.numel(), env.device, command.cfg.partial_turn_start_probability)
    ids = env_ids[rows]
    env._rubik_partial_turn[ids] = partial
    env._rubik_partial_goal_pending[ids] = partial
    env._rubik_partial_goal[ids] = goal
    for row, angle, is_partial in zip(rows.tolist(), start.tolist(), partial.tolist(), strict=True):
      if is_partial:
        state = dict(zip(command.joint_names, joint_pos[row].tolist(), strict=True))
        state = apply_face_turn_angle(state, 4, angle, command.joint_names)
        joint_pos[row] = torch.tensor([state[name] for name in command.joint_names], device=env.device)
  asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

  env._rubik_full_plan_index[env_ids] = plan_ids
  env._rubik_full_target_face[env_ids] = target_faces
  env._rubik_reset_mode[env_ids] = (group == FULL_CUBE_TURN).long()
  env._rubik_reset_initial_angle[env_ids] = 0.0
  env._rubik_reset_goal_angle[env_ids] = 0.0
  env._rubik_reset_yaw_goal_quat[env_ids] = root_quat
  env._rubik_reset_flip_start_dot[env_ids] = 0.0
  env._rubik_reset_bottom[env_ids] = False
  env._rubik_reset_turn180[env_ids] = False


class RubikFullCubeCommand(RubikUFaceCommand):
  """Independent per-env mixed-task scheduler over the existing Rubik command."""

  cfg: RubikFullCubeCommandCfg

  def __init__(self, cfg: RubikFullCubeCommandCfg, env):
    if not cfg.solver_mode:
      raise ValueError("Full-cube command requires solver_mode for arbitrary faces")
    super().__init__(cfg, env)
    self.joint_names = tuple(self.object.joint_names)
    if len(self.joint_names) != 66:
      raise ValueError(
        f"Full-cube Rubik scene must expose 66 hinge joints, got {len(self.joint_names)}"
      )
    joint_index = {name: i for i, name in enumerate(self.joint_names)}
    self._driver_joint_indices = torch.tensor(
      [joint_index[name] for name in _DRIVER_NAMES], device=self.device
    )
    self._piece_joint_indices = torch.tensor(
      [
        [joint_index[f"cubelet:rot{axis}:{piece}"] for axis in "xyz"]
        for piece in _BLOCK_NAMES
      ],
      device=self.device,
    )
    self._block_joint_indices = self._piece_joint_indices.reshape(-1)
    self._piece_home_coords = torch.tensor(
      [
        [
          (1.0 if f"pos_{axis}" in piece else -1.0 if f"neg_{axis}" in piece else 0.0)
          for axis in "xyz"
        ]
        for piece in _BLOCK_NAMES
      ],
      device=self.device,
      dtype=self.object.data.joint_pos.dtype,
    )
    cubelet_body_names = _CENTER_BODY_NAMES + tuple(
      f"cubelet:{piece}" for piece in _BLOCK_NAMES
    )
    cubelet_body_ids, matched_body_names = self.object.find_bodies(cubelet_body_names)
    if len(matched_body_names) != len(cubelet_body_names):
      raise ValueError("Full-cube Rubik scene must expose all 26 cubelet bodies")
    body_id_by_name = dict(zip(matched_body_names, cubelet_body_ids, strict=True))
    self._cubelet_body_indices = torch.tensor(
      [body_id_by_name[name] for name in cubelet_body_names],
      device=self.device,
      dtype=torch.long,
    )
    self.plan_pool = load_solver_plan_pool(cfg.plan_pool_path)
    for plan in self.plan_pool.plans:
      if not plan["solution"]:
        raise ValueError("Training plan pool cannot contain already-solved states")
    defaults = self.object.data.default_joint_pos[0].detach().cpu().numpy()
    if len(defaults) != len(self.joint_names):
      raise ValueError("Rubik hinge names and default joint state widths differ")
    default_map = dict(zip(self.joint_names, map(float, defaults), strict=True))
    self._plan_state_rows: list[np.ndarray] = []
    self._plan_step_rows: list[list[np.ndarray]] = []
    self._plan_step_tokens: list[tuple[str, ...]] = []
    self._plan_step_face_ids: list[tuple[int, ...]] = []
    for plan in self.plan_pool.plans:
      initial = build_solver_joint_state(plan, default_map, self.joint_names)
      self._plan_state_rows.append(initial)
      state = dict(zip(self.joint_names, map(float, initial), strict=True))
      tokens = []
      for move in plan["solution"]:
        quarter = move[0] + "'" if move.endswith("2") else move
        tokens.extend([quarter] * (2 if move.endswith("2") else 1))
      state = dict(zip(self.joint_names, map(float, initial), strict=True))
      row = []
      for token in tokens:
        state = apply_face_move(state, token, self.joint_names)
        row.append(np.asarray([state[name] for name in self.joint_names]))
      self._plan_step_rows.append(row)
      self._plan_step_tokens.append(tuple(tokens))
      self._plan_step_face_ids.append(
        tuple(face_id_for_move(token) for token in tokens)
      )
    self._plan_states = torch.as_tensor(
      np.stack(self._plan_state_rows),
      device=self.device,
      dtype=self.object.data.joint_pos.dtype,
    )
    max_steps = max(map(len, self._plan_step_rows))
    self._plan_step_state_tensor = torch.zeros(
      (len(self._plan_step_rows), max_steps, len(self.joint_names)),
      device=self.device,
      dtype=self.object.data.joint_pos.dtype,
    )
    self._plan_step_face_tensor = torch.zeros(
      (len(self._plan_step_rows), max_steps), device=self.device, dtype=torch.long
    )
    self._plan_step_count = torch.tensor(
      [len(row) for row in self._plan_step_rows], device=self.device, dtype=torch.long
    )
    effective_action_counts = []
    for face_ids in self._plan_step_face_ids:
      counts = []
      for initial_top in range(6):
        top, count = initial_top, len(face_ids)
        for face_id in face_ids:
          if face_id != top:
            count += 2 if face_id == (top ^ 1) else 1
            top = face_id
        counts.append(count)
      effective_action_counts.append(counts)
    self._plan_effective_action_count = torch.tensor(
      effective_action_counts, device=self.device, dtype=torch.long
    )
    for plan_id, row in enumerate(self._plan_step_rows):
      if row:
        self._plan_step_state_tensor[plan_id, : len(row)] = torch.as_tensor(
          np.stack(row), device=self.device, dtype=self.object.data.joint_pos.dtype
        )
        self._plan_step_face_tensor[plan_id, : len(row)] = torch.tensor(
          self._plan_step_face_ids[plan_id], device=self.device, dtype=torch.long
        )

    self.episode_kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.plan_index = torch.zeros_like(self.episode_kind)
    self.move_index = torch.zeros_like(self.episode_kind)
    self.turn_index = torch.zeros_like(self.episode_kind)
    self.current_top_face = torch.full_like(self.episode_kind, 4)
    self.requested_face = torch.full_like(self.episode_kind, 4)
    self.bottom_intermediate = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.bottom_final_face = torch.full_like(self.episode_kind, 4)
    self.expected_joint_pos = torch.zeros_like(self.object.data.joint_pos)
    self.phase_start_joint_pos = torch.zeros_like(self.object.data.joint_pos)
    self.episode_initial_joint_pos = torch.zeros_like(self.object.data.joint_pos)
    self.failure_reason = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.facelet_check_mismatch = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._goal_start_step = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._episode_budget_steps = torch.zeros_like(self._goal_start_step)
    self._solved_event = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._turn_program_complete = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._goal_attempt_count = torch.zeros(
      (self.num_envs, 2, 6), dtype=torch.float, device=self.device
    )
    self._goal_success_count = torch.zeros_like(self._goal_attempt_count)
    self._smart_skill_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
    if getattr(self.cfg, "smart_cube_curriculum", None) is not None:
      from .smart_cube_curriculum import get_smart_cube_curriculum
      get_smart_cube_curriculum(env, self.cfg.smart_cube_curriculum)
    self._goal_terminal = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._internal_error_max = torch.zeros(self.num_envs, device=self.device)
    if getattr(self.cfg, 'flip_specialization', False):
      from .flip_specialization import FlipPreservation
      self.flip_preservation = FlipPreservation(self.num_envs, self.device)
      self._flip_progress_best = torch.zeros(self.num_envs, device=self.device)
      self._flip_request_event = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
      env._flip_preservation = self.flip_preservation
    self._ensure_reset_buffers(env)
    env._rubik_full_task_groups = assign_full_cube_task_groups(
      self.num_envs, device=self.device, fractions=self.cfg.task_fractions
    )
    env._rubik_full_pool_joint_states = self._plan_states
    env._rubik_full_plan_first_face = torch.tensor(
      [face_id_for_move(plan["solution"][0]) for plan in self.plan_pool.plans],
      device=self.device,
      dtype=torch.long,
    )
    from .rubik_solver import solver_face_frame_quaternion

    env._rubik_full_face_quats = solver_face_frame_quaternion(
      torch.arange(6, device=self.device), dtype=self.object.data.joint_pos.dtype
    )
    env._rubik_full_plan_pool = self.plan_pool

  def _ensure_reset_buffers(self, env) -> None:
    n, device = env.num_envs, env.device
    if not hasattr(env, "_rubik_curriculum"):
      names = (
        "flip_stage", "turn_stage", "side_trials", "side_success",
        "bottom_trials", "bottom_success", "turn90_trials", "turn90_success",
        "turn180_trials", "turn180_success", "yaw_trials", "yaw_success",
      )
      env._rubik_curriculum = {
        name: torch.zeros((), dtype=torch.long, device=device) for name in names
      }
    if hasattr(env, "_rubik_reset_mode"):
      return
    env._rubik_reset_mode = torch.zeros(n, dtype=torch.long, device=device)
    env._rubik_reset_initial_angle = torch.zeros(n, device=device)
    env._rubik_reset_goal_angle = torch.zeros(n, device=device)
    env._rubik_reset_yaw_goal_quat = torch.zeros((n, 4), device=device)
    env._rubik_reset_yaw_goal_quat[:, 0] = 1.0
    env._rubik_reset_flip_start_dot = torch.zeros(n, device=device)
    env._rubik_reset_bottom = torch.zeros(n, dtype=torch.bool, device=device)
    env._rubik_reset_turn180 = torch.zeros(n, dtype=torch.bool, device=device)
    env._rubik_full_plan_index = torch.zeros(n, dtype=torch.long, device=device)
    env._rubik_full_target_face = torch.full((n,), 4, dtype=torch.long, device=device)

  def _internal_state_before_current_move(self, env_ids: torch.Tensor) -> torch.Tensor:
    initial = self.episode_initial_joint_pos[env_ids]
    previous_index = (self.move_index[env_ids] - 1).clamp_min(0)
    previous = self._plan_step_state_tensor[self.plan_index[env_ids], previous_index]
    use_previous = (self.episode_kind[env_ids] == FULL_CUBE_SOLVE) & (
      self.move_index[env_ids] > 0
    )
    return torch.where(use_previous.unsqueeze(-1), previous, initial)

  def _sample_next_flip_goal(self, env_ids: torch.Tensor) -> None:
    """Sample any of the five faces other than the current top face."""
    if env_ids.numel() == 0:
      return
    current = self._read_top_face_id()[env_ids]
    offset = torch.randint(1, 6, (env_ids.numel(),), device=self.device)
    target = (current + offset) % 6
    self.target_face_id[env_ids] = target
    self._flip_goal_bottom[env_ids] = target == (current ^ 1)
    self.flip_start_dot[env_ids] = self._read_flip_target_dot()[env_ids]
    self.u_target_angle[env_ids] = self.u_angle[env_ids]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    env_ids = resolve_env_ids(self._env, env_ids)
    if env_ids.numel() == 0:
      return
    super()._resample_command(env_ids)
    env = self._env
    self.episode_kind[env_ids] = env._rubik_full_task_groups[env_ids]
    self.plan_index[env_ids] = env._rubik_full_plan_index[env_ids]
    self.move_index[env_ids] = 0
    self.turn_index[env_ids] = 0
    self.failure_reason[env_ids] = 0
    self.facelet_check_mismatch[env_ids] = False
    self._goal_attempt_count[env_ids] = 0
    self._goal_success_count[env_ids] = 0
    self._solved_event[env_ids] = False
    self._turn_program_complete[env_ids] = False
    self._goal_terminal[env_ids] = False
    self._smart_skill_recorded[env_ids] = False
    initial = self._plan_states[self.plan_index[env_ids]]
    if getattr(self.cfg, 'flip_specialization', False):
      # The reset may choose solved OR scrambled states. Preserve what was
      # actually installed, never an unrelated solver plan or a later drift.
      initial = self.object.data.joint_pos[env_ids].clone()
      rotations = _matrix_from_xyz(initial[:, self._piece_joint_indices])
      self.flip_preservation.reset(
        env_ids, rotations, initial[:, self._driver_joint_indices]
      )
      self._flip_request_event[env_ids] = False
    self.episode_initial_joint_pos[env_ids] = initial
    self.expected_joint_pos[env_ids] = initial
    self.phase_start_joint_pos[env_ids] = self.object.data.joint_pos[env_ids]
    self._goal_start_step[env_ids] = env.episode_length_buf[env_ids]

    selected_face = env._rubik_full_target_face[env_ids]
    self.requested_face[env_ids] = selected_face
    self.current_top_face[env_ids] = self._read_top_face_id()[env_ids]
    turn_ids = env_ids[self.episode_kind[env_ids] == FULL_CUBE_TURN]
    if turn_ids.numel():
      self._install_basic_turn_goal(turn_ids)

    flip_ids = env_ids[self.episode_kind[env_ids] == FULL_CUBE_FLIP]
    if flip_ids.numel():
      requested = self.requested_face[flip_ids]
      if getattr(self.cfg, 'flip_specialization', False):
        self._install_flip_or_turn_start(flip_ids, requested)
      else:
        self._install_goal(flip_ids, requested, 0, self.object.data.joint_pos[flip_ids])
      bottom = requested == (self.current_top_face[flip_ids] ^ 1)
      self._flip_goal_bottom[flip_ids] = bottom
      self._env._rubik_curriculum["side_trials"] += (~bottom).sum()
      self._env._rubik_curriculum["bottom_trials"] += bottom.sum()

    solve_ids = env_ids[self.episode_kind[env_ids] == FULL_CUBE_SOLVE]
    if solve_ids.numel():
      requested = self._plan_step_face_tensor[
        self.plan_index[solve_ids], self.move_index[solve_ids]
      ]
      self.requested_face[solve_ids] = requested
      self._install_flip_or_turn_start(solve_ids, requested)

    effective_actions = self._plan_effective_action_count[
      self.plan_index[env_ids], self.current_top_face[env_ids]
    ]
    solve_seconds = torch.clamp(
      effective_actions.float() * self.cfg.solve_seconds_per_action,
      min=self.cfg.solve_min_episode_seconds,
      max=self.cfg.solve_max_episode_seconds,
    )
    base_seconds = torch.full_like(solve_seconds, self.cfg.base_episode_seconds)
    seconds = torch.where(
      self.episode_kind[env_ids] == FULL_CUBE_SOLVE, solve_seconds, base_seconds
    )
    if self.cfg.turn_episode_seconds is not None:
      seconds = torch.where(
        self.episode_kind[env_ids] == FULL_CUBE_TURN,
        self.cfg.turn_episode_seconds, seconds,
      )
    self._episode_budget_steps[env_ids] = torch.ceil(seconds / self._env.step_dt).long()

  def _install_basic_turn_goal(self, env_ids: torch.Tensor) -> None:
    """Reuse legacy random +/-90/180 degree goals, independently of Node plans."""
    self._sample_next_turn_goal(env_ids)
    partial_pending = getattr(self._env, '_rubik_partial_goal_pending', None)
    if partial_pending is not None:
      ids = env_ids[partial_pending[env_ids]]
      self.u_target_angle[ids] = self._env._rubik_partial_goal[ids]
      self._turn_goal_180[ids] = False
      partial_pending[ids] = False
    self.turn_start_error[env_ids] = torch.atan2(
      torch.sin(self.u_target_angle[env_ids] - self.u_angle[env_ids]),
      torch.cos(self.u_target_angle[env_ids] - self.u_angle[env_ids]),
    ).abs()
    angles = self.u_target_angle[env_ids].clone()
    half_turn = self._turn_goal_180[env_ids].clone()
    delta = torch.atan2(
      torch.sin(angles - self.u_angle[env_ids]),
      torch.cos(angles - self.u_angle[env_ids]),
    )
    states = []
    actual = self.object.data.joint_pos[env_ids].detach().cpu().numpy()
    faces = self.target_face_id[env_ids].cpu().tolist()
    for row, face_id, angle in zip(actual, faces, delta.cpu().tolist(), strict=True):
      state = dict(zip(self.joint_names, map(float, row), strict=True))
      sign = -1.0 if face_id % 2 else 1.0
      state = apply_face_turn_angle(state, face_id, angle * sign, self.joint_names)
      states.append([state[name] for name in self.joint_names])
    target = torch.as_tensor(states, device=self.device, dtype=self.expected_joint_pos.dtype)
    self._install_goal(env_ids, self.target_face_id[env_ids], 1, target)
    self.u_target_angle[env_ids] = angles
    self._turn_goal_180[env_ids] = half_turn

  def _install_flip_or_turn_start(
    self, env_ids: torch.Tensor, requested_face: torch.Tensor
  ) -> None:
    if getattr(self.cfg, 'flip_specialization', False):
      # Physical top can differ from the previous target after alignment timeout.
      top = self._read_top_face_id()[env_ids]
      self.current_top_face[env_ids] = top
      requested_face = requested_face.clone()
      same = requested_face == top
      offsets = torch.randint(1, 6, (env_ids.numel(),), device=self.device)
      requested_face[same] = (top[same] + offsets[same]) % 6
      self.requested_face[env_ids] = requested_face
    top = self.current_top_face[env_ids]
    already_up = requested_face == top
    turn_ids = env_ids[already_up]
    if turn_ids.numel():
      if (self.episode_kind[turn_ids] == FULL_CUBE_SOLVE).any():
        solve_turn_ids = turn_ids[self.episode_kind[turn_ids] == FULL_CUBE_SOLVE]
        step = self.move_index[solve_turn_ids]
        state = self._plan_step_state_tensor[self.plan_index[solve_turn_ids], step]
        self._install_goal(solve_turn_ids, requested_face[already_up], 1, state)
      flip_turn_ids = turn_ids[self.episode_kind[turn_ids] == FULL_CUBE_FLIP]
      if flip_turn_ids.numel():
        self._install_next_flip_turn(flip_turn_ids)

    flip_ids = env_ids[~already_up]
    if flip_ids.numel() == 0:
      return
    requested = requested_face[~already_up]
    bottom = requested == (self.current_top_face[flip_ids] ^ 1)
    if (getattr(self.cfg, 'bottom_tip_observer', False)
        and not self.cfg.bottom_tip_early_per_leg):
      from .flip_bottom_tip_runtime import runtime
      runtime(self).start(flip_ids)
    self.bottom_intermediate[flip_ids] = bottom
    self.bottom_final_face[flip_ids] = requested
    waypoint = torch.where(bottom, (requested + 2) % 6, requested)
    preserved_state = self._internal_state_before_current_move(flip_ids)
    self._install_goal(
      flip_ids,
      waypoint,
      0,
      preserved_state,
    )

  def _install_next_flip_turn(self, env_ids: torch.Tensor) -> None:
    is_solve = self.episode_kind[env_ids] == FULL_CUBE_SOLVE
    if is_solve.any():
      ids = env_ids[is_solve]
      step = self.move_index[ids]
      faces = self._plan_step_face_tensor[self.plan_index[ids], step]
      states = self._plan_step_state_tensor[self.plan_index[ids], step]
      self._install_goal(ids, faces, 1, states)
    is_flip = ~is_solve
    if is_flip.any():
      ids = env_ids[is_flip]
      faces = self.requested_face[ids]
      self._install_goal(ids, faces, 1, self.episode_initial_joint_pos[ids])

  def _install_goal(
    self,
    env_ids: torch.Tensor,
    face_ids: torch.Tensor,
    phase: int,
    target_states: torch.Tensor,
  ) -> None:
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    if env_ids.numel() == 0:
      return
    target_states = target_states.to(
      device=self.device, dtype=self.expected_joint_pos.dtype
    )
    if getattr(self.cfg, 'flip_specialization', False) and phase == 0:
      target_states = self.episode_initial_joint_pos[env_ids]
      self.flip_preservation.start_goal(env_ids, int(self._env.common_step_counter))
      self._flip_progress_best[env_ids] = 0
    if target_states.ndim == 1:
      target_states = target_states.unsqueeze(0)
    self.expected_joint_pos[env_ids] = target_states
    self._goal_attempt_count[env_ids, phase, face_ids] += 1.0
    self.phase_start_joint_pos[env_ids] = self.object.data.joint_pos[env_ids]
    self._goal_start_step[env_ids] = self._env.episode_length_buf[env_ids]
    self._goal_terminal[env_ids] = False
    self._smart_skill_recorded[env_ids] = False
    if getattr(self.cfg, "turn_specialization", False):
      self.smart_move_pending[env_ids] = False
      self._smart_move_state_step[env_ids] = int(self._env.common_step_counter)
    driver_ids = self._face_driver_ids[face_ids]
    raw_angle = target_states.gather(1, driver_ids.unsqueeze(-1)).squeeze(-1)
    sign = torch.where((face_ids % 2) == 1, -1.0, 1.0).to(raw_angle.dtype)
    target_angle = raw_angle * sign
    self.set_solver_goal(
      face_ids,
      phase,
      target_angle,
      target_states,
      env_ids=env_ids,
    )
    if self.cfg.bottom_tip_observer and self.cfg.bottom_tip_early_per_leg and phase == 0:
      # The target and flip_start_dot must be installed before opening this leg.
      # This also resets evidence/patch identity for the second half of 180°.
      from .flip_bottom_tip_runtime import runtime
      runtime(self).start(env_ids)
    if getattr(self.cfg, 'flip_specialization', False):
      # Installing a next goal must not count a second hold step or clear the
      # completed request pulse during another read of this control frame.
      self._last_update_step[env_ids] = int(self._env.common_step_counter)
    self.task_mode[env_ids] = (self.episode_kind[env_ids] == FULL_CUBE_TURN).long()
    self._reset_face_relative_observation_history(env_ids)

  def _reset_face_relative_observation_history(self, env_ids: torch.Tensor) -> None:
    manager = self._env.observation_manager
    manager._obs_buffer = None
    face_relative_terms = {
      "rubik_cube_orientation",
      "rubik_compat_cube_ori_error",
      "rubik_u_face_angle",
    }
    delay_buffers = manager._group_obs_term_delay_buffer
    for group_name, term_buffers in manager._group_obs_term_history_buffer.items():
      for term_name in face_relative_terms & term_buffers.keys():
        term_buffers[term_name].reset(batch_ids=env_ids)
        delay = delay_buffers[group_name].get(term_name)
        if delay is not None:
          delay.reset(batch_ids=env_ids)

  def _evaluate_current_step(self) -> None:
    step = int(self._env.common_step_counter)
    fresh = self._last_update_step != step
    if not fresh.any():
      return
    specialist = getattr(self.cfg, 'flip_specialization', False)
    super()._evaluate_current_step()
    if specialist:
      state = self.flip_preservation
      actual = self.object.data.joint_pos
      sensor = getattr(self._env, '_smart_cube_state', None)
      face_valid = self._read_flip_target_dot() >= math.cos(self.cfg.continuous_flip_upright_tolerance_rad)
      state.update(step, self._flip_event, face_valid,
        _matrix_from_xyz(actual[:, self._piece_joint_indices]),
        actual[:, self._driver_joint_indices], self._env.step_dt,
        contact=(self.tip_contact_count >= self.cfg.min_contact_fingertips),
        move_event=sensor.move_event if sensor is not None else None,
        move_step=sensor.move_step if sensor is not None else None,
        window_seconds=self.cfg.flip_alignment_seconds,
        tolerance_rad=self.cfg.flip_alignment_tolerance_rad,
        hold_steps=self.cfg.flip_alignment_hold_steps)
      progress = ((self._read_flip_target_dot() - self.flip_start_dot)
                  / (1 - self.flip_start_dot).clamp_min(.1)).clamp(0, 1)
      valid = ((self.tip_contact_count >= self.cfg.min_contact_fingertips)
               & ~state.arrival_seen & (self.phase == 0))
      self.progress_delta = torch.where(fresh & valid,
        (progress - self._flip_progress_best).clamp_min(0), 0.)
      self._flip_progress_best = torch.where(fresh & valid,
        torch.maximum(self._flip_progress_best, progress), self._flip_progress_best)
    raw_flip = self._flip_event.clone()
    raw_turn = self._turn_event.clone()

    if specialist:
      # Shape truth is sampled at arrival/inside alignment only, not while
      # performing a flip. Keep the diagnostic cached between those samples.
      self._internal_error_max = state.error.clone()
    else:
      actual = self.object.data.joint_pos
      expected = self.expected_joint_pos
      phase_start = self.phase_start_joint_pos
      _, goal_error = _rotation_errors(
        actual,
        expected,
        self.joint_names,
        piece_joint_indices=self._piece_joint_indices,
        driver_joint_indices=self._driver_joint_indices,
      )
      goal_delta = torch.atan2(
        torch.sin(self.u_target_angle - self.u_initial_angle),
        torch.cos(self.u_target_angle - self.u_initial_angle),
      )
      current_delta = torch.atan2(
        torch.sin(self.u_angle - self.u_initial_angle),
        torch.cos(self.u_angle - self.u_initial_angle),
      )
      safe_goal_delta = torch.where(
        goal_delta.abs() > 1.0e-5, goal_delta, torch.ones_like(goal_delta)
      )
      fraction = current_delta / safe_goal_delta
      fraction = torch.where(
        goal_delta.abs() > 1.0e-5, fraction.clamp(0.0, 1.0), torch.zeros_like(fraction)
      )
      fraction = torch.where(self.phase == 1, fraction, torch.zeros_like(fraction))
      path_error = _face_turn_path_error(
        actual,
        phase_start,
        expected,
        self.target_face_id,
        fraction,
        piece_joint_indices=self._piece_joint_indices,
        driver_joint_indices=self._driver_joint_indices,
        piece_home_coords=self._piece_home_coords,
      )
      path_error = torch.where((self.phase == 1).unsqueeze(-1), path_error, goal_error)
      # Internal state is diagnostic/critic information, never a skill-success gate.
      self._internal_error_max = path_error.amax(dim=-1)
    dropped = self._env.termination_manager.get_term("cage_drop")
    self.failure_reason = torch.where(
      fresh & dropped, 4,
      torch.where(fresh & (self.failure_reason == 4), 0, self.failure_reason),
    )
    active = ~self._goal_terminal
    accepted_flip = raw_flip & active
    accepted_turn = raw_turn & active
    if specialist:
      completed = state.done & active
      # Complete a flip when its requested face is still at the target after
      # the assessment window. Internal layer alignment is a separate metric.
      accepted_flip = completed & face_valid & ~dropped
      successful_ids = accepted_flip.nonzero().flatten()
      state.finish(successful_ids, step, self._env.step_dt, success=True)
      timeout_ids = (completed & ~accepted_flip).nonzero().flatten()
      state.finish(timeout_ids, step, self._env.step_dt, timeout=True)
      self._goal_terminal[timeout_ids] = True
      self.failure_reason[timeout_ids] = 3
      requested_done = successful_ids[~self.bottom_intermediate[successful_ids] & face_valid[successful_ids]]
      self._flip_request_event[:] = False
      self._flip_request_event[requested_done] = True
      state.request_count[requested_done] += 1
      state._totals['completed_requests'] += requested_done.numel()
      if getattr(self.cfg, 'bottom_tip_observer', False):
        from .flip_bottom_tip_runtime import runtime
        tracker = runtime(self).requests
        tracker.goal_event.zero_()
        tracker.success_event.zero_()
        if self.cfg.bottom_tip_early_per_leg:
          tracker.freeze(state.arrival_event.nonzero().flatten())
          ended = (state.done | dropped).nonzero().flatten()
          tracker.finish(ended, accepted_flip[ended] & face_valid[ended])
        else:
          tracker.freeze((state.arrival_event & ~self.bottom_intermediate).nonzero().flatten())
          ended = ((state.done & ~self.bottom_intermediate) | dropped).nonzero().flatten()
          tracker.finish(ended, self._flip_request_event[ended] & ~dropped[ended])
    strict_facelet_check = getattr(self.cfg, "verify_physical_facelets", False)
    final_facelet_check = getattr(self.cfg, "verify_final_physical_facelets", False)
    if strict_facelet_check or final_facelet_check:
      candidates = ((accepted_flip | accepted_turn)
        & (self.episode_kind == FULL_CUBE_SOLVE)).nonzero().flatten()
      rejected = self._verify_physical_facelets(candidates)
      if rejected.numel():
        self.facelet_check_mismatch[rejected] = True
      if not strict_facelet_check and rejected.numel():
        final_candidates = candidates[
          self.move_index[candidates] + 1
          >= self._plan_step_count[self.plan_index[candidates]]
        ]
        rejected = rejected[torch.isin(rejected, final_candidates)]
      accepted_flip[rejected] = False
      accepted_turn[rejected] = False
      # Intermediate mismatches are diagnostic only. Strict mode and final
      # move validation retain the old failure path.
      self._goal_terminal[rejected] = True
      self.failure_reason[rejected] = 5
    curriculum = getattr(self._env, "_rubik_curriculum", None)
    if curriculum is not None:
      flip = accepted_flip & (self.episode_kind == FULL_CUBE_FLIP)
      turn = accepted_turn & (self.episode_kind == FULL_CUBE_TURN)
      curriculum["side_success"] += (flip & ~self._flip_goal_bottom).sum()
      curriculum["bottom_success"] += (flip & self._flip_goal_bottom).sum()
      curriculum["turn90_success"] += (turn & ~self._turn_goal_180).sum()
      curriculum["turn180_success"] += (turn & self._turn_goal_180).sum()
      self._promote_curriculum()
    successful_goals = (accepted_flip | accepted_turn).nonzero().flatten()
    if getattr(self.cfg, "smart_cube_curriculum", None) is not None:
      from .smart_cube_curriculum import record_smart_cube_skill_successes
      record_smart_cube_skill_successes(
        self._env, self, successful_goals, dropped | (self.failure_reason > 0))
    if successful_goals.numel():
      self._goal_success_count[
        successful_goals,
        self.phase[successful_goals],
        self.target_face_id[successful_goals],
      ] += 1.0
    self.flip_to_top_count += accepted_flip.float() - raw_flip.float()
    self.turn_success_count -= (raw_turn & ~accepted_turn).float()
    self.goal_reach_count += accepted_flip.float() - raw_flip.float()
    self.goal_reach_count -= (raw_turn & ~accepted_turn).float()
    self._flip_event = torch.where(fresh, accepted_flip, self._flip_event)
    self._turn_event = torch.where(fresh, accepted_turn, self._turn_event)
    self._pending_goal_advance = torch.where(
      fresh, accepted_flip | accepted_turn, self._pending_goal_advance
    )
    final_solve_turn = (
      accepted_turn
      & (self.episode_kind == FULL_CUBE_SOLVE)
      & (self.move_index + 1 >= self._plan_step_count[self.plan_index])
    )
    self._solved_event = torch.where(fresh, final_solve_turn, self._solved_event)
    if getattr(self.cfg, "turn_specialization", False):
      self.update_smart_move_pending()
      stats = getattr(self, '_turn_move_stats', None)
      if stats is None:
        from .turn_specialization import TurnMoveStats
        stats = self._turn_move_stats = TurnMoveStats(self.num_envs, self.device)
        self._env._turn_move_stats = stats
      pending = self.smart_move_pending & (self.phase == 1) & (self.episode_kind == FULL_CUBE_TURN)
      stats.observe(step, pending, accepted_turn, self.u_angle_error, self._env.step_dt)

  @property
  def solved_achieved(self) -> torch.Tensor:
    return self._solved_event

  @property
  def turn_program_complete(self) -> torch.Tensor:
    return self._turn_program_complete

  @property
  def failure_event(self) -> torch.Tensor:
    failure = (self.episode_kind == FULL_CUBE_SOLVE) & (
      (self.failure_reason == 1)
      | (self.failure_reason == 2)
      | (self.failure_reason == 3)
      | (self.failure_reason == 5)
    )
    if getattr(self.cfg, 'flip_specialization', False):
      failure |= (self.episode_kind == FULL_CUBE_FLIP) & (self.failure_reason > 0)
    return failure

  def _verify_physical_facelets(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Simulated FACELETS check at solver skill boundaries, never every step.

    Decode physical cubies without snapping them. Off-grid states fail the
    boundary check. Host recovery uses smart_cube_facelets.replan_from_facelets;
    Node is never invoked inside this vectorized training loop.
    """
    from .rubik_solver import decode_actual_facelets
    rejected = []
    actual = self.object.data.joint_pos[env_ids].detach().cpu().tolist()
    expected = self.expected_joint_pos[env_ids].detach().cpu().tolist()
    for env_id, row, target in zip(env_ids.cpu().tolist(), actual, expected, strict=True):
      observed = decode_actual_facelets(dict(zip(self.joint_names, row, strict=True)), self.joint_names)
      wanted = decode_actual_facelets(dict(zip(self.joint_names, target, strict=True)), self.joint_names)
      if observed is None or wanted is None or observed != wanted:
        rejected.append(env_id)
    return torch.tensor(rejected, device=env_ids.device, dtype=torch.long)

  @property
  def internal_deviation(self) -> torch.Tensor:
    return self._internal_error_max.clamp(max=math.pi) / math.pi

  def update_stage_timeout(self) -> torch.Tensor:
    elapsed = self._env.episode_length_buf - self._goal_start_step
    goal_limit = math.ceil(self.cfg.phase_timeout_seconds / self._env.step_dt)
    goal_timeout = elapsed >= goal_limit
    timeout = goal_timeout & (self.failure_reason == 0) & ~self._solved_event
    timeout &= ~self._pending_goal_advance & ~self._turn_program_complete
    eligible = self.episode_kind == FULL_CUBE_SOLVE
    if getattr(self.cfg, "turn_specialization", False):
      eligible |= self.episode_kind == FULL_CUBE_TURN
    if getattr(self.cfg, 'flip_specialization', False):
      eligible |= (self.episode_kind == FULL_CUBE_FLIP) & ~self.flip_preservation.settling
    timeout &= eligible
    self.failure_reason = torch.where(timeout, 3, self.failure_reason)
    return timeout

  def update_episode_timeout(self) -> torch.Tensor:
    elapsed = self._env.episode_length_buf - self._goal_start_step
    goal_limit = math.ceil(self.cfg.phase_timeout_seconds / self._env.step_dt)
    episode_timeout = self._env.episode_length_buf >= self._episode_budget_steps
    pure_budget_timeout = episode_timeout & ((elapsed < goal_limit) | (self.episode_kind != FULL_CUBE_SOLVE))
    timeout = pure_budget_timeout & (self.failure_reason == 0) & ~self._solved_event
    timeout &= ~self._pending_goal_advance & ~self._turn_program_complete
    self.failure_reason = torch.where(timeout, 3, self.failure_reason)
    return timeout

  def update_timeout(self) -> torch.Tensor:
    return self.update_stage_timeout() | self.update_episode_timeout()

  def _advance_pending_goals(self) -> None:
    env_ids = self._pending_goal_advance.nonzero().flatten()
    if env_ids.numel() == 0:
      return
    flip_ids = env_ids[self.phase[env_ids] == 0]
    turn_ids = env_ids[self.phase[env_ids] == 1]
    self._pending_goal_advance[env_ids] = False

    if flip_ids.numel():
      is_intermediate = self.bottom_intermediate[flip_ids]
      intermediate = flip_ids[is_intermediate]
      remaining = flip_ids[~is_intermediate]
      if intermediate.numel():
        final_face = self.bottom_final_face[intermediate]
        self.bottom_intermediate[intermediate] = False
        self.requested_face[intermediate] = final_face
        self.current_top_face[intermediate] = self.target_face_id[intermediate]
        preserved_state = self._internal_state_before_current_move(intermediate)
        self._install_goal(
          intermediate,
          final_face,
          0,
          preserved_state,
        )
      if remaining.numel():
        self.current_top_face[remaining] = self.target_face_id[remaining]
        solve = self.episode_kind[remaining] == FULL_CUBE_SOLVE
        if solve.any():
          self._install_next_flip_turn(remaining[solve])
        basic_flip = remaining[~solve]
        if basic_flip.numel():
          self._sample_next_flip_goal(basic_flip)
          requested = self.target_face_id[basic_flip].clone()
          bottom = self._flip_goal_bottom[basic_flip].clone()
          self.requested_face[basic_flip] = requested
          if getattr(self.cfg, 'flip_specialization', False):
            self._install_flip_or_turn_start(basic_flip, requested)
          else:
            self._install_goal(
              basic_flip, requested, 0, self.object.data.joint_pos[basic_flip]
            )
          self._flip_goal_bottom[basic_flip] = bottom

    if turn_ids.numel():
      solve = self.episode_kind[turn_ids] == FULL_CUBE_SOLVE
      solve_ids = turn_ids[solve]
      if solve_ids.numel():
        self.current_top_face[solve_ids] = self.target_face_id[solve_ids]
        next_move = self.move_index[solve_ids] + 1
        has_next = next_move < self._plan_step_count[self.plan_index[solve_ids]]
        self.move_index[solve_ids] = next_move
        self._install_solver_next(solve_ids[has_next])
        self._goal_terminal[solve_ids[~has_next]] = True
      basic_ids = turn_ids[~solve]
      if basic_ids.numel():
        self.current_top_face[basic_ids] = self.target_face_id[basic_ids]
        self.turn_index[basic_ids] += 1
        self._install_basic_turn_goal(basic_ids)

  def _install_solver_next(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    requested = self._plan_step_face_tensor[
      self.plan_index[env_ids], self.move_index[env_ids]
    ]
    self.requested_face[env_ids] = requested
    self._install_flip_or_turn_start(env_ids, requested)

  def _update_command(self) -> None:
    self._evaluate_current_step()
    self._advance_pending_goals()

  def _update_metrics(self) -> None:
    super()._update_metrics()
    # The base class expands scalar curriculum stages; resets need owned buffers.
    self.metrics["rubik_flip_stage"] = self.metrics["rubik_flip_stage"].clone()
    self.metrics["rubik_turn_stage"] = self.metrics["rubik_turn_stage"].clone()
    self.metrics["rubik_episode_kind"] = self.episode_kind.float()
    self.metrics["rubik_solver_plan_length"] = torch.where(
      self.episode_kind == FULL_CUBE_SOLVE,
      self._plan_step_count[self.plan_index].float(), 0.0
    )
    self.metrics["rubik_solver_moves_completed"] = self.move_index.float()
    self.metrics["rubik_failure_reason"] = self.failure_reason.float()
    self.metrics["rubik_facelet_check_mismatch"] = self.facelet_check_mismatch.float()
    self.metrics["rubik_solved"] = self._solved_event.float()
    self.metrics["rubik_internal_error_rad"] = self._internal_error_max
    if getattr(self.cfg, 'flip_specialization', False):
      state = self.flip_preservation
      self.metrics['flip_align/max_error_deg'] = torch.rad2deg(state.error)
      self.metrics['flip_align/mean_error_deg'] = torch.rad2deg(state.mean_error)
      self.metrics['flip_align/settling'] = state.settling.float()
      self.metrics['flip_align/completed_requests'] = state.request_count
      self.metrics['flip_align/face_error_deg'] = torch.rad2deg(torch.acos(self._read_flip_target_dot()))
    self.metrics["rubik_failure_phase"] = torch.where(
      self.failure_reason > 0, self.phase, -1
    ).float()
    self.metrics["rubik_failure_move"] = torch.where(
      self.failure_reason > 0, self.move_index, -1
    ).float()


@dataclass(kw_only=True)
class RubikFullCubeCommandCfg(RubikUFaceCommandCfg):
  plan_pool_path: str
  task_fractions: tuple[float, float, float] | None = None
  partial_turn_start_probability: float = 0.0
  turn_180_probability: float | None = None
  turn_specialization: bool = False
  turn_episode_seconds: float | None = None
  flip_specialization: bool = False
  bottom_tip_observer: bool = False
  bottom_tip_early_per_leg: bool = False
  bottom_tip_compact_contact: bool = False
  bottom_tip_debug_diagnostics: bool = True
  bottom_tip_contact_window_deg: float = 45.
  bottom_tip_success_rewards: tuple[float, float] | None = None
  flip_scramble_probability: float = .5
  flip_alignment_seconds: float = 1.
  flip_alignment_tolerance_rad: float = math.radians(5)
  flip_alignment_hold_steps: int = 5
  verify_physical_facelets: bool = False
  verify_final_physical_facelets: bool = False
  base_episode_seconds: float = 50.0
  smart_cube_curriculum: SmartCubeCurriculumCfg | None = None
  phase_timeout_seconds: float = 20.0
  solve_seconds_per_action: float = 10.0
  solve_min_episode_seconds: float = 60.0
  solve_max_episode_seconds: float = 600.0
  solver_mode: bool = True
  continuous_flip_task_kind: int = FULL_CUBE_FLIP
  continuous_flip_upright_tolerance_rad: float = math.radians(4.0)
  continuous_flip_angle_tolerance_rad: float = math.radians(4.0)
  continuous_flip_hold_steps: int = 10

  def build(self, env) -> RubikFullCubeCommand:
    return RubikFullCubeCommand(self, env)


def rubik_full_cube_internal_privileged(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  return internal_state_error_features(
    command.object.data.joint_pos,
    command.expected_joint_pos,
    command.joint_names,
    piece_joint_indices=command._piece_joint_indices,
    driver_joint_indices=command._driver_joint_indices,
  )


def rubik_full_cube_solved(env, command_name: str = "reorient_command") -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.solved_achieved.clone()


def rubik_full_cube_failed(env, command_name: str = "reorient_command") -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.failure_event.clone()


def rubik_full_cube_turn_program_complete(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.turn_program_complete.clone()


def rubik_full_cube_timeout(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.update_episode_timeout()


def rubik_full_cube_stage_timeout(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.update_stage_timeout()


def rubik_full_cube_solved_reward(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.solved_achieved.float() / max(float(env.step_dt), 1e-6)


def rubik_full_cube_failure_reward(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.failure_event.float() / max(float(env.step_dt), 1e-6)


def rubik_full_cube_internal_penalty(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term(command_name)
  command.evaluate_task()
  return command.internal_deviation


def rubik_task_fraction_metric(env, task_kind: int) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term("reorient_command")
  return (command.episode_kind == task_kind).float()


def rubik_solver_plan_length_metric(env) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term("reorient_command")
  return torch.where(
    command.episode_kind == FULL_CUBE_SOLVE,
    command._plan_step_count[command.plan_index].float(), 0.0
  )


def rubik_solver_moves_completed_metric(env) -> torch.Tensor:
  return env.command_manager.get_term("reorient_command").move_index.float()


def rubik_failure_reason_metric(env) -> torch.Tensor:
  return env.command_manager.get_term("reorient_command").failure_reason.float()


def rubik_facelet_check_mismatch_metric(env) -> torch.Tensor:
  return env.command_manager.get_term("reorient_command").facelet_check_mismatch.float()


def rubik_failure_phase_metric(env) -> torch.Tensor:
  command = env.command_manager.get_term("reorient_command")
  return torch.where(command.failure_reason > 0, command.phase, -1).float()


def rubik_failure_move_metric(env) -> torch.Tensor:
  command = env.command_manager.get_term("reorient_command")
  return torch.where(command.failure_reason > 0, command.move_index, -1).float()


def rubik_solved_metric(env) -> torch.Tensor:
  return env.command_manager.get_term("reorient_command").solved_achieved.float()


def rubik_internal_error_metric(env) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term("reorient_command")
  return command._internal_error_max


def rubik_goal_count_metric(
  env, face_id: int, phase: int, *, success: bool = False
) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term("reorient_command")
  count = command._goal_success_count if success else command._goal_attempt_count
  return count[:, phase, face_id]


def rubik_solver_attempt_metric(env) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term("reorient_command")
  command.evaluate_task()
  dropped = env.termination_manager.get_term("cage_drop")
  ended = (
    command.solved_achieved
    | command.turn_program_complete
    | (command.failure_reason > 0)
    | dropped
  )
  return ((command.episode_kind == FULL_CUBE_SOLVE) & ended).float()


def rubik_solver_success_metric(env) -> torch.Tensor:
  command: RubikFullCubeCommand = env.command_manager.get_term("reorient_command")
  command.evaluate_task()
  return command.solved_achieved.float()
