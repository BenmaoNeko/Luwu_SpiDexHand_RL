# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_mul

from wuji_mjlab.tasks.reorient.reorient_constants import (
  TAG_IN_PALM_POS,
  TAG_IN_PALM_QUAT_WXYZ,
)

if TYPE_CHECKING:
  import viser
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_GOAL_VIS_Z_OFFSET = 0.15
_STATUS_VIS_X_OFFSET = 0.06
_STATUS_VIS_RADIUS = 0.012
_FRAME_TRIAD_LEN = 0.10
_FRAME_TRIAD_RADIUS = 0.005
_TAG_TRIAD_COLORS = (
  (0.0, 0.85, 0.85),
  (0.85, 0.0, 0.85),
  (0.9, 0.85, 0.0),
)


def goal_vis_position_above_object(
  object_pos: np.ndarray,
  z_offset: float = _GOAL_VIS_Z_OFFSET,
) -> np.ndarray:
  """Place the goal pose visualization directly above the live object pose."""
  vis_pos = np.array(object_pos, copy=True)
  vis_pos[2] += z_offset
  return vis_pos


def goal_vis_env_indices(num_envs: int) -> list[int]:
  """Draw reorient 3D goal/status markers for every environment."""
  return list(range(num_envs))


def reorient_status_color_rgba(success: bool) -> tuple[float, float, float, float]:
  """Color-code current policy status for visualization."""
  if success:
    return (0.2, 0.9, 0.2, 0.95)
  return (0.95, 0.2, 0.2, 0.95)


def format_reorient_status_markdown(
  ori_error_rad: float,
  success: bool,
  target_face_id: int | None = None,
  consecutive_flip_count: int | None = None,
  face_coverage_count: int | None = None,
) -> str:
  """Format current env reorient status for the Viser GUI."""
  ori_error_deg = float(np.degrees(ori_error_rad))
  status = (
    "### Reorient Policy Status\n"
    f"- ori_error: {ori_error_deg:.1f} deg\n"
    f"- success: {success}\n"
  )
  if target_face_id is not None:
    status += f"- target_face: {target_face_id}\n"
  if consecutive_flip_count is not None:
    status += f"- flip_chain: {consecutive_flip_count}\n"
  if face_coverage_count is not None:
    status += f"- face_coverage: {face_coverage_count}/6\n"
  return status


def update_reorient_status_markdown(
  status_markdown,
  get_env_idx: Callable[[], int],
  num_envs: int,
  policy_status_for_env: Callable[[int], tuple],
) -> None:
  """Refresh the Viser markdown widget for the currently selected env."""
  if num_envs <= 0:
    return
  env_idx = int(np.clip(get_env_idx(), 0, num_envs - 1))
  status = policy_status_for_env(env_idx)
  ori_error_rad, is_success = status[:2]
  extra = status[2:]
  status_markdown.content = format_reorient_status_markdown(
    ori_error_rad=ori_error_rad,
    success=is_success,
    target_face_id=extra[0] if len(extra) > 0 else None,
    consecutive_flip_count=extra[1] if len(extra) > 1 else None,
    face_coverage_count=extra[2] if len(extra) > 2 else None,
  )


def _tag_pose_from_palm_pose(
  palm_pos_w: "torch.Tensor",
  palm_quat_w: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
  """Compose palm world pose with the public tag-in-palm rigid transform."""
  tag_in_palm_pos = torch.tensor(
    TAG_IN_PALM_POS,
    device=palm_pos_w.device,
    dtype=palm_pos_w.dtype,
  ).expand_as(palm_pos_w)
  tag_in_palm_quat = torch.tensor(
    TAG_IN_PALM_QUAT_WXYZ,
    device=palm_quat_w.device,
    dtype=palm_quat_w.dtype,
  ).expand_as(palm_quat_w)
  tag_pos_w = palm_pos_w + quat_apply(palm_quat_w, tag_in_palm_pos)
  tag_quat_w = quat_mul(palm_quat_w, tag_in_palm_quat)
  return tag_pos_w, tag_quat_w


def draw_reorient_frame_triads(
  *,
  visualizer: "DebugVisualizer",
  num_envs: int,
  palm_pose_w: "torch.Tensor",
) -> None:
  """Draw palm-frame (RGB) and tag-frame (CMY) triads per env."""
  palm_pos_t = palm_pose_w[:, 0:3]
  palm_quat_t = palm_pose_w[:, 3:7]
  palm_R = matrix_from_quat(palm_quat_t).detach().cpu().numpy()
  palm_pos = palm_pos_t.detach().cpu().numpy()

  tag_pos_t, tag_quat_t = _tag_pose_from_palm_pose(palm_pos_t, palm_quat_t)
  tag_R = matrix_from_quat(tag_quat_t).detach().cpu().numpy()
  tag_pos = tag_pos_t.detach().cpu().numpy()

  for env_idx in goal_vis_env_indices(num_envs):
    visualizer.add_frame(
      position=palm_pos[env_idx].astype(np.float32),
      rotation_matrix=palm_R[env_idx].astype(np.float32),
      scale=_FRAME_TRIAD_LEN,
      axis_radius=_FRAME_TRIAD_RADIUS,
      alpha=1.0,
      label=f"palm_frame_env{env_idx}",
    )
    visualizer.add_frame(
      position=tag_pos[env_idx].astype(np.float32),
      rotation_matrix=tag_R[env_idx].astype(np.float32),
      scale=_FRAME_TRIAD_LEN,
      axis_radius=_FRAME_TRIAD_RADIUS,
      alpha=1.0,
      axis_colors=_TAG_TRIAD_COLORS,
      label=f"tag_frame_env{env_idx}",
    )


@dataclass
class ReorientCommandVisualization:
  entity_name: str
  _goal_cube_model: object | None = field(default=None, init=False, repr=False)
  _native_mesh_id: int | None = field(default=None, init=False, repr=False)
  _native_mat_id: int = field(default=-1, init=False, repr=False)
  status_markdown: object | None = None
  get_env_idx: Callable[[], int] | None = None

  def create_gui(
    self,
    *,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    with server.gui.add_folder(name.capitalize()):
      self.status_markdown = server.gui.add_markdown("")
    self.get_env_idx = get_env_idx

  def attach_status_targets(
    self,
    *,
    status_markdown,
    get_env_idx: Callable[[], int],
  ) -> None:
    """Test seam for injecting GUI targets without constructing a real Viser UI."""
    self.status_markdown = status_markdown
    self.get_env_idx = get_env_idx

  def update_status_gui(
    self,
    *,
    num_envs: int,
    policy_status_for_env: Callable[[int], tuple],
  ) -> None:
    if self.status_markdown is None or self.get_env_idx is None:
      return
    update_reorient_status_markdown(
      status_markdown=self.status_markdown,
      get_env_idx=self.get_env_idx,
      num_envs=num_envs,
      policy_status_for_env=policy_status_for_env,
    )

  def _ensure_goal_cube_model(self):
    """Build a standalone visual cube model for native and Viser viewers."""
    if self._goal_cube_model is not None:
      return self._goal_cube_model

    import mujoco

    # Use a standalone primitive instead of the full environment model. The
    # Viser backend renders debug ghosts through its generic primitive-mesh
    # path; it cannot consume a raw ``MjvScene`` geom inserted by a command
    # term. The dimensions match the physical 54 mm cube.
    model = mujoco.MjModel.from_xml_string(
      """
      <mujoco model="reorient_goal_cube">
        <worldbody>
          <body name="goal_cube">
            <freejoint name="goal_cube_freejoint"/>
            <geom name="goal_cube_visual" type="box"
              size="0.027 0.027 0.027"
              contype="0" conaffinity="0"
              rgba="0.05 0.55 1.0 1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    self._goal_cube_model = model
    return model

  def _ensure_native_cube_mesh(self, mj_model):
    """Find the namespaced physical cube mesh in the full native model."""
    if self._native_mesh_id is not None:
      return self._native_mesh_id, self._native_mat_id

    import mujoco

    mesh_id = mujoco.mj_name2id(
      mj_model, mujoco.mjtObj.mjOBJ_MESH, f"{self.entity_name}/cube_mesh"
    )
    if mesh_id < 0:
      return None

    self._native_mesh_id = mesh_id
    for geom_id in range(mj_model.ngeom):
      if (
        mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        and mj_model.geom_dataid[geom_id] == mesh_id
      ):
        self._native_mat_id = int(mj_model.geom_matid[geom_id])
        break
    return self._native_mesh_id, self._native_mat_id

  def draw_debug_visuals(
    self,
    *,
    visualizer: "DebugVisualizer",
    num_envs: int,
    palm_pose_w: "torch.Tensor",
    object_pos_w: "torch.Tensor",
    goal_quat_w: "torch.Tensor",
    policy_status_for_env: Callable[[int], tuple],
  ) -> None:
    """Draw frame triads, target cube, and status spheres for all envs."""
    draw_reorient_frame_triads(
      visualizer=visualizer,
      num_envs=num_envs,
      palm_pose_w=palm_pose_w,
    )

    is_native_visualizer = hasattr(visualizer, "scn") and hasattr(
      visualizer, "mj_model"
    )
    goal_cube_model = (
      None if is_native_visualizer else self._ensure_goal_cube_model()
    )
    goal_rot = matrix_from_quat(goal_quat_w)
    native_mesh = (
      self._ensure_native_cube_mesh(visualizer.mj_model)
      if is_native_visualizer
      else None
    )

    for env_idx in goal_vis_env_indices(num_envs):
      status = policy_status_for_env(env_idx)
      ori_error_rad, is_success = status[:2]
      vis_pos = goal_vis_position_above_object(
        object_pos_w[env_idx].detach().cpu().numpy()
      )
      if is_native_visualizer:
        import mujoco

        scene = visualizer.scn
        if scene.ngeom >= scene.maxgeom:
          break
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom_type = mujoco.mjtGeom.mjGEOM_BOX.value
        geom_size = np.array([0.027, 0.027, 0.027], dtype=np.float64)
        if native_mesh is not None:
          mesh_id, mat_id = native_mesh
          geom_type = mujoco.mjtGeom.mjGEOM_MESH.value
          geom_size = np.array([0.03, 0.03, 0.03], dtype=np.float64)
        mujoco.mjv_initGeom(
          geom,
          geom_type,
          geom_size,
          vis_pos.astype(np.float64),
          goal_rot[env_idx].detach().cpu().numpy().reshape(-1).astype(np.float64),
          np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32),
        )
        if native_mesh is not None:
          geom.dataid = 2 * mesh_id
          geom.matid = mat_id
          geom.texcoord = 1
      else:
        goal_qpos = np.concatenate(
          (vis_pos, goal_quat_w[env_idx].detach().cpu().numpy())
        )
        visualizer.add_ghost_mesh(
          qpos=goal_qpos,
          model=goal_cube_model,
          alpha=0.72,
          label=f"reorient_goal_cube_env{env_idx}",
        )

      visualizer.add_frame(
        position=vis_pos,
        rotation_matrix=goal_rot[env_idx].detach().cpu().numpy(),
        scale=0.06,
        axis_radius=0.003,
        alpha=0.9,
        axis_colors=((1.0, 0.15, 0.15), (0.15, 0.95, 0.15), (0.15, 0.35, 1.0)),
        label=f"reorient_goal_frame_env{env_idx}",
      )

      status_pos = vis_pos.copy()
      status_pos[0] += _STATUS_VIS_X_OFFSET
      visualizer.add_sphere(
        center=status_pos,
        radius=_STATUS_VIS_RADIUS,
        color=reorient_status_color_rgba(is_success),
        label=(
          f"reorient_status_env{env_idx}"
          f"_err_{np.degrees(ori_error_rad):.1f}deg"
          f"_success_{is_success}"
        ),
      )
