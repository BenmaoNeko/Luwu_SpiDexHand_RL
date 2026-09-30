# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import mujoco
from wuji_mjlab.tasks.reorient import mdp as reorient_mdp
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
  draw_reorient_frame_triads,
  format_reorient_status_markdown,
  goal_vis_env_indices,
  goal_vis_position_above_object,
  reorient_status_color_rgba,
  update_reorient_status_markdown,
)
from wuji_mjlab.tasks.reorient.tooling.scene_builder import set_goal_mocap


class _StubScene:
  def __init__(self, cube_pos: np.ndarray):
    self._cube_pos = np.array(cube_pos, dtype=np.float64)
    self.goal_mocap_id = 0
    self.data = SimpleNamespace(
      mocap_pos=np.zeros((1, 3), dtype=np.float64),
      mocap_quat=np.zeros((1, 4), dtype=np.float64),
    )

  @property
  def cube_pos(self) -> np.ndarray:
    return self._cube_pos.copy()


class _StubVisualizer:
  def __init__(self) -> None:
    self.frames = []
    self.ghosts = []
    self.spheres = []

  def add_frame(
    self,
    *,
    position,
    rotation_matrix,
    scale,
    axis_radius,
    alpha,
    label,
    axis_colors=None,
  ) -> None:
    self.frames.append(
      {
        "position": position,
        "rotation_matrix": rotation_matrix,
        "scale": scale,
        "axis_radius": axis_radius,
        "alpha": alpha,
        "label": label,
        "axis_colors": axis_colors,
      }
    )

  def add_ghost_mesh(self, *, qpos, model, alpha, label) -> None:
    self.ghosts.append(
      {
        "qpos": np.asarray(qpos),
        "model": model,
        "alpha": alpha,
        "label": label,
      }
    )

  def add_sphere(self, *, center, radius, color, label) -> None:
    self.spheres.append(
      {
        "center": center,
        "radius": radius,
        "color": color,
        "label": label,
      }
    )


def test_set_goal_mocap_places_goal_visualization_above_current_cube():
  scene = _StubScene(cube_pos=np.array([0.12, -0.07, 0.63], dtype=np.float64))
  goal_quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

  # _StubScene is intentionally minimal and structurally compatible with SceneMetadata.
  set_goal_mocap(scene, goal_quat)  # type: ignore[arg-type]

  np.testing.assert_allclose(
    scene.data.mocap_pos[scene.goal_mocap_id],
    np.array([0.12, -0.07, 0.78], dtype=np.float64),
  )
  np.testing.assert_allclose(scene.data.mocap_quat[scene.goal_mocap_id], goal_quat)


def test_goal_vis_position_above_object_uses_live_object_position():
  object_pos = np.array([0.01, 0.02, 0.73], dtype=np.float32)

  got = goal_vis_position_above_object(object_pos)

  np.testing.assert_allclose(got, np.array([0.01, 0.02, 0.88], dtype=np.float32))


def test_wuji_hand_reorient_play_env_cfg_enables_debug_vis_only_for_play():
  train_cfg = wuji_hand_reorient_env_cfg(play=False)
  play_cfg = wuji_hand_reorient_env_cfg(play=True)

  assert train_cfg.commands["reorient_command"].debug_vis is False
  assert play_cfg.commands["reorient_command"].debug_vis is True


def test_format_reorient_status_markdown_shows_ori_error_and_success():
  got = format_reorient_status_markdown(ori_error_rad=np.pi / 6, success=True)

  assert "ori_error" in got
  assert "30.0 deg" in got
  assert "success" in got
  assert "True" in got


def test_format_reorient_status_markdown_shows_continuous_face_progress():
  got = format_reorient_status_markdown(
    ori_error_rad=np.pi / 12,
    success=False,
    target_face_id=3,
    consecutive_flip_count=4,
    face_coverage_count=5,
  )

  assert "target_face: 3" in got
  assert "flip_chain: 4" in got
  assert "face_coverage: 5/6" in got


def test_reorient_status_color_rgba_maps_success_to_green_and_failure_to_red():
  assert reorient_status_color_rgba(True) == (0.2, 0.9, 0.2, 0.95)
  assert reorient_status_color_rgba(False) == (0.95, 0.2, 0.2, 0.95)


def test_goal_vis_env_indices_draws_all_envs_not_just_selected_env():
  assert goal_vis_env_indices(num_envs=4) == [0, 1, 2, 3]


def test_visualization_helpers_remain_reachable_via_mdp_reexports():
  assert reorient_mdp.goal_vis_env_indices(3) == [0, 1, 2]
  assert reorient_mdp.format_reorient_status_markdown is format_reorient_status_markdown


def test_update_reorient_status_markdown_clamps_env_selection():
  markdown = SimpleNamespace(content="")

  update_reorient_status_markdown(
    status_markdown=markdown,
    get_env_idx=lambda: 99,
    num_envs=2,
    policy_status_for_env=lambda env_idx: (np.pi / 3, env_idx == 1),
  )

  assert "60.0 deg" in markdown.content
  assert "True" in markdown.content


def test_update_reorient_status_markdown_fast_returns_for_empty_envs():
  markdown = SimpleNamespace(content="unchanged")

  update_reorient_status_markdown(
    status_markdown=markdown,
    get_env_idx=lambda: 0,
    num_envs=0,
    policy_status_for_env=lambda env_idx: (_ for _ in ()).throw(
      AssertionError("should not run")
    ),
  )

  assert markdown.content == "unchanged"


def test_draw_reorient_frame_triads_adds_palm_and_tag_frames():
  visualizer = _StubVisualizer()
  palm_pose_w = torch.tensor(
    [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]],
    dtype=torch.float32,
  )

  draw_reorient_frame_triads(
    visualizer=visualizer,
    num_envs=1,
    palm_pose_w=palm_pose_w,
  )

  assert len(visualizer.frames) == 2
  assert visualizer.frames[0]["label"] == "palm_frame_env0"
  assert visualizer.frames[1]["label"] == "tag_frame_env0"


def test_viser_debug_visualization_emits_a_target_cube_ghost():
  visualizer = _StubVisualizer()
  visualization = ReorientCommandVisualization(entity_name="object")

  visualization.draw_debug_visuals(
    visualizer=visualizer,
    num_envs=1,
    palm_pose_w=torch.tensor(
      [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    ),
    object_pos_w=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
    goal_quat_w=torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], dtype=torch.float32),
    policy_status_for_env=lambda _env_idx: (np.pi / 2, False),
  )

  assert len(visualizer.ghosts) == 1
  assert visualizer.ghosts[0]["qpos"].shape == (7,)
  np.testing.assert_allclose(
    visualizer.ghosts[0]["qpos"][:3], np.array([0.1, 0.2, 0.55])
  )
  assert visualizer.ghosts[0]["label"] == "reorient_goal_cube_env0"


def test_native_target_uses_cube_mesh_so_face_rotation_is_visible():
  model = mujoco.MjModel.from_xml_string(
    """
    <mujoco model="goal_mesh_test">
      <asset>
        <mesh name="object/cube_mesh"
          vertex="-1 -1 -1 1 -1 -1 1 1 -1 -1 1 -1 -1 -1 1 1 -1 1 1 1 1 -1 1 1"/>
      </asset>
      <worldbody>
        <body name="cube">
          <freejoint name="cube_freejoint"/>
          <geom type="mesh" mesh="object/cube_mesh"
            contype="0" conaffinity="0"/>
        </body>
      </worldbody>
    </mujoco>
    """
  )
  scene = mujoco.MjvScene(model, maxgeom=64)
  from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

  visualizer = MujocoNativeDebugVisualizer(
    scn=scene,
    mj_model=model,
    env_idx=0,
  )
  visualization = ReorientCommandVisualization(entity_name="object")

  visualization.draw_debug_visuals(
    visualizer=visualizer,
    num_envs=1,
    palm_pose_w=torch.tensor(
      [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    ),
    object_pos_w=torch.tensor([[0.1, 0.2, 0.4]], dtype=torch.float32),
    goal_quat_w=torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], dtype=torch.float32),
    policy_status_for_env=lambda _env_idx: (np.pi / 2, False),
  )

  assert any(
    geom.type == mujoco.mjtGeom.mjGEOM_MESH.value and geom.texcoord == 1
    for geom in scene.geoms[: scene.ngeom]
  )


def test_continuous_status_tuple_is_accepted_by_debug_visualization():
  from wuji_mjlab.tasks.reorient.tests.test_continuous_face_command import (
    _make_command,
  )

  command, env = _make_command()
  command._resample_command(torch.tensor([0]))
  visualization = ReorientCommandVisualization(entity_name="object")
  visualizer = _StubVisualizer()

  visualization.draw_debug_visuals(
    visualizer=visualizer,
    num_envs=1,
    palm_pose_w=env.scene["robot"].data.body_link_pose_w[:, 0],
    object_pos_w=env.scene["object"].data.root_link_pos_w,
    goal_quat_w=command.goal_quat_w,
    policy_status_for_env=command._policy_status_for_env,
  )

  assert len(visualizer.ghosts) == 1


def test_visualization_adapter_owns_gui_refresh_state():
  visualization = ReorientCommandVisualization(entity_name="object")
  markdown = SimpleNamespace(content="")
  visualization.attach_status_targets(
    status_markdown=markdown,
    get_env_idx=lambda: 99,
  )

  visualization.update_status_gui(
    num_envs=2,
    policy_status_for_env=lambda env_idx: (np.pi / 4, env_idx == 1),
  )

  assert "45.0 deg" in markdown.content
  assert "True" in markdown.content
