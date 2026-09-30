from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from tools.robogym_cube_face_preview import (
  DEFAULT_TORQUE_PER_CUBIE_NM,
  apply_face_torque,
  build_preview_model,
  face_turn_parameters,
  reset_cube,
  reset_viewer_camera,
  select_face_cubies,
)


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  assert joint_id >= 0
  return float(data.qpos[model.jnt_qposadr[joint_id]])


def test_face_labels_map_to_robogym_axes_and_clockwise_signs():
  expected = {
    "L": (0, 0, math.pi / 2),
    "R": (0, 1, -math.pi / 2),
    "F": (1, 0, math.pi / 2),
    "B": (1, 1, -math.pi / 2),
    "D": (2, 0, math.pi / 2),
    "U": (2, 1, -math.pi / 2),
  }

  for face, clockwise in expected.items():
    axis, side, angle = face_turn_parameters(face, "cw")
    assert (axis, side) == (clockwise[0], clockwise[1])
    assert math.isclose(angle, clockwise[2])
    assert math.isclose(face_turn_parameters(face, "ccw")[2], -angle)


@pytest.mark.parametrize(
  ("face", "driver_name", "clockwise_angle"),
  [
    ("L", "cubelet:driver:neg_x", math.pi / 2),
    ("R", "cubelet:driver:pos_x", -math.pi / 2),
    ("F", "cubelet:driver:neg_y", math.pi / 2),
    ("B", "cubelet:driver:pos_y", -math.pi / 2),
    ("D", "cubelet:driver:neg_z", math.pi / 2),
    ("U", "cubelet:driver:pos_z", -math.pi / 2),
  ],
)
def test_face_torque_targets_layer_without_teleporting_qpos(
  face: str, driver_name: str, clockwise_angle: float
):
  model, data = build_preview_model()
  reset_cube(model, data)
  qpos_before_torque = data.qpos.copy()
  data.qfrc_applied[:] = 0.0

  selected_joint_count = apply_face_torque(model, data, face, "cw", 0.001)
  driver_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driver_name)
  driver_dof = model.jnt_dofadr[driver_id]

  assert selected_joint_count == 9
  assert math.isclose(float(data.qfrc_applied[driver_dof]), math.copysign(0.001, clockwise_angle))
  np.testing.assert_array_equal(data.qpos, qpos_before_torque)
  assert np.count_nonzero(np.abs(data.qfrc_applied) > 1e-9) == 9


def test_sustained_face_torque_moves_the_layer_through_physics_steps():
  model, data = build_preview_model()
  reset_cube(model, data)
  driver_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "cubelet:driver:pos_z"
  )
  initial_angle = _joint_qpos(model, data, "cubelet:driver:pos_z")

  for _ in range(5):
    data.qfrc_applied[:] = 0.0
    apply_face_torque(model, data, "U", "cw", 0.001)
    mujoco.mj_step(model, data)

  final_angle = _joint_qpos(model, data, "cubelet:driver:pos_z")
  assert final_angle < initial_angle
  assert abs(final_angle) < math.pi / 2
  assert driver_id >= 0


def test_default_torque_moves_a_cubie_in_the_ground_contacting_row():
  model, data = build_preview_model()
  reset_cube(model, data)
  selected = select_face_cubies(model, data, "B")
  bottom_row_joint = "cubelet:roty:neg_x_pos_y_neg_z"
  initial_angle = _joint_qpos(model, data, bottom_row_joint)

  for _ in range(30):
    apply_face_torque(
      model,
      data,
      "B",
      "cw",
      DEFAULT_TORQUE_PER_CUBIE_NM,
      selected_cubies=selected,
    )
    mujoco.mj_step(model, data, nstep=2)

  ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "preview_ground")
  floor_contact = any(
    data.contact[index].geom1 == ground_id or data.contact[index].geom2 == ground_id
    for index in range(data.ncon)
  )
  assert floor_contact
  assert abs(_joint_qpos(model, data, bottom_row_joint) - initial_angle) > 0.1


def test_held_turn_keeps_its_original_nine_cubies_if_one_deforms():
  model, data = build_preview_model()
  reset_cube(model, data)
  selected = select_face_cubies(model, data, "F")
  assert len(selected) == 9
  assert "neg_y_pos_z" in selected

  moved_piece_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "cubelet:rotz:neg_y_pos_z",
  )
  data.qpos[model.jnt_qposadr[moved_piece_joint]] = math.pi / 2
  mujoco.mj_forward(model, data)
  data.qfrc_applied[:] = 0.0

  still_selected = apply_face_torque(
    model, data, "F", "cw", 0.001, selected_cubies=selected
  )

  assert still_selected == 9
  piece_joint_names = (
    "cubelet:rotx:neg_y_pos_z",
    "cubelet:roty:neg_y_pos_z",
    "cubelet:rotz:neg_y_pos_z",
  )
  assert any(
    abs(
      float(
        data.qfrc_applied[
          model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
          ]
        ]
      )
    )
    > 1e-9
    for joint_name in piece_joint_names
  )


def test_preview_model_contains_ground_and_starts_cube_above_it():
  model, data = build_preview_model()
  ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "preview_ground")
  assert ground_id >= 0
  assert int(model.geom_type[ground_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
  assert math.isclose(_joint_qpos(model, data, "cube:tz"), 0.029, abs_tol=1e-7)


def test_reset_restores_all_cubelet_spring_positions():
  model, data = build_preview_model()
  spring_joint_ids = [
    joint_id
    for joint_id in range(model.njnt)
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith(
      "cubelet:spring:"
    )
  ]
  assert len(spring_joint_ids) == 20
  for index, joint_id in enumerate(spring_joint_ids):
    data.qpos[model.jnt_qposadr[joint_id]] = (index + 1) * 1e-5
  reset_cube(model, data)

  assert all(abs(float(data.qpos[model.jnt_qposadr[jid]])) < 1e-9 for jid in spring_joint_ids)


def test_reset_viewer_camera_restores_three_quarter_cube_view():
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(camera)
  camera.lookat[:] = (1.0, 1.0, 1.0)
  camera.distance = 0.8
  camera.azimuth = 0.0
  camera.elevation = 0.0

  reset_viewer_camera(camera)

  np.testing.assert_allclose(camera.lookat, (0.0, 0.0, 0.029))
  assert math.isclose(camera.distance, 0.23)
  assert math.isclose(camera.azimuth, 135.0)
  assert math.isclose(camera.elevation, -22.0)


def test_default_per_cubie_torque_is_large_enough_to_show_ground_contact_response():
  assert math.isclose(DEFAULT_TORQUE_PER_CUBIE_NM, 0.005)
