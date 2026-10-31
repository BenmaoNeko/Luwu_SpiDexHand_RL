from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg
from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
  apply_face_move,
  apply_face_turn_angle,
)
from wuji_mjlab.utils.rubik_layer_lock import RubikLayerLock


def test_original_u_layer_matches_training_torque_response():
  training = get_rubik_cfg().spec_fn().compile()
  dynamic = get_rubik_cfg(full_cube=True, dynamic_layers=True).spec_fn().compile()
  # Isolate the equality constraints from the preview's friction tuning.
  dynamic.geom_friction[:] = training.geom_friction
  dynamic.dof_frictionloss[:] = training.dof_frictionloss
  a, b = mujoco.MjData(training), mujoco.MjData(dynamic)
  names = tuple(
    dynamic.joint(i).name
    for i in range(dynamic.njnt)
    if dynamic.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  )
  sim = SimpleNamespace(
    mj_model=dynamic,
    mj_data=b,
    model=SimpleNamespace(eq_data=torch.from_numpy(dynamic.eq_data.copy()[None])),
    data=SimpleNamespace(
      qpos=torch.from_numpy(b.qpos[None]),
      qvel=torch.from_numpy(b.qvel[None]),
      eq_active=torch.from_numpy(b.eq_active.copy()[None]),
    ),
  )
  RubikLayerLock(sim, names).select(dict.fromkeys(names, 0.0), 4)
  b.eq_active[:] = sim.data.eq_active[0].numpy()
  for model, data in ((training, a), (dynamic, b)):
    model.opt.gravity[:] = 0
    data.qfrc_applied[model.joint("cubelet:driver:pos_z").dofadr] = 0.002
    for _ in range(30):
      mujoco.mj_step(model, data)
  np.testing.assert_allclose(a.qpos, b.qpos, atol=1e-6)
  np.testing.assert_allclose(a.qvel, b.qvel, atol=1e-5)


def test_solver_cube_friction_is_lower_without_changing_training_cube():
  training = get_rubik_cfg().spec_fn().compile()
  solver = get_rubik_cfg(full_cube=True, dynamic_layers=True).spec_fn().compile()
  cube_geoms = [
    i for i in range(solver.ngeom) if solver.geom(i).name.startswith("cubelet:")
  ]
  cube_dofs = [
    solver.joint(i).dofadr[0]
    for i in range(solver.njnt)
    if solver.joint(i).name.startswith("cubelet:driver:")
  ]
  np.testing.assert_allclose(
    training.geom_friction[cube_geoms],
    np.tile([0.2, 0.0025, 0.0005], (len(cube_geoms), 1)),
  )
  np.testing.assert_allclose(
    solver.geom_friction[cube_geoms],
    np.tile([0.16, 0.002, 0.0004], (len(cube_geoms), 1)),
  )
  np.testing.assert_allclose(training.dof_frictionloss[cube_dofs], 0.0005)
  np.testing.assert_allclose(solver.dof_frictionloss[cube_dofs], 0.0003)


@pytest.mark.parametrize("residual", [0.0, 2e-5, 4e-4, 2e-3])
def test_layer_switch_preserves_scramble_and_allows_only_selected_face(residual):
  spec = get_rubik_cfg(full_cube=True, dynamic_layers=True).spec_fn()
  model = spec.compile()
  assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
  data = mujoco.MjData(model)
  names = tuple(
    model.joint(i).name
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  )
  sim = SimpleNamespace(
    mj_model=model,
    mj_data=data,
    model=SimpleNamespace(eq_data=torch.from_numpy(model.eq_data.copy()[None])),
    data=SimpleNamespace(
      qpos=torch.from_numpy(data.qpos[None]),
      qvel=torch.from_numpy(data.qvel[None]),
      eq_active=torch.from_numpy(data.eq_active.copy()[None]),
    ),
  )
  lock = RubikLayerLock(sim, names)
  state = dict.fromkeys(names, 0.0)
  for move in ("R", "U", "F2"):
    state = apply_face_move(state, move, names)
  # Physics leaves tiny errors around singular Euler charts, unlike exact
  # reset snapshots. Layer switching must tolerate these numerical residuals.
  for name in names:
    if name.startswith("cubelet:roty:") and abs(np.cos(state[name])) < 1e-8:
      state[name] += residual

  def write_state(values):
    for name, angle in values.items():
      data.qpos[model.joint(name).qposadr[0]] = angle

  def equality_error():
    data.eq_active[:] = sim.data.eq_active[0].numpy()
    mujoco.mj_forward(model, data)
    return np.max(
      np.abs(data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]),
      initial=0,
    )

  for face in range(6):
    write_state(state)
    before = data.qpos.copy()
    lock.select(state)
    np.testing.assert_array_equal(data.qpos, before)
    assert sim.data.eq_active.sum() == 66
    assert equality_error() < 1e-8
    mujoco.mj_forward(model, data)
    before_pose = data.xmat.copy()
    lock.select(state, face)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xmat, before_pose, atol=max(1e-7, 5 * residual))
    assert sim.data.eq_active.sum() == 65  # 57 fixed hinges + 8 joint followers.
    assert equality_error() < 1e-8
    moving = apply_face_turn_angle(state, face, 0.2, names)
    write_state(moving)
    mujoco.mj_forward(model, data)
    expected_pose = data.xmat.copy()
    write_state(state)
    # Advance the selected driver and its followers in their chosen Euler chart.
    active_ids = np.flatnonzero(sim.data.eq_active[0].numpy())
    followers = [i for i in active_ids if model.eq_obj2id[i] >= 0]
    assert len(followers) == 8
    driver = model.eq_obj2id[followers[0]]
    data.qpos[model.jnt_qposadr[driver]] += 0.2
    for i in followers:
      data.qpos[model.jnt_qposadr[model.eq_obj1id[i]]] += 0.2 * model.eq_data[i, 1]
    assert equality_error() < 1e-7
    np.testing.assert_allclose(data.xmat, expected_pose, atol=max(1e-7, 5 * residual))
    # A different face cannot turn under the selected constraints.
    wrong = apply_face_turn_angle(state, (face + 2) % 6, 0.2, names)
    write_state(wrong)
    assert equality_error() > 1e-3
    state = apply_face_turn_angle(state, face, np.pi / 2, names)
