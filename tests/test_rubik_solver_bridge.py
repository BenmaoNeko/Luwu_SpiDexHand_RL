from __future__ import annotations

import numpy as np
from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
  aligned_target_state,
  apply_face_move,
  apply_face_turn_angle,
  face_id_for_move,
  face_layer_joint_names,
  face_turn_parameters,
  invert_moves,
  joint_state_max_error,
  solver_face_frame_quaternion,
)


def test_full_cube_friction_is_restored_without_changing_grip():
  import mujoco
  from wuji_mjlab.assets.objects.rubik.rubik_cfg import (
    get_rubik_cfg,
    get_rubik_target_cfg,
  )

  full = get_rubik_cfg(full_cube=True).spec_fn().compile()
  training = get_rubik_cfg().spec_fn().compile()
  target = get_rubik_target_cfg(full_cube=True).spec_fn().compile()
  for model, resistance in ((full, 0.0005), (training, 0.0005)):
    hinge_dofs = model.jnt_dofadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    np.testing.assert_allclose(model.dof_damping[hinge_dofs], resistance)
    np.testing.assert_allclose(model.dof_frictionloss[hinge_dofs], 0.0005)
    np.testing.assert_allclose(model.dof_armature[hinge_dofs], 0.0001)
  cube_ids = [i for i in range(full.ngeom) if full.geom(i).name.startswith("cubelet:")]
  np.testing.assert_allclose(
    full.geom_friction[cube_ids],
    np.tile([0.2, 0.0025, 0.0005], (len(cube_ids), 1)),
  )
  assert full.npair == training.npair == target.npair == 0
  assert np.all(full.geom_condim[cube_ids] == 6)
  np.testing.assert_allclose(full.body_mass, training.body_mass)
  np.testing.assert_allclose(full.body_inertia, training.body_inertia)


def test_full_cube_uses_original_geometry_and_inertia():
  from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg

  full_spec = get_rubik_cfg(full_cube=True).spec_fn()
  base_spec = get_rubik_cfg().spec_fn()
  full = full_spec.compile()
  base = base_spec.compile()
  np.testing.assert_allclose(full.body_mass, base.body_mass)
  np.testing.assert_allclose(full.body_inertia, base.body_inertia)
  np.testing.assert_allclose(full.body_ipos, base.body_ipos)
  assert full_spec.mesh("rounded_cube_clearance") is None
  assert full.ngeom == base.ngeom
  np.testing.assert_allclose(full.mesh_vert, base.mesh_vert)
  for body in full_spec.bodies:
    for geom in body.geoms:
      if not geom.name.startswith("cubelet:"):
        continue
      original = base_spec.geom(geom.name)
      assert full_spec.geom(f"visual:{geom.name}") is None
      assert geom.meshname == original.meshname == "rounded_cube"
      assert geom.material == original.material
      assert geom.contype == original.contype
      assert geom.conaffinity == original.conaffinity
      np.testing.assert_allclose(geom.pos, original.pos)
      np.testing.assert_allclose(geom.rgba, original.rgba)
      np.testing.assert_allclose(geom.density, original.density)


def test_solver_face_turns_use_outward_normal_signs():
  assert face_turn_parameters("U") == (2, 1, -90)
  assert face_turn_parameters("D") == (2, -1, 90)
  assert face_turn_parameters("R'") == (0, 1, 90)
  assert face_turn_parameters("L") == (0, -1, 90)
  assert [face_id_for_move(token) for token in ("R", "L", "B", "F", "U", "D")] == [
    0,
    1,
    2,
    3,
    4,
    5,
  ]


def test_actor_face_frame_maps_each_selected_normal_to_local_positive_z():
  import torch
  from wuji_mjlab.tasks.reorient.mdp.cube_faces import FACE_NORMALS

  face_ids = torch.arange(6)
  quats = solver_face_frame_quaternion(face_ids, dtype=torch.float32)
  rotated_z = torch.stack(
    (
      2 * (quats[:, 1] * quats[:, 3] + quats[:, 0] * quats[:, 2]),
      2 * (quats[:, 2] * quats[:, 3] - quats[:, 0] * quats[:, 1]),
      1 - 2 * (quats[:, 1].square() + quats[:, 2].square()),
    ),
    dim=-1,
  )
  assert torch.allclose(rotated_z, FACE_NORMALS, atol=1e-6)


def test_inverse_moves_reverse_order_and_invert_quarter_turns():
  assert invert_moves(("R", "U'", "F2")) == ("F2", "U", "R'")


def test_four_quarter_turns_restore_cubelet_joint_state():
  joint_names = [
    "cubelet:driver:pos_x",
    "cubelet:driver:neg_x",
    "cubelet:driver:pos_y",
    "cubelet:driver:neg_y",
    "cubelet:driver:pos_z",
    "cubelet:driver:neg_z",
  ]
  joint_names.extend(
    f"cubelet:rot{axis}:{piece}"
    for piece in (
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
    for axis in "xyz"
  )
  state = {name: 0.0 for name in joint_names}
  initial = {name: np.asarray((state[name],)) for name in state}
  initial_state = {name: value[0] for name, value in initial.items()}
  for face in "URFDLB":
    for _ in range(4):
      state = apply_face_move(state, face, joint_names)
    assert joint_state_max_error(initial_state, state, joint_names) < 1e-7

  sequence = ("R", "U", "R'", "U'")
  for token in sequence + invert_moves(sequence):
    state = apply_face_move(state, token, joint_names)
  assert joint_state_max_error(initial_state, state, joint_names) < 1e-7


def test_full_cube_spec_unlocks_all_face_and_piece_hinges_without_changing_default():
  from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg

  constrained = get_rubik_cfg().spec_fn()
  unlocked = get_rubik_cfg(full_cube=True).spec_fn()
  assert len(constrained.equalities) > 0
  assert len(unlocked.equalities) == 0
  assert len(constrained.joints) == len(unlocked.joints)
  assert constrained.compile().neq > 0
  assert unlocked.compile().neq == 0


def test_face_layer_joint_names_follow_the_current_scrambled_cubelet_positions():
  joint_names = [
    "cubelet:driver:pos_x",
    "cubelet:driver:neg_x",
    "cubelet:driver:pos_y",
    "cubelet:driver:neg_y",
    "cubelet:driver:pos_z",
    "cubelet:driver:neg_z",
  ]
  pieces = (
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
  joint_names.extend(f"cubelet:rot{axis}:{piece}" for piece in pieces for axis in "xyz")
  state = {name: 0.0 for name in joint_names}

  u_layer = set(face_layer_joint_names(4, state, joint_names))
  assert "cubelet:driver:pos_z" in u_layer
  assert len(u_layer) == 9
  assert all(
    name.startswith("cubelet:rotz:") for name in u_layer - {"cubelet:driver:pos_z"}
  )

  scrambled = apply_face_move(state, "R", joint_names)
  scrambled_u_layer = set(face_layer_joint_names(4, scrambled, joint_names))
  assert len(scrambled_u_layer) == 9
  assert scrambled_u_layer != u_layer


def test_solver_preview_uses_dynamic_layers_only_for_physical_object(
  monkeypatch, tmp_path
):
  import hashlib
  import sys
  from types import SimpleNamespace

  from scripts.play import play_rubik_solver as preview

  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"test checkpoint")
  monkeypatch.setattr(preview, "CHECKPOINT", checkpoint)
  monkeypatch.setattr(
    preview,
    "CHECKPOINT_SHA256",
    hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
  )

  env_cfg = SimpleNamespace(
    scene=SimpleNamespace(num_envs=1, entities={}),
    sim=SimpleNamespace(nconmax=1, njmax=1),
    commands={"reorient_command": SimpleNamespace(solver_mode=False)},
    episode_length_s=50.0,
  )
  captured = {}
  monkeypatch.setattr(
    preview, "prepare_task_cfgs", lambda *_args, **_kwargs: (env_cfg, object())
  )
  monkeypatch.setattr(
    preview,
    "run_play_with_cfg",
    lambda _task, _play_cfg, cfg, _agent_cfg, **_kwargs: captured.update(env=cfg),
  )
  monkeypatch.setattr(sys, "argv", ["play_rubik_solver.py", "--random"])

  preview.main()

  for entity_name in ("object", "target"):
    spec = captured["env"].scene.entities[entity_name].spec_fn()
    assert len(spec.equalities) == (426 if entity_name == "object" else 0)
    assert all(not eq.active for eq in spec.equalities)


def test_actual_facelet_decoder_matches_multimove_scramble_and_rejects_half_turn():
  import mujoco
  from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
    decode_actual_facelets,
    make_solver_plan,
  )

  model = get_rubik_cfg(full_cube=True).spec_fn().compile()
  joint_names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    and (
      (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
      and (name.startswith("cubelet:driver:") or name.startswith("cubelet:rot"))
    )
  )
  state = {name: 0.0 for name in joint_names}
  scramble = "R U F2 L' D B"
  for move in scramble.split():
    state = apply_face_move(state, move, joint_names)
  snapshot = dict(state)

  facelets = decode_actual_facelets(state, joint_names)
  assert facelets == make_solver_plan(scramble=scramble)["initialFacelets"]
  assert state == snapshot

  # A nearly aligned layer should decode to its nearest legal quarter turn.
  # Check all faces on a scrambled cube, without changing the physical snapshot.
  for face_id in range(6):
    for degrees in (-14.9, -10.0, 10.0, 14.9):
      near = apply_face_turn_angle(state, face_id, np.deg2rad(degrees), joint_names)
      near_snapshot = dict(near)
      assert decode_actual_facelets(near, joint_names) == facelets
      target = aligned_target_state(near, joint_names)
      assert decode_actual_facelets(target, joint_names) == facelets
      assert joint_state_max_error(target, state, joint_names) < 1e-6
      turned_target = apply_face_move(target, "U", joint_names)
      assert decode_actual_facelets(
        turned_target, joint_names
      ) == decode_actual_facelets(apply_face_move(state, "U", joint_names), joint_names)
      assert near == near_snapshot
    for degrees in (-45.0, -16.0, 16.0, 45.0):
      off_grid = apply_face_turn_angle(state, face_id, np.deg2rad(degrees), joint_names)
      assert decode_actual_facelets(off_grid, joint_names) is None

  halfway = {name: 0.0 for name in joint_names}
  halfway["cubelet:driver:pos_x"] = np.pi / 4
  for name in joint_names:
    if name.startswith("cubelet:rotx:") and "pos_x" in name:
      piece = name.removeprefix("cubelet:rotx:")
      halfway[f"cubelet:rotx:{piece}"] = np.pi / 4
  assert decode_actual_facelets(halfway, joint_names) is None


def test_non_grid_recovery_target_is_constructed_without_mutating_actual_state():
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
    decode_actual_facelets,
    face_id_for_driver_joint,
    make_solver_plan,
  )

  joint_names = [
    "cubelet:driver:pos_x",
    "cubelet:driver:neg_x",
    "cubelet:driver:pos_y",
    "cubelet:driver:neg_y",
    "cubelet:driver:pos_z",
    "cubelet:driver:neg_z",
  ]
  pieces = (
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
  joint_names.extend(f"cubelet:rot{axis}:{piece}" for piece in pieces for axis in "xyz")
  legal_state = {name: 0.0 for name in joint_names}
  face_id = face_id_for_driver_joint("cubelet:driver:pos_x")
  halfway_state = apply_face_turn_angle(
    legal_state, face_id, np.deg2rad(45.0), joint_names
  )
  snapshot = dict(halfway_state)

  assert decode_actual_facelets(halfway_state, joint_names) is None
  recovered_state = apply_face_turn_angle(
    halfway_state, face_id, np.deg2rad(45.0), joint_names
  )
  assert (
    decode_actual_facelets(recovered_state, joint_names)
    == make_solver_plan(scramble="R'")["initialFacelets"]
  )
  assert halfway_state == snapshot

  # A completed R turn plus a partial F turn has only one unaligned layer.
  from types import SimpleNamespace

  from wuji_mjlab.utils.play_runner import _RubikSolverPolicy

  actual = apply_face_move(legal_state, "R", joint_names)
  actual = apply_face_turn_angle(actual, 3, np.deg2rad(35.0), joint_names)
  original = dict(actual)
  installed = []
  policy = object.__new__(_RubikSolverPolicy)
  policy._joint_names = joint_names
  policy._last_legal_state = legal_state
  policy._recovery_count = 0
  policy._phase_steps = 100
  policy._phase = "turn"
  policy._status = "running"
  policy._solution = ("F",)
  policy._move_index = 0
  policy._command = SimpleNamespace(u_angle=[0.0])
  policy._install_goal = lambda **kwargs: installed.append(kwargs)
  policy._begin_recovery(actual, None)
  assert policy._status == "running"
  assert policy._phase == "recover_flip"
  assert installed[0]["face_id"] == 3
  assert np.isclose(policy._recovery_delta, np.deg2rad(-35.0))
  assert actual == original

  # Two genuinely off-grid opposite layers cannot be repaired as one layer.
  multiple = apply_face_turn_angle(legal_state, 0, np.deg2rad(35.0), joint_names)
  multiple = apply_face_turn_angle(multiple, 1, np.deg2rad(35.0), joint_names)
  policy._phase = "turn"
  policy._phase_steps = 100
  policy._begin_recovery(multiple, None)
  assert policy._status == "running"
  assert policy._phase == "turn"
  assert len(installed) == 1
  policy._phase_steps = policy._PHASE_STEP_LIMIT + 1
  policy._begin_recovery(multiple, None)
  assert policy._status == "failed"
  assert "alignment deltas deg" in policy._result


def test_unfinished_turn_is_not_interrupted_by_off_grid_stall(monkeypatch):
  from types import SimpleNamespace

  from wuji_mjlab.utils import play_runner

  policy = object.__new__(play_runner._RubikSolverPolicy)
  policy._raw_env = SimpleNamespace(step_dt=0.1)
  policy._monitor_accumulator = 0.0
  policy._invalid_duration = 0.0
  policy._phase = "turn"
  policy._turn_boundary_pending = False
  policy._flip_boundary_pending = False
  policy._joint_names = ()
  policy._joint_state = lambda: {}
  policy._stationary = lambda: True
  attempts = []
  policy._begin_recovery = lambda *args: attempts.append(args)
  monkeypatch.setattr(play_runner, "decode_actual_facelets", lambda *args: None)
  for _ in range(30):
    policy._monitor_actual_state(None)
  assert attempts == []
  policy._turn_boundary_pending = True
  policy._monitor_actual_state(None)
  assert len(attempts) == 1


def test_stable_actual_facelet_change_is_the_state_used_for_replanning(monkeypatch):
  from types import SimpleNamespace

  import torch
  from wuji_mjlab.utils import play_runner

  actual_state = {"cubelet:driver:pos_x": 0.25}
  policy = object.__new__(play_runner._RubikSolverPolicy)
  policy._raw_env = SimpleNamespace(step_dt=0.1)
  policy._object = SimpleNamespace(
    data=SimpleNamespace(
      joint_vel=torch.zeros((1, 1)),
      root_link_vel_w=torch.zeros((1, 6)),
    )
  )
  policy._joint_names = tuple(actual_state)
  policy._phase_start_facelets = "planned-state"
  policy._candidate_facelets = "planned-state"
  policy._candidate_duration = 0.0
  policy._invalid_duration = 0.0
  policy._monitor_accumulator = 0.0
  policy._phase = "turn"
  policy._replan_count = 0
  policy._joint_state = lambda: actual_state
  policy._stationary = lambda: True
  accepted = []
  policy._accept_actual_state = lambda state, facelets, **_kwargs: accepted.append(
    (dict(state), facelets)
  )
  monkeypatch.setattr(
    play_runner,
    "decode_actual_facelets",
    lambda _state, _names: "actual-physical-facelets",
  )

  policy._monitor_actual_state(None)
  policy._monitor_actual_state(None)

  # An intermediate legal detent must not interrupt an active turn.
  assert accepted == []
  policy._phase = "flip"
  policy._monitor_actual_state(None)
  assert accepted == [(actual_state, "actual-physical-facelets")]


def test_solver_plan_is_requested_from_the_observed_facelets(monkeypatch):
  from types import SimpleNamespace

  import torch
  from wuji_mjlab.utils import play_runner

  # This scheduling test uses a synthetic one-joint state, not a physical cube.
  monkeypatch.setattr(play_runner, "aligned_target_state", lambda state, names: state)

  observed = "actual-legal-cube-state"
  calls = []
  installed_goals = []
  policy = object.__new__(play_runner._RubikSolverPolicy)
  policy._command = SimpleNamespace(
    u_angle=torch.zeros(1),
    _read_top_face_id=lambda: [4],
    set_solver_goal=lambda *args: installed_goals.append(args),
  )
  policy._raw_env = SimpleNamespace(
    device="cpu", sim=SimpleNamespace(forward=lambda: None)
  )
  policy._target = SimpleNamespace(
    data=SimpleNamespace(default_joint_pos=torch.zeros((1, 1)))
  )
  policy._target_joint_names = ("cubelet:driver:pos_x",)
  policy._target_tensor = lambda _state: torch.zeros((1, 1))
  policy._layer_lock = SimpleNamespace(select=lambda *args: None)
  policy._joint_state = lambda: {}
  policy._refresh_object_observation_history = lambda: None
  policy._replan_count = 4
  policy._flip_boundary_pending = True
  policy._turn_boundary_pending = True
  monkeypatch.setattr(
    play_runner,
    "make_solver_plan",
    lambda **kwargs: calls.append(kwargs) or {"solution": ["R"]},
  )

  policy._accept_actual_state({"cubelet:driver:pos_x": 0.3}, observed)

  assert calls == [{"facelets": observed}]
  assert policy._solution == ("R",)
  assert policy._status == "running"
  assert installed_goals[0][0:2] == (face_id_for_move("R"), 0)
  assert policy._flip_boundary_pending is False
  assert policy._turn_boundary_pending is False

  # No fresh completion event: a new turn must retain its goal over many steps.
  policy._phase = "turn"
  policy._command.turn_completed = torch.tensor([False])
  policy._geometric_goal_reached = lambda **_: False
  policy._raw_env.termination_manager = SimpleNamespace(
    get_term=lambda _name: torch.tensor([False])
  )
  policy._raw_env.step_dt = 0.05
  policy._monitor_actual_state = lambda _viewer: None
  for _ in range(100):
    policy.after_step(None)
  assert policy._phase == "turn"
  assert policy._phase_steps == 100
  assert len(calls) == 1
  assert len(installed_goals) == 1

  policy._install_goal(0, 1, np.deg2rad(100.0), {"cubelet:driver:pos_x": 0.3})
  assert np.isclose(installed_goals[-1][2], np.pi / 2)
