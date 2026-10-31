from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp import observations


def sensor_module():
  from wuji_mjlab.tasks.reorient.mdp import smart_cube_observations

  return smart_cube_observations


def test_six_face_moves_delay_hysteresis_reversal_and_partial_reset():
  m = sensor_module()
  profile = m.SmartCubeProfile(threshold_degrees=(48.0, 48.0))
  state = m.create_smart_cube_state(2, "cpu", profile)
  angles = torch.zeros(2, 6)

  def read(step, degrees):
    angles[0, 0] = torch.deg2rad(torch.tensor(float(degrees)))
    return torch.rad2deg(m.step_move_model(angles, state, profile, step_id=step))

  assert read(0, 0)[0, 0] == 0
  assert read(1, 47)[0, 0] == 0
  assert read(2, 49)[0, 0] == 0
  assert read(2, 49)[0, 0] == 0  # same step must not advance buffer
  assert read(3, 47)[0, 0] == pytest.approx(90)
  assert read(4, 41)[0, 0] == pytest.approx(90)
  assert read(5, 0)[0, 0] == pytest.approx(0)
  read(6, -49)
  assert read(7, -90)[0, 0] == pytest.approx(-90)
  angles[0, 1] = torch.deg2rad(torch.tensor(180.0))
  m.step_move_model(angles, state, profile, step_id=8)
  result = m.step_move_model(angles, state, profile, step_id=9)
  assert torch.rad2deg(result[0, 1]) == pytest.approx(180)
  m.reset_smart_cube_state(state, torch.tensor([1]), profile)
  assert state.move_initialized.tolist() == [True, False]
  assert torch.rad2deg(state.delivered_angles[0, 1]) == pytest.approx(180)


def test_actor_angle_selects_delayed_face_and_keeps_critic_continuous():
  m = sensor_module()
  profile = m.SmartCubeProfile(threshold_degrees=(48.0, 48.0))
  joints = torch.zeros(1, 6)
  command = SimpleNamespace(
    cfg=SimpleNamespace(solver_mode=True),
    object=SimpleNamespace(data=SimpleNamespace(joint_pos=joints)),
    _face_driver_ids=torch.arange(6),
    target_face_id=torch.tensor([0]),
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene=SimpleNamespace(device="cpu"),
    common_step_counter=0,
    _smart_cube_profile=profile,
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  observations.rubik_smart_cube_angle_sincos(env)
  joints[0, 0] = torch.deg2rad(torch.tensor(49.0))
  env.common_step_counter = 1
  assert observations.rubik_smart_cube_angle_sincos(env)[0, 0] == 0
  env.common_step_counter = 2
  assert observations.rubik_smart_cube_angle_sincos(env)[0, 0] == pytest.approx(1)
  command.target_face_id[:] = 1
  assert observations.rubik_smart_cube_angle_sincos(env)[0, 0] == 0
  # Sensor processing must never mutate the continuous physical state.
  assert torch.rad2deg(joints[0, 0]) == pytest.approx(49)


def test_imu_shared_sample_drop_age_and_drift():
  m = sensor_module()
  p = m.SmartCubeProfile(
    imu_delay_steps=0,
    imu_noise_rad=0,
    calibration_noise_rad=0,
    yaw_drift_rad_per_s=0.1,
    imu_dropout_probability=0,
  )
  state = m.create_smart_cube_state(2, "cpu", p)
  state.yaw_rate[:] = 0.1
  q = torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1)
  first = m.step_imu_model(q, state, p, step_id=0, step_dt=0.05)
  second = m.step_imu_model(q, state, p, step_id=1, step_dt=0.05)
  assert second[0, 3] > first[0, 3]
  assert torch.equal(second, m.step_imu_model(q, state, p, step_id=1, step_dt=0.05))
  before_age = m.smart_cube_metadata(state, p)[:, 1].clone()
  dropped = replace(p, imu_dropout_probability=1)
  held = m.step_imu_model(q, state, dropped, step_id=2, step_dt=0.05)
  assert torch.equal(held, second)
  assert (m.smart_cube_metadata(state, p)[:, 1] > before_age).all()
  m.reset_smart_cube_state(state, torch.tensor([1]), p)
  assert state.imu_initialized.tolist() == [True, False]


def test_smart_task_preserves_actor_order_and_critic_terms():
  from wuji_mjlab.tasks.reorient.config.luwu_hand import rubik_env_cfg as cfgs

  base = cfgs.luwu_hand_rubik_full_cube_solver_transfer_env_cfg(num_envs=4)
  smart = cfgs.luwu_hand_rubik_smart_cube_transfer_env_cfg(num_envs=4)
  assert tuple(base.observations["policy"].terms) == tuple(
    smart.observations["policy"].terms
  )
  assert set(base.observations["critic"].terms) < set(smart.observations["critic"].terms)
  assert "rubik_smart_move_residual" in smart.observations["critic"].terms
  assert smart.observations["policy"].terms["rubik_u_face_angle"].history_length == 3
  assert not smart.commands["reorient_command"].verify_physical_facelets
  assert smart.commands["reorient_command"].verify_final_physical_facelets


def test_invalid_sensor_parameters_rejected():
  m = sensor_module()
  with pytest.raises(ValueError):
    m.SmartCubeProfile(threshold_degrees=(20, 60))


def test_exact_45_degree_boundary_does_not_chatter():
  m = sensor_module()
  p = m.SmartCubeProfile(threshold_degrees=(45, 45), move_delay_steps=0)
  state = m.create_smart_cube_state(1, "cpu", p)
  angles = torch.zeros(1, 6)
  m.step_move_model(angles, state, p, step_id=0)
  angles[0, 0] = torch.pi / 4
  values = [m.step_move_model(angles, state, p, step_id=i)[0, 0] for i in range(1, 4)]
  assert all(value == 0 for value in values)


def test_physical_boundary_verification_rejects_wrong_or_unaligned_cubies(monkeypatch):
  from wuji_mjlab.tasks.reorient.mdp import rubik_solver
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import RubikFullCubeCommand

  command = object.__new__(RubikFullCubeCommand)
  command.joint_names = ("test_joint",)
  command.object = SimpleNamespace(
    data=SimpleNamespace(joint_pos=torch.tensor([[0.0], [1.0], [2.0]]))
  )
  command.expected_joint_pos = torch.zeros(3, 1)
  monkeypatch.setattr(
    rubik_solver,
    "decode_actual_facelets",
    lambda row, _: {0.0: "expected", 1.0: "wrong", 2.0: None}[row["test_joint"]],
  )
  assert command._verify_physical_facelets(torch.arange(3)).tolist() == [1, 2]


def test_shared_actor_imu_and_critic_bypass():
  from wuji_mjlab.tasks.reorient.mdp import vision_observations

  m = sensor_module()
  p = m.SmartCubeProfile(imu_delay_steps=0, imu_noise_rad=0, calibration_noise_rad=0)
  palm = torch.tensor([[[0.0, 0, 0, 1, 0, 0, 0]]])
  command = SimpleNamespace(
    cfg=SimpleNamespace(solver_mode=False),
    robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=palm)),
    palm_body_id=0,
    object=SimpleNamespace(
      data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]]))
    ),
    target_face_id=torch.tensor([4]),
    phase=torch.tensor([0]),
    root_target_quat_w=torch.tensor([[1.0, 0, 0, 0]]),
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene=SimpleNamespace(device="cpu"),
    common_step_counter=0,
    step_dt=0.05,
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  clean = observations.rubik_cube_orientation_6d_in_tag(env, vision_dr=False)
  observations.rubik_cube_orientation_6d_in_tag(
    env, vision_dr=True, smart_cube_profile=p
  )
  state = m.get_smart_cube_state(env)
  state.calibration[:, 2] = 0.2
  env.common_step_counter = 1
  noisy = observations.rubik_cube_orientation_6d_in_tag(
    env, vision_dr=True, smart_cube_profile=p
  )
  before = state.orientation.clone()
  observations.rubik_compat_cube_orientation_error_6d(env, smart_cube_profile=p)
  assert torch.equal(state.orientation, before)
  assert not torch.allclose(clean, noisy)
  assert torch.equal(
    clean, observations.rubik_cube_orientation_6d_in_tag(env, vision_dr=False)
  )
  assert vision_observations.visual_pose_metadata(env).shape == (1, 4)


def test_physical_check_decodes_real_cubies_without_snapping():
  import mujoco
  from wuji_mjlab.assets.objects.rubik.rubik_cfg import get_rubik_cfg
  from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import RubikFullCubeCommand
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import (
    apply_face_move,
    apply_face_turn_angle,
  )

  model = get_rubik_cfg(full_cube=True).spec_fn().compile()
  names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
  )
  home = dict.fromkeys(names, 0.0)
  correct = apply_face_move(home, "R", names)
  halfway = apply_face_turn_angle(home, 0, torch.pi / 4, names)
  actual = torch.tensor([[row[n] for n in names] for row in (home, correct, halfway)])
  command = object.__new__(RubikFullCubeCommand)
  command.joint_names = names
  command.object = SimpleNamespace(data=SimpleNamespace(joint_pos=actual))
  command.expected_joint_pos = torch.tensor([[correct[n] for n in names]]).repeat(3, 1)
  before = actual.clone()
  assert command._verify_physical_facelets(torch.arange(3)).tolist() == [0, 2]
  assert torch.equal(actual, before)


def test_irregular_imu_transport_holds_and_never_reverses_packets():
  torch.manual_seed(42)
  s = sensor_module()
  profile = s.SmartCubeProfile(imu_latency_range_seconds=(.03, .12), imu_latency_mode_seconds=.09, imu_dropout_probability=0.,
    imu_noise_rad=0., calibration_noise_rad=0., yaw_drift_rad_per_s=0.)
  state = s.create_smart_cube_state(64, 'cpu', profile)
  previous = torch.zeros(64)
  held = 0
  ages = []
  for step in range(40):
    angle = torch.full((64,), step * .01)
    quat = torch.zeros(64, 4)
    quat[:, 0] = torch.cos(angle / 2)
    quat[:, 3] = torch.sin(angle / 2)
    result = s.step_imu_model(quat, state, profile, step_id=step, step_dt=.05)
    observed = 2 * torch.atan2(result[:, 3], result[:, 0])
    assert (observed >= previous - 1e-6).all()
    if step > 4:
      held += int(torch.isclose(observed, previous).sum())
      ages.append(state.age_seconds.clone())
    previous = observed
  assert held > 0
  assert torch.cat(ages).min() >= .05 - 1e-5
  assert torch.cat(ages).max() <= .2 + 1e-5


def test_imu_latency_triangular_distribution_and_cached_partial_reset():
  s = sensor_module()
  torch.manual_seed(13)
  p = s.SmartCubeProfile(imu_latency_range_seconds=(.03, .12),
    imu_latency_mode_seconds=.09, imu_dropout_probability=0.)
  state = s.create_smart_cube_state(20000, 'cpu', p)
  q = torch.zeros(20000, 4)
  q[:, 0] = 1
  s.step_imu_model(q, state, p, step_id=0, step_dt=.05)
  latency = state.imu_arrival_times[:, 0]
  assert latency.min() >= .03
  assert latency.max() <= .12
  assert latency.mean().item() == pytest.approx(.08, abs=.001)
  bins = torch.histc(latency, bins=9, min=.03, max=.12)
  assert int(bins.argmax()) in (5, 6)  # density peak near90 ms
  cached = state.imu_arrival_times.clone()
  s.step_imu_model(q, state, p, step_id=0, step_dt=.05)
  assert torch.equal(cached, state.imu_arrival_times)
  s.reset_smart_cube_state(state, torch.tensor([0]), p)
  assert torch.isinf(state.imu_arrival_times[0]).all()
  assert (state.imu_sample_steps[0] == -1).all()
  assert torch.equal(cached[1:], state.imu_arrival_times[1:])
