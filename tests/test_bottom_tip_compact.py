from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip_runtime import BottomTipRuntime


class _FakeEntity:
  def __init__(self, pose, names):
    self.data = SimpleNamespace(body_link_pose_w=pose)
    self._names = names

  def find_bodies(self, names):
    if isinstance(names, str):
      names = (names,)
    ids = [self._names.index(name) for name in names]
    return ids, list(names)


class _FakeSensor:
  def __init__(self, pos, normal, force, found):
    self.data = SimpleNamespace(pos=pos, normal=normal, force=force, found=found)


class _FakeScene(dict):
  """Scene supports the mapping and attribute access used by the runtime."""

  def __init__(self, **entities):
    super().__init__(entities)
    self.sensors = {}


def _quat_z(angle):
  half = angle / 2
  return torch.stack((torch.cos(half), torch.zeros_like(half),
                      torch.zeros_like(half), torch.sin(half)), dim=-1)


def _make_fixture(*, compact=False, debug=True, num_envs=2, randomized=False):
  device = torch.device("cpu")
  cubelet_pose = torch.zeros((num_envs, 26, 7), device=device)
  cubelet_pose[..., 3] = 1.0
  if randomized:
    angles = torch.linspace(-1.1, 1.2, num_envs * 26).reshape(num_envs, 26)
    cubelet_pose[..., 3:] = _quat_z(angles)
    cubelet_pose[..., :3] = torch.randn((num_envs, 26, 3), generator=torch.Generator().manual_seed(7)) * .03
  root_pose = torch.zeros((num_envs, 7), device=device)
  root_pose[:, 3] = 1.0
  object_entity = SimpleNamespace(data=SimpleNamespace(
    body_link_pose_w=cubelet_pose,
    root_link_pose_w=root_pose,
  ))

  robot_names = ["right_palm_link"] + [f"right_finger{i}_link4" for i in range(1, 5)]
  robot_pose = torch.zeros((num_envs, len(robot_names), 7), device=device)
  robot_pose[..., 3] = 1.0
  robot_entity = _FakeEntity(robot_pose, robot_names)

  env = SimpleNamespace(
    num_envs=num_envs,
    device=device,
    physics_dt=.01,
    step_dt=.02,
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    action_manager=SimpleNamespace(_terms={}),
    scene=_FakeScene(robot=robot_entity, object=object_entity),
  )
  cfg = SimpleNamespace(
    bottom_tip_contact_window_deg=45.,
    bottom_tip_success_rewards=None,
    bottom_tip_early_per_leg=False,
    bottom_tip_compact_contact=compact,
    bottom_tip_debug_diagnostics=debug,
  )
  command = SimpleNamespace(
    _env=env,
    cfg=cfg,
    object=object_entity,
    _cubelet_body_indices=torch.arange(26),
    flip_start_dot=torch.ones(num_envs),
  )

  runtime = BottomTipRuntime(command)
  # Exactly twelve selected patches: four finger channels on each of three
  # cubelets.  Every selected local normal points along +x.
  runtime.selected.zero_()
  runtime.selected[:, :3] = True
  runtime.local_normal.zero_()
  runtime.local_normal[:, :3, :, 0] = 1.
  runtime.initialized[:] = True
  runtime.geometry_valid[:] = True
  runtime._cache_patches(torch.arange(num_envs))

  # Every fingertip contacts cubelet zero's +x face.  Move each tip pose so
  # the same point is in the distal-cap region, including under rotation.
  from mjlab.utils.lab_api.math import quat_apply
  local_point = torch.zeros((num_envs, 3))
  local_point[:, 0] = .0095
  cube_zero = cubelet_pose[:, 0]
  point = cube_zero[:, :3] + quat_apply(cube_zero[:, 3:], local_point)
  tip_pose = robot_pose[:, 1:]
  tip_pose[..., :3] = point[:, None, :] + torch.tensor([.05, 0., 0.])
  robot_entity.data.body_link_pose_w = robot_pose
  normal = quat_apply(cube_zero[:, 3:], torch.tensor([[1., 0., 0.]]).expand(num_envs, -1))
  sensors = {}
  for finger in range(4):
    sensors[f"bottom_tip_{finger + 1}"] = _FakeSensor(
      point[:, None, :].clone(),
      normal[:, None, :].clone(),
      torch.tensor([[[.1, 0., 0.]]]).expand(num_envs, -1, -1).clone(),
      torch.ones((num_envs, 1)),
    )
  env.scene.sensors = sensors
  return runtime, env, command


def _start_and_sample(runtime, num_envs, frames=3):
  ids = torch.arange(num_envs)
  runtime.start(ids)
  runtime.initialized[:] = True
  runtime.geometry_valid[:] = True
  runtime._cache_patches(ids)
  for _ in range(frames):
    runtime.substep()


def _state(runtime):
  return (
    runtime.requests.supported.clone(),
    runtime.requests.duration.clone(),
    runtime.geometry_valid.clone(),
    runtime.overflow.clone(),
  )


def test_compact_runtime_allocates_twelve_patch_buffers_and_start_clears_cache():
  runtime, _, _ = _make_fixture(compact=True)
  assert runtime.compact_contact is True
  assert runtime.debug_diagnostics is True
  assert runtime.patch_cubelets.shape == (2, 12)
  assert runtime.patch_normals.shape == (2, 12, 3)
  assert runtime.patch_valid.shape == (2, 12)
  expected_cubelets = torch.arange(3).repeat_interleave(4).expand(2, -1)
  assert torch.equal(torch.sort(runtime.patch_cubelets, dim=1).values, expected_cubelets)
  assert torch.allclose(runtime.patch_normals[..., 0], torch.ones((2, 12)))
  assert runtime.patch_valid.all()

  runtime.start(torch.tensor([1]))
  assert not runtime.patch_valid[1].any()
  assert runtime.patch_valid[0].all()


def test_compact_contact_matches_dense_contact_on_randomly_rotated_cubelets():
  dense, _, _ = _make_fixture(compact=False, randomized=True)
  compact, _, _ = _make_fixture(compact=True, randomized=True)
  _start_and_sample(dense, 2)
  _start_and_sample(compact, 2)

  dense_state = _state(dense)
  compact_state = _state(compact)
  for expected, actual in zip(dense_state, compact_state, strict=True):
    assert torch.equal(expected, actual)
  assert dense.diagnostics.equal(compact.diagnostics)
  assert compact.requests.supported.all()


@pytest.mark.parametrize("case", ["threshold", "non_distal", "weakforce", "overflow"])
def test_compact_contact_fails_closed_for_contact_gate_cases(case):
  dense, dense_env, _ = _make_fixture(compact=False)
  compact, compact_env, _ = _make_fixture(compact=True)
  for env in (dense_env, compact_env):
    for sensor_wrapper in env.scene.sensors.values():
      sensor = sensor_wrapper.data
      if case == "threshold":
        sensor.pos[..., 1] = .011
      elif case == "non_distal":
        sensor.pos[..., 0] += .01
      elif case == "weakforce":
        sensor.force.zero_()
        sensor.force[..., 0] = .01
      else:
        sensor.pos = sensor.pos.expand(-1, 17, -1).clone()
        sensor.normal = sensor.normal.expand(-1, 17, -1).clone()
        sensor.force = sensor.force.expand(-1, 17, -1).clone()
        sensor.found = torch.full((2, 17), 17.)

  _start_and_sample(dense, 2)
  _start_and_sample(compact, 2)
  assert not dense.requests.supported.any()
  assert not compact.requests.supported.any()
  assert torch.equal(dense.requests.duration, compact.requests.duration)
  assert torch.equal(dense.overflow, compact.overflow)
  if case == "overflow":
    assert compact.overflow.all()


def test_invalid_geometry_is_fail_closed_in_compact_mode():
  runtime, env, _ = _make_fixture(compact=True)
  runtime.start(torch.arange(2))
  runtime.initialized[:] = True
  runtime.geometry_valid[:] = False
  runtime.patch_valid[:] = False
  runtime.substep()
  assert not runtime.requests.supported.any()
  assert not runtime.requests.duration.any()
  assert not runtime.geometry_valid.any()
  assert not runtime.patch_valid.any()
  assert env.scene.sensors["bottom_tip_1"].data.found.any()


def test_debug_diagnostics_toggle_does_not_change_contact_behavior():
  enabled, _, _ = _make_fixture(compact=True, debug=True)
  disabled, _, _ = _make_fixture(compact=True, debug=False)
  _start_and_sample(enabled, 2)
  _start_and_sample(disabled, 2)
  enabled_state, disabled_state = _state(enabled), _state(disabled)
  for expected, actual in zip(enabled_state, disabled_state, strict=True):
    assert torch.equal(expected, actual)
  assert enabled.diagnostics.sum() > 0
  assert disabled.diagnostics.sum() == 0


def test_relock_rebuilds_compact_cache_after_start():
  runtime, _, _ = _make_fixture(compact=True)
  calls = []
  original = runtime._cache_patches

  def record(ids):
    calls.append(ids.clone())
    original(ids)

  runtime._cache_patches = record
  runtime.start(torch.tensor([0]))
  assert not runtime.patch_valid[0].any()
  runtime._lock(torch.tensor([0]))
  assert calls and torch.equal(calls[-1], torch.tensor([0]))
  assert runtime.patch_valid.shape == (2, 12)


def test_real_cube_geometry_lock_and_moving_patches_match_dense():
  from mjlab.utils.lab_api.math import quat_apply

  trackers = []
  for compact in (False, True):
    runtime, env, command = _make_fixture(compact=compact)
    centers = torch.cartesian_prod(*(torch.tensor([-.02, 0., .02]),) * 3)
    centers = centers[centers.abs().sum(-1) > 0]
    rotation = _quat_z(torch.tensor([.3, -.7]))
    poses = command.object.data.body_link_pose_w
    poses[..., :3] = quat_apply(rotation[:, None].expand(-1, 26, -1), centers[None].expand(2, -1, -1))
    poses[..., 3:] = rotation[:, None]
    command.object.data.root_link_pose_w[:, 3:] = rotation
    env.scene['robot'].data.body_link_pose_w[:, 0, 3:] = rotation
    runtime.start(torch.arange(2))
    runtime._lock(torch.arange(2))
    assert runtime.geometry_valid.all()
    assert runtime.selected.sum((-1, -2)).tolist() == [12, 12]
    for world in range(2):
      cubelet, face = runtime.selected[world].nonzero()[0]
      normal = quat_apply(poses[world, cubelet, 3:], runtime.local_normal[world, cubelet, face])
      point = poses[world, cubelet, :3] + normal * .0095
      for sensor in env.scene.sensors.values():
        sensor.data.pos[world, 0] = point
        sensor.data.normal[world, 0] = normal
      env.scene['robot'].data.body_link_pose_w[world, 1:, :3] = point + torch.tensor([.05, 0., 0.])
    # Move the locked physical patches: a cached world-space patch would miss.
    poses[..., :3] += .1
    env.scene['robot'].data.body_link_pose_w[:, 1:, :3] += .1
    for sensor in env.scene.sensors.values():
      sensor.data.pos += .1
    for _ in range(3):
      runtime.substep()
    assert runtime.requests.supported.all()
    trackers.append(runtime)
  for expected, actual in zip(_state(trackers[0]), _state(trackers[1]), strict=True):
    assert torch.equal(expected, actual)
