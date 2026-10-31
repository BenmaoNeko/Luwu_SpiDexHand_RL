from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp import smart_cube_curriculum as m


def _good_totals(requests: int = 100) -> tuple[int, int, int, int]:
  return (requests, int(requests * 0.85), int(requests * 0.70), int(requests * 0.60))


def test_low_friction_controller_starts_with_persistable_state_and_gated_progress():
  cfg = m.SmartCubeCurriculumCfg(low_friction_transition_steps=100)
  controller = m.SmartCubeCurriculum(cfg)

  assert controller.low_friction_progress == 0.0
  assert controller.low_friction_last_step == 0
  assert controller.low_friction_last_totals == (0, 0, 0, 0)
  assert {
    "low_friction_progress",
    "low_friction_last_step",
    "low_friction_last_totals",
  } <= vars(controller).keys()

  # The first settled window is held until the 25-step cooldown.
  good = _good_totals()
  m.update_low_friction_curriculum(controller, 24, good)
  assert controller.low_friction_progress == 0.0
  m.update_low_friction_curriculum(controller, 25, good)
  assert controller.low_friction_progress == pytest.approx(0.25)

  # A window with no new requests freezes and must not bank elapsed steps.
  m.update_low_friction_curriculum(controller, 50, good)
  assert controller.low_friction_progress == pytest.approx(0.25)
  m.update_low_friction_curriculum(controller, 75, _good_totals(200))
  assert controller.low_friction_progress == pytest.approx(0.50)

  # Both settled-success gates are required; 84/100 fails both gates here.
  m.update_low_friction_curriculum(controller, 100, (300, 254, 210, 170))
  assert controller.low_friction_progress == pytest.approx(0.50)


def test_low_friction_controller_rewinds_reset_baseline_without_advancing():
  controller = m.SmartCubeCurriculum(
    m.SmartCubeCurriculumCfg(low_friction_transition_steps=100)
  )
  m.update_low_friction_curriculum(controller, 25, _good_totals())
  assert controller.low_friction_progress == pytest.approx(0.25)

  # Cumulative counters rewound (for example after loading an older runner state).
  rewound_totals = (90, 70, 60, 50)
  m.update_low_friction_curriculum(controller, 50, rewound_totals)
  assert controller.low_friction_progress == pytest.approx(0.25)
  assert controller.low_friction_last_totals == rewound_totals

  m.update_low_friction_curriculum(controller, 75, (190, 162, 130, 114))
  assert controller.low_friction_progress == pytest.approx(0.50)

  # A step-counter rewind also resets the baseline and cannot advance progress.
  m.update_low_friction_curriculum(controller, 60, (200, 170, 140, 120))
  assert controller.low_friction_progress == pytest.approx(0.50)
  m.update_low_friction_curriculum(controller, 85, (300, 255, 210, 180))
  assert controller.low_friction_progress == pytest.approx(0.75)


def test_low_friction_controller_requires_each_settled_success_gate_and_caps():
  controller = m.SmartCubeCurriculum(
    m.SmartCubeCurriculumCfg(low_friction_transition_steps=100)
  )
  m.update_low_friction_curriculum(controller, 25, (100, 85, 70, 60))
  assert controller.low_friction_progress == pytest.approx(0.25)

  # Success rate passes, but supported-success rate is below .60.
  m.update_low_friction_curriculum(controller, 50, (200, 170, 140, 119))
  assert controller.low_friction_progress == pytest.approx(0.25)
  m.update_low_friction_curriculum(controller, 75, (300, 255, 210, 179))
  assert controller.low_friction_progress == pytest.approx(0.50)

  # Supported-success rate passes, but settled-success rate is below .85.
  m.update_low_friction_curriculum(controller, 100, (400, 339, 280, 239))
  assert controller.low_friction_progress == pytest.approx(0.50)
  m.update_low_friction_curriculum(controller, 125, (500, 424, 350, 299))
  assert controller.low_friction_progress == pytest.approx(0.75)

  # A late check advances by at most one cooldown window, then progress caps at 1.
  m.update_low_friction_curriculum(controller, 250, (600, 509, 420, 359))
  assert controller.low_friction_progress == pytest.approx(1.0)
  m.update_low_friction_curriculum(controller, 1000, (700, 594, 490, 419))
  assert controller.low_friction_progress == pytest.approx(1.0)


def test_low_friction_controller_does_not_bank_steps_across_empty_windows():
  controller = m.SmartCubeCurriculum(
    m.SmartCubeCurriculumCfg(low_friction_transition_steps=100)
  )
  m.update_low_friction_curriculum(controller, 25, (100, 85, 70, 60))
  m.update_low_friction_curriculum(controller, 50, (100, 85, 70, 60))
  assert controller.low_friction_progress == pytest.approx(0.25)

  # The long gap cannot bank 175 steps; the bounded increment is still 25/100.
  m.update_low_friction_curriculum(controller, 250, (200, 170, 140, 120))
  assert controller.low_friction_progress == pytest.approx(0.50)


def test_low_friction_controller_rejects_bad_windows_and_play_mode_is_fixed():
  with pytest.raises(ValueError):
    m.SmartCubeCurriculumCfg(low_friction_transition_steps=-1)

  play = m.SmartCubeCurriculum(
    m.SmartCubeCurriculumCfg(play=True, low_friction_transition_steps=100)
  )
  m.update_low_friction_curriculum(play, 100, _good_totals())
  assert play.low_friction_progress == 0.0

  disabled = m.SmartCubeCurriculum(
    m.SmartCubeCurriculumCfg(low_friction_transition_steps=0)
  )
  m.update_low_friction_curriculum(disabled, 100, _good_totals())
  assert disabled.low_friction_progress == 0.0


@pytest.mark.parametrize("finger", [False, True])
def test_low_friction_bounds_are_finite_and_reach_absolute_endpoints(finger):
  default = torch.tensor([0.20, 0.40, 0.80], dtype=torch.float32)
  start_low, start_high = m.low_friction_bounds(default, 0.0, finger=finger)
  end_low, end_high = m.low_friction_bounds(default, 1.0, finger=finger)

  assert torch.isfinite(start_low).all() and torch.isfinite(start_high).all()
  assert torch.isfinite(end_low).all() and torch.isfinite(end_high).all()
  if finger:
    torch.testing.assert_close(start_low, default * 0.05)
    torch.testing.assert_close(start_high, default * 0.10)
  else:
    torch.testing.assert_close(start_low, default)
    torch.testing.assert_close(start_high, default)
  torch.testing.assert_close(end_low, torch.full_like(default, 0.001))
  torch.testing.assert_close(end_high, torch.full_like(default, 0.005))

  midpoint_low, midpoint_high = m.low_friction_bounds(default, 0.5, finger=finger)
  torch.testing.assert_close(midpoint_low, torch.sqrt(start_low * end_low))
  torch.testing.assert_close(midpoint_high, torch.sqrt(start_high * end_high))

  with pytest.raises(ValueError):
    m.low_friction_bounds(torch.tensor([float("nan")]), 0.5, finger=finger)
  with pytest.raises(ValueError):
    m.low_friction_bounds(default, float("nan"), finger=finger)


def test_low_friction_surface_randomization_is_partial_reset_and_sliding_only():
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  defaults = torch.tensor([
    [[0.40, 0.70, 0.90], [0.20, 0.80, 1.10], [0.60, 0.90, 1.20],
     [0.30, 0.60, 1.00], [0.50, 0.75, 1.30]],
  ]).repeat(4, 1, 1)
  model = SimpleNamespace(geom_friction=defaults.clone())
  default_field = defaults[0].clone()
  robot = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.arange(5)))
  cube = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.arange(5)))
  env = SimpleNamespace(
    device="cpu",
    num_envs=4,
    sim=SimpleNamespace(
      model=model,
      get_default_field=lambda field: default_field,
    ),
    scene={"robot": robot, "object": cube},
    _smart_cube_curriculum=SimpleNamespace(low_friction_progress=0.5),
  )
  ids = torch.tensor([1, 3])
  finger_cfg = SceneEntityCfg("robot", geom_names=(r".*finger.*_col",))
  cube_cfg = SceneEntityCfg("object", geom_names=(r"cubelet:.*",))
  finger_cfg.geom_ids = [1, 2]
  cube_cfg.geom_ids = [3, 4]

  torch.manual_seed(7)
  before = model.geom_friction.clone()
  m.randomize_low_friction_surfaces(env, ids, finger_cfg, cube_cfg)

  # Only the selected envs and geom groups change, and only sliding friction (axis 0).
  assert torch.equal(model.geom_friction[0], before[0])
  assert torch.equal(model.geom_friction[2], before[2])
  assert torch.equal(model.geom_friction[:, 0], before[:, 0])  # palm remains untouched
  assert torch.equal(model.geom_friction[:, :, 1:], before[:, :, 1:])
  assert not torch.equal(
    model.geom_friction[ids][:, [1, 2, 3, 4], 0],
    before[ids][:, [1, 2, 3, 4], 0],
  )
  assert set(env._low_friction_bounds) == {
    "curriculum/finger_sliding_min",
    "curriculum/finger_sliding_max",
    "curriculum/cube_sliding_min",
    "curriculum/cube_sliding_max",
  }


def test_low_friction_surface_randomization_uses_absolute_targets_at_final_progress():
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  default = torch.tensor([
    [0.40, 0.70, 0.90], [0.20, 0.80, 1.10], [0.60, 0.90, 1.20],
    [0.30, 0.60, 1.00], [0.50, 0.75, 1.30],
  ])
  model = SimpleNamespace(geom_friction=default.unsqueeze(0).repeat(2, 1, 1))
  robot = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.arange(5)))
  cube = SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.arange(5)))
  env = SimpleNamespace(
    device="cpu",
    num_envs=2,
    sim=SimpleNamespace(
      model=model,
      get_default_field=lambda field: default,
    ),
    scene={"robot": robot, "object": cube},
    _smart_cube_curriculum=SimpleNamespace(low_friction_progress=1.0),
  )
  finger_cfg = SceneEntityCfg("robot", geom_names=(r".*finger.*_col",))
  cube_cfg = SceneEntityCfg("object", geom_names=(r"cubelet:.*",))
  finger_cfg.geom_ids = [1, 2]
  cube_cfg.geom_ids = [3, 4]

  torch.manual_seed(11)
  m.randomize_low_friction_surfaces(
    env, torch.tensor([0, 1]), finger_cfg, cube_cfg
  )
  selected = model.geom_friction[:, [1, 2, 3, 4], 0]
  assert (selected >= 0.001).all() and (selected <= 0.005).all()
  assert env._low_friction_bounds["curriculum/finger_sliding_min"].item() == pytest.approx(0.001)
  assert env._low_friction_bounds["curriculum/finger_sliding_max"].item() == pytest.approx(0.005)
  assert env._low_friction_bounds["curriculum/cube_sliding_min"].item() == pytest.approx(0.001)
  assert env._low_friction_bounds["curriculum/cube_sliding_max"].item() == pytest.approx(0.005)


def test_training_config_wires_low_friction_reset_event_and_transition():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_bottom_tip_train_env_cfg,
  )

  cfg = luwu_hand_rubik_flip_bottom_tip_train_env_cfg(play=False, num_envs=4)
  command_cfg = cfg.commands["reorient_command"]
  assert command_cfg.smart_cube_curriculum.low_friction_transition_steps == 20_000

  event = cfg.events["low_friction_surfaces"]
  assert event.mode == "reset"
  finger_cfg = event.params["finger_cfg"]
  cube_cfg = event.params["cube_cfg"]
  assert finger_cfg.name == "robot"
  assert cube_cfg.name == "object"
  assert event.params['palm_cfg'].name == 'robot'
  assert event.params['palm_cfg'].geom_names == r'.*palm_.*collision'
  finger_patterns = (finger_cfg.geom_names,) if isinstance(finger_cfg.geom_names, str) else finger_cfg.geom_names
  cube_patterns = (cube_cfg.geom_names,) if isinstance(cube_cfg.geom_names, str) else cube_cfg.geom_names
  assert finger_patterns and all(
    "finger" in pattern.lower() and "palm" not in pattern.lower()
    for pattern in finger_patterns
  )
  assert cube_patterns and all(
    "cubelet" in pattern.lower() for pattern in cube_patterns
  )


def test_palm_half_friction_uses_defaults_and_does_not_compound():
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  defaults = torch.tensor([[.8, .05, .01], [.8, .05, .01], [.2, .0025, .0005]])
  env = SimpleNamespace(device='cpu', num_envs=2,
    sim=SimpleNamespace(model=SimpleNamespace(geom_friction=defaults.repeat(2, 1, 1)),
      get_default_field=lambda _: defaults),
    scene={'robot': SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.arange(3))),
           'object': SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.arange(3)))},
    _smart_cube_curriculum=SimpleNamespace(alpha=1., low_friction_progress=0.))
  palm, finger, cube = SceneEntityCfg('robot'), SceneEntityCfg('robot'), SceneEntityCfg('object')
  palm.geom_ids, finger.geom_ids, cube.geom_ids = [0], [1], [2]
  before = env.sim.model.geom_friction.clone()
  for _ in range(3):
    m.randomize_low_friction_surfaces(env, torch.tensor([1]), finger, cube, palm_cfg=palm)
    actual = env.sim.model.geom_friction
    assert .02 - 1e-7 <= actual[1, 0, 0] <= .04 + 1e-7
    assert torch.equal(actual[0], before[0])
    assert torch.equal(actual[:, :, 1:], before[:, :, 1:])
