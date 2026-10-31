from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp import smart_cube_curriculum as m
from wuji_mjlab.tasks.reorient.mdp.smart_cube_observations import SmartCubeProfile


def test_beta_alpha_cooldown_hysteresis_and_empty_windows():
  c = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg(ema_smoothing=1))
  assert (c.beta, c.alpha) == (0.3, 0)
  c.record_outcomes(8, 10)
  c.update(24)
  assert c.beta == 0.3
  c.update(25)
  assert c.beta == pytest.approx(0.4)
  assert c.alpha == 0
  for step in range(50, 176, 25):
    c.record_outcomes(8, 10)
    c.update(step)
  assert c.beta == 1
  assert c.alpha == 0  # unlock on the following check
  c.record_outcomes(8, 10)
  c.update(200)
  assert c.alpha == pytest.approx(0.05)
  c.record_outcomes(6, 10)
  c.update(225)
  assert c.alpha == pytest.approx(0.05)
  c.record_outcomes(4, 10)
  c.update(250)
  assert c.alpha == pytest.approx(0.025)
  c.update(275)
  assert c.alpha == pytest.approx(0.025)  # stale EMA cannot promote
  c.update(275)
  assert c.alpha == pytest.approx(0.025)


def test_beta_time_fallback_and_fixed_play():
  c = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg(beta_transition_steps=100))
  c.update(100)
  assert c.beta == 1 and c.alpha == 0
  p = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg(play=True, play_alpha=0.9))
  p.record_outcomes(0, 10)
  p.update(10000)
  assert (p.beta, p.alpha) == (1, 0.9)
  with pytest.raises(ValueError):
    m.SmartCubeCurriculumCfg(play_alpha=1.1)


def test_sensor_profile_endpoints():
  low = m.interpolate_smart_cube_profile(SmartCubeProfile(), 0)
  high = m.interpolate_smart_cube_profile(SmartCubeProfile(), 1)
  assert low.threshold_degrees == (43.0, 53.0)
  assert high.threshold_degrees == (43, 53)
  assert low.move_delay_range == (0, 1)
  assert high.move_delay_range == (1, 2)
  assert low.imu_dropout_probability == 0
  assert high.imu_dropout_probability == 0.03
  assert low.calibration_noise_rad == pytest.approx(
    torch.deg2rad(torch.tensor(0.2)).item()
  )
  assert high.calibration_noise_rad == pytest.approx(
    torch.deg2rad(torch.tensor(3.0)).item()
  )


def test_reset_failure_only_unfinished_skills():
  c = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg())
  command = SimpleNamespace(_smart_skill_recorded=torch.tensor([True, False, False]))
  env = SimpleNamespace(
    device="cpu",
    num_envs=3,
    _smart_cube_curriculum=c,
    common_step_counter=10,
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  m.record_reset_skill_failures(env, torch.tensor([0, 1]))
  m.record_reset_skill_failures(env, torch.tensor([0, 1]))
  assert c.pending_attempts == 1 and c.pending_successes == 0
  assert command._smart_skill_recorded.tolist() == [True, True, False]


def test_step_log_tags_and_ema():
  c = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg(ema_smoothing=0.1))
  c.record_outcomes(5, 10)
  env = SimpleNamespace(_smart_cube_curriculum=c, common_step_counter=25, extras={})
  m.update_smart_cube_curriculum(env, None)
  assert c.skill_sr_ema == pytest.approx(0.5)
  c.record_outcomes(10, 10)
  env.common_step_counter = 50
  m.update_smart_cube_curriculum(env, None)
  assert c.skill_sr_ema == pytest.approx(0.55)
  assert set(env.extras["log"]) == {
    "curriculum/beta",
    "curriculum/dr_alpha",
    "curriculum/skill_sr_ema",
  }


def test_threshold_below_45_stationary_and_reversal_do_not_chatter():
  from wuji_mjlab.tasks.reorient.mdp import smart_cube_observations as s

  p = SmartCubeProfile(threshold_degrees=(43, 43), move_delay_steps=0)
  state = s.create_smart_cube_state(1, "cpu", p)
  angles = torch.zeros(1, 6)
  s.step_move_model(angles, state, p, step_id=0)
  angles[:, 0] = torch.deg2rad(torch.tensor(44.0))
  for step in (1, 2, 3):
    assert s.step_move_model(angles, state, p, step_id=step)[0, 0] == pytest.approx(
      torch.pi / 2
    )
  angles[:, 0] = torch.deg2rad(torch.tensor(43.5))
  assert s.step_move_model(angles, state, p, step_id=4)[0, 0] == 0
  assert s.step_move_model(angles, state, p, step_id=5)[0, 0] == 0


def test_beta_blends_unwrapped_continuous_and_discrete_angles():
  from wuji_mjlab.tasks.reorient.mdp import observations

  p = SmartCubeProfile(move_delay_steps=0)
  joints = torch.zeros(1, 6)
  command = SimpleNamespace(
    cfg=SimpleNamespace(solver_mode=True),
    object=SimpleNamespace(data=SimpleNamespace(joint_pos=joints)),
    _face_driver_ids=torch.arange(6),
    target_face_id=torch.tensor([0]),
  )
  c = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg())
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene=SimpleNamespace(device="cpu"),
    common_step_counter=0,
    _smart_cube_profile=p,
    _smart_cube_curriculum=c,
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  observations.rubik_smart_cube_angle_sincos(env)
  joints[:, 0] = torch.deg2rad(torch.tensor(30.0))
  env.common_step_counter = 1
  result = observations.rubik_smart_cube_angle_sincos(env)
  assert result[0, 0] == pytest.approx(
    torch.sin(torch.deg2rad(torch.tensor(21.0))).item()
  )
  c.beta = 1
  assert observations.rubik_smart_cube_angle_sincos(env)[0, 0] == 0


def test_action_dr_endpoints_and_environment_coherent_delay():
  from wuji_mjlab.tasks.reorient.mdp.actions import apply_smart_cube_action_dr

  queue = torch.stack((torch.ones(1000, 16), torch.zeros(1000, 16)), dim=1)
  assert torch.equal(apply_smart_cube_action_dr(queue, 0), queue[:, 0])
  torch.manual_seed(7)
  target = apply_smart_cube_action_dr(queue, 1)
  assert ((target < 0.5).all(dim=-1).float().mean().item()) == pytest.approx(
    0.7, abs=0.04
  )
  baseline = (target > 0.5).float()
  assert (target - baseline).abs().max() <= 0.020001


def test_face_resistance_default_relative_partial_reset():
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  default = torch.arange(1, 9).float()
  model = SimpleNamespace(
    dof_damping=default.repeat(3, 1), dof_frictionloss=default.repeat(3, 1)
  )
  asset = SimpleNamespace(indexing=SimpleNamespace(joint_v_adr=torch.arange(8)))
  env = SimpleNamespace(
    device="cpu",
    num_envs=3,
    sim=SimpleNamespace(model=model, get_default_field=lambda _: default),
    scene={"object": asset},
    _smart_cube_curriculum=m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg()),
  )
  cfg = SceneEntityCfg("object")
  cfg.joint_ids = list(range(1, 7))
  env._smart_cube_curriculum.alpha = 1
  m.randomize_smart_cube_face_resistance(env, torch.tensor([1]), cfg)
  ratios = model.dof_damping[1, 1:7] / default[1:7]
  assert ((ratios >= 0.5) & (ratios <= 1.8)).all()
  assert torch.equal(model.dof_damping[0], default)
  assert model.dof_damping[1, 0] == default[0]
  env._smart_cube_curriculum.alpha = 0
  m.randomize_smart_cube_face_resistance(env, torch.tensor([1]), cfg)
  assert torch.equal(model.dof_damping, default.repeat(3, 1))


def test_play_configuration_and_opt_in():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_full_cube_solver_transfer_env_cfg as base,
  )
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_smart_cube_transfer_env_cfg as smart,
  )

  train = smart(num_envs=4)
  play = smart(play=True, play_alpha=0.9)
  assert train.commands["reorient_command"].smart_cube_curriculum.play is False
  assert play.commands["reorient_command"].smart_cube_curriculum.play_alpha == 0.9
  assert train.actions["joint_pos"].smart_cube_dr
  assert not base(num_envs=4).actions["joint_pos"].smart_cube_dr
  assert train.events["smart_cube_face_resistance"].mode == "reset"
  assert train.events["smart_cube_curriculum_step"].mode == "step"


def test_success_outcome_is_once_and_drop_overrides_success():
  c = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg())
  command = SimpleNamespace(_smart_skill_recorded=torch.zeros(3, dtype=torch.bool))
  env = SimpleNamespace(
    device="cpu",
    num_envs=3,
    _smart_cube_curriculum=c,
    common_step_counter=1,
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  ids = torch.tensor([0, 1])
  dropped = torch.tensor([False, True, False])
  m.record_smart_cube_skill_successes(env, command, ids, dropped)
  m.record_smart_cube_skill_successes(env, command, ids, dropped)
  assert (c.pending_successes, c.pending_attempts) == (1, 1)
  m.record_reset_skill_failures(env, ids)
  assert (c.pending_successes, c.pending_attempts) == (1, 2)


def test_move_jitter_packets_stay_ordered_and_same_step_is_cached():
  from wuji_mjlab.tasks.reorient.mdp import smart_cube_observations as s

  torch.manual_seed(12)
  p = m.interpolate_smart_cube_profile(SmartCubeProfile(), 1)
  state = s.create_smart_cube_state(128, "cpu", p)
  angles = torch.zeros(128, 6)
  s.step_move_model(angles, state, p, step_id=0)
  previous = state.delivered_angles.clone()
  observed_delays = set()
  for step in range(1, 12):
    angles[:, 0] += torch.pi / 2
    current = s.step_move_model(angles, state, p, step_id=step)
    assert (current[:, 0] >= previous[:, 0]).all()
    assert torch.equal(current, s.step_move_model(angles, state, p, step_id=step))
    observed_delays.update((step - state.delivered_move_step).tolist())
    previous = current
  assert observed_delays == {1, 2}


def test_flip_specialist_smooth_cube_dr_configuration():
  import math

  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_specialist_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.mdp.smart_cube_observations import (
    effective_smart_cube_profile,
  )

  for play in (False, True):
    cfg = luwu_hand_rubik_flip_specialist_env_cfg(play=play, num_envs=2)
    assert 'robot_friction' not in cfg.events
    assert cfg.events['smart_cube_hand_friction'].mode == 'reset'
    resistance = cfg.events['smart_cube_face_resistance'].params
    assert resistance['scale_range'] == (.1, 1.)
    controller = m.SmartCubeCurriculum(cfg.commands['reorient_command'].smart_cube_curriculum)
    env = SimpleNamespace(_smart_cube_curriculum=controller)
    for alpha in (0., .5, 1.):
      controller.alpha = alpha
      profile = effective_smart_cube_profile(env, SmartCubeProfile())
      assert math.degrees(profile.imu_noise_rad) == pytest.approx(.1 + 4.9 * alpha)
    # Other smart-cube tasks retain their existing one-degree endpoint.
    assert math.degrees(m.interpolate_smart_cube_profile(SmartCubeProfile(), 1).imu_noise_rad) == pytest.approx(1.)


@pytest.mark.parametrize('alpha,expected', [(0., (.7, 1.1)), (.5, (.375, .6)), (1., (.05, .1))])
def test_hand_friction_curriculum_uses_reset_ids_and_default_relative_scaling(monkeypatch, alpha, expected):
  from mjlab.envs.mdp import dr
  calls = []
  monkeypatch.setattr(dr, 'geom_friction', lambda env, ids, **kwargs: calls.append((ids, kwargs)))
  env = SimpleNamespace(_smart_cube_curriculum=SimpleNamespace(alpha=alpha))
  ids = torch.tensor([1])
  m.randomize_smart_cube_hand_friction(env, ids)
  selected, kwargs = calls[0]
  assert selected is ids
  assert kwargs['ranges'] == pytest.approx(expected)
  assert kwargs['operation'] == 'scale'
  assert kwargs['axes'] == [0]


def test_smooth_driver_resistance_partial_reset_and_no_compounding():
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  default = torch.ones(8)
  model = SimpleNamespace(dof_damping=default.repeat(3, 1), dof_frictionloss=default.repeat(3, 1))
  asset = SimpleNamespace(indexing=SimpleNamespace(joint_v_adr=torch.arange(8)))
  env = SimpleNamespace(device='cpu', num_envs=3, sim=SimpleNamespace(model=model, get_default_field=lambda _: default),
    scene={'object': asset}, _smart_cube_curriculum=SimpleNamespace(alpha=1.))
  cfg = SceneEntityCfg('object')
  cfg.joint_ids = list(range(1, 7))
  for _ in range(10):
    m.randomize_smart_cube_face_resistance(env, torch.tensor([1]), cfg, scale_range=(.1, 1.))
    assert ((model.dof_frictionloss[1, 1:7] >= .1) & (model.dof_frictionloss[1, 1:7] <= 1.)).all()
    assert torch.equal(model.dof_damping[0], default)
  env._smart_cube_curriculum.alpha = 0.
  m.randomize_smart_cube_face_resistance(env, torch.tensor([1]), cfg, scale_range=(.1, 1.))
  assert torch.equal(model.dof_damping, default.repeat(3, 1))


def test_physical_smooth_cube_dr_reset_keeps_other_envs_and_friction_axes(tmp_path):
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_specialist_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool

  pool = tmp_path / 'pool.json'
  generate_solver_plan_pool(pool, count=2, seed=42, min_scramble_moves=2, max_scramble_moves=2)
  cfg = luwu_hand_rubik_flip_specialist_env_cfg(play=True, plan_pool_path=str(pool))
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
  try:
    env.reset(seed=42)
    term = env.event_manager.get_term_cfg('smart_cube_hand_friction')
    asset_cfg = term.params['asset_cfg']
    geom_ids = env.scene['robot'].indexing.geom_ids[asset_cfg.geom_ids].long()
    default = env.sim.get_default_field('geom_friction')[geom_ids]
    controller = m.get_smart_cube_curriculum(env)
    untouched = env.sim.model.geom_friction[1].clone()
    for alpha in (0., .5, 1.):
      controller.alpha = alpha
      for _ in range(3):
        term.func(env, torch.tensor([0]), **term.params)
        actual = env.sim.model.geom_friction[0, geom_ids]
        ratio = actual[:, 0] / default[:, 0]
        assert (ratio >= .7 - .65 * alpha - 1e-6).all()
        assert (ratio <= 1.1 - alpha + 1e-6).all()
        assert torch.equal(actual[:, 1:], default[:, 1:])
        assert torch.equal(env.sim.model.geom_friction[1], untouched)
    obs, reward, _, _, _ = env.step(torch.zeros(2, 16))
    assert obs['policy'].shape == (2, 204)
    assert torch.isfinite(reward).all()
  finally:
    env.close()


def test_flip_imu_latency_curriculum_endpoints_and_midpoint():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_specialist_env_cfg,
  )
  cfg = luwu_hand_rubik_flip_specialist_env_cfg()
  assert cfg.commands['reorient_command'].smart_cube_curriculum.irregular_imu_latency
  for alpha, bounds, mode in ((0., (.05, .05), .05), (.5, (.04, .085), .07), (1., (.03, .12), .09)):
    p = m.interpolate_smart_cube_profile(SmartCubeProfile(), alpha, 5., True)
    assert p.imu_latency_range_seconds == pytest.approx(bounds)
    assert p.imu_latency_mode_seconds == pytest.approx(mode)
  assert m.interpolate_smart_cube_profile(SmartCubeProfile(), 1.).imu_latency_range_seconds is None
