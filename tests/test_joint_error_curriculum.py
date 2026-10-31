import math
from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp import actions, observations
from wuji_mjlab.tasks.reorient.mdp import smart_cube_curriculum as m


@pytest.mark.parametrize('progress', [0., .5, 1.])
def test_execution_noise_uses_signed_progressive_angle_range(progress):
    torch.manual_seed(44)
    queue = torch.zeros(2048, 3, 16)
    result = actions.apply_smart_cube_action_dr(queue, 1., joint_dr_progress=progress)
    lower = math.radians(5) * progress
    upper = .02 * (1-progress) + math.radians(15) * progress
    assert result.abs().min() >= lower - 1e-7
    assert result.abs().max() <= upper + 1e-7
    assert (result > 0).any() and (result < 0).any()
    assert not torch.equal(result[:, 0], result[:, 1])
    assert result.abs().max() > .99 * upper


def test_step_dr_advances_without_successes_or_attempts_and_respects_resume():
    cfg = m.SmartCubeCurriculumCfg(joint_dr_transition_steps=100,
        low_friction_transition_steps=100, dr_success_gated=False)
    c = m.SmartCubeCurriculum(cfg)
    c.joint_dr_progress, c.low_friction_progress = .2, .7
    c.joint_dr_last_step = c.low_friction_last_step = 1000
    for step, joint, friction in [(1024,.2,.7),(1025,.45,.95),
                                  (1050,.7,1.),(1100,1.,1.),(900,1.,1.)]:
        m.update_joint_dr_curriculum(c, step)
        m.update_low_friction_curriculum(c, step, (0,0,0,0))
        assert c.joint_dr_progress == pytest.approx(joint)
        assert c.low_friction_progress == pytest.approx(friction)
    c.low_friction_skill_attempts, c.low_friction_skill_successes = 100, 0
    c.joint_dr_progress = c.low_friction_progress = 0
    m.update_joint_dr_curriculum(c, 925)
    m.update_low_friction_curriculum(c, 925, (100,0,0,0))
    assert c.joint_dr_progress == c.low_friction_progress == .25


def test_step_dr_freezes_in_play_mode():
    c = m.SmartCubeCurriculum(m.SmartCubeCurriculumCfg(play=True,
        joint_dr_transition_steps=100, low_friction_transition_steps=100,
        dr_success_gated=False))
    m.update_joint_dr_curriculum(c, 100)
    m.update_low_friction_curriculum(c, 100, (0,0,0,0))
    assert c.joint_dr_progress == c.low_friction_progress == 0


@pytest.mark.parametrize('progress', [0., 1.])
def test_observation_noise_has_same_physical_degrees_for_all_joints(progress):
    torch.manual_seed(123)
    spans = torch.tensor([1.05, 1., 1.5, 1.]).repeat(4)
    robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.zeros(4096, 16),
        soft_joint_pos_limits=torch.stack((-spans, spans), dim=-1).repeat(4096, 1, 1)))
    env = SimpleNamespace(scene={'robot': robot}, _smart_cube_curriculum=SimpleNamespace(
        joint_dr_progress=progress))
    result = observations.joint_pos_limit_normalized_with_dr(env)
    angular_error = result * spans
    bound = math.radians(1 + 2 * progress)
    assert angular_error.abs().max() <= bound + 1e-6
    assert (angular_error.abs().amax(0) > .99 * bound).all()


def test_joint_curriculum_uses_fresh_successes_without_banking_or_rewinds():
    cfg = m.SmartCubeCurriculumCfg(joint_dr_transition_steps=100)
    c = m.SmartCubeCurriculum(cfg)
    assert c.joint_dr_progress == 0
    c.low_friction_progress = .7
    for step, attempts, successes, expected in [
        (24, 100, 85, 0), (25, 100, 85, .25),
        (50, 100, 85, .25), (75, 200, 169, .25),
        (100, 300, 254, .5), (125, 10, 10, .5),
        (150, 110, 95, .75), (1000, 210, 180, 1),
    ]:
        c.low_friction_skill_attempts, c.low_friction_skill_successes = attempts, successes
        m.update_joint_dr_curriculum(c, step)
        assert c.joint_dr_progress == pytest.approx(expected)
    assert c.low_friction_progress == .7
    m.update_joint_dr_curriculum(c, 900)
    assert c.joint_dr_progress == 1


def test_new_joint_dr_is_opt_in_and_replaces_old_normalized_noise():
    from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
        luwu_hand_rubik_flip_low_friction_env_cfg,
        luwu_hand_rubik_flip_specialist_env_cfg,
    )
    cfg = luwu_hand_rubik_flip_low_friction_env_cfg(num_envs=4)
    old = luwu_hand_rubik_flip_specialist_env_cfg(num_envs=4)
    assert cfg.commands['reorient_command'].smart_cube_curriculum.realistic_dr
    assert cfg.commands['reorient_command'].smart_cube_curriculum.joint_dr_transition_steps == 0
    assert old.commands['reorient_command'].smart_cube_curriculum.joint_dr_transition_steps == 0
    assert not cfg.commands['reorient_command'].smart_cube_curriculum.dr_success_gated
    assert old.commands['reorient_command'].smart_cube_curriculum.dr_success_gated
    term = cfg.observations['policy'].terms['noisy_joint_angles']
    assert term.noise is None
    assert term.func is observations.joint_pos_limit_normalized_with_dr
    assert cfg.observations['policy'].terms['qpos_error'].func is observations.joint_pos_target_error_with_dr
    assert old.observations['policy'].terms['noisy_joint_angles'].noise is not None
    assert set(cfg.rewards) == set(old.rewards)


def test_joint_error_observation_reuses_same_noisy_measurement():
    limits = torch.tensor([[[-1., 1.]]]).repeat(32, 16, 1)
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(32, 16),
        soft_joint_pos_limits=limits))
    action = SimpleNamespace(processed_action=torch.full((32, 16), .2))
    env = SimpleNamespace(scene={'robot': robot},
        action_manager=SimpleNamespace(get_term=lambda name: action),
        _smart_cube_curriculum=SimpleNamespace(joint_dr_progress=1.))
    angles = observations.joint_pos_limit_normalized_with_dr(env)
    error = observations.joint_pos_target_error_with_dr(env)
    torch.testing.assert_close(error, angles - .2 / (1 + 1e-6))
    assert torch.count_nonzero(angles) > 0


def test_imu_drift_accumulates_over_time_once_per_sample_and_resets():
    from dataclasses import replace

    from wuji_mjlab.tasks.reorient.mdp import smart_cube_observations as sensor
    profile = m.interpolate_smart_cube_profile(sensor.SmartCubeProfile(), 1., 5., True)
    assert math.degrees(profile.yaw_drift_rad_per_s) == pytest.approx(.05)
    assert math.degrees(profile.imu_noise_rad) == pytest.approx(5.)
    assert math.degrees(profile.calibration_noise_rad) == pytest.approx(3.)
    isolated = replace(profile, imu_noise_rad=0., calibration_noise_rad=0.,
        imu_delay_steps=0, imu_latency_range_seconds=None, imu_dropout_probability=0.)
    state = sensor.create_smart_cube_state(2, 'cpu', isolated)
    state.yaw_rate[:] = math.radians(.05)
    quat = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    sensor.step_imu_model(quat, state, isolated, step_id=0, step_dt=.05)
    for step, degrees in [(1200, 3.), (2400, 6.)]:
        sensor.step_imu_model(quat, state, isolated, step_id=step, step_dt=.05)
        torch.testing.assert_close(state.yaw_bias, torch.full((2,), math.radians(degrees)))
        before = state.yaw_bias.clone()
        sensor.step_imu_model(quat, state, isolated, step_id=step, step_dt=.05)
        torch.testing.assert_close(state.yaw_bias, before)
    sensor.reset_smart_cube_state(state, torch.tensor([1]), isolated)
    assert state.yaw_bias[1] == 0 and state.yaw_bias[0] > 0
