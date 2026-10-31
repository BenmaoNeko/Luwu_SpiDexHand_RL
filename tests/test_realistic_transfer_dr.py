"""Contract for the next model900 transfer curriculum."""
import math
from types import SimpleNamespace

import pytest
import torch

from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_low_friction_env_cfg,
    luwu_hand_rubik_flip_specialist_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp import actions, observations
from wuji_mjlab.tasks.reorient.mdp import smart_cube_curriculum as dr


@pytest.mark.parametrize('step,expected', [(0,(0.,0.)), (4000,(0.,0.)),
    (14000,(.5,0.)), (24000,(1.,0.)), (32000,(1.,.5)), (40000,(1.,1.))])
def test_realistic_course_uses_absolute_control_steps(step, expected):
    assert dr.realistic_stage_progress(step) == pytest.approx(expected)


def test_new_transfer_config_keeps_source_reward_and_control_period():
    cfg = luwu_hand_rubik_flip_low_friction_env_cfg(num_envs=4)
    old = luwu_hand_rubik_flip_specialist_env_cfg(num_envs=4)
    assert cfg.actions['joint_pos'].ema_alpha == .75
    assert cfg.actions['joint_pos'].realistic_dr
    assert cfg.commands['reorient_command'].smart_cube_curriculum.realistic_dr
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 10
    assert cfg.sim.mujoco.cone == 'elliptic' and cfg.sim.mujoco.solver == 'cg'
    assert set(cfg.rewards) == set(old.rewards)
    assert {k: v.weight for k, v in cfg.rewards.items()} == {k: v.weight for k, v in old.rewards.items()}


def test_realistic_action_bias_is_held_per_episode_and_clamped():
    torch.manual_seed(1)
    bias = actions.sample_realistic_action_bias(4096, 16, 1., 'cpu')
    assert bias.shape == (4096,16)
    assert bias.abs().max() <= math.radians(3) + 1e-7
    assert bias.abs().max() >= .99 * math.radians(3)
    target = torch.zeros_like(bias)
    assert torch.equal(actions.apply_realistic_action_bias(target, bias), bias)
    torch.testing.assert_close(actions.apply_realistic_action_bias(target, bias, -0.02, 0.02).abs().max(), torch.tensor(.02))


def test_realistic_joint_observation_noise_is_small_and_shared():
    spans = torch.tensor([1.05, 1., 1.5, 1.]).repeat(4)
    robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.zeros(4096,16),
        soft_joint_pos_limits=torch.stack((-spans,spans),-1).repeat(4096,1,1)))
    action = SimpleNamespace(processed_action=torch.zeros(4096,16))
    controller = SimpleNamespace(cfg=SimpleNamespace(realistic_dr=True), realistic_steps=24000)
    env = SimpleNamespace(scene={'robot':robot}, _smart_cube_curriculum=controller,
        action_manager=SimpleNamespace(get_term=lambda _:action))
    normalized = observations.joint_pos_limit_normalized_with_dr(env)
    error = observations.joint_pos_target_error_with_dr(env)
    assert torch.equal(normalized,error)
    degrees = normalized * spans * 180 / math.pi
    assert degrees.abs().max() <= .5 + 1e-4
    assert degrees.abs().max() >= .49


def test_realistic_contact_bounds_avoid_near_zero_and_keep_nominal_anchor():
    assert dr.realistic_friction_bounds(0) == pytest.approx((.20,.30))
    assert dr.realistic_friction_bounds(24000) == pytest.approx((.15,.35))
    assert dr.realistic_friction_bounds(40000) == pytest.approx((.10,.50))
    assert dr.realistic_palm_sliding() == pytest.approx(.30)
