import pytest
import torch
from wuji_mjlab.tasks.reorient.mdp.gentle_flip import gentle_velocity_cost


def test_velocity_cost_is_symmetric_continuous_and_uses_mean():
    velocities = torch.tensor([[0., 0.], [1., -1.], [2., -2.]])
    value = gentle_velocity_cost(velocities, (1., 1.), (1., 1.))
    torch.testing.assert_close(value, torch.tensor([0., .1, 1.4]))
    torch.testing.assert_close(value, gentle_velocity_cost(-velocities, 1., 1.))
    torch.testing.assert_close(value, gentle_velocity_cost(velocities.repeat(1, 8), 1., 1.))
    near = gentle_velocity_cost(torch.tensor([[.99999], [1.00001]]), 1., 1.)
    assert abs(float(near[1] - near[0])) < .00001


@pytest.mark.parametrize('reference,soft', [(0., 1.), (-1., 1.), (1., -1.), (float('nan'), 1.)])
def test_invalid_speed_parameters_are_rejected(reference, soft):
    with pytest.raises(ValueError):
        gentle_velocity_cost(torch.zeros(2, 16), reference, soft)


def test_integrated_cost_is_time_step_independent_and_nonfinite_is_bounded():
    cost = gentle_velocity_cost(torch.full((1, 16), 2.), 1., 1.)
    assert float(cost * .05 * 40) == pytest.approx(float(cost * .025 * 80))
    bad = gentle_velocity_cost(torch.tensor([[float('nan'), float('inf')]]), 1., 1.)
    assert torch.isfinite(bad).all() and (bad >= 0).all()


def test_real_reward_manager_integrates_once_and_warmup_is_excluded():
    from types import SimpleNamespace

    from mjlab.managers import RewardManager, RewardTermCfg
    from wuji_mjlab.tasks.reorient.mdp.gentle_flip import GentleFlipState
    from wuji_mjlab.tasks.reorient.mdp.rewards import gentle_joint_velocity_penalty

    def accumulated(dt, steps):
        action = SimpleNamespace(cfg=SimpleNamespace(warmup_time_s=.4))
        robot = SimpleNamespace(joint_names=[f'j{i}' for i in range(16)],
            data=SimpleNamespace(joint_vel=torch.full((2, 16), 2.)))
        env = SimpleNamespace(num_envs=2, device='cpu', scene={'robot': robot},
            episode_length_buf=torch.zeros(2), step_dt=dt,
            action_manager=SimpleNamespace(get_term=lambda _: action),
            _gentle_flip_state=GentleFlipState(speed_scale=.5))
        manager = RewardManager({'speed': RewardTermCfg(func=gentle_joint_velocity_penalty,
            weight=-1., params={'v_ref': 1., 'v_soft': 1., 'coefficient': 2.})}, env)
        assert torch.equal(manager.compute(dt), torch.zeros(2))
        env.episode_length_buf[:] = 1000
        total = sum(manager.compute(dt).clone() for _ in range(steps))
        return total

    first, second = accumulated(.05, 40), accumulated(.025, 80)
    torch.testing.assert_close(first, torch.full((2,), -2.8))
    torch.testing.assert_close(first, second)
