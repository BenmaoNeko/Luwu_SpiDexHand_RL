from types import SimpleNamespace

import pytest
import torch
from wuji_mjlab.rl.nonfinite_capture import capture_failure, install_failure_capture


def fixture_env():
    data = SimpleNamespace(qpos=torch.ones(3, 7), qvel=torch.zeros(3, 6),
        qacc=torch.zeros(3, 6), xquat=torch.tensor([[[1., 0, 0, 0]]]).repeat(3, 1, 1))
    return SimpleNamespace(common_step_counter=42, sim=SimpleNamespace(data=data))


def test_finite_step_creates_no_dump(tmp_path):
    assert capture_failure(fixture_env(), torch.zeros(3, 2),
        {'policy': torch.zeros(3, 4)}, torch.ones(3), tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_failure_retains_bad_field_and_does_not_sanitize(tmp_path):
    obs = torch.zeros(3, 4)
    obs[1, 2] = float('nan')
    path = capture_failure(fixture_env(), torch.zeros(3, 2),
        {'policy': obs}, torch.ones(3), tmp_path)
    saved = torch.load(path, weights_only=False)
    assert saved['env_ids'].tolist() == [1]
    assert saved['bad_observation_indices']['policy'].tolist() == [[1, 2]]
    assert saved['step'] == 42
    assert torch.isnan(saved['observations']['policy'][0, 2])
    assert torch.isnan(obs[1, 2])
    assert torch.isfinite(saved['physics']['qpos']).all()


def test_reward_only_failure_and_inf_are_captured(tmp_path):
    path = capture_failure(fixture_env(), torch.zeros(3, 2),
        {'policy': torch.zeros(3, 4)}, torch.tensor([1., 1., float('inf')]), tmp_path)
    saved = torch.load(path, weights_only=False)
    assert saved['env_ids'].tolist() == [2]
    assert torch.isinf(saved['rewards']).all()


def test_bad_action_is_captured_before_physics_step(tmp_path):
    env = fixture_env()
    env.unwrapped = env
    env.get_observations = lambda: {'policy': torch.zeros(3, 4)}
    called = []
    env.step = lambda actions: called.append(True)
    install_failure_capture(env, tmp_path)
    actions = torch.zeros(3, 2)
    actions[1, 0] = float('nan')
    with pytest.raises(FloatingPointError, match='before physics'):
        env.step(actions)
    assert called == []
    assert len(list(tmp_path.glob('*.pt'))) == 1


def test_bad_observation_is_captured_before_runner_consumes_it(tmp_path):
    env = fixture_env()
    env.unwrapped = env
    obs = torch.zeros(3, 4)
    obs[2, 1] = float('nan')
    env.step = lambda actions: ({'policy': obs}, torch.ones(3), torch.zeros(3), {})
    install_failure_capture(env, tmp_path)
    with pytest.raises(FloatingPointError, match='after physics'):
        env.step(torch.zeros(3, 2))
    assert len(list(tmp_path.glob('*.pt'))) == 1
