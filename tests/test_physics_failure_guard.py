from types import SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from wuji_mjlab.rl.physics_failure_guard import (
    install_physics_failure_guard,
    physics_failure,
)


def make_env():
    data = SimpleNamespace(qpos=torch.zeros(2, 7), qvel=torch.zeros(2, 6),
        qacc=torch.zeros(2, 6), qacc_warmstart=torch.zeros(2, 6),
        ctrl=torch.ones(2, 2), xquat=torch.ones(2, 1, 4), sensordata=torch.zeros(2, 2))
    counts = {'forward': 0}

    def step():
        data.qpos[0, 0] += 1
        data.qvel[1, 0] = float('nan')
        data.qacc[1] = float('nan')

    def forward():
        counts['forward'] += 1

    sim = SimpleNamespace(data=data, step=step, forward=forward)
    env = SimpleNamespace(num_envs=2, device='cpu', sim=sim, common_step_counter=7, extras={})
    env.step = lambda actions: (sim.step(), sim.step())
    return env, data, counts


def test_single_failed_world_is_saved_quarantined_and_not_relabelled_success(tmp_path):
    env, data, counts = make_env()
    guard = install_physics_failure_guard(env, tmp_path)
    env.step(torch.zeros(2, 2))
    assert env._physics_failure_mask.tolist() == [False, True]
    assert data.qpos[0, 0] == 2  # Healthy world keeps both physical steps.
    assert torch.isfinite(data.qvel).all()
    assert torch.isfinite(data.qacc).all()
    assert guard.total_failures == 1  # Not once per remaining substep.
    assert counts['forward'] == 2
    saved = torch.load(next(tmp_path.glob('*.pt')), weights_only=False)
    assert torch.isnan(saved['after']['qvel']).any()  # Original fault retained.
    assert saved['env_ids'].tolist() == [1]


def test_finite_runaway_is_detected_before_it_becomes_nan(tmp_path):
    env, data, _ = make_env()
    env.sim.step = lambda: data.qvel[1].fill_(201.)
    guard = install_physics_failure_guard(env, tmp_path)
    env.step(torch.zeros(2, 2))
    assert guard.total_failures == 1
    assert env._physics_failure_mask.tolist() == [False, True]


def test_guard_can_keep_existing_drop_penalty_without_new_reward(tmp_path):
    env, _, _ = make_env()
    drop = SimpleNamespace(func=lambda env: torch.tensor([0., 20.]), params={})
    success = SimpleNamespace(func=lambda env: torch.tensor([20., 20.]), params={})
    env.reward_manager = SimpleNamespace(_term_names=['cage_drop', 'success'],
        _term_cfgs=[drop, success])
    install_physics_failure_guard(env, tmp_path, preserve_failure_rewards=('cage_drop',))
    env.step(torch.zeros(2, 2))
    assert drop.func(env).tolist() == [0., 20.]
    assert success.func(env).tolist() == [20., 0.]


def test_fault_mask_clears_at_next_control_step(tmp_path):
    env, data, _ = make_env()
    install_physics_failure_guard(env, tmp_path)
    env.step(torch.zeros(2, 2))
    # Mimic the normal termination/reset, then a healthy physical step.
    data.qvel.zero_()
    env.sim.step = lambda: None
    env.step(torch.zeros(2, 2))
    assert not env._physics_failure_mask.any()


def test_healthy_path_has_no_dynamic_gather_and_one_host_decision_per_substep(tmp_path):
    env, _, _ = make_env()
    env.sim.step = lambda: None
    install_physics_failure_guard(env, tmp_path)
    operations = []

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            operations.append(str(func))
            return func(*args, **(kwargs or {}))

    with Capture():
        env.step(torch.zeros(2, 2))
    assert not any(op.startswith('aten.index.') for op in operations)
    assert operations.count('aten._local_scalar_dense.default') == 2  # Two physics steps.


def test_existing_failure_mask_does_not_allocate_an_unused_default(tmp_path, monkeypatch):
    env, _, _ = make_env()
    install_physics_failure_guard(env, tmp_path)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError('Allocated fallback despite existing mask')

    monkeypatch.setattr(torch, 'zeros', unexpected_allocation)
    assert physics_failure(env) is env._physics_failure_mask


def test_builtin_nan_diagnostics_scan_only_after_guard_detects_a_fault(tmp_path):
    env, data, _ = make_env()
    calls = []
    observer = SimpleNamespace(enabled=True, check_and_dump=lambda data: calls.append('scan'))
    env.sim.nan_guard = observer
    fail = False

    def physical_step():
        if fail:
            data.qvel[1, 0] = float('nan')
        observer.check_and_dump(data)

    env.sim.step = physical_step
    install_physics_failure_guard(env, tmp_path)
    env.step(torch.zeros(2, 2))
    assert calls == []
    fail = True
    env.step(torch.zeros(2, 2))
    assert calls == ['scan']  # Still invokes the original diagnostics before quarantine.


def test_builtin_nan_guard_keeps_history_and_original_fault_dump(tmp_path):
    from collections import deque

    from mjlab.utils.nan_guard import NanGuard
    env, data, _ = make_env()
    observer = object.__new__(NanGuard)
    observer.enabled, observer._dumped = True, False
    observer.buffer, observer.step_counter = deque(maxlen=5), 0
    observer.mj_model = SimpleNamespace(na=0, nmocap=0)
    dumped = []
    observer._dump_buffer = lambda ids: dumped.append((ids, list(observer.buffer)))
    env.sim.nan_guard = observer
    fail = False

    def step():
        with observer.watch(data):
            data.qpos[0, 0] += 1
            if fail:
                data.qvel[1, 0] = float('nan')

    env.sim.step = step
    guard = install_physics_failure_guard(env, tmp_path)
    env.step(torch.zeros(2, 2))
    assert observer.step_counter == 2
    assert not observer._dumped
    fail = True
    env.step(torch.zeros(2, 2))
    assert observer._dumped
    assert dumped[0][0] == [1]
    assert len(dumped[0][1]) == 3
    assert torch.isfinite(dumped[0][1][-1]['qvel']).all()
    assert torch.isfinite(data.qvel).all()
    assert guard.total_failures == 1


def test_contact_metrics_remain_on_device_without_copying_totals_to_host():
    from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import BottomTipRequests
    from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip_runtime import bottom_tip_metric
    requests = BottomTipRequests(2, 'cpu')
    requests.totals[:] = torch.tensor([10, 8, 6, 5])
    runtime = SimpleNamespace(requests=requests)
    command = SimpleNamespace(bottom_tip_runtime=runtime)
    env = SimpleNamespace(num_envs=2, device='cpu',
        command_manager=SimpleNamespace(get_term=lambda _: command))
    # Production logging must not call the analysis-only Python scalar API.
    requests.metrics = lambda: (_ for _ in ()).throw(AssertionError('Host metric conversion'))
    for key, expected in (('requests', 10.), ('success_rate', .8),
        ('success_after_bottom_contact', 5/6), ('bottom_contact_given_success', 5/8),
        ('supported_success_rate', .5)):
        assert torch.allclose(bottom_tip_metric(env, key), torch.full((2,), expected))
    requests.totals.zero_()
    assert torch.isnan(bottom_tip_metric(env, 'success_after_bottom_contact')).all()
