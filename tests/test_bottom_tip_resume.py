import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from wuji_mjlab.rl.bottom_tip_ppo import BottomTipPPO


def make_algorithm():
    obs = TensorDict({'policy': torch.zeros(2, 4), 'critic': torch.zeros(2, 6)}, batch_size=[2])
    groups = {'actor': ['policy'], 'critic': ['critic']}
    actor = MLPModel(obs, groups, 'actor', 2, hidden_dims=(8,), obs_normalization=True,
        distribution_cfg={'class_name': 'SoftplusGaussianDistribution', 'init_std': .5})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=(8,), obs_normalization=True)
    storage = RolloutStorage('rl', 2, 2, obs, [2], 'cpu')
    return BottomTipPPO(actor, critic, storage)


def test_resume_keeps_original_teacher_and_does_not_repeat_warmup():
    source = make_algorithm()
    source.prepare_transfer()
    teacher_before = copy.deepcopy(source.teacher.state_dict())
    with torch.no_grad():
        source.actor.mlp[0].weight.add_(1.)
    source.updates = 301
    checkpoint = source.save() | {'iter': 300}
    target = make_algorithm()
    assert target.load(checkpoint, None, True)
    assert target.updates == 301
    assert target.teacher is not None
    assert not target.teacher.training
    assert all(not p.requires_grad for p in target.teacher.parameters())
    assert all(torch.equal(v, teacher_before[k]) for k, v in target.teacher.state_dict().items())
    assert not torch.equal(target.teacher.mlp[0].weight, target.actor.mlp[0].weight)


def test_legacy_resume_restores_update_count_and_explicit_original_teacher(tmp_path):
    original = make_algorithm()
    path = tmp_path / 'original_teacher.pt'
    torch.save({'actor_state_dict': original.actor.state_dict()}, path)
    source = make_algorithm()
    checkpoint = source.save() | {'iter': 300}
    checkpoint.pop('bottom_tip_state', None)
    target = make_algorithm()
    target.load(checkpoint, None, True)
    assert target.updates == 301
    target.prepare_transfer(teacher_checkpoint=path)
    assert torch.equal(target.teacher.mlp[0].weight, original.actor.mlp[0].weight)
    assert target.actor.obs_normalizer.until == 0


def test_actor_only_load_does_not_restore_training_state():
    source = make_algorithm()
    source.prepare_transfer()
    source.updates = 301
    target = make_algorithm()
    assert not target.load(source.save(), {'actor': True}, True)
    assert target.updates == 0
    assert target.teacher is None


def test_runner_roundtrip_restores_curriculum_and_next_iteration(tmp_path):
    from types import SimpleNamespace

    from wuji_mjlab.rl.runner import WujiOnPolicyRunner

    def runner():
        value = object.__new__(WujiOnPolicyRunner)
        value.alg = make_algorithm()
        value.cfg = {'upload_model': False}
        value.current_learning_iteration = 300
        value.env = SimpleNamespace(unwrapped=SimpleNamespace(
            common_step_counter=12040,
            _smart_cube_curriculum=SimpleNamespace(cfg=None, alpha=.2, beta=.3,
                low_friction_progress=0., low_friction_last_step=0,
                low_friction_last_totals=(0, 0, 0, 0))))
        return value

    source = runner()
    source.alg.prepare_transfer()
    source.alg.updates = 301
    source.env.unwrapped._smart_cube_curriculum.alpha = 1.
    source.env.unwrapped._smart_cube_curriculum.beta = 1.
    source.env.unwrapped._smart_cube_curriculum.low_friction_progress = .35
    source.env.unwrapped._smart_cube_curriculum.low_friction_last_step = 12025
    source.env.unwrapped._smart_cube_curriculum.low_friction_last_totals = (100, 95, 90, 85)
    path = tmp_path / 'model_300.pt'
    source.save(str(path))
    target = runner()
    target.load(str(path), map_location='cpu')
    assert target.current_learning_iteration == 301
    assert target.alg.updates == 301
    assert target.env.unwrapped._smart_cube_curriculum.alpha == 1.
    assert target.env.unwrapped._smart_cube_curriculum.beta == 1.
    assert target.env.unwrapped._smart_cube_curriculum.low_friction_progress == .35
    assert target.env.unwrapped._smart_cube_curriculum.low_friction_last_step == 12025
    assert target.env.unwrapped._smart_cube_curriculum.low_friction_last_totals == (100, 95, 90, 85)
    assert target.env.unwrapped.common_step_counter == 12040


def test_regular_ppo_resume_restores_low_friction_without_actor_only_side_effects(tmp_path):
    from types import SimpleNamespace

    from rsl_rl.algorithms import PPO
    from wuji_mjlab.rl.runner import WujiOnPolicyRunner

    def runner():
        value = object.__new__(WujiOnPolicyRunner)
        base = make_algorithm()
        value.alg = PPO(base.actor, base.critic, base.storage)
        value.cfg = {'upload_model': False}
        value.current_learning_iteration = 300
        value.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=12040,
            _smart_cube_curriculum=SimpleNamespace(cfg=None, alpha=.2, beta=.3,
                low_friction_progress=0., low_friction_last_step=0,
                low_friction_last_totals=(0, 0, 0, 0),
                low_friction_skill_attempts=0, low_friction_skill_successes=0)))
        return value

    source = runner()
    state = source.env.unwrapped._smart_cube_curriculum
    state.low_friction_progress = .4
    state.low_friction_last_step = 12025
    state.low_friction_last_totals = (100, 90, 0, 0)
    state.low_friction_skill_attempts = 100
    state.low_friction_skill_successes = 90
    path = tmp_path / 'regular.pt'
    source.save(str(path))
    target = runner()
    target.load(str(path), map_location='cpu')
    restored = target.env.unwrapped._smart_cube_curriculum
    assert target.current_learning_iteration == 301
    assert vars(restored) == vars(state)
    target.env.unwrapped._smart_cube_curriculum.low_friction_progress = .1
    target.load(str(path), load_cfg={'actor': True}, map_location='cpu')
    assert target.env.unwrapped._smart_cube_curriculum.low_friction_progress == .1


def test_joint_dr_old_checkpoint_starts_at_zero_and_new_checkpoint_restores(tmp_path):
    from types import SimpleNamespace

    from rsl_rl.algorithms import PPO
    from wuji_mjlab.rl.runner import WujiOnPolicyRunner
    from wuji_mjlab.tasks.reorient.mdp.smart_cube_curriculum import (
        SmartCubeCurriculum,
        SmartCubeCurriculumCfg,
    )

    def runner():
        value = object.__new__(WujiOnPolicyRunner)
        base = make_algorithm()
        value.alg = PPO(base.actor, base.critic, base.storage)
        value.cfg = {'upload_model': False}
        value.current_learning_iteration = 200
        value.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=8040,
            _smart_cube_curriculum=SmartCubeCurriculum(SmartCubeCurriculumCfg(
                joint_dr_transition_steps=20000))))
        return value

    source = runner()
    state = source.env.unwrapped._smart_cube_curriculum
    state.low_friction_progress = .7
    state.low_friction_skill_attempts, state.low_friction_skill_successes = 1000, 900
    path = tmp_path / 'legacy.pt'
    source.save(str(path))
    saved = torch.load(path, weights_only=False)
    for name in ('joint_dr_progress', 'joint_dr_last_step', 'joint_dr_last_totals'):
        saved['infos']['smart_cube_curriculum'].pop(name)
    torch.save(saved, path)
    target = runner()
    target.load(str(path), map_location='cpu')
    restored = target.env.unwrapped._smart_cube_curriculum
    assert restored.joint_dr_progress == 0
    assert restored.joint_dr_last_step == 8040
    assert restored.joint_dr_last_totals == (1000, 900)
    assert restored.low_friction_progress == .7
    restored.joint_dr_progress = .4
    restored.joint_dr_last_step = 8065
    restored.joint_dr_last_totals = (1100, 990)
    modern = tmp_path / 'modern.pt'
    target.save(str(modern))
    resumed = runner()
    resumed.load(str(modern), map_location='cpu')
    assert vars(resumed.env.unwrapped._smart_cube_curriculum) == vars(restored)
