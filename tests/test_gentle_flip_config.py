from dataclasses import asdict

import pytest
from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_gentle_env_cfg,
    luwu_hand_rubik_flip_low_friction_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.gentle_flip import (
    GentleFlipSchedule,
    GentleFlipState,
)


def test_schedule_finishes_ema_before_penalty_and_requires_fresh_successes():
    cfg = GentleFlipSchedule(ema_steps=100, settle_steps=100, check_steps=50, min_attempts=10)
    state = GentleFlipState()
    state.advance(50, 0, 0, cfg)
    assert state.alpha(cfg) == pytest.approx(.625)
    assert state.speed_scale == 0
    state.advance(200, 20, 20, cfg)
    assert state.alpha(cfg) == .5 and state.speed_scale == 0
    state.advance(250, 30, 29, cfg)
    assert state.speed_scale == 0
    state.advance(300, 40, 39, cfg)
    assert state.speed_scale == .25
    state.advance(350, 40, 39, cfg)
    assert state.speed_scale == .25
    restored = GentleFlipState(**asdict(state))
    assert restored == state
    restored.advance(0, 40, 39, cfg)
    assert restored.elapsed_steps == state.elapsed_steps  # reset never restarts the course


def test_gentle_task_preserves_source_contract_and_freezes_full_dr():
    source = luwu_hand_rubik_flip_low_friction_env_cfg(num_envs=2)
    cfg = luwu_hand_rubik_flip_gentle_env_cfg(num_envs=2)
    assert cfg.commands['reorient_command'].smart_cube_curriculum.realistic_initial_progress == 1.
    assert cfg.actions['joint_pos'].ema_alpha == .75
    assert set(cfg.rewards) == set(source.rewards) | {'gentle_joint_velocity'}
    assert {k: v.weight for k, v in source.rewards.items()} == {
        k: cfg.rewards[k].weight for k in source.rewards}
    assert cfg.sim.mujoco == source.sim.mujoco
    assert cfg.decimation == source.decimation
    assert cfg.observations == source.observations
    assert source.commands['reorient_command'].smart_cube_curriculum.realistic_initial_progress == .2
    play = luwu_hand_rubik_flip_gentle_env_cfg(play=True)
    assert play.actions['joint_pos'].ema_alpha == .5
    assert play.events['gentle_flip_schedule'].params['schedule'].frozen


def test_calibration_has_no_guessed_nonzero_penalty(tmp_path):
    with pytest.raises(ValueError, match='calibration'):
        luwu_hand_rubik_flip_gentle_env_cfg(calibration_path=str(tmp_path / 'missing.json'))
    cfg = luwu_hand_rubik_flip_gentle_env_cfg()
    assert cfg.rewards['gentle_joint_velocity'].params['coefficient'] == 0


def test_checkpoint_restores_runtime_parameters_and_freezes_preview():
    from copy import deepcopy
    from types import SimpleNamespace

    from wuji_mjlab.tasks.reorient.mdp.gentle_flip import (
        apply_gentle_ema,
        gentle_checkpoint_metadata,
        restore_gentle_checkpoint,
    )

    def environment():
        cfg = luwu_hand_rubik_flip_gentle_env_cfg(num_envs=1)
        action = SimpleNamespace(cfg=deepcopy(cfg.actions['joint_pos']), _ema_alpha=.75)
        events = deepcopy(cfg.events)
        rewards = deepcopy(cfg.rewards)
        return SimpleNamespace(cfg=cfg,
            action_manager=SimpleNamespace(get_term=lambda _: action),
            event_manager=SimpleNamespace(get_term_cfg=lambda key: events[key]),
            reward_manager=SimpleNamespace(get_term_cfg=lambda key: rewards[key]))

    source = environment()
    source._gentle_flip_state = GentleFlipState(elapsed_steps=2000, last_step=2000, speed_scale=.4)
    source.cfg.rewards['gentle_joint_velocity'].params['coefficient'] = 3.
    source.cfg.rewards['flip_align_streak'].weight = 1.
    source.cfg.rewards['rubik_cube_position_stability'].weight = .2
    source._gentle_fraction_calibration = {'target_fraction': .4, 'coefficient': 3.}
    apply_gentle_ema(source, source.cfg.events['gentle_flip_schedule'].params['schedule'])
    infos = gentle_checkpoint_metadata(source)
    target = environment()
    restore_gentle_checkpoint(target, infos)
    assert target._gentle_flip_state == source._gentle_flip_state
    assert target.action_manager.get_term('joint_pos')._ema_alpha == .625
    assert target.reward_manager.get_term_cfg('gentle_joint_velocity').params['coefficient'] == 3.
    assert target.event_manager.get_term_cfg('gentle_flip_schedule').params['coefficient'] == 3.
    assert target.reward_manager.get_term_cfg('flip_align_streak').weight == 1.
    assert target.cfg.rewards['rubik_cube_position_stability'].weight == .2
    assert gentle_checkpoint_metadata(target)['gentle_flip_runtime']['fraction_calibration'] == source._gentle_fraction_calibration
    restore_gentle_checkpoint(target, infos, preview=True)
    schedule = target.event_manager.get_term_cfg('gentle_flip_schedule').params['schedule']
    assert schedule.frozen and schedule.ema_end == .625 and schedule.max_speed_scale == .4
    assert target.cfg.actions['joint_pos'].ema_alpha == .625


def test_export_uses_checkpoint_ema_not_initial_yaml():
    from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import (
        _apply_checkpoint_runtime_config,
    )
    assert _apply_checkpoint_runtime_config({'ema_alpha': .75},
        {'infos': {'gentle_flip_action_ema': .5}})['ema_alpha'] == .5
    assert _apply_checkpoint_runtime_config({'ema_alpha': .75}, {})['ema_alpha'] == .75
    with pytest.raises(ValueError):
        _apply_checkpoint_runtime_config({}, {'infos': {'gentle_flip_action_ema': float('nan')}})


def test_calibration_stops_at_approved_initial_penalty(tmp_path):
    import json
    calibration = dict(schema_version=1, joint_names=[f'j{i}' for i in range(16)],
        v_ref=[1.] * 16, v_soft=[.5] * 16, lambda_initial=2., lambda_target=5.)
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(calibration))
    cfg = luwu_hand_rubik_flip_gentle_env_cfg(calibration_path=str(path))
    schedule = cfg.events['gentle_flip_schedule'].params['schedule']
    assert schedule.speed_increment == .4 and schedule.max_speed_scale == .4
    calibration['approved_max_penalty'] = 2.5
    path.write_text(json.dumps(calibration))
    cfg = luwu_hand_rubik_flip_gentle_env_cfg(calibration_path=str(path))
    assert cfg.events['gentle_flip_schedule'].params['schedule'].max_speed_scale == 1.


def test_export_preserves_actual_control_period_for_new_and_legacy_yaml():
    from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import _build_config
    assert _build_config('', {'decimation': 10, 'sim': {'mujoco': {'timestep': .005}}})['ctrl_dt'] == .05
    assert _build_config('', {'decimation': 5, 'sim': {'timestep': .01}})['ctrl_dt'] == .05


def test_high_geometric_success_cannot_promote_poor_alignment():
    cfg = GentleFlipSchedule(ema_steps=1, settle_steps=0, check_steps=1, min_attempts=10)
    state = GentleFlipState()
    state.advance(1, 0, 0, cfg, aligned=0, alignment_attempts=0)
    state.advance(2, 10, 10, cfg, aligned=5, alignment_attempts=10)
    assert state.speed_scale == 0
    state.advance(3, 20, 20, cfg, aligned=15, alignment_attempts=20)
    assert state.speed_scale == .25


def test_success_only_course_promotes_without_alignment_and_keeps_small_steps():
    cfg = GentleFlipSchedule(ema_start=.5, ema_end=.5, ema_steps=1,
        settle_steps=0, check_steps=2000, min_attempts=256,
        require_alignment=False, speed_increment=.1, max_speed_scale=.4)
    state = GentleFlipState(elapsed_steps=4000, last_step=4000, last_check=4000,
        last_attempts=1000, last_successes=950, speed_scale=.1)
    state.advance(6000, 2000, 1930, cfg, aligned=0, alignment_attempts=0)
    assert state.speed_scale == pytest.approx(.2)
    state.advance(8000, 3000, 2830, cfg, aligned=0, alignment_attempts=1000)
    assert state.speed_scale == pytest.approx(.2)  # 90% cannot promote
    state.advance(10000, 4000, 3830, cfg, aligned=0, alignment_attempts=2000)
    state.advance(12000, 5000, 4830, cfg, aligned=0, alignment_attempts=3000)
    assert state.speed_scale == .4


@pytest.mark.parametrize('fraction_mode', [False, True, 'fixed'])
def test_strong_speed_resume_preserves_learning_state_and_uses_fresh_windows(tmp_path, fraction_mode):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    import torch

    source = tmp_path / 'source.pt'
    output = tmp_path / 'strong.pt'
    checkpoint = dict(iter=245, model_state_dict={'actor': torch.tensor([1., 2.])},
        optimizer_state_dict={'step': 123}, infos={
            'gentle_flip_runtime': {'schedule': asdict(GentleFlipSchedule()),
                'reward': {'coefficient': 16.121217325804828, 'v_ref': [2.] * 16}},
            'gentle_flip': asdict(GentleFlipState(elapsed_steps=9800, last_step=9800)),
            'smart_cube_curriculum': {'low_friction_skill_attempts': 10000,
                'low_friction_skill_successes': 9800, 'preserved_dr': .7}})
    torch.save(checkpoint, source)
    flags = ['--strong-speed']
    if fraction_mode is True:
        calibration = dict(schema_version=1, target_fraction=.4, achieved_fraction=.4,
            coefficient=120., checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            v_ref=[2.] * 16, v_soft=[1.] * 16, joint_names=[f'j{i}' for i in range(16)],
            reward_weights={'flip_align_streak': 1., 'rubik_cube_position_stability': .2},
            mean_success_positive_reward=30., mean_success_velocity_cost=.1,
            successful_requests=300, ema=.5)
        checkpoint['infos']['gentle_flip_runtime']['reward'].update(
            v_soft=calibration['v_soft'], joint_names=tuple(calibration['joint_names']))
        torch.save(checkpoint, source)
        calibration['checkpoint_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        calibration_path = tmp_path / 'ratio.json'
        calibration_path.write_text(json.dumps(calibration))
        flags = ['--reward-fraction-calibration', str(calibration_path)]
    if fraction_mode == 'fixed':
        flags = ['--speed-coefficient', '365.36716160378206']
        checkpoint['infos']['gentle_flip_runtime']['fraction_calibration'] = {'target_fraction': .4}
        torch.save(checkpoint, source)
    subprocess.run([sys.executable, 'scripts/train/prepare_gentle_success_resume.py',
        str(source), str(output), *flags], check=True,
        cwd=Path(__file__).resolve().parents[1])
    result = torch.load(output, weights_only=False)
    torch.testing.assert_close(result['model_state_dict']['actor'], checkpoint['model_state_dict']['actor'])
    assert result['optimizer_state_dict'] == checkpoint['optimizer_state_dict']
    assert result['iter'] == 245
    infos = result['infos']
    assert infos['smart_cube_curriculum'] == checkpoint['infos']['smart_cube_curriculum']
    runtime = infos['gentle_flip_runtime']
    assert runtime['reward']['v_ref'] == [2.] * 16
    assert runtime['reward_weights'] == {'flip_align_streak': 1., 'rubik_cube_position_stability': .2}
    cfg = GentleFlipSchedule(**runtime['schedule'])
    state = GentleFlipState(**infos['gentle_flip'])
    assert infos['gentle_flip_action_ema'] == .5
    assert not cfg.require_alignment
    if fraction_mode == 'fixed':
        assert state.speed_scale == cfg.max_speed_scale == 1.
        assert runtime['reward']['coefficient'] == pytest.approx(365.3671616)
        assert 'fraction_calibration' not in runtime
        return
    if fraction_mode is True:
        assert state.speed_scale == cfg.max_speed_scale == 1.
        assert runtime['reward']['coefficient'] == 120.
        assert runtime['fraction_calibration']['target_fraction'] == .4
        calibration['checkpoint_sha256'] = 'wrong'
        calibration_path.write_text(json.dumps(calibration))
        rejected = subprocess.run([sys.executable, 'scripts/train/prepare_gentle_success_resume.py',
            str(source), str(tmp_path / 'wrong.pt'), *flags], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1])
        assert rejected.returncode != 0 and 'checkpoint' in rejected.stderr
        assert not (tmp_path / 'wrong.pt').exists()
        return
    assert state.speed_scale * runtime['reward']['coefficient'] == pytest.approx(16.1212173)
    state.advance(11800, 11000, 10700, cfg)  # 90%: hold
    assert state.speed_scale == .25
    for step, attempts, successes, scale in [(13800, 12000, 11700, .5),
            (15800, 13000, 12700, .75), (17800, 14000, 13700, 1.),
            (19800, 15000, 14700, 1.)]:
        state.advance(step, attempts, successes, cfg)
        assert state.speed_scale == scale
    assert runtime['reward']['coefficient'] == pytest.approx(64.4848693)
    assert torch.load(source, weights_only=False)['infos']['gentle_flip_runtime'] == checkpoint['infos']['gentle_flip_runtime']
