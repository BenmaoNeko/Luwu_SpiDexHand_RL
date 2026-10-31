import pytest
import torch
from scripts.eval.eval_gentle_flip import RequestRecorder, calibrate, summarize_requests


def test_request_outcomes_groups_and_intermediate_legs_are_not_double_counted():
    recorder = RequestRecorder(2, ['a', 'b'], .05)
    recorder.begin(0, face=1, half_turn=True)
    recorder.begin(1, face=2, half_turn=False)
    recorder.sample(torch.tensor([[1., -1.], [2., 0.]]), torch.zeros(2, 2),
                    [True, False], [2, 0])
    recorder.finish(0, 'success', intermediate=True, aligned=True)
    assert not recorder.rows
    recorder.finish(0, 'success', aligned=True)
    recorder.finish(1, 'drop')
    recorder.begin(1, face=3, half_turn=False)
    recorder.finish(1, 'timeout')
    recorder.begin(0, face=4, half_turn=False)
    recorder.flush()
    summary = summarize_requests(recorder.rows)
    assert summary['success'] == 1 and summary['drop'] == 1
    assert summary['timeout'] == 1 and summary['unfinished'] == 1
    assert summary['attempts'] == 4 and summary['success_rate'] == .25
    assert summary['nominal']['success'] == 1
    assert summary['non_nominal']['drop'] == 1
    assert recorder.rows[0]['legs'] == 2
    assert recorder.rows[0]['duration_s'] == .05


def test_calibration_is_per_joint_and_integrates_dt():
    recorder = RequestRecorder(1, ['a', 'b'], .05)
    recorder.begin(0, face=0, half_turn=False)
    for _ in range(20):
        recorder.sample(torch.tensor([[1., 2.]]), torch.zeros(1, 2), [False], [2])
    recorder.finish(0, 'success')
    calibration = calibrate(recorder.rows, ['a', 'b'])
    assert calibration['v_ref'] == pytest.approx([1., 2.])
    assert calibration['v_soft'] == pytest.approx([1., 2.])
    assert calibration['median_success_cost'] == pytest.approx(.1)
    assert calibration['lambda_initial'] == pytest.approx(10.)
    assert calibration['lambda_target'] == pytest.approx(25.)


def test_calibration_refuses_zero_motion_and_absent_successes():
    with pytest.raises(ValueError):
        calibrate([], ['a'])
    recorder = RequestRecorder(1, ['a'], .05)
    recorder.begin(0, face=1, half_turn=False)
    recorder.sample(torch.zeros(1, 1), torch.zeros(1, 1), [False], [1])
    recorder.finish(0, 'success')
    with pytest.raises(ValueError):
        calibrate(recorder.rows, ['a'])


def test_positive_reward_attribution_survives_finish_and_new_request():
    recorder = RequestRecorder(1, ['a'], .05)
    recorder.begin(0, face=1, half_turn=False)
    recorder.sample([[1.]], [[0.]], [False], [1])
    recorder.finish(0, 'success')
    recorder.begin(0, face=2, half_turn=False)
    recorder.add_rewards([[400., 20., -10.]], ['arrival', 'progress', 'penalty'])
    assert recorder.rows[0]['positive_reward'] == pytest.approx(21.)
    assert recorder.rows[0]['positive_reward_terms'] == {'arrival': 20., 'progress': 1.}
    assert recorder.active[0]['positive_reward'] == 0.
    assert recorder.rows[0]['reward_samples'] == 1


def test_fraction_calibration_uses_all_positive_rewards_and_fixed_thresholds():
    from scripts.eval.eval_gentle_flip import calibrate_reward_fraction
    recorder = RequestRecorder(1, ['a'], .05)
    for reward, outcome in [(20., 'success'), (40., 'success'), (1000., 'drop')]:
        recorder.begin(0, face=1, half_turn=False)
        for _ in range(20):
            recorder.sample([[1.]], [[0.]], [False], [1])
            recorder.add_rewards([[reward]], ['positive'])
        recorder.finish(0, outcome)
    fixed = dict(joint_names=['a'], v_ref=[1.], v_soft=[1.])
    result = calibrate_reward_fraction(recorder.rows, fixed, ['a'], .4, min_requests=2)
    assert result['coefficient'] == pytest.approx(120.)
    assert result['mean_success_positive_reward'] == pytest.approx(30.)
    assert result['mean_success_velocity_cost'] == pytest.approx(.1)
    assert result['achieved_fraction'] == pytest.approx(.4)
    assert result['v_ref'] == [1.] and result['v_soft'] == [1.]
    with pytest.raises(ValueError, match='samples'):
        calibrate_reward_fraction(recorder.rows, fixed, ['a'], .4)
    with pytest.raises(ValueError):
        calibrate_reward_fraction(recorder.rows, fixed, ['wrong'], .4, min_requests=2)
    recorder.rows[0]['reward_samples'] = 0
    with pytest.raises(ValueError, match='samples'):
        calibrate_reward_fraction(recorder.rows, fixed, ['a'], .4, min_requests=2)
