"""Fail closed with the first offending observations and physical state intact."""
from pathlib import Path

import torch


def capture_failure(env, actions, observations, rewards, output_dir):
    bad = ~torch.isfinite(rewards).reshape(rewards.shape[0], -1).all(-1)
    bad |= ~torch.isfinite(actions).reshape(actions.shape[0], -1).all(-1)
    for value in observations.values():
        invalid = ~torch.isfinite(value)
        bad |= invalid.reshape(value.shape[0], -1).any(-1)
    if not bad.any():
        return None
    indices = {name: (~torch.isfinite(value)).nonzero().cpu()
        for name, value in observations.items()}
    ids = bad.nonzero().flatten()[:5]
    physics = {}
    for name in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'xquat', 'sensordata', 'ctrl'):
        value = getattr(env.sim.data, name, None)
        if value is not None:
            physics[name] = value[ids].detach().cpu()
    state = getattr(env, '_smart_cube_state', None)
    snapshot = dict(step=env.common_step_counter, env_ids=ids.cpu(),
        bad_observation_indices=indices, physics=physics,
        observations={k:v[ids].detach().cpu() for k,v in observations.items()},
        actions=actions[ids].detach().cpu(), rewards=rewards[ids].detach().cpu(),
        imu={} if state is None else {k:v[ids].detach().cpu() for k,v in vars(state).items()})
    path = Path(output_dir) / f'observation_failure_step_{env.common_step_counter}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)
    return path


def install_failure_capture(env, output_dir):
    """Stop at the first invalid action/output, preserving the unsanitized state."""
    original_step = env.step

    def guarded_step(actions):
        if not torch.isfinite(actions).all():
            path = capture_failure(env.unwrapped, actions, env.get_observations(),
                actions.new_zeros(actions.shape[0]), output_dir)
            raise FloatingPointError(f'Non-finite action before physics; snapshot: {path}')
        result = original_step(actions)
        observations, rewards = result[:2]
        path = capture_failure(env.unwrapped, actions, observations, rewards, output_dir)
        if path is not None:
            raise FloatingPointError(f'Non-finite output after physics; snapshot: {path}')
        return result

    env.step = guarded_step
