"""Explicit failure termination for individual numerically invalid physics worlds.

Not a dynamics repair: preserve the fault, quarantine it before observation/contact
sampling, then let normal episode termination/reset account for it as a failure.
Install once per environment; the guard owns the simulation step and NanGuard
diagnostic hook for that environment's remaining lifetime.
"""
from pathlib import Path

import torch


def physics_failure(env):
    mask = getattr(env, '_physics_failure_mask', None)
    return mask if mask is not None else torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)


def guarded_cage_drop(env, original, **params):
    return original(env, **params) | physics_failure(env)


def guarded_reward(env, reward_func, **params):
    value = reward_func(env, **params)
    return torch.where(physics_failure(env), 0., value)


def physics_failure_penalty(env):
    return physics_failure(env).float() / env.step_dt


class GuardedClassReward:
    def __init__(self, original):
        self.original = original

    def __call__(self, env, **params):
        return guarded_reward(env, self.original, **params)

    def reset(self, env_ids=None):
        return self.original.reset(env_ids=env_ids)


class PhysicsFailureGuard:
    def __init__(self, env, output_dir, max_velocity=200.):
        self.env = env
        self.output_dir = Path(output_dir)
        self.max_velocity = max_velocity
        self.total_failures = 0
        self.control_steps = 0
        self.substep = 0
        env._physics_failure_mask = torch.zeros(env.num_envs,
            dtype=torch.bool, device=env.device)
        self.original_step = env.sim.step
        self.safe_qpos = env.sim.data.qpos.clone()
        self.safe_qvel = env.sim.data.qvel.clone()
        self.has_failures = False
        observer = getattr(env.sim, 'nan_guard', None)
        self.diagnostic_check = None
        if observer is not None and observer.enabled:
            # Keep NanGuard's rolling history, but avoid scanning the same five
            # tensors twice per substep. Its original dump runs only on a fault,
            # before quarantine overwrites the invalid physical state.
            self.diagnostic_check = observer.check_and_dump
            observer.check_and_dump = lambda data: False

    def step(self):
        env, data = self.env, self.env.sim.data
        mask = env._physics_failure_mask
        if self.has_failures:
            healthy = ~mask
            self.safe_qpos[healthy] = data.qpos[healthy]
            self.safe_qvel[healthy] = data.qvel[healthy]
        else:
            self.safe_qpos.copy_(data.qpos)
            self.safe_qvel.copy_(data.qvel)
        self.original_step()
        self.substep += 1
        bad = ~torch.isfinite(data.qpos).all(-1)
        for name in ('qvel', 'qacc', 'qacc_warmstart', 'sensordata'):
            value = getattr(data, name, None)
            if value is not None:
                bad |= ~torch.isfinite(value).reshape(env.num_envs, -1).all(-1)
        bad |= data.qvel.abs().amax(-1) > self.max_velocity
        # Exactly one device-to-host decision on the common healthy path.
        if not (bad | mask).any():
            return
        fresh = bad & ~mask
        if fresh.any():
            if self.diagnostic_check is not None:
                self.diagnostic_check(data)
            ids = fresh.nonzero().flatten()
            self.total_failures += ids.numel()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f'physics_failure_substep_{self.substep}.pt'
            # Keep all episode counts exact, but bound repetitive disk snapshots.
            if self.total_failures <= 20:
                model_fields = {}
                for name in getattr(env.sim, '_expanded_fields', ()):
                    value = getattr(env.sim.model, name)
                    if value.shape[0] == env.num_envs:
                        model_fields[name] = value[ids].detach().cpu()
                torch.save(dict(step=env.common_step_counter, env_ids=ids.cpu(),
                    before=dict(qpos=self.safe_qpos[ids].cpu(), qvel=self.safe_qvel[ids].cpu()),
                    after={name: getattr(data, name)[ids].detach().cpu() for name in
                        ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl')},
                    randomized_model_fields=model_fields,
                    max_velocity=self.max_velocity), path)
            print(f'[PhysicsFailure] envs={ids.tolist()} substep={self.substep} '
                f'total={self.total_failures}; FAILED episode, no success reward; '
                f'snapshot={path if self.total_failures <= 20 else "snapshot limit reached"}', flush=True)
        mask |= bad
        self.has_failures = True
        # Only quarantined worlds are rolled back. They cannot continue their
        # task: cage_drop includes this mask and the normal reset must run.
        data.qpos[mask] = self.safe_qpos[mask]
        data.qvel[mask] = 0
        data.qacc[mask] = 0
        data.qacc_warmstart[mask] = 0
        data.ctrl[mask] = 0
        env.sim.forward()


def install_physics_failure_guard(env, output_dir, *, preserve_failure_rewards=()):
    guard = PhysicsFailureGuard(env, output_dir)
    original_env_step = env.step
    manager = getattr(env, 'reward_manager', None)
    if manager is not None:
        for name, cfg in zip(manager._term_names, manager._term_cfgs, strict=True):
            if (name != 'physics_failure' and name not in preserve_failure_rewards
                    and cfg.func is not guarded_reward):
                cfg.func = GuardedClassReward(cfg.func)

    def step(actions):
        env._physics_failure_mask.zero_()
        guard.has_failures = False
        result = original_env_step(actions)
        guard.control_steps += 1
        env.extras.setdefault('log', {})['physics_failure/total'] = guard.total_failures
        env.extras['log']['physics_failure/attempt_rate'] = (
            guard.total_failures / (guard.control_steps * env.num_envs))
        return result

    env.sim.step = guard.step
    env.step = step
    env._physics_failure_guard = guard
    return guard
