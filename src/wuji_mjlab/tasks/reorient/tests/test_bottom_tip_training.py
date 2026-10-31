import torch
from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
  luwu_hand_rubik_flip_bottom_tip_train_env_cfg,
)


def test_training_replaces_unbounded_positive_rewards_only():
  cfg = luwu_hand_rubik_flip_bottom_tip_train_env_cfg(num_envs=4)
  assert cfg.commands['reorient_command'].bottom_tip_observer
  assert cfg.commands['reorient_command'].bottom_tip_early_per_leg
  assert cfg.commands['reorient_command'].bottom_tip_compact_contact
  assert not cfg.commands['reorient_command'].bottom_tip_debug_diagnostics
  assert cfg.commands['reorient_command'].smart_cube_curriculum.low_friction_transition_steps == 20000
  assert cfg.events['low_friction_surfaces'].mode == 'reset'
  assert cfg.commands['reorient_command'].bottom_tip_contact_window_deg == 40.
  assert cfg.commands['reorient_command'].bottom_tip_success_rewards == (-1.,30.)
  import math
  assert cfg.commands['reorient_command'].continuous_flip_upright_tolerance_rad == math.radians(4.)
  assert 'rubik_flip_to_top' not in cfg.rewards
  assert 'flip_align_streak' not in cfg.rewards
  assert cfg.rewards['bottom_tip_goal'].weight == 1.
  assert cfg.rewards['flip_align_boundary'].weight == -2.
  assert cfg.rewards['flip_align_improvement'].weight == .25
  assert cfg.rewards['flip_align_move'].weight == -2.
  assert cfg.rewards['rubik_task_progress'].params['limit'] == 2.
  assert cfg.rewards['approach_progress'].params['limit'] == .25
  assert cfg.rewards['contact_milestones'].params['limit'] == .25
  assert cfg.sim.mujoco.timestep == .005
  assert cfg.decimation == 10
  assert cfg.sim.mujoco.timestep * cfg.decimation == .05
  assert 'physics_failure' in cfg.terminations
  assert cfg.rewards['physics_failure'].weight == -10.


def test_smaller_alignment_incentive_is_local_to_bottom_tip_training():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_specialist_env_cfg,
  )
  source = luwu_hand_rubik_flip_specialist_env_cfg(num_envs=2)
  trained = luwu_hand_rubik_flip_bottom_tip_train_env_cfg(num_envs=2)
  assert source.rewards['flip_align_improvement'].weight == 1.
  assert trained.rewards['flip_align_improvement'].weight == .25
  for name in ('flip_align_boundary', 'flip_align_move', 'rubik_cube_position_stability'):
    assert trained.rewards[name].weight == source.rewards[name].weight


def test_physics_failure_is_terminal_without_success_reward(tmp_path):
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.rl.physics_failure_guard import install_physics_failure_guard
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool
  pool = tmp_path/'pool.json'
  generate_solver_plan_pool(pool, count=2, seed=42, min_scramble_moves=2, max_scramble_moves=2)
  cfg = luwu_hand_rubik_flip_bottom_tip_train_env_cfg(num_envs=2, plan_pool_path=str(pool))
  env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
  try:
    env.reset(seed=42)
    original_step = env.sim.step
    injected = False
    def inject_once():
      nonlocal injected
      original_step()
      if not injected:
        env.sim.data.qvel[1, 0] = float('nan')
        injected = True
    env.sim.step = inject_once
    guard = install_physics_failure_guard(env, tmp_path/'faults')
    tracker = env.command_manager.get_term('reorient_command').bottom_tip_runtime.requests
    successes = tracker.totals[1].clone()
    obs, reward, terminated, truncated, _ = env.step(torch.zeros(2, 16))
    assert terminated.tolist() == [False, True]
    assert not truncated[1]
    assert reward[1] == -10.
    assert guard.total_failures == 1
    assert tracker.totals[1] == successes
    assert tracker.totals[0] >= 1  # Failed attempt counted, not discarded.
    assert env.episode_length_buf[1] == 0
    assert all(torch.isfinite(v).all() for v in obs.values())
    assert not tracker.goal_event[1]
  finally:
    env.close()


def test_training_budget_resets_per_request_and_stability_pays_only_success():
  from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip import BottomTipRewardLedger
  ledger = BottomTipRewardLedger(2, 'cpu')
  ids = torch.arange(2)
  ledger.start(ids)
  assert torch.allclose(ledger.pay('progress', torch.tensor([3.,1.]), 2.), torch.tensor([2.,1.]))
  assert torch.allclose(ledger.pay('progress', torch.tensor([3.,2.]), 2.), torch.tensor([0.,1.]))
  ledger.stability(torch.tensor([.5,1.]))
  ledger.stability(torch.tensor([1.,.5]))
  assert torch.allclose(ledger.stability_payout(torch.tensor([True,False])), torch.tensor([.75,0.]))
  ledger.start(torch.tensor([0]))
  assert ledger.pay('progress', torch.ones(2), 2.).tolist() == [1.,0.]


def test_safe_optimizer_keeps_actor_fixed_during_critic_warmup():
  from rsl_rl.models import MLPModel
  from rsl_rl.storage import RolloutStorage
  from tensordict import TensorDict
  from wuji_mjlab.rl.bottom_tip_ppo import BottomTipPPO
  obs = TensorDict({'policy': torch.randn(8,4), 'critic': torch.randn(8,6)}, batch_size=[8])
  groups = {'actor': ['policy'], 'critic': ['critic']}
  actor = MLPModel(obs, groups, 'actor', 2, hidden_dims=(8,), obs_normalization=True,
    distribution_cfg={'class_name':'SoftplusGaussianDistribution', 'init_std':.5})
  critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=(8,), obs_normalization=True)
  storage = RolloutStorage('rl', 8, 2, obs, [2], 'cpu')
  algo = BottomTipPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=2)
  algo.prepare_transfer()
  before = {k:v.clone() for k,v in actor.state_dict().items()}
  critic_before = critic.mlp[0].weight.clone()
  def rollout():
    for _ in range(2):
      with torch.no_grad():
        algo.act(obs)
        algo.process_env_step(obs, torch.ones(8), torch.zeros(8,dtype=torch.bool), {})
    algo.compute_returns(obs)
  rollout()
  result = algo.update()
  assert result['actor_enabled'] == 0
  assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
  assert not torch.equal(critic_before, critic.mlp[0].weight)
  algo.updates = 10
  rollout()
  result = algo.update()
  assert result['actor_enabled'] == 1
  assert not torch.equal(actor.mlp[0].weight, before['mlp.0.weight'])
  assert torch.equal(actor.obs_normalizer._mean, before['obs_normalizer._mean'])


def test_training_reward_manager_steps_with_finite_rewards(tmp_path, monkeypatch):
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool
  pool = tmp_path/'pool.json'
  generate_solver_plan_pool(pool, count=2, seed=42, min_scramble_moves=2, max_scramble_moves=2)
  cfg = luwu_hand_rubik_flip_bottom_tip_train_env_cfg(num_envs=2, plan_pool_path=str(pool))
  env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
  try:
    obs, _ = env.reset(seed=42)
    assert obs['policy'].shape == (2,204)
    import wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip_runtime as contact_module
    original_apply = contact_module.quat_apply
    contact_runtime = env.command_manager.get_term('reorient_command').bottom_tip_runtime
    normal_transforms = []
    def count_normal_transforms(quat, value):
      if value is contact_runtime.patch_normals:
        assert value.shape == (2,12,3)
        normal_transforms.append(1)
      return original_apply(quat, value)
    monkeypatch.setattr(contact_module, 'quat_apply', count_normal_transforms)
    for _ in range(12):
      _, reward, _, _, _ = env.step(torch.zeros(2,16))
      assert torch.isfinite(reward).all()
    assert len(normal_transforms) == 12 * cfg.decimation
    runtime = env.command_manager.get_term('reorient_command').bottom_tip_runtime
    assert runtime.requests.active.all()
    assert runtime.ledger.budgets['rubik_task_progress'].remaining.le(2).all()
    _check_early_contact_per_leg(env, runtime)
  finally:
    env.close()


def _check_early_contact_per_leg(env, runtime):
  """Exercise real goal installation/settlement, replacing only sensed evidence."""
  import math
  command = env.command_manager.get_term('reorient_command')
  ids = torch.arange(2)
  opposite = command._read_top_face_id() ^ 1
  command._install_flip_or_turn_start(ids, opposite)
  assert command.bottom_intermediate.all()
  assert not runtime.requests.supported.any()
  valid = torch.ones(2,4,dtype=torch.bool)
  original = command._read_flip_target_dot
  initial = runtime.initial_angle.clone()
  command._read_flip_target_dot = lambda: torch.cos(initial-torch.deg2rad(torch.tensor([39.,41.])))
  try:
    for _ in range(math.ceil(.03 / env.physics_dt)):
      runtime._record_contact(valid)
    assert runtime.requests.supported.tolist() == [True,False]
    command._read_flip_target_dot = lambda: torch.cos(initial-.2)
    for _ in range(math.ceil(.03 / env.physics_dt)):
      runtime._record_contact(valid)
    assert runtime.requests.supported.tolist() == [True,False]
    # Both intermediate 90° legs settle, without waiting for the 180° request.
    state = command.flip_preservation
    state.done[:] = True
    state.arrival_event[:] = True
    command._read_flip_target_dot = lambda: torch.ones(2)
    totals = runtime.requests.totals.clone()
    command._last_update_step[:] = -1
    update = state.update
    state.update = lambda *args, **kwargs: None
    try:
      command._evaluate_current_step()
    finally:
      state.update = update
    assert runtime.requests.goal_event.tolist() == [30.,-1.]
    assert runtime.requests.success_event.all()
    assert (runtime.requests.totals-totals).tolist() == [2,2,1,1]
    assert not command._flip_request_event.any()
    from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip_runtime import (
      bottom_tip_stability_reward,
    )
    runtime.ledger.stability(torch.full((2,),.5))
    # A negative completion reward is still a successful flip for auxiliary payout.
    assert (bottom_tip_stability_reward(env) > 0).all()
    runtime.ledger.pay('rubik_task_progress',torch.full((2,),3.),2.)
    command._advance_pending_goals()
    assert not command.bottom_intermediate.any()
    assert runtime.requests.active.all()
    assert not runtime.requests.supported.any()
    assert not runtime.requests.success_event.any()
    assert not runtime.window.closed.any()
    assert not runtime.initialized.any()
    assert runtime.ledger.budgets['rubik_task_progress'].remaining.eq(2.).all()
    assert torch.all(runtime.initial_angle <= math.pi)
  finally:
    command._read_flip_target_dot = original
