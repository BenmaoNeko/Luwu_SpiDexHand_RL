from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
  luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg,
  luwu_hand_rubik_flip_specialist_env_cfg,
)


def test_observe_task_preserves_source_rewards_and_actor_terms():
  old = luwu_hand_rubik_flip_specialist_env_cfg(num_envs=4)
  cfg = luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(num_envs=4)
  assert not old.commands['reorient_command'].bottom_tip_observer
  assert cfg.commands['reorient_command'].bottom_tip_observer
  assert cfg.observations['policy'] == old.observations['policy']
  assert cfg.rewards == old.rewards
  assert cfg.metrics['bottom_tip/success_after_bottom_contact'].reduce == 'last'
  assert cfg.metrics['bottom_tip_substep'].per_substep
  sensors = [s for s in cfg.scene.sensors if s.name.startswith('bottom_tip_')]
  assert len(sensors) == 4
  assert all(s.num_slots == 16 and s.reduce == 'none' for s in sensors)


def test_observer_runs_in_physics_loop_with_204_actor(tmp_path):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool
  pool=tmp_path/'pool.json'
  generate_solver_plan_pool(pool,count=2,seed=42,min_scramble_moves=2,max_scramble_moves=2)
  cfg=luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(num_envs=4,plan_pool_path=str(pool))
  env=ManagerBasedRlEnv(cfg=cfg,device='cpu')
  try:
    obs,_=env.reset(seed=42)
    assert obs['policy'].shape == (4,204)
    for _ in range(12): env.step(torch.zeros(4,16))
    observer=env.command_manager.get_term('reorient_command').bottom_tip_runtime
    assert observer.initialized.all()
    assert not observer.overflow.any()
    assert observer.requests.duration.shape == (4,4)
  finally: env.close()
