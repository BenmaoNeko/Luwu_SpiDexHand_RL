# SPDX-License-Identifier: Apache-2.0
"""Luwu Hand reorientation task registration."""

from copy import deepcopy

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)

from .continuous_env_cfgs import (
  luwu_hand_80g_visual_transfer_env_cfg,
  luwu_hand_continuous_faces_env_cfg,
  luwu_hand_from_scratch_curriculum_env_cfg,
)
from .env_cfgs import luwu_hand_reorient_env_cfg
from .rubik_env_cfg import (
  luwu_hand_rubik_flip_bottom_tip_train_env_cfg,
  luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg,
  luwu_hand_rubik_flip_gentle_env_cfg,
  luwu_hand_rubik_flip_low_friction_env_cfg,
  luwu_hand_rubik_flip_specialist_env_cfg,
  luwu_hand_rubik_full_cube_solver_transfer_env_cfg,
  luwu_hand_rubik_smart_cube_transfer_env_cfg,
  luwu_hand_rubik_u_face_continuous_transfer_env_cfg,
  luwu_hand_rubik_u_face_env_cfg,
  luwu_hand_rubik_u_face_transfer_env_cfg,
)

register_mjlab_task(
  task_id="LuwuHand_Reorient",
  env_cfg=luwu_hand_reorient_env_cfg(num_envs=8192),
  play_env_cfg=luwu_hand_reorient_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Luwu_Reorient", max_iterations=5000
  ),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="LuwuHand_Reorient_Light",
  env_cfg=luwu_hand_reorient_env_cfg(num_envs=4096),
  play_env_cfg=luwu_hand_reorient_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Luwu_Reorient_Light", max_iterations=7500
  ),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="LuwuHand_Reorient_ContinuousFaces",
  env_cfg=luwu_hand_continuous_faces_env_cfg(num_envs=4096),
  play_env_cfg=luwu_hand_continuous_faces_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Luwu_ContinuousFaces_Transfer",
    max_iterations=3000,
    learning_rate=5.0e-5,
  ),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="LuwuHand_Reorient_FromScratch_Curriculum",
  env_cfg=luwu_hand_from_scratch_curriculum_env_cfg(num_envs=4096),
  play_env_cfg=luwu_hand_from_scratch_curriculum_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Luwu_FromScratch_Curriculum",
    max_iterations=3000,
  ),
  runner_cls=WujiOnPolicyRunner,
)

register_mjlab_task(
  task_id="LuwuHand_Reorient_80g_VisualTransfer",
  env_cfg=luwu_hand_80g_visual_transfer_env_cfg(num_envs=4096),
  play_env_cfg=luwu_hand_80g_visual_transfer_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Luwu_80g_VisualTransfer",
    max_iterations=3000,
    learning_rate=5.0e-5,
  ),
  runner_cls=WujiOnPolicyRunner,
)

rubik_rl_cfg = wuji_hand_reorient_ppo_runner_cfg(
  run_name="Luwu_Rubik_StableContact_3000",
  max_iterations=3000,
)
rubik_rl_cfg.actor.distribution_cfg["max_std"] = 0.8
rubik_rl_cfg.algorithm.entropy_coef = 0.0001

register_mjlab_task(
  task_id="LuwuHand_Rubik_UFace_Stage1",
  env_cfg=luwu_hand_rubik_u_face_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_u_face_env_cfg(play=True),
  rl_cfg=rubik_rl_cfg,
  runner_cls=WujiOnPolicyRunner,
)

rubik_transfer_rl_cfg = wuji_hand_reorient_ppo_runner_cfg(
  run_name="Luwu_Rubik_2999_Transfer",
  max_iterations=3000,
  learning_rate=5.0e-5,
)
rubik_transfer_rl_cfg.actor.distribution_cfg["max_std"] = 0.8
rubik_transfer_rl_cfg.algorithm.entropy_coef = 0.0001

register_mjlab_task(
  task_id="LuwuHand_Rubik_UFace_2999Transfer",
  env_cfg=luwu_hand_rubik_u_face_transfer_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_u_face_transfer_env_cfg(play=True),
  rl_cfg=rubik_transfer_rl_cfg,
  runner_cls=WujiOnPolicyRunner,
)

rubik_continuous_transfer_rl_cfg = wuji_hand_reorient_ppo_runner_cfg(
  run_name="Luwu_Rubik_Continuous_Transfer",
  max_iterations=3000,
  learning_rate=5.0e-5,
)
rubik_continuous_transfer_rl_cfg.actor.distribution_cfg["max_std"] = 0.8
rubik_continuous_transfer_rl_cfg.algorithm.entropy_coef = 0.0001

register_mjlab_task(
  task_id="LuwuHand_Rubik_UFace_ContinuousTransfer",
  env_cfg=luwu_hand_rubik_u_face_continuous_transfer_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_u_face_continuous_transfer_env_cfg(play=True),
  rl_cfg=rubik_continuous_transfer_rl_cfg,
  runner_cls=WujiOnPolicyRunner,
)

rubik_full_cube_transfer_rl_cfg = wuji_hand_reorient_ppo_runner_cfg(
  run_name="Luwu_Rubik_FullCube_SolverTransfer",
  max_iterations=3000,
  learning_rate=5.0e-5,
)
rubik_full_cube_transfer_rl_cfg.actor.distribution_cfg["max_std"] = 0.8
rubik_full_cube_transfer_rl_cfg.algorithm.entropy_coef = 0.0001

register_mjlab_task(
  task_id="LuwuHand_Rubik_FullCube_SolverTransfer",
  env_cfg=luwu_hand_rubik_full_cube_solver_transfer_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_full_cube_solver_transfer_env_cfg(play=True),
  rl_cfg=rubik_full_cube_transfer_rl_cfg,
  runner_cls=WujiOnPolicyRunner,
)

smart_cube_rl_cfg = wuji_hand_reorient_ppo_runner_cfg(
  run_name='Luwu_Rubik_SmartCube_Transfer', max_iterations=3000, learning_rate=2.0e-5,
)
smart_cube_rl_cfg.actor.distribution_cfg['max_std'] = 0.8
smart_cube_rl_cfg.algorithm.entropy_coef = 0.0001
register_mjlab_task(
  task_id='LuwuHand_Rubik_FullCube_SmartCubeTransfer',
  env_cfg=luwu_hand_rubik_smart_cube_transfer_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_smart_cube_transfer_env_cfg(play=True),
  rl_cfg=smart_cube_rl_cfg, runner_cls=WujiOnPolicyRunner,
)

flip_specialist_rl_cfg = wuji_hand_reorient_ppo_runner_cfg(
  run_name='Luwu_Rubik_CleanFlip', max_iterations=5000, learning_rate=2.0e-5,
)

register_mjlab_task(
  task_id='LuwuHand_Rubik_FlipBottomTipObserve',
  env_cfg=luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(play=True),
  rl_cfg=flip_specialist_rl_cfg, runner_cls=WujiOnPolicyRunner,
)
flip_specialist_rl_cfg.actor.distribution_cfg['max_std'] = .8
flip_specialist_rl_cfg.algorithm.entropy_coef = .0001
bottom_tip_rl_cfg = wuji_hand_reorient_ppo_runner_cfg(
  run_name='Luwu_Rubik_BottomTip_StageA', max_iterations=3000, learning_rate=5e-6,
)
bottom_tip_rl_cfg.actor.distribution_cfg['max_std'] = .8
bottom_tip_rl_cfg.algorithm.class_name = 'wuji_mjlab.rl.bottom_tip_ppo:BottomTipPPO'
bottom_tip_rl_cfg.algorithm.num_learning_epochs = 2
bottom_tip_rl_cfg.algorithm.clip_param = .1
bottom_tip_rl_cfg.algorithm.entropy_coef = .0001
register_mjlab_task(
  task_id='LuwuHand_Rubik_FlipBottomTipTransfer',
  env_cfg=luwu_hand_rubik_flip_bottom_tip_train_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_flip_bottom_tip_train_env_cfg(play=True),
  rl_cfg=bottom_tip_rl_cfg, runner_cls=WujiOnPolicyRunner,
)
register_mjlab_task(
  task_id='LuwuHand_Rubik_FlipSpecialist',
  env_cfg=luwu_hand_rubik_flip_specialist_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_flip_specialist_env_cfg(play=True),
  rl_cfg=flip_specialist_rl_cfg, runner_cls=WujiOnPolicyRunner,
)

# Active low-friction experiment: original model900 rewards and ordinary PPO.
flip_low_friction_rl_cfg = deepcopy(flip_specialist_rl_cfg)
flip_low_friction_rl_cfg.run_name = 'Luwu_Flip900Rewards_LowFriction_SymmetricJ1'
flip_low_friction_rl_cfg.algorithm.learning_rate = 1e-5
register_mjlab_task(
  task_id='LuwuHand_Rubik_FlipLowFrictionTransfer',
  env_cfg=luwu_hand_rubik_flip_low_friction_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_flip_low_friction_env_cfg(play=True),
  rl_cfg=flip_low_friction_rl_cfg, runner_cls=WujiOnPolicyRunner,
)

gentle_flip_rl_cfg = deepcopy(flip_low_friction_rl_cfg)
gentle_flip_rl_cfg.run_name = 'Luwu_GentleFlip_EMA050'
register_mjlab_task(
  task_id='LuwuHand_Rubik_FlipGentleTransfer',
  env_cfg=luwu_hand_rubik_flip_gentle_env_cfg(num_envs=256),
  play_env_cfg=luwu_hand_rubik_flip_gentle_env_cfg(play=True),
  rl_cfg=gentle_flip_rl_cfg, runner_cls=WujiOnPolicyRunner,
)
