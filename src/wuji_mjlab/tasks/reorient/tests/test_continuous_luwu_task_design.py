# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from mjlab.tasks.registry import list_tasks, load_rl_cfg
from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
  luwu_hand_continuous_faces_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.luwu_hand.env_cfgs import (
  luwu_hand_reorient_env_cfg,
)


def test_continuous_task_uses_random_faces_and_keeps_episode_alive():
  cfg = luwu_hand_continuous_faces_env_cfg(num_envs=4096)
  command = cfg.commands["reorient_command"]

  assert command.goal_sampling_mode == "continuous_yaw_or_random_face"
  assert command.yaw_goal_probability == 0.50
  assert command.success_hold_steps == 10
  assert command.goal_switch_delay == 0
  assert command.require_tip_contact is True
  assert "flip_completed" not in cfg.terminations
  assert "adaptive_episode" not in cfg.curriculum
  assert cfg.curriculum["success_curriculum"].params["flip_count_threshold"] == 2


def test_continuous_task_uses_transfer_rewards_and_full_joint_limits():
  cfg = luwu_hand_continuous_faces_env_cfg(num_envs=1)

  assert cfg.rewards["finger_collision"].weight == -0.25
  assert cfg.rewards["action_rate"].weight == -0.10
  assert cfg.rewards["flip_success"].weight == 30.0
  assert cfg.actions["joint_pos"].use_joint_limits is True
  assert cfg.actions["joint_pos"].action_scale == 1.0


def test_continuous_task_reenables_the_approved_transfer_dr_ranges():
  cfg = luwu_hand_continuous_faces_env_cfg(num_envs=1)

  assert cfg.events["object_com"].params["ranges"] == (-0.001, 0.001)
  assert cfg.events["object_size"].params["scale_range"] == (0.98, 1.02)
  assert cfg.events["object_mass"].params["scale_range"] == (0.9, 1.1)
  assert cfg.events["robot_friction"].params["ranges"] == (0.65, 1.35)
  assert cfg.events["pd_gains"].params["kp_range"] == (0.75, 1.25)
  assert cfg.events["pd_gains"].params["kd_range"] == (0.75, 1.25)
  assert cfg.events["encoder_bias"].params["bias_range"] == (-0.015, 0.015)


def test_original_baseline_config_is_not_mutated():
  cfg = luwu_hand_reorient_env_cfg(num_envs=1)
  command = cfg.commands["reorient_command"]

  assert command.goal_sampling_mode == "incremental_yaw_or_flip"
  assert command.yaw_goal_probability == 0.5
  assert command.success_hold_steps == 5
  assert cfg.rewards["finger_collision"].weight == -0.08
  assert cfg.rewards["action_rate"].weight == -0.04
  assert "flip_completed" in cfg.terminations


def test_continuous_task_is_registered_with_transfer_runner_settings():
  task_id = "LuwuHand_Reorient_ContinuousFaces"
  assert task_id in list_tasks()
  agent = load_rl_cfg(task_id)

  assert agent.max_iterations == 3000
  assert agent.algorithm.learning_rate == 5.0e-5
  assert agent.run_name == "Luwu_ContinuousFaces_Transfer"


def test_from_scratch_curriculum_task_uses_fresh_final_interface():
  from mjlab.tasks.registry import list_tasks, load_rl_cfg
  from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
    luwu_hand_from_scratch_curriculum_env_cfg,
  )

  task_id = "LuwuHand_Reorient_FromScratch_Curriculum"
  assert task_id in list_tasks()
  cfg = luwu_hand_from_scratch_curriculum_env_cfg(num_envs=1)
  agent = load_rl_cfg(task_id)
  command = cfg.commands["reorient_command"]

  assert command.curriculum_enabled is True
  assert command.curriculum_stage == 0
  assert command.goal_sampling_mode == "curriculum_flip"
  assert command.yaw_goal_probability == 0.3
  assert command.success_hold_steps == 5
  assert cfg.rewards["finger_collision"].weight == -0.08
  assert "flip_completed" in cfg.terminations
  assert cfg.actions["joint_pos"].action_scale == 1.0
  assert cfg.actions["joint_pos"].use_joint_limits is True
  assert cfg.observations["policy"].terms["cube_ori_error"].params["vision_dr"] is True
  assert set(cfg.observations["policy"].terms) >= {
    "vision_valid",
    "vision_age",
    "vision_confidence",
    "vision_innovation",
  }
  assert agent.max_iterations == 3000
  assert agent.run_name == "Luwu_FromScratch_Curriculum"
  assert "transfer_checkpoint_file" not in agent.__dict__


def test_from_scratch_play_uses_final_stage_without_training_curriculum():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
    luwu_hand_from_scratch_curriculum_env_cfg,
  )

  cfg = luwu_hand_from_scratch_curriculum_env_cfg(play=True)
  command = cfg.commands["reorient_command"]

  assert cfg.scene.num_envs == 8
  assert command.curriculum_enabled is False
  assert command.curriculum_stage == 3
  assert command.goal_sampling_mode == "curriculum_flip"
  assert "flip_completed" in cfg.terminations


def test_curriculum_stage_updates_action_dr_and_safety_scales():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
    luwu_hand_from_scratch_curriculum_env_cfg,
  )
  from wuji_mjlab.tasks.reorient.mdp.flip_curriculum import (
    apply_stage_parameters,
    stage_parameters,
  )

  cfg = luwu_hand_from_scratch_curriculum_env_cfg(num_envs=1)
  apply_stage_parameters(cfg, stage_parameters(3))

  assert cfg.actions["joint_pos"].command_delay_steps == 2
  assert cfg.actions["joint_pos"].command_noise_std == 0.01
  assert cfg.rewards["action_rate"].weight == -0.10
  assert cfg.rewards["finger_collision"].weight == -0.25
  assert cfg.events["object_com"].params["ranges"] == (-0.003, 0.003)
  assert cfg.events["object_size"].params["scale_range"] == (0.97, 1.03)
  assert cfg.events["object_mass"].params["scale_range"] == (0.85, 1.15)
  assert cfg.events["robot_friction"].params["ranges"] == (0.5, 1.5)


def test_from_scratch_training_defers_visual_pose_dr_until_after_skill_learning():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
    luwu_hand_from_scratch_curriculum_env_cfg,
  )

  cfg = luwu_hand_from_scratch_curriculum_env_cfg(num_envs=1)
  policy_ori = cfg.observations["policy"].terms["cube_ori_error"]
  critic_ori = cfg.observations["critic"].terms["cube_ori_error"]

  assert policy_ori.params["vision_dr"] is True
  assert policy_ori.noise is None
  assert "vision_dr" not in critic_ori.params
  assert "cube_pos_in_tag" not in cfg.observations["policy"].terms


def test_from_scratch_play_disables_visual_pose_dr_without_changing_terms():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
    luwu_hand_from_scratch_curriculum_env_cfg,
  )

  cfg = luwu_hand_from_scratch_curriculum_env_cfg(play=True)
  policy_ori = cfg.observations["policy"].terms["cube_ori_error"]

  assert policy_ori.params["vision_dr"] is False
  assert policy_ori.noise is not None
  assert "cube_pos_in_tag" not in cfg.observations["policy"].terms
