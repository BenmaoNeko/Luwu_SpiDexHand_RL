# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import mujoco

from mjlab.tasks.registry import list_tasks, load_rl_cfg
from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.tasks.reorient.config.luwu_hand.continuous_env_cfgs import (
  luwu_hand_80g_visual_transfer_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.vision_observations import (
  create_visual_pose_state,
  vision_stage_parameters,
)


def test_mass_override_changes_only_the_requested_cube_spec():
  baseline = get_inhand_object_cfg().spec_fn().compile()
  transfer = get_inhand_object_cfg(mass_kg=0.080).spec_fn().compile()
  cube_id = mujoco.mj_name2id(transfer, mujoco.mjtObj.mjOBJ_BODY, "cube")

  assert math.isclose(float(baseline.body_mass[cube_id]), 0.120, abs_tol=1e-6)
  assert math.isclose(float(transfer.body_mass[cube_id]), 0.080, abs_tol=1e-6)


def test_transfer_visual_profile_expands_only_approved_fields_and_has_buffer():
  stage3 = vision_stage_parameters(3)
  stage4 = vision_stage_parameters(4)
  state = create_visual_pose_state(num_envs=1, device="cpu")

  assert math.isclose(stage4.noise_range_rad, math.radians(7.0))
  assert stage4.bias_limit_rad == stage3.bias_limit_rad
  assert stage4.bias_step_rad == stage3.bias_step_rad
  assert stage4.calibration_limit_rad == stage3.calibration_limit_rad
  assert stage4.delay_steps == (1, 8)
  assert stage4.dropout_probability == 0.08
  assert stage4.max_hold_steps == 6
  assert state.pose_history.shape == (1, 9, 4)


def test_80g_transfer_task_preserves_checkpoint_actor_and_action_contract():
  cfg = luwu_hand_80g_visual_transfer_env_cfg(num_envs=1)
  policy = cfg.observations["policy"].terms
  ori = policy["cube_ori_error"]
  cube_spec = cfg.scene.entities["object"].spec_fn().compile()
  cube_id = mujoco.mj_name2id(cube_spec, mujoco.mjtObj.mjOBJ_BODY, "cube")

  assert set(policy) == {
    "noisy_joint_angles",
    "qpos_error",
    "cube_ori_error",
    "action_history",
  }
  assert ori.params["vision_dr"] is True
  assert ori.params["vision_stage_override"] == 4
  assert math.isclose(float(cube_spec.body_mass[cube_id]), 0.080, abs_tol=1e-6)
  assert 3 * (16 + 16 + 6 + 16) == 162

  command = cfg.commands["reorient_command"]
  assert command.curriculum_enabled is False
  assert command.curriculum_stage == 3
  assert command.goal_sampling_mode == "curriculum_flip"
  assert command.yaw_goal_probability == 0.30
  assert cfg.events["object_mass"].params["scale_range"] == (0.85, 1.15)
  assert cfg.rewards["rotation_success"].weight == 5.0
  assert cfg.rewards["flip_success"].weight == 40.0
  assert cfg.rewards["action_rate"].weight == -0.10
  assert cfg.rewards["finger_collision"].weight == -0.25
  assert cfg.actions["joint_pos"].action_scale == 1.0
  assert cfg.actions["joint_pos"].use_joint_limits is True


def test_80g_transfer_task_is_registered_with_actor_transfer_runner():
  task_id = "LuwuHand_Reorient_80g_VisualTransfer"
  assert task_id in list_tasks()
  agent = load_rl_cfg(task_id)

  assert agent.max_iterations == 3000
  assert agent.algorithm.learning_rate == 5.0e-5
  assert agent.run_name == "Luwu_80g_VisualTransfer"
