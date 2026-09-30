# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from wuji_mjlab.tasks.reorient.mdp import observations, vision_observations


def _rubik_env(num_envs: int = 2):
  palm_pose = torch.zeros((num_envs, 1, 7))
  palm_pose[..., 3] = 1.0
  cube_quat = torch.zeros((num_envs, 4))
  cube_quat[:, 0] = 1.0
  robot = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=palm_pose))
  rubik = SimpleNamespace(data=SimpleNamespace(root_link_quat_w=cube_quat))
  command = SimpleNamespace(
    object=rubik,
    robot=robot,
    palm_body_id=0,
  )
  return SimpleNamespace(
    num_envs=num_envs,
    device=torch.device("cpu"),
    common_step_counter=7,
    scene=SimpleNamespace(device=torch.device("cpu")),
    command_manager=SimpleNamespace(get_term=lambda _name: command),
  )


def test_rubik_orientation_terms_share_one_visual_sample_per_step(monkeypatch):
  env = _rubik_env()
  profile = vision_observations.VisionPoseStage(
    stage=3,
    noise_range_rad=0.0,
    bias_limit_rad=0.0,
    bias_step_rad=0.0,
    calibration_limit_rad=0.0,
    delay_steps=(0, 0),
    dropout_probability=0.0,
    max_hold_steps=0,
  )
  monkeypatch.setattr(observations, "vision_stage_parameters", lambda _stage: profile)
  original_step = observations.step_visual_pose_model
  batch_sizes = []

  def count_visual_advances(current, state, selected_profile, **kwargs):
    step_id = kwargs.get("step_id")
    if step_id is not None:
      pending = int((state.last_update_step != step_id).sum())
      if pending:
        batch_sizes.append(pending)
    return original_step(current, state, selected_profile, **kwargs)

  monkeypatch.setattr(observations, "step_visual_pose_model", count_visual_advances)

  first = observations.rubik_cube_orientation_6d_in_tag(env, vision_dr=True)
  assert torch.equal(
    env._visual_pose_state.last_update_step,
    torch.full((2,), env.common_step_counter, dtype=torch.long),
  )
  repeated = observations.rubik_cube_orientation_6d_in_tag(env, vision_dr=True)
  assert torch.equal(first, repeated)
  assert batch_sizes == [2]

  env.common_step_counter += 1
  observations.rubik_cube_orientation_6d_in_tag(env, vision_dr=True)
  assert batch_sizes == [2, 2]

  vision_observations.reset_visual_pose_state(
    env, torch.tensor([1], dtype=torch.long)
  )
  after_partial_reset = observations.rubik_cube_orientation_6d_in_tag(
    env, vision_dr=True
  )
  assert torch.equal(after_partial_reset[0], first[0])
  assert batch_sizes == [2, 2, 1]


def test_transfer_rubik_env_builder_is_available_without_changing_scratch_builder():
  from wuji_mjlab.tasks.reorient.config.luwu_hand import rubik_env_cfg

  builder = getattr(rubik_env_cfg, "luwu_hand_rubik_u_face_transfer_env_cfg", None)
  assert callable(builder)

  scratch_cfg = rubik_env_cfg.luwu_hand_rubik_u_face_env_cfg(num_envs=1)
  scratch_terms = tuple(scratch_cfg.observations["policy"].terms)
  cfg = builder(num_envs=1)
  assert cfg.scene.num_envs == 1
  assert tuple(cfg.observations["policy"].terms) == (
    "noisy_joint_angles",
    "qpos_error",
    "rubik_compat_cube_ori_error",
    "action_history",
    "rubik_cube_orientation",
    "vision_valid",
    "vision_age",
    "vision_confidence",
    "vision_innovation",
    "rubik_u_face_angle",
    "rubik_u_face_goal",
    "rubik_face_goal",
    "rubik_root_goal",
    "rubik_task_phase",
  )
  assert tuple(scratch_cfg.observations["policy"].terms) == scratch_terms
  assert "rubik_compat_cube_ori_error" not in scratch_terms


def test_transfer_task_is_registered_as_separate_task():
  import wuji_mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  tasks = list_tasks()
  assert "LuwuHand_Rubik_UFace_Stage1" in tasks
  assert "LuwuHand_Rubik_UFace_2999Transfer" in tasks
  assert "LuwuHand_Rubik_UFace_ContinuousTransfer" in tasks


def test_rubik_tip_contact_sensor_matches_the_entire_object_subtree():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_u_face_env_cfg,
  )

  cfg = luwu_hand_rubik_u_face_env_cfg(num_envs=1)
  sensor_cfgs = {sensor.name: sensor for sensor in cfg.scene.sensors}
  tip_sensor = sensor_cfgs["tip_object_contact"]
  assert tip_sensor.secondary.mode == "subtree"
  assert tip_sensor.secondary.pattern == "middle"
  assert tip_sensor.secondary.entity == "object"
  assert not any(name.startswith("tip_object_contact_") for name in sensor_cfgs)


def test_compat_orientation_error_uses_root_goal_and_handles_opposite_face():
  env = _rubik_env()
  command = env.command_manager.get_term("reorient_command")
  command.phase = torch.tensor([2, 0], dtype=torch.long)
  command.target_face_id = torch.tensor([4, 5], dtype=torch.long)
  command.root_target_quat_w = torch.zeros((2, 4))
  command.root_target_quat_w[:, 0] = 1.0

  error = observations.rubik_compat_cube_orientation_error_6d(
    env, vision_dr=False
  )

  expected_identity_6d = torch.tensor((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
  assert error.shape == (2, 6)
  assert torch.isfinite(error).all()
  assert torch.allclose(error[0], expected_identity_6d, atol=1e-6)
