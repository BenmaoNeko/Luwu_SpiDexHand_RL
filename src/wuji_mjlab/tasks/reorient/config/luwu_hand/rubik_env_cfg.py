# SPDX-License-Identifier: Apache-2.0
"""Stage-one Luwu task: continuous 45/45/10 Rubik manipulation."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch

from wuji_mjlab.assets.objects.rubik.rubik_cfg import (
  RUBIK_INIT_STATE,
  get_rubik_cfg,
  get_rubik_target_cfg,
)
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.mdp.rubik_command import RubikUFaceCommandCfg

from .env_cfgs import luwu_hand_reorient_env_cfg


def luwu_hand_rubik_u_face_env_cfg(
  play: bool = False,
  num_envs: int = 256,
):
  """Build the constrained full-cube task with a fresh actor/critic policy."""
  cfg = luwu_hand_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    enable_visual_pose_dr=True,
    object_body_pattern=r"cubelet:.*",
    object_geom_pattern=r"cubelet:.*",
    tip_object_secondary=ContactMatch(
      mode="subtree", pattern="middle", entity="object"
    ),
  )
  if play:
    cfg.scene.num_envs = 1

  cfg.scene.entities["object"] = get_rubik_cfg()
  cfg.scene.entities["object"].init_state = RUBIK_INIT_STATE
  if play:
    cfg.scene.entities["target"] = get_rubik_target_cfg()
  cfg.commands["reorient_command"] = RubikUFaceCommandCfg(
    entity_name="object",
    target_entity_name="target" if play else "",
    resampling_time_range=(1.0e6, 1.0e6),
    upright_tolerance_rad=0.20,
    upright_hold_steps=8,
    flip_angle_tolerance_rad=0.1745329252,
    turn_tolerance_rad=0.0872664626,
    turn_hold_steps=8,
    max_u_face_speed=0.45,
    combine_after_steps=40_000,  # Iteration 1000 of a 3000-iteration run.
  )
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=mdp.reset_rubik_u_face,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("object"),
      "robot_cfg": SceneEntityCfg("robot", body_names=("right_palm_link",)),
      "palm_clearance_m": 0.006,
    },
  )
  if not play:
    cfg.events["rubik_mass"] = EventTermCfg(
      func=mdp.randomize_body_mass_and_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "object", body_names=("middle", "cubelet:.*")
        ),
        "scale_range": (0.7875, 1.0),
        "shared_scale_per_env": True,
      },
    )

  policy = cfg.observations["policy"].terms
  policy.pop("cube_ori_error", None)
  policy["rubik_cube_orientation"] = ObservationTermCfg(
    func=mdp.rubik_cube_orientation_6d_in_tag,
    params={"vision_dr": True},
    history_length=3,
  )
  policy["rubik_u_face_angle"] = ObservationTermCfg(
    func=mdp.rubik_u_face_angle_sincos,
    params={"observation_noise_rad": 0.05},
    history_length=3,
  )
  policy["rubik_u_face_goal"] = ObservationTermCfg(
    func=mdp.rubik_u_face_goal_sincos,
    history_length=0,
  )
  policy["rubik_face_goal"] = ObservationTermCfg(
    func=mdp.rubik_face_goal_local,
    history_length=0,
  )
  policy["rubik_root_goal"] = ObservationTermCfg(
    func=mdp.rubik_root_goal_6d_in_tag,
    history_length=0,
  )
  policy["rubik_task_phase"] = ObservationTermCfg(
    func=mdp.rubik_task_phase,
    history_length=0,
  )

  critic = cfg.observations["critic"].terms
  for name in (
    "cube_ori_error",
    "cube_ori_error_clean",
    "state_progress",
  ):
    critic.pop(name, None)
  critic["rubik_state_privileged"] = ObservationTermCfg(
    func=mdp.rubik_state_privileged,
    history_length=0,
  )

  for name in (
    "goal_progress",
    "rotation_success",
    "flip_success",
  ):
    cfg.rewards.pop(name, None)
  cfg.rewards["approach_progress"].params.update(
    object_cfg=SceneEntityCfg("object"),
    box_half_size_m=0.02855,  # Robogym cubelet envelope: 57.1 mm edge.
  )
  cfg.rewards.update(
    {
      "rubik_task_progress": RewardTermCfg(
        func=mdp.rubik_task_progress_reward,
        weight=10.0,
      ),
      "rubik_flip_to_top": RewardTermCfg(
        func=mdp.rubik_flip_to_top_reward,
        weight=5.0,
      ),
      "rubik_u_turn_success": RewardTermCfg(
        func=mdp.rubik_u_turn_success_reward,
        weight=10.0,
      ),
      "rubik_cube_yaw_success": RewardTermCfg(
        func=mdp.rubik_cube_yaw_success_reward,
        weight=1.0,
      ),
    }
  )
  cfg.rewards["finger_collision"].weight = -0.15

  cfg.terminations.pop("flip_completed", None)
  cfg.curriculum.clear()

  for name in (
    "success_interval",
    "goal_reach_count",
    "rotation_goal_reach_count",
    "flip_success_count",
  ):
    cfg.metrics.pop(name, None)
  cfg.metrics.update(
    {
      "rubik_task_phase": MetricsTermCfg(func=mdp.rubik_phase_metric),
      "rubik_turn_task_fraction": MetricsTermCfg(
        func=mdp.rubik_turn_task_fraction,
      ),
      "rubik_yaw_task_fraction": MetricsTermCfg(func=mdp.rubik_yaw_task_fraction),
      "rubik_upright_dot": MetricsTermCfg(func=mdp.rubik_upright_dot),
      "rubik_u_face_angle_error": MetricsTermCfg(
        func=mdp.rubik_u_face_angle_error,
      ),
      "rubik_u_face_speed": MetricsTermCfg(func=mdp.rubik_u_face_speed),
      "rubik_flip_to_top_count": MetricsTermCfg(
        func=mdp.rubik_flip_to_top_count,
      ),
      "rubik_turn_success_count": MetricsTermCfg(
        func=mdp.rubik_turn_success_count,
      ),
      "rubik_yaw_success_count": MetricsTermCfg(func=mdp.rubik_yaw_success_count),
      "rubik_goal_reach_count": MetricsTermCfg(func=mdp.rubik_goal_reach_count),
      "rubik_flip_stage": MetricsTermCfg(func=mdp.rubik_flip_stage),
      "rubik_turn_stage": MetricsTermCfg(func=mdp.rubik_turn_stage),
      "rubik_side_success_rate": MetricsTermCfg(func=mdp.rubik_side_success_rate),
      "rubik_bottom_success_rate": MetricsTermCfg(func=mdp.rubik_bottom_success_rate),
      "rubik_turn90_success_rate": MetricsTermCfg(func=mdp.rubik_turn90_success_rate),
      "rubik_turn180_success_rate": MetricsTermCfg(func=mdp.rubik_turn180_success_rate),
      "rubik_yaw_success_rate": MetricsTermCfg(func=mdp.rubik_yaw_success_rate),
    }
  )
  cfg.episode_length_s = 50.0
  return cfg


def luwu_hand_rubik_u_face_transfer_env_cfg(
  play: bool = False,
  num_envs: int = 256,
):
  """Build a Rubik environment with the successful cube actor input as prefix."""
  cfg = luwu_hand_rubik_u_face_env_cfg(play=play, num_envs=num_envs)
  # Transfer-specific balance: strengthen all three task completions and
  # soften the self-collision cost while adapting the pretrained actor.
  cfg.rewards["rubik_flip_to_top"].weight = 10.0
  cfg.rewards["rubik_u_turn_success"].weight = 20.0
  cfg.rewards["rubik_cube_yaw_success"].weight = 3.0
  cfg.rewards["finger_collision"].weight = -0.07

  terms = cfg.observations["policy"].terms
  legacy_joint_angles = terms["noisy_joint_angles"]
  legacy_qpos_error = terms["qpos_error"]
  legacy_action_history = terms["action_history"]
  compat_orientation = ObservationTermCfg(
    func=mdp.rubik_compat_cube_orientation_error_6d,
    params={"vision_dr": True},
    history_length=3,
  )
  extra_order = (
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
  missing = [name for name in extra_order if name not in terms]
  if missing:
    raise ValueError(f"Rubik transfer observation terms are missing: {missing}")
  extras = {name: terms[name] for name in extra_order}
  terms.clear()
  terms.update(
    {
      "noisy_joint_angles": legacy_joint_angles,
      "qpos_error": legacy_qpos_error,
      "rubik_compat_cube_ori_error": compat_orientation,
      "action_history": legacy_action_history,
      **extras,
    }
  )
  return cfg


def luwu_hand_rubik_u_face_continuous_transfer_env_cfg(
  play: bool = False,
  num_envs: int = 256,
):
  """Initialize Rubik control from the six-face continuous cube actor."""
  cfg = luwu_hand_rubik_u_face_transfer_env_cfg(play=play, num_envs=num_envs)
  cfg.rewards["rubik_u_turn_progress"] = RewardTermCfg(
    func=mdp.rubik_u_turn_progress_reward,
    weight=5.0,
  )
  return cfg
