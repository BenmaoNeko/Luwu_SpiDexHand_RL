# SPDX-License-Identifier: Apache-2.0
"""Luwu four-finger binding for the shared reorientation task."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.luwu_hand.luwu_hand_cfg import get_luwu_hand_cfg
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg

from .luwu_constants import (
  LUWU_CUBE_INIT_STATE,
  LUWU_CUBE_RESET_QUATERNIONS_WXYZ,
  LUWU_FLIP_GOAL_QUAT_WXYZ,
  LUWU_ROBOT_INIT_STATE,
  LUWU_ROTATION_STEP_DEGREES,
)

LUWU_TIP_SITE_NAMES = (r".*_finger[1-4]_tip",)
LUWU_TIP_BODY_NAMES = (r".*_finger[1-4]_link4",)
LUWU_TIP_COLLISION_GEOMS = (r".*_finger[1-4]_link4_col",)
LUWU_UNDESIRED_OBJECT_CONTACT_BODIES = (
  r".*_palm_link",
  r".*_finger[1-4]_link1",
  r".*_finger[1-4]_link2",
)


def luwu_hand_reorient_env_cfg(
  play: bool = False,
  num_envs: int = 8192,
  *,
  enable_visual_pose_dr: bool = False,
  object_body_pattern: str = "cube",
  object_geom_pattern: str = "cube",
  tip_object_secondary: ContactMatch | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create the fixed-arm Luwu hand reorientation configuration."""
  cfg = make_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    tip_site_names=LUWU_TIP_SITE_NAMES,
    tip_body_names=LUWU_TIP_BODY_NAMES,
    tip_collision_geoms=LUWU_TIP_COLLISION_GEOMS,
    undesired_object_contact_bodies=LUWU_UNDESIRED_OBJECT_CONTACT_BODIES,
    include_cube_position_in_policy=False,
    enable_visual_pose_dr=enable_visual_pose_dr,
    object_body_pattern=object_body_pattern,
    object_geom_pattern=object_geom_pattern,
    tip_object_secondary=tip_object_secondary,
  )
  cfg.scene.entities = {
    "robot": get_luwu_hand_cfg(),
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = LUWU_ROBOT_INIT_STATE
  cfg.scene.entities["object"].init_state = LUWU_CUBE_INIT_STATE

  # Start every episode with a stationary cube at the calibrated palm pose.
  # Face A is always upward; the command, not the reset, defines whether the
  # episode is an incremental yaw or an A→B flip task.
  reset_object_pose = cfg.events["reset_object_pose"]
  reset_object_pose.params["pos_noise"] = 0.0
  reset_object_pose.params[
    "orientation_quaternions"
  ] = LUWU_CUBE_RESET_QUATERNIONS_WXYZ
  for event_name in (
    "object_disturbance_force",
    "reset_object_disturbance_force",
    "object_com",
    "object_size",
    "object_mass",
  ):
    cfg.events.pop(event_name, None)

  # The arm is baked into the fixed MJCF at J4=+1.5 rad. Keep the root reset
  # event with zero pose noise: for fixed-base robots MJLab uses this event to
  # place each mocap replica at its own environment origin. Removing it would
  # stack every arm at world (0, 0, 0) while the cube still resets per origin.
  reset_robot_pose = cfg.events["reset_robot_pose"]
  reset_robot_pose.params["pose_range"] = {}
  reset_robot_joints = cfg.events["reset_robot_joints"]
  reset_robot_joints.params.update(
    position_range={},
    velocity_range={},
    use_default_offset=False,
    operation="abs",
  )

  # The Luwu policy can command the full per-joint URDF range around home.
  cfg.actions["joint_pos"].action_scale = 1.0
  cfg.actions["joint_pos"].use_joint_limits = True

  # Use only the two manipulation behaviors defined for the real experiment:
  # 50% incremental +15° yaw targets, 50% one-shot +90° A→B flip.
  command_cfg = cfg.commands["reorient_command"]
  command_cfg.goal_sampling_mode = "incremental_yaw_or_flip"
  command_cfg.yaw_goal_probability = 0.5
  command_cfg.rotation_step_degrees = LUWU_ROTATION_STEP_DEGREES
  command_cfg.flip_goal_quat_wxyz = LUWU_FLIP_GOAL_QUAT_WXYZ
  command_cfg.success_threshold = math.radians(4.0)
  command_cfg.goal_switch_delay = 0
  command_cfg.require_tip_contact = True
  command_cfg.min_contact_fingertips = 1

  # Use a soft nominal PD controller and only a narrow 0.8–1.2 startup
  # randomization band around it.
  pd_event = cfg.events.get("pd_gains")
  if pd_event is not None:
    pd_event.params["kp_range"] = (0.8, 1.2)
    pd_event.params["kd_range"] = (0.8, 1.2)
  # Luwu collision geoms are mesh assets: the arm uses convex hulls while the
  # hand uses the original STL meshes. MuJoCo's generic geom-size randomizer
  # only supports primitive geoms, so this event must not reach the Luwu
  # training environment.
  cfg.events.pop("robot_geom_size", None)

  # The four-finger hand has no Wuji thumb softbody geom. Reuse the common
  # rigid-finger contact randomization for all four fingers instead.
  cfg.events.pop("contact_params_palm_thumb", None)
  contact_event = cfg.events.get("contact_params_fingers")
  if contact_event is not None:
    contact_event.params["robot_cfg"].geom_names = (
      r".*finger[1-4]_link[2-4]_col",
    )

  # Luwu uses bounded contact-driven shaping. The generic orientation reward
  # is too generous for a 15-degree goal and the old hold reward was unbounded.
  for reward_name in (
    "orientation_alignment",
    "hold_escalation",
    "palm_detach",
    "hand_pose",
    "tip_slide",
  ):
    cfg.rewards.pop(reward_name, None)
  tip_cfg = SceneEntityCfg("robot", site_names=LUWU_TIP_SITE_NAMES)
  cube_cfg = SceneEntityCfg("object", geom_names=("cube",))
  contact_cfg = SceneEntityCfg("tip_object_contact")
  cfg.rewards.update(
    {
      "approach_progress": RewardTermCfg(
        func=mdp.approach_progress_reward,
        weight=1.0,
        params={
          "robot_cfg": tip_cfg,
          "object_cfg": cube_cfg,
          "step_budget": 2.0,
        },
      ),
      "contact_milestones": RewardTermCfg(
        func=mdp.contact_milestone_reward,
        weight=1.0,
        params={
          "sensor_cfg": contact_cfg,
          "single_reward": 1.0,
          "double_reward": 1.0,
        },
      ),
      "goal_progress": RewardTermCfg(
        func=mdp.contact_gated_goal_progress_reward,
        weight=1.0,
        params={"sensor_cfg": contact_cfg, "step_budget": 10.0},
      ),
      "rotation_success": RewardTermCfg(
        func=mdp.rotation_success_reward,
        weight=20.0,
      ),
      "flip_success": RewardTermCfg(
        func=mdp.flip_success_reward,
        weight=30.0,
      ),
      "cage_drop": RewardTermCfg(
        func=mdp.drop_penalty_sparse,
        weight=-10.0,
        params={"term_name": "cage_drop"},
      ),
    }
  )
  cfg.rewards["action_rate"].func = mdp.action_target_smoothness_penalty
  cfg.rewards["action_rate"].weight = -0.04
  cfg.rewards["action_rate"].params.update(
    action_name="joint_pos",
    asset_cfg=SceneEntityCfg("robot"),
  )
  cfg.rewards["torque"].weight = -0.02
  cfg.rewards["torque"].params.update(
    torque_limit=0.5,
    reduce_mean=True,
  )
  cfg.rewards["finger_collision"].weight = -0.08
  cfg.rewards["finger_collision"].params.update(
    normalize=True,
  )
  cfg.rewards["cage_escape"].weight = -0.2
  cfg.rewards["cage_escape"].params["bounded_escape_penalty"] = True

  cfg.terminations["flip_completed"] = TerminationTermCfg(
    func=mdp.flip_completed,
    params={"command_name": "reorient_command"},
    time_out=False,
  )
  cfg.commands["reorient_command"].synchronized_contact_task = True
  if "adaptive_episode" in cfg.curriculum:
    cfg.curriculum["adaptive_episode"].params["success_term"] = "flip_completed"
  if "success_curriculum" in cfg.curriculum:
    cfg.curriculum["success_curriculum"].params.update(
      task_aware=True,
      count_threshold=3,
    )
  cfg.metrics.update(
    {
      "rotation_goal_reach_count": MetricsTermCfg(
        func=mdp.rotation_goal_reach_count,
        params={"command_name": "reorient_command"},
      ),
      "flip_success_count": MetricsTermCfg(
        func=mdp.flip_success_count,
        params={"command_name": "reorient_command"},
      ),
    }
  )
  cfg.viewer.body_name = "right_palm_link"
  return cfg
