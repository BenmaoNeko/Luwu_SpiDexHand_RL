# SPDX-License-Identifier: Apache-2.0
"""Initial states for the fixed-arm Luwu reorientation task."""

from mjlab.entity import EntityCfg

from wuji_mjlab.assets.robots.luwu_hand.luwu_hand_cfg import HAND_HOME_FIRST_JOINT_RAD


# Keep the reset deterministic: face A starts upward in every episode.
LUWU_CUBE_RESET_QUATERNIONS_WXYZ = ((1.0, 0.0, 0.0, 0.0),)

# Relative goals are expressed from the cube reset/world frame, in MuJoCo
# WXYZ order. The common task advances by +15° yaw after each successful goal;
# the minority task is a one-shot +90° A→B flip.
LUWU_ROTATION_STEP_DEGREES = 15.0

# Kept for compatibility with the legacy fixed +90° yaw sampler.
LUWU_YAW_GOAL_QUAT_WXYZ = (
  0.7071067811865476,
  0.0,
  0.0,
  0.7071067811865476,
)
LUWU_FLIP_GOAL_QUAT_WXYZ = (
  0.7071067811865476,
  0.7071067811865476,
  0.0,
  0.0,
)


LUWU_ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    f"right_finger{finger}_joint1": HAND_HOME_FIRST_JOINT_RAD
    for finger in range(1, 5)
  },
  joint_vel={".*": 0.0},
)


# Generated from the approved fixed-arm J4=+1.5 rad pose. The cube is a
# separate dynamic entity in the Wuji scene, so this is an absolute world pose.
LUWU_CUBE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(-0.0264848718, 8.59749199e-08, 0.40649396),
  rot=(1.0, 0.0, 0.0, 0.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
)
