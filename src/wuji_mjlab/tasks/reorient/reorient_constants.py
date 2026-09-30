# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reorient task-specific initial pose + frame constants.

``right_mjlab.xml`` has the palm at origin with no rotation (fingers point
along +Z). The wrist tag is mounted such that the palm frame and tag frame
differ by a clean **R_y(-90°)** rotation. We replicate this -90° here via
the robot root quaternion so sim's "palm-up" pose matches the real geometry.
"""

import math

from mjlab.entity import EntityCfg

# ---------------------------------------------------------------------------
# Robot root pose — palm rotated -90° around Y so palm faces up
# ---------------------------------------------------------------------------
# Quaternion for R_y(-90°): (cos(-45°), 0, sin(-45°), 0) ≈ (0.7071, 0, -0.7071, 0)
REORIENT_ROBOT_ROOT_POS = (0.0, 0.0, 0.5)
REORIENT_ROBOT_ROOT_ROT = (0.70710678, 0.0, -0.70710678, 0.0)

# ---------------------------------------------------------------------------
# Cube initial pose — in palm-up frame, above the fingertip cage
# ---------------------------------------------------------------------------
# Cube-relative-to-palm offset ≈ (0.060, 0.004, 0.075) in palm body local
# frame, transformed through R_y(-90°) into world coordinates so the cube
# sits above the fingertip cage in the palm-up world pose.
REORIENT_CUBE_INIT_POS = (-0.0967, 0.0100, 0.5599)
REORIENT_CUBE_INIT_ROT = (1.0, 0.0, 0.0, 0.0)  # identity; training randomises

REORIENT_CUBE_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REORIENT_CUBE_INIT_POS,
  rot=REORIENT_CUBE_INIT_ROT,
)

# Tag (wrist marker) ↔ palm_link rigid transform
#     palm_in_tag pos = (-0.0563, 0, -0.0262), quat (wxyz) = R_y(-90°)
#     tag_in_palm pos = ( 0.0262, 0, -0.0563), quat (wxyz) = R_y(+90°)
TAG_IN_PALM_POS = (0.0262, 0.0, -0.0563)
TAG_IN_PALM_QUAT_WXYZ = (
  math.cos(math.radians(45.0)),
  0.0,
  math.sin(math.radians(45.0)),
  0.0,
)

# ---------------------------------------------------------------------------
# Hand joint angles (active keyframe)
# ---------------------------------------------------------------------------
REORIENT_JOINT_POS: dict[str, float] = {
  ".*_finger1_joint1": 0.8,
  ".*_finger1_joint2": -0.0215,
  ".*_finger1_joint3": 0.285,
  ".*_finger1_joint4": 0.582,
  ".*_finger2_joint1": 0.441,
  ".*_finger2_joint2": 0.163,
  ".*_finger2_joint3": 0.822,
  ".*_finger2_joint4": 0.494,
  ".*_finger3_joint1": 0.448,
  ".*_finger3_joint2": -0.0832,
  ".*_finger3_joint3": 0.883,
  ".*_finger3_joint4": 0.305,
  ".*_finger4_joint1": 0.594,
  ".*_finger4_joint2": -0.268,
  ".*_finger4_joint3": 1.13,
  ".*_finger4_joint4": 0.18,
  ".*_finger5_joint1": 1.2,
  ".*_finger5_joint2": -0.228,
  ".*_finger5_joint3": 0.794,
  ".*_finger5_joint4": 0.407,
}

REORIENT_ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=REORIENT_ROBOT_ROOT_POS,
  rot=REORIENT_ROBOT_ROOT_ROT,
  joint_pos=REORIENT_JOINT_POS,
  joint_vel={".*": 0.0},
)
