# SPDX-License-Identifier: Apache-2.0
"""MJLab configuration for the fixed, horizontal Luwu four-finger hand."""

from functools import partial
from pathlib import Path

import mujoco
try:
  from mjlab.actuator import XmlActuatorCfg
except ImportError:  # mjlab 1.2 names the same config XmlPositionActuatorCfg.
  from mjlab.actuator import XmlPositionActuatorCfg as XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


_ASSET_DIR = Path(__file__).resolve().parent
LUWU_HAND_XML = _ASSET_DIR / "mjcf" / "luwu_mjlab.xml"

HAND_HOME_FIRST_JOINT_RAD = 0.610865238
LUWU_HAND_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={
    f"right_finger{finger}_joint1": HAND_HOME_FIRST_JOINT_RAD
    for finger in range(1, 5)
  },
  joint_vel={".*": 0.0},
)

LUWU_HAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=(r"right_finger[1-4]_joint[1-4]",),
    ),
  ),
  # Keep the normalized limits equal to the URDF limits. The XML itself
  # carries the exact lower/upper limits and the ±0.5 Nm force caps.
  soft_joint_pos_limit_factor=1.0,
)


def _update_assets(assets: dict[str, bytes], path: Path, meshdir: str) -> None:
  for file_path in path.rglob("*"):
    if file_path.is_file():
      assets[f"{meshdir}/{file_path.relative_to(path).as_posix()}"] = file_path.read_bytes()


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(LUWU_HAND_XML))
  mesh_root = (_ASSET_DIR / "meshes").resolve()
  assets: dict[str, bytes] = {}
  _update_assets(assets, mesh_root / "visual", "../meshes/visual")
  _update_assets(assets, mesh_root / "collision", "../meshes/collision")
  spec.assets = assets
  return spec


def get_luwu_hand_cfg() -> EntityCfg:
  """Return a fresh fixed-base Luwu hand entity configuration."""
  return EntityCfg(
    init_state=LUWU_HAND_HOME,
    spec_fn=partial(_get_spec),
    articulation=LUWU_HAND_ARTICULATION,
  )
