# SPDX-License-Identifier: MIT
"""Robogym Rubik's Cube model configured for the first U-face task."""

from functools import partial
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent
_XML_PATH = _ASSET_DIR / "xmls" / "rubik_perpendicular.xml"
_U_FACE_JOINT = "cubelet:driver:pos_z"
_ROUNDCUBE_SOURCE_MASS_KG = 0.0921676

RUBIK_MASS_KG = 0.080
RUBIK_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(-0.0264848718, 8.59749199e-08, 0.40649396),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)
RUBIK_TARGET_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0635151282, 8.59749199e-08, 0.40649396),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


def _get_spec(mass_kg: float, visual_only: bool = False) -> mujoco.MjSpec:
  if mass_kg <= 0.0:
    raise ValueError(f"mass_kg must be positive, got {mass_kg}")

  spec = mujoco.MjSpec.from_file(str(_XML_PATH))
  assets = {"roundcube.msh": (_ASSET_DIR / "meshes" / "roundcube.msh").read_bytes()}
  assets.update(
    {
      f"cube/{path.name}": path.read_bytes()
      for path in (_ASSET_DIR / "textures" / "cube").glob("*.png")
    }
  )
  spec.assets = assets

  # Replace Robogym's four-joint root with the freejoint expected by MJLab.
  middle = spec.body("middle")
  for joint in list(middle.joints):
    spec.delete(joint)
  middle.add_freejoint(name="cube:free")

  # Robogym removes these internal spring sliders in its full-cube environment.
  for joint in list(spec.joints):
    if joint.name.startswith("cubelet:spring:"):
      spec.delete(joint)

  # This decorative internal sphere is not part of the hand-contact surface.
  center = spec.geom("center_sphere")
  center.contype = 0
  center.conaffinity = 0

  for joint in list(spec.joints):
    name = joint.name
    if name.startswith("cubelet:driver:"):
      if name != _U_FACE_JOINT:
        _add_joint_equality(spec, f"lock_{name}", name)
    elif name.startswith("cubelet:rot"):
      if name.startswith("cubelet:rotz:") and _in_initial_u_layer(name):
        _add_joint_equality(spec, f"follow_u_{name}", name, _U_FACE_JOINT)
      else:
        _add_joint_equality(spec, f"lock_{name}", name)

  # Scale density uniformly to the measured 80 g cube while keeping geometry
  # and inertia ratios intact.
  mass_scale = float(mass_kg) / _ROUNDCUBE_SOURCE_MASS_KG
  for geom in spec.geoms:
    if geom.density > 0.0:
      geom.density *= mass_scale
    if visual_only:
      geom.contype = 0
      geom.conaffinity = 0

  if visual_only:
    for body in spec.bodies:
      if body.name != "world":
        body.gravcomp = 1.0

  return spec


def _in_initial_u_layer(joint_name: str) -> bool:
  piece = joint_name.rsplit(":", 1)[-1]
  return piece == "pos_z" or piece.endswith("_pos_z")


def _add_joint_equality(
  spec: mujoco.MjSpec,
  name: str,
  joint1: str,
  joint2: str | None = None,
) -> None:
  equality = spec.add_equality(
    name=name,
    type=mujoco.mjtEq.mjEQ_JOINT,
    name1=joint1,
    name2=joint2,
    solref=[0.002, 1.0],
    solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
  )
  data = list(equality.data)
  data[:5] = [0.0, 1.0 if joint2 else 0.0, 0.0, 0.0, 0.0]
  equality.data = data


def get_rubik_cfg(*, mass_kg: float = RUBIK_MASS_KG) -> EntityCfg:
  return EntityCfg(
    init_state=RUBIK_INIT_STATE,
    spec_fn=partial(_get_spec, mass_kg),
  )


def get_rubik_target_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=RUBIK_TARGET_INIT_STATE,
    spec_fn=partial(_get_spec, RUBIK_MASS_KG, True),
  )
