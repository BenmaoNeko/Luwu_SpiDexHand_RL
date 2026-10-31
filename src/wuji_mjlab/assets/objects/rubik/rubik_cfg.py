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
_RUBIK_OUTER_FRICTION = (0.4, 0.005, 0.001)
_RUBIK_INTERNAL_FRICTION = (0.08, 0.08, 0.001, 0.0002, 0.0002)

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


def _get_spec(
  mass_kg: float,
  visual_only: bool = False,
  full_cube: bool = False,
  dynamic_layers: bool = False,
) -> mujoco.MjSpec:
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

  if dynamic_layers:
    if not full_cube or visual_only:
      raise ValueError("Dynamic layers require a physical full cube")
    pieces = [
      j.name.split(":", 2)[2] for j in spec.joints if j.name.startswith("cubelet:rotx:")
    ]
    drivers = [j.name for j in spec.joints if j.name.startswith("cubelet:driver:")]
    for joint in spec.joints:
      if joint.name.startswith(("cubelet:driver:", "cubelet:rot")):
        _add_joint_equality(spec, f"dynamic_lock_{joint.name}", joint.name)
        spec.equality(f"dynamic_lock_{joint.name}").active = False
    for driver in drivers:
      face = driver.split(":", 2)[2]
      for piece in pieces:
        for axis in "xyz":
          name = f"dynamic_follow_{face}:{axis}:{piece}"
          _add_joint_equality(spec, name, f"cubelet:rot{axis}:{piece}", driver)
          spec.equality(name).active = False
    cubelet_geoms = sorted(
      geom.name for geom in spec.geoms if geom.name.startswith("cubelet:")
    )
    for first in range(len(cubelet_geoms)):
      for second in range(first + 1, len(cubelet_geoms)):
        spec.add_pair(
          name=f"internal_friction_{first}_{second}",
          geomname1=cubelet_geoms[first],
          geomname2=cubelet_geoms[second],
          condim=6,
          friction=_RUBIK_INTERNAL_FRICTION,
        )

  if not full_cube:
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
    if dynamic_layers and geom.name.startswith("cubelet:"):
      geom.friction = _RUBIK_OUTER_FRICTION
    if visual_only:
      geom.contype = 0
      geom.conaffinity = 0

  if dynamic_layers:
    for joint in spec.joints:
      if joint.name.startswith(("cubelet:driver:", "cubelet:rot")):
        joint.frictionloss *= 0.6

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


def get_rubik_cfg(
  *,
  mass_kg: float = RUBIK_MASS_KG,
  full_cube: bool = False,
  dynamic_layers: bool = False,
) -> EntityCfg:
  return EntityCfg(
    init_state=RUBIK_INIT_STATE,
    spec_fn=partial(
      _get_spec,
      mass_kg,
      full_cube=full_cube,
      dynamic_layers=dynamic_layers,
    ),
  )


def get_rubik_target_cfg(*, full_cube: bool = False) -> EntityCfg:
  return EntityCfg(
    init_state=RUBIK_TARGET_INIT_STATE,
    spec_fn=partial(_get_spec, RUBIK_MASS_KG, True, full_cube),
  )
