# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Common in-hand object config helpers."""

from functools import partial
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent

INHAND_OBJECT_XML: Path = _ASSET_DIR / "xmls" / "cube.xml"
assert INHAND_OBJECT_XML.exists(), f"Missing in-hand object XML: {INHAND_OBJECT_XML}"

# Baseline cube — reflects values inside cube.xml. If cube.xml changes, update here.
BASELINE_EDGE_M = 0.054
BASELINE_MASS_KG = 0.120
CUBE_DENSITY = BASELINE_MASS_KG / (BASELINE_EDGE_M**3)  # ≈ 683 kg/m^3

INHAND_OBJECT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(-0.1404, 0.0042, 0.52),
  rot=(0.3894, -0.3796, -0.5980, -0.5888),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
)


def _get_spec(
  edge_m: float | None = None,
  mass_kg: float | None = None,
) -> mujoco.MjSpec:
  """Load the cube spec; optionally scale to a different edge length.

  When ``edge_m`` is None, return the spec exactly as the XML defines it
  (54 mm baseline). When set, mutate the named ``cube`` geom's box size +
  mass and the ``cube_mesh`` mesh scale so that the cube becomes a uniform
  same-density solid with the requested edge length. ``mass_kg`` overrides
  only the cube's mass and can be used without changing its geometry.
  """
  if mass_kg is not None and mass_kg <= 0.0:
    raise ValueError(f"mass_kg must be positive, got {mass_kg}")
  spec = mujoco.MjSpec.from_file(str(INHAND_OBJECT_XML))
  if edge_m is None and mass_kg is None:
    return spec

  cube_geom = next((g for g in spec.geoms if g.name == "cube"), None)
  if cube_geom is None:
    raise ValueError(f"{INHAND_OBJECT_XML} must contain a geom named 'cube'")

  if edge_m is not None:
    if edge_m <= 0.0:
      raise ValueError(f"edge_m must be positive, got {edge_m}")
    half = edge_m / 2.0
    cube_geom.size = [half, half, half]
    if mass_kg is None:
      cube_geom.mass = CUBE_DENSITY * (edge_m**3)

    cube_mesh = next((m for m in spec.meshes if m.name == "cube_mesh"), None)
    if cube_mesh is None:
      raise ValueError(f"{INHAND_OBJECT_XML} must contain a mesh named 'cube_mesh'")
    cube_mesh.scale = [half, half, half]

  if mass_kg is not None:
    cube_geom.mass = mass_kg

  return spec


def get_inhand_object_cfg(
  edge_m: float | None = None,
  *,
  mass_kg: float | None = None,
) -> EntityCfg:
  """Create the in-hand cube entity, optionally overriding its mass."""
  return EntityCfg(
    init_state=INHAND_OBJECT_INIT_STATE,
    spec_fn=partial(_get_spec, edge_m, mass_kg),
  )
