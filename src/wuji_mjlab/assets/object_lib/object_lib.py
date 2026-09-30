# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Single-body multi-variant object library for per-world-mesh mjlab scenes.

``ObjectLib`` owns the discovered ``ObjectAsset`` set, builds per-variant
``VariantEntityCfg``, and serves runtime queries (per-env variant assignment,
per-env half-extents, startup logging). A class-level registry keyed by the
scene entity name lets downstream MDP helpers locate the right ``ObjectLib``
without stashing state on the env itself.

Each variant is a single floating-base body called ``object`` with:
  * one visual mesh geom (``object_visual``, contype=0/conaffinity=0,
    mass=0). The body's explicit inertial properties are computed from the
    watertight ``high`` mesh so concave/empty objects do not get MuJoCo's
    convexified mesh mass/inertia.
  * N *collision* mesh geoms named ``object_col_<idx>`` (contype=1/conaffinity=1,
    mass=0) — one per V-HACD convex hull.

All variants share the same body / joint structure so mjlab's per-world-mesh
path can pad the geom slots and assign one variant per simulation world.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import trimesh

from .discovery import ObjectAsset, discover_objects

if TYPE_CHECKING:
  from mjlab.entity import EntityCfg, VariantEntityCfg

OBJECT_COLLISION_GEOM_PATTERN = r"object_col_\d+"

_COLLISION_FRICTION = (1.0, 0.02, 0.002)
_COLLISION_SOLREF = (0.015, 1.0)
_COLLISION_SOLIMP = (0.85, 0.85, 0.001, 0.5, 2.0)
_VISUAL_GROUP = 2
_COLLISION_GROUP = 3
_VISUAL_RGBA = (0.85, 0.6, 0.2, 1.0)

# Tiny tetrahedron (1 mm edges) used as the mesh for padding-slot collision
# geoms. Embedded at the body origin, so it cannot reach any external surface
# (body origin = bbox centre, filtered to ``min_axis >= 3 cm``); the only
# thing it contributes is a valid mesh reference that lets mjlab's merged
# template carry ``contype=1`` on every slot. Smaller (sub-mm) tets fail
# MuJoCo's "mesh volume too small" check at compile time.
_PAD_MESH_NAME = "col_pad"
_PAD_VERT = np.array(
  [[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]], dtype=np.float32
).ravel()
_PAD_FACE = np.array(
  [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
).ravel()


@dataclass(frozen=True)
class ObjectLibConfig:
  """Knobs for selecting and parametrising the variant set."""

  roots: tuple[Path, ...]
  """Per-dataset directories to scan. Supplied by the caller (no default — the
  grasp-object dataset is .gitignored, so its location must come from the
  train/play CLI, not from import-time magic)."""
  bbox_min: float | None = 0.04
  bbox_max: float | None = 0.22
  min_axis: float | None = 0.015
  """Smallest bbox axis floor (metres). ``None`` disables the gate. The default
  admits pens / markers / thin lids while still rejecting paper-thin objects."""
  max_objects: int | None = None
  """Cap the number of variants (after filtering and shuffling). ``None`` keeps all."""
  shuffle_seed: int | None = 0
  """Deterministically shuffle objects before truncation; ``None`` preserves order."""
  density: float = 500.0
  """Default density (kg / m^3) used to derive each object's nominal mass."""
  mass_range: tuple[float, float] | None = (0.005, 3.0)
  """Accepted nominal-mass window (kg), computed as ``volume * density``.
  Filters out featherweights that would fly off on contact and bricks the
  wrist can't physically lift. ``None`` disables."""


def _load_mesh(path: Path) -> trimesh.Trimesh:
  tm = trimesh.load(str(path), force="mesh")
  assert isinstance(tm, trimesh.Trimesh)
  return tm


def _to_user_arrays(
  tm: trimesh.Trimesh,
  offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Return ``(uservert, userface)`` flat arrays shifted by ``-offset``.

  mjlab's ``Entity._build_merged_spec`` merges variant specs via
  ``copy_mesh_data``, which does NOT merge each variant's ``spec.assets`` dict.
  Every dataset is expected to use identical basenames
  (``mesh_high.stl``, ``hull_NN.stl``), so a file-based path would collapse all
  variants onto variant 0's bytes. Embedding vertices / faces directly avoids
  the collision and also makes ``spec.compile()`` refresh ``geom_aabb`` /
  ``body_mass`` / ``body_inertia`` when meshname is rebound per world.
  """
  verts = (np.asarray(tm.vertices, dtype=np.float32) - offset).ravel()
  faces = np.asarray(tm.faces, dtype=np.int32).ravel()
  return verts, faces


def _quat_from_mat(mat: np.ndarray) -> np.ndarray:
  quat = np.zeros(4, dtype=np.float64)
  mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
  if quat[0] < 0:
    quat *= -1.0
  return quat


def _mesh_inertial(
  tm: trimesh.Trimesh,
  offset: np.ndarray,
  density: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """Return explicit inertial fields from the true watertight mesh volume."""
  mesh = tm.copy()
  if float(mesh.volume) < 0.0:
    mesh.invert()

  volume = float(mesh.volume)
  if not np.isfinite(volume) or volume <= 0.0:
    raise ValueError("Cannot compute object inertial from a non-positive mesh volume.")

  mass = float(volume * density)
  center_mass = np.asarray(mesh.center_mass, dtype=np.float64) - offset.astype(
    np.float64
  )
  inertia_tensor = np.asarray(mesh.moment_inertia, dtype=np.float64) * density
  inertia_tensor = 0.5 * (inertia_tensor + inertia_tensor.T)

  principal, axes = np.linalg.eigh(inertia_tensor)
  order = np.argsort(principal)
  principal = principal[order]
  axes = axes[:, order]
  if np.linalg.det(axes) < 0.0:
    axes[:, 0] *= -1.0

  principal = np.maximum(principal, np.finfo(np.float64).eps)
  iquat = _quat_from_mat(axes)
  return mass, center_mass, principal, iquat


def _spec_factory(asset: ObjectAsset, density: float, max_hulls: int):
  """Return a ``spec_fn`` that materialises the MjSpec lazily.

  Meshes are re-centered on the visual AABB so body origin = bbox centre.
  Every variant emits exactly ``max_hulls`` collision geoms, padding short
  variants with a 1 mm tetrahedron (see module docstring).
  """
  visual_path = asset.visual_mesh
  hull_paths = asset.collision_files
  n_real = len(hull_paths)
  if not 1 <= n_real <= max_hulls:
    raise ValueError(
      f"Asset '{asset.variant_name}' has {n_real} hulls, expected 1..{max_hulls}."
    )

  def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()

    tm_visual = _load_mesh(visual_path)
    lo, hi = tm_visual.bounds
    offset = ((lo + hi) * 0.5).astype(np.float32)
    mass, ipos, inertia, iquat = _mesh_inertial(tm_visual, offset, density)

    visual_mesh_name = "visual"
    spec_visual = spec.add_mesh()
    spec_visual.name = visual_mesh_name
    verts, faces = _to_user_arrays(tm_visual, offset)
    spec_visual.uservert = verts
    spec_visual.userface = faces

    hull_mesh_names: list[str] = []
    for idx, hull_path in enumerate(hull_paths):
      mesh_name = f"col_{idx:02d}"
      hull_mesh_names.append(mesh_name)
      hm = spec.add_mesh()
      hm.name = mesh_name
      hverts, hfaces = _to_user_arrays(_load_mesh(hull_path), offset)
      hm.uservert = hverts
      hm.userface = hfaces

    if n_real < max_hulls:
      pad = spec.add_mesh()
      pad.name = _PAD_MESH_NAME
      pad.uservert = _PAD_VERT
      pad.userface = _PAD_FACE

    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_freejoint")
    body.explicitinertial = 1
    body.mass = mass
    body.ipos = ipos
    body.inertia = inertia
    body.iquat = iquat

    # Keep all geom mass at zero. The body-level explicit inertial above is
    # propagated per variant by mjlab's per-world mesh metadata.
    body.add_geom(
      name="object_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=visual_mesh_name,
      contype=0,
      conaffinity=0,
      group=_VISUAL_GROUP,
      rgba=_VISUAL_RGBA,
      mass=0.0,
    )
    for idx in range(max_hulls):
      is_padding = idx >= n_real
      body.add_geom(
        name=f"object_col_{idx:02d}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=_PAD_MESH_NAME if is_padding else hull_mesh_names[idx],
        contype=1,
        conaffinity=1,
        group=_COLLISION_GROUP,
        condim=3,
        friction=_COLLISION_FRICTION,
        solref=_COLLISION_SOLREF,
        solimp=_COLLISION_SOLIMP,
        mass=0.0,
        rgba=(0.0, 0.0, 0.0, 0.0),
      )

    return spec

  return build_spec


class ObjectLib:
  """Library of graspable object variants bound to one scene entity name.

  Typical lifecycle::

      lib = ObjectLib(ObjectLibConfig(max_objects=32, density=500.0))
      object_cfg = lib.build_variant_cfg(entity_name="object", init_state=...)
      # ... build the rest of the env_cfg ...
      cfg.events["log_object_lib_assignment"] = EventTermCfg(
          mode="startup", func=lib.log_assignment, params={"limit": 16},
      )

  Downstream MDP helpers retrieve the library via
  ``ObjectLib.for_entity("object")``. The class registers itself on
  ``build_variant_cfg``; callers cache any per-env derived tensors themselves.
  """

  _by_entity: dict[str, "ObjectLib"] = {}

  def __init__(self, cfg: ObjectLibConfig):
    assets = discover_objects(cfg.roots)
    if cfg.mass_range is not None:
      mass_lo, mass_hi = cfg.mass_range
      assets = [a for a in assets if mass_lo <= a.volume * cfg.density <= mass_hi]
    if cfg.shuffle_seed is not None:
      random.Random(cfg.shuffle_seed).shuffle(assets)
    if cfg.max_objects is not None:
      assets = assets[: cfg.max_objects]
    if len(assets) < 2:
      raise RuntimeError(
        f"Found only {len(assets)} objects with watertight high-mesh + "
        f"V-HACD collision manifests under {[str(r) for r in cfg.roots]}. "
        "Each root must contain assets/<obj>/mesh_high.stl plus "
        "assets/<obj>/collision/manifest.json."
      )

    self.cfg = cfg
    self.assets: list[ObjectAsset] = assets
    self._by_name: dict[str, ObjectAsset] = {a.variant_name: a for a in assets}
    self._entity_name: str | None = None

  @classmethod
  def for_entity(cls, entity_name: str) -> "ObjectLib":
    """Return the ``ObjectLib`` bound to ``entity_name`` via ``build_variant_cfg``."""
    try:
      return cls._by_entity[entity_name]
    except KeyError:
      raise KeyError(
        f"No ObjectLib registered for entity '{entity_name}'. "
        f"Did you call build_variant_cfg(entity_name=...)?"
      ) from None

  def build_variant_cfg(
    self,
    *,
    entity_name: str = "object",
    init_state: "EntityCfg.InitialStateCfg | None" = None,
  ) -> "VariantEntityCfg":
    """Build a ``VariantEntityCfg`` and register ``self`` under ``entity_name``."""
    # Lazy: mjlab.entity import triggers task entry-point discovery that walks
    # wuji_mjlab.tasks and re-imports this module. Deferring avoids the cycle.
    from mjlab.entity import EntityCfg, VariantCfg, VariantEntityCfg

    max_hulls = max(len(a.collision_files) for a in self.assets)
    variants = {
      a.variant_name: VariantCfg(
        spec_fn=_spec_factory(a, self.cfg.density, max_hulls),
        weight=1.0,
      )
      for a in self.assets
    }
    init_state = init_state or EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.10),
      rot=(1.0, 0.0, 0.0, 0.0),
    )
    self._entity_name = entity_name
    ObjectLib._by_entity[entity_name] = self
    return VariantEntityCfg(init_state=init_state, variants=variants)

  def variant_assignment(self, env) -> list[tuple[int, str]]:
    """Return ``[(env_id, variant_name), ...]`` via the deterministic
    largest-remainder allocation used by the per-world-mesh path."""
    if self._entity_name is None:
      raise RuntimeError("ObjectLib has no bound entity; call build_variant_cfg first.")
    from mjlab.scene.per_world_mesh import allocate_worlds

    variant_info = env.scene.collect_variant_info()
    prefix = f"{self._entity_name}/"
    metadata = next((m for p, m in variant_info if p == prefix), None)
    if metadata is None:
      raise RuntimeError(
        f"Entity '{self._entity_name}' has no variant metadata; "
        f"is it a VariantEntityCfg?"
      )
    assignment = allocate_worlds(metadata.variant_weights, env.num_envs)
    return [(w, metadata.variant_names[v]) for w, v in enumerate(assignment)]

  def per_world_half_extents(
    self,
    env,
    *,
    device: torch.device | str | None = None,
  ) -> torch.Tensor:
    """Return ``(num_envs, 3)`` half-extents (m) for the bound entity.

    Reads from the authoritative asset side-table (manifest bbox_size / 2),
    not from ``sim.model.geom_aabb``. The latter is per-world-static after
    compile, so per-env DR changes to geom_size do not propagate to it.
    """
    rows = self.variant_assignment(env)
    bbox_arr = [self._by_name[name].bbox_size for _, name in rows]
    return (
      torch.tensor(bbox_arr, device=device or env.device, dtype=torch.float32) * 0.5
    )

  def log_assignment(self, env, env_ids=None, *, limit: int = 32) -> None:
    """Print the per-env variant assignment. Usable as an ``EventTermCfg``
    startup hook (``env_ids`` is ignored)."""
    del env_ids
    name = self._entity_name or "object"
    rows = self.variant_assignment(env)
    print(
      f"[object_lib] '{name}' per-env variant assignment "
      f"({len(rows)} envs; showing first {min(limit, len(rows))}):"
    )
    for w, vn in rows[:limit]:
      asset = self._by_name[vn]
      bbox = [round(x, 3) for x in asset.bbox_size]
      mass = round(self.cfg.density * asset.volume, 3)
      print(f"  env {w:>3}: variant={vn:<48s}  bbox(m)={bbox}  mass~={mass}kg")
    if len(rows) > limit:
      print(f"  ... +{len(rows) - limit} more")
    if env.num_envs > 1:
      print("[object_lib] viewer tip: ',' / '.' step env_idx to focus a world.")
