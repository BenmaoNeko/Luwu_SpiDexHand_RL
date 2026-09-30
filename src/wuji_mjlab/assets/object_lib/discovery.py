# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Walk the dataset roots to find decomposed objects with collision manifests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DATASET_NAMES: tuple[str, ...] = (
  "dexycbassets",
  "hocapassets",
  "hot3dassets",
  "oakink2assets",
)
PACKAGE_DATASETS_DIR = (
  Path(__file__).resolve().parents[1] / "datasets" / "grasp_objects"
)


def resolve_dataset_roots(base: Path | str) -> tuple[Path, ...]:
  """Expand a dataset base directory into per-dataset roots."""
  base_path = Path(base).expanduser()
  return tuple(base_path / name for name in DATASET_NAMES)


@dataclass(frozen=True)
class ObjectAsset:
  """One graspable object discovered on disk."""

  dataset: str
  object_id: str
  visual_mesh: Path
  collision_files: tuple[Path, ...]
  bbox_size: tuple[float, float, float]
  volume: float

  @property
  def variant_name(self) -> str:
    return f"{self.dataset}__{self.object_id}"


def _physics_valid(manifest: dict) -> bool:
  """Reject manifests whose physical fields are degenerate.

  * ``watertight`` is required because ``_mesh_inertial`` uses
    ``trimesh.volume`` / ``moment_inertia``, both undefined for open surfaces.
  * bbox components and volume must be finite and positive — any NaN/Inf/zero
    makes mass, inertia, or keypoint scaling blow up downstream.
  * At least one collision hull must exist (no naked visual-only objects).
  """
  inp = manifest.get("input") or {}
  if not inp.get("watertight", False):
    return False
  bbox = inp.get("bbox_size") or []
  if len(bbox) != 3 or not all(math.isfinite(v) and v > 0.0 for v in bbox):
    return False
  volume = inp.get("volume")
  if not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume <= 0.0:
    return False
  hulls = manifest.get("hulls") or []
  if not hulls:
    return False
  return True


def discover_objects(roots: tuple[Path, ...]) -> list[ObjectAsset]:
  """Discover objects whose ``high`` mesh is watertight and decomposed.

  Each root must contain ``assets/<obj>/mesh_high.stl`` plus
  ``assets/<obj>/collision/manifest.json``.

  Physics gates (see :func:`_physics_valid`):
    * Mesh is watertight (inertia tensor needs this).
    * ``bbox_size`` / ``volume`` are finite and positive.
    * At least one convex collision hull.
  """
  found: list[ObjectAsset] = []
  for root in roots:
    if not root.exists():
      continue
    assets_root = root / "assets" if (root / "assets").is_dir() else root
    for obj_dir in sorted(assets_root.iterdir()):
      manifest_path = obj_dir / "collision" / "manifest.json"
      visual = obj_dir / "mesh_high.stl"
      if not (obj_dir.is_dir() and manifest_path.exists() and visual.exists()):
        continue
      try:
        manifest = json.loads(manifest_path.read_text())
      except json.JSONDecodeError:
        continue

      if not _physics_valid(manifest):
        continue

      bbox = manifest["input"]["bbox_size"]
      collision_dir = obj_dir / "collision"
      hull_files = tuple(sorted(collision_dir / h["file"] for h in manifest["hulls"]))

      found.append(
        ObjectAsset(
          dataset=root.name,
          object_id=obj_dir.name,
          visual_mesh=visual,
          collision_files=hull_files,
          bbox_size=tuple(bbox),
          volume=float(manifest["input"]["volume"]),
        )
      )
  return found
