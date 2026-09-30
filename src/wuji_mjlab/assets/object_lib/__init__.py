# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Multi-variant graspable object library for mjlab per-world-mesh scenes."""

from .discovery import (
  DATASET_NAMES,
  PACKAGE_DATASETS_DIR,
  ObjectAsset,
  discover_objects,
  resolve_dataset_roots,
)
from .object_lib import OBJECT_COLLISION_GEOM_PATTERN, ObjectLib, ObjectLibConfig

__all__ = [
  "DATASET_NAMES",
  "OBJECT_COLLISION_GEOM_PATTERN",
  "ObjectAsset",
  "ObjectLib",
  "ObjectLibConfig",
  "PACKAGE_DATASETS_DIR",
  "discover_objects",
  "resolve_dataset_roots",
]
