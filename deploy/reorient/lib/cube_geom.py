# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared helpers for cube geometry: cube_tags JSON resolution + runtime scaling.

Used by both the vision pipeline (cube_world_observer.py — needs cube/tag sizes
for AprilTag/ArUco PnP) and the sim/benchmark pipeline (play_real.py — needs to
patch the MuJoCo cube body so visualization and physics match the real cube).
"""
from __future__ import annotations

import json
import os
from typing import Any

import mujoco

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CUBE_CONFIG_FILE = os.path.join(ROOT_DIR, "config", "cube_tags.json")


def resolve_cube_config_path(arg: str | None) -> str:
    """Resolve a --cube CLI argument to an absolute cube_tags JSON path.

    Accepts:
      - None / "" / "default"  -> ``config/cube_tags.json`` (the 54mm baseline).
      - An existing path (absolute or relative to cwd) -> used as-is.
      - A short size suffix like ``"36"`` / ``"40_5"`` -> ``config/cube_tags<suffix>.json``.

    Raises FileNotFoundError if the resolved path does not exist.
    """
    if not arg or arg == "default":
        return DEFAULT_CUBE_CONFIG_FILE
    if os.path.exists(arg):
        return os.path.abspath(arg)
    candidate = os.path.join(ROOT_DIR, "config", f"cube_tags{arg}.json")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"--cube={arg!r}: not a path nor a known size suffix. Tried {candidate!r}."
    )


def load_cube_config(path: str) -> dict[str, Any]:
    """Load a cube_tags JSON file."""
    with open(path) as f:
        return json.load(f)


def scale_cube_in_model(
    model,
    target_size_m: float,
    *,
    geom_name: str = "cube",
    mesh_name: str = "cube_mesh",
) -> tuple[float, float] | None:
    """Rescale the cube body's geometry to ``target_size_m`` (full edge length).

    Patches an already-compiled MuJoCo model in place:
      - Box collision geom's half-size  -> ``target_size_m / 2``
      - Visual mesh vertices            -> multiplied by the matching factor
      - Body inertia                    -> recomputed for the new box (mass kept)

    Mass is preserved (whatever the XML declared). Returns
    ``(old_half_size, new_half_size)`` on success, or ``None`` if the cube
    geom isn't present in the model (e.g. a cube-less scene).
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return None

    cur_half = float(model.geom_size[geom_id, 0])
    new_half = target_size_m / 2.0
    factor = new_half / cur_half
    if abs(factor - 1.0) < 1e-6:
        return cur_half, new_half  # already at target; nothing to do

    # Collision box
    model.geom_size[geom_id, :3] = new_half

    # Visual mesh vertices (only if there's a mesh — robust to mesh-less scenes)
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    if mesh_id >= 0:
        v0 = int(model.mesh_vertadr[mesh_id])
        n = int(model.mesh_vertnum[mesh_id])
        model.mesh_vert[v0:v0 + n] *= factor

    # Body inertia for a uniform-density box of mass m and half-size h:
    #   Ixx = Iyy = Izz = 2 * m * h^2 / 3
    body_id = int(model.geom_bodyid[geom_id])
    mass = float(model.body_mass[body_id])
    iaxis = mass * 2.0 * (new_half ** 2) / 3.0
    model.body_inertia[body_id] = [iaxis, iaxis, iaxis]

    return cur_half, new_half
