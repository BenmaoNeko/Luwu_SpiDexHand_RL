#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Deploy hardware/runtime configuration loader.

Backs ``deploy/reorient/config/control.yaml``. Only deploy-side hardware and
runtime defaults live there — model-intrinsic params (action_scale,
ema_alpha, warmup_time_s, ctrl_dt, history_len, control_mode) come from the
ONNX sidecar JSON (``onnx/<stem>.json``), are read by
``ONNXPolicy.config``, and are applied via
``make_real_hand_env_cfg(policy_config=...)`` overrides.

Public API:
    load_yaml(file_path)
    get_control_config()
    goal_port() / cube_port()
    effort_limit_nm() / lowpass_cutoff_hz()

Example:
    >>> from deploy.reorient.lib.config_loader import goal_port, effort_limit_nm
    >>> goal_port()
    5556
    >>> effort_limit_nm()
    0.5
"""

from __future__ import annotations

import os
from typing import Any

import yaml

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
"""Directory containing this script."""

ROOT_DIR: str = os.path.dirname(SCRIPT_DIR)
"""Root directory of deploy/reorient."""

CONFIG_DIR: str = os.path.join(ROOT_DIR, "config")
"""Directory containing configuration files."""

CONTROL_CONFIG_FILE: str = os.path.join(CONFIG_DIR, "control.yaml")
"""Path to control.yaml."""

# Cached config (loaded once per process)
_ctrl_cfg: dict[str, Any] | None = None


def load_yaml(file_path: str) -> dict[str, Any]:
    """Load a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config not found: {file_path}")
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_control_config() -> dict[str, Any]:
    """Load and return ``control.yaml`` as a dict."""
    return load_yaml(CONTROL_CONFIG_FILE)


def _ensure_loaded() -> dict[str, Any]:
    global _ctrl_cfg
    if _ctrl_cfg is None:
        _ctrl_cfg = get_control_config()
    return _ctrl_cfg


def goal_port() -> int:
    """ZMQ port for goal publishing (play_real)."""
    try:
        return int(_ensure_loaded()["zmq"]["goal_port"])
    except KeyError as e:
        raise KeyError(
            f"Missing key {e} in deploy/reorient/config/control.yaml; "
            f"check that the yaml has the expected structure (zmq.goal_port). "
            f"If you copied an older control.yaml, sync from the repo's current version."
        ) from e


def cube_port() -> int:
    """ZMQ port for cube observation (cube_world_observer)."""
    try:
        return int(_ensure_loaded()["zmq"]["cube_port"])
    except KeyError as e:
        raise KeyError(
            f"Missing key {e} in deploy/reorient/config/control.yaml; "
            f"check that the yaml has the expected structure (zmq.cube_port). "
            f"If you copied an older control.yaml, sync from the repo's current version."
        ) from e


def effort_limit_nm() -> float:
    """Per-joint torque cap (Nm) for WujiHandDriver."""
    try:
        return float(_ensure_loaded()["hardware"]["effort_limit_nm"])
    except KeyError as e:
        raise KeyError(
            f"Missing key {e} in deploy/reorient/config/control.yaml; "
            f"check that the yaml has the expected structure (hardware.effort_limit_nm). "
            f"If you copied an older control.yaml, sync from the repo's current version."
        ) from e


def lowpass_cutoff_hz() -> float:
    """SDK LowPass filter cutoff frequency (Hz)."""
    try:
        return float(_ensure_loaded()["hardware"]["lowpass_cutoff_hz"])
    except KeyError as e:
        raise KeyError(
            f"Missing key {e} in deploy/reorient/config/control.yaml; "
            f"check that the yaml has the expected structure (hardware.lowpass_cutoff_hz). "
            f"If you copied an older control.yaml, sync from the repo's current version."
        ) from e


if __name__ == "__main__":
    print("=" * 50)
    print("Config Loader smoke check")
    print("=" * 50)

    ctrl = get_control_config()
    print(f"\ncontrol.yaml ({CONTROL_CONFIG_FILE}):")
    print(f"  hardware: {ctrl.get('hardware')}")
    print(f"  zmq:      {ctrl.get('zmq')}")

    print("\nQuick accessors:")
    print(f"  goal_port():         {goal_port()}")
    print(f"  cube_port():         {cube_port()}")
    print(f"  effort_limit_nm():   {effort_limit_nm()}")
    print(f"  lowpass_cutoff_hz(): {lowpass_cutoff_hz()}")
