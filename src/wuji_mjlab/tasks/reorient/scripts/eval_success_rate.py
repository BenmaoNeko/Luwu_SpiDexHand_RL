#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Automated success rate evaluation for the reorient policy in MuJoCo (mjlab).

Thin CLI wrapper. The importable core lives at
``wuji_mjlab.tasks.reorient.tooling.eval_core``.

Usage:
    pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <onnx_path>
    pixi run python -m wuji_mjlab.tasks.reorient.scripts.eval_success_rate <onnx_path> --num-trials 50
"""

from __future__ import annotations

import os

# Select a windowed GL backend before any mujoco GL state is initialised.
os.environ.setdefault("MUJOCO_GL", "glfw")

from wuji_mjlab.tasks.reorient.tooling.eval_core import main  # noqa: E402

if __name__ == "__main__":
  main()
