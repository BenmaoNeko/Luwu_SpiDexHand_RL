#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Export a trained reorient policy checkpoint to ONNX format.

Thin CLI wrapper. The importable core lives at
``wuji_mjlab.tasks.reorient.tooling.onnx_export_core`` and is unit-tested
without spawning a sim.

Usage:
    python -m wuji_mjlab.tasks.reorient.scripts.export_onnx path/to/model_10000.pt
    python -m wuji_mjlab.tasks.reorient.scripts.export_onnx path/to/model_10000.pt --filename policy_deploy.onnx
"""

from __future__ import annotations

from wuji_mjlab.tasks.reorient.tooling.onnx_export_core import main

if __name__ == "__main__":
  main()
