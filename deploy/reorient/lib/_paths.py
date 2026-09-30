# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared path helpers for deploy/reorient scripts and tests.

Resolves the wuji-mjlab logs/ directory at the repo root. Walks up from
the current file to find the first ancestor that contains a `logs/`
directory. Override via WUJI_LOGS_ROOT env var.
"""
from __future__ import annotations

import os
from pathlib import Path


def find_logs_root() -> Path:
    """Locate wuji-mjlab logs/ directory.

    Resolution order:
      1. ``WUJI_LOGS_ROOT`` env var, if set (returned verbatim, no existence
         check — the caller may want to construct a path before the dir is
         created)
      2. First ancestor of this file with a child ``logs/`` directory

    Raises:
        FileNotFoundError: if ``WUJI_LOGS_ROOT`` is unset and no ancestor of
            this module file contains ``logs/``. Loud failure is intentional:
            the old magic-number fallback (``here.parents[5] / "logs"``)
            silently returned a nonexistent path and hid the real
            configuration bug.
    """
    env = os.environ.get("WUJI_LOGS_ROOT")
    if env:
        return Path(env)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "logs"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate logs/ ancestor of {here}. "
        "Set WUJI_LOGS_ROOT to override."
    )
