# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Linear interpolator for smooth motor command output.

Converts low-frequency policy targets (e.g. 20Hz) to higher-frequency
SDK commands (e.g. 100Hz) via linear interpolation between consecutive targets.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class ActionInterpolator:
    """Linearly interpolate between consecutive motor targets.

    Each call to set_target() registers a new target. steps() then yields
    n_substeps evenly spaced values from the previous target to the new one.

    On the first call (no previous target), steps() yields the new target
    repeated n_substeps times (no interpolation).

    Example:
        >>> interp = ActionInterpolator(n_substeps=5)
        >>> interp.set_target(np.array([0.0, 0.0]))
        >>> interp.set_target(np.array([1.0, 2.0]))
        >>> list(interp.steps())  # 5 linearly spaced values from [0,0] to [1,2]
        [array([0.2, 0.4]), array([0.4, 0.8]), ..., array([1.0, 2.0])]
    """

    def __init__(self, n_substeps: int = 5) -> None:
        self._n_substeps = n_substeps
        self._prev_target: np.ndarray | None = None
        self._new_target: np.ndarray | None = None

    def set_target(self, new_target: np.ndarray) -> None:
        """Register a new motor target from the policy."""
        self._prev_target = self._new_target
        self._new_target = new_target.copy()

    def steps(self) -> Iterator[np.ndarray]:
        """Yield n_substeps linearly interpolated targets.

        The last yielded value is always exactly the new target.
        """
        assert self._new_target is not None, "call set_target() before steps()"
        if self._prev_target is None:
            for _ in range(self._n_substeps):
                yield self._new_target.copy()
        else:
            for i in range(self._n_substeps):
                alpha = (i + 1) / self._n_substeps
                yield self._prev_target + (self._new_target - self._prev_target) * alpha
