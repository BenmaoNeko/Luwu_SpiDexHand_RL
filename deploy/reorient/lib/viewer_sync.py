# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Async viewer-sync helper to keep MuJoCo viewer updates off the hot loop.

The passive viewer's ``sync()`` call transfers state to a GL render thread
and can take 5-14ms, which is larger than the slack the 50Hz control loop
has after its substep scheduling. :class:`AsyncViewerSync` runs the user's
render callback on a daemon thread with **latest-wins** semantics: calling
:meth:`submit` swaps the pending snapshot, so the viewer thread always sees
the most recent state and intermediate snapshots drop silently.
"""
from __future__ import annotations

import threading
from typing import Any, Callable


class AsyncViewerSync:
    """Run a render callback on a background thread, latest-wins.

    Typical usage in a control loop::

        viewer = AsyncViewerSync(my_render_fn)
        viewer.start()
        while running:
            ...
            viewer.submit({"joints": joints, "cube_pos": cube_pos, ...})
        viewer.stop()

    The worker thread's lifecycle is event-driven: it sleeps on a condition
    until ``submit`` signals fresh state or ``stop`` is called.
    """

    def __init__(self, render_fn: Callable[[dict[str, Any]], None]) -> None:
        self._render_fn = render_fn
        self._pending: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="viewer-sync"
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stopped = True
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, snapshot: dict[str, Any]) -> None:
        """Non-blocking: replace pending snapshot and wake the worker.

        If the worker has not yet picked up the previous pending snapshot,
        that snapshot is dropped and this one takes its place — the viewer
        always ends up showing the most recent state.
        """
        with self._lock:
            self._pending = snapshot
        self._event.set()

    def _run(self) -> None:
        while not self._stopped:
            self._event.wait(timeout=0.2)
            self._event.clear()
            if self._stopped:
                break
            with self._lock:
                snap = self._pending
                self._pending = None
            if snap is None:
                continue
            try:
                self._render_fn(snap)
            except Exception as exc:  # render bugs must not kill the worker
                print(f"[viewer-sync] render error: {exc!r}", flush=True)
