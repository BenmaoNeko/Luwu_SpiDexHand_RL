# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import sys
import types


def _stub_mediapy() -> None:
  if "mediapy" in sys.modules:
    return
  mediapy_stub = types.ModuleType("mediapy")
  mediapy_stub.set_ffmpeg = lambda *args, **kwargs: None  # type: ignore[attr-defined]
  sys.modules["mediapy"] = mediapy_stub


_stub_mediapy()
