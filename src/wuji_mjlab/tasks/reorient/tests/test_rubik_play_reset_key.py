# SPDX-License-Identifier: Apache-2.0
from mjlab.viewer.native.keys import KEY_ENTER, KEY_R

from wuji_mjlab.utils import play_runner


def test_rubik_native_preview_r_and_enter_queue_full_reset():
  viewer_type = getattr(play_runner, "_RubikResettableNativeMujocoViewer")
  viewer = object.__new__(viewer_type)
  calls = []
  viewer.request_reset = lambda: calls.append("reset")
  viewer.user_key_callback = None

  viewer._safe_key_callback(KEY_R)
  viewer._safe_key_callback(KEY_ENTER)

  assert calls == ["reset", "reset"]
