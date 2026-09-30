#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Open an interactive viewer of the release-bundled printable cube.

Loads the textured cube mesh (``release-assets/hardware/cube/cube.obj``
+ sibling ``cube.mtl`` + ``cube.png``) from the unpacked release asset
zip and displays it with trimesh's pyglet-backed OpenGL viewer. Use this
to verify the 24 ArUco tag layout matches your printed cube before
sticking decals or kicking off a multi-material print.

Run:
    pixi run python deploy/reorient/tools/view_release_cube.py \\
        [--cube path/to/cube.obj]

The default path is ``release-assets/hardware/cube/cube.obj`` relative
to the current working directory — unzip + rename the release asset
(see docs/sim2real/setup.md section 3) so the tree lives under
``release-assets/`` in your shell's cwd before running.

Close the viewer window to exit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(
    description="View the release-bundled printable cube mesh."
  )
  parser.add_argument(
    "--cube",
    type=Path,
    default=Path("release-assets/hardware/cube/cube.obj"),
    help=(
      "Path to cube.obj. Defaults to release-assets/hardware/cube/cube.obj "
      "relative to cwd (the layout produced by the unzip + rename "
      "step in docs/sim2real/setup.md §3). Sibling cube.mtl and cube.png "
      "must be alongside."
    ),
  )
  args = parser.parse_args()

  if not args.cube.is_file():
    print(
      f"cube file not found: {args.cube}\n"
      "Download the release asset zip per docs/sim2real/setup.md §3\n"
      "  https://github.com/wuji-technology/wuji-mjlab/releases/latest\n"
      "and unzip + rename so 'release-assets/hardware/cube/cube.obj'\n"
      f"lives at {args.cube.resolve()},\n"
      "or pass --cube <path> explicitly.",
      file=sys.stderr,
    )
    return 1

  import trimesh

  mesh = trimesh.load(str(args.cube))
  mesh.show()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
