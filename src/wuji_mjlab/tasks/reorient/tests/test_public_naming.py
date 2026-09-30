# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Verify the source tree contains no forbidden symbol names.

The test walks the filesystem under the repo root rather than ``git ls-files``
so it still works inside source zips, sdists/wheels, or any checkout missing a
``.git`` directory. Local / generated directories are skipped explicitly.
"""

from __future__ import annotations

from pathlib import Path

_ALLOWED_BINARY_SUFFIXES = {
  ".gif",
  ".png",
  ".jpg",
  ".jpeg",
  ".STL",
  ".stl",
  ".obj",
  ".pyc",
}
# Local / generated / vendored caches that may legally contain the forbidden
# token (e.g. pixi-resolved deps, compiled bytecode, IDE state). These are not
# tracked by git, so leaks here are not a public-surface concern.
_SKIP_DIR_NAMES = {
  ".git",
  ".pixi",
  ".pytest_cache",
  ".ruff_cache",
  ".mypy_cache",
  ".venv",
  "__pycache__",
  "node_modules",
  "logs",
  "build",
  "dist",
  "outputs",
  "wandb",
}
# Concatenated to avoid this file matching itself.
_FORBIDDEN_NAME = "wh" + "110"


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[5]


def _iter_repo_files(root: Path):
  """Yield every regular file under ``root`` minus skipped dirs."""
  pending: list[Path] = [root]
  while pending:
    cur = pending.pop()
    try:
      entries = list(cur.iterdir())
    except (FileNotFoundError, PermissionError):
      continue
    for entry in entries:
      if entry.is_symlink():
        continue
      if entry.is_dir():
        if entry.name in _SKIP_DIR_NAMES or entry.name.endswith(".egg-info"):
          continue
        pending.append(entry)
      elif entry.is_file():
        yield entry


def test_internal_robot_codename_is_not_exposed_in_paths_or_text() -> None:
  root = _repo_root()
  leaks: list[str] = []
  for path in _iter_repo_files(root):
    rel = path.relative_to(root)
    if _FORBIDDEN_NAME in rel.as_posix().lower():
      leaks.append(str(rel))
      continue
    if path.suffix in _ALLOWED_BINARY_SUFFIXES:
      continue
    try:
      text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
      continue
    if _FORBIDDEN_NAME in text.lower():
      leaks.append(str(rel))

  assert leaks == []
