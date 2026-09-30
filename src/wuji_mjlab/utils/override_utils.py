# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Thin IsaacLab-style override helpers for env/agent cfg objects."""

from __future__ import annotations

import ast
from dataclasses import is_dataclass
from typing import Any


def parse_override_value(raw: str) -> Any:
  lowered = raw.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  if lowered in {"none", "null"}:
    return None

  try:
    return ast.literal_eval(raw)
  except (ValueError, SyntaxError):
    return raw


def _is_index(part: str) -> bool:
  try:
    int(part)
  except ValueError:
    return False
  return True


def _get_child(target: Any, part: str) -> Any:
  if isinstance(target, dict):
    if part not in target:
      raise KeyError(f"Missing key '{part}'")
    return target[part]
  if isinstance(target, list | tuple):
    if not _is_index(part):
      raise KeyError(f"Expected integer index for sequence access, got '{part}'")
    index = int(part)
    return target[index]
  if hasattr(target, part):
    return getattr(target, part)
  raise KeyError(f"Missing attribute '{part}' on {type(target).__name__}")


def _set_child(target: Any, part: str, value: Any) -> None:
  if isinstance(target, dict):
    if part not in target:
      raise KeyError(f"Missing key '{part}'")
    target[part] = value
    return
  if isinstance(target, list):
    if not _is_index(part):
      raise KeyError(f"Expected integer index for list access, got '{part}'")
    target[int(part)] = value
    return
  if isinstance(target, tuple):
    raise TypeError(
      "Tuple item overrides are not supported by this thin override layer."
    )
  if hasattr(target, part):
    setattr(target, part, value)
    return
  raise KeyError(f"Missing attribute '{part}' on {type(target).__name__}")


def _assert_supported_root(obj: Any) -> None:
  if isinstance(obj, (dict, list, tuple)):
    return
  if is_dataclass(obj) or hasattr(obj, "__dict__"):
    return
  raise TypeError(f"Unsupported override root type: {type(obj).__name__}")


def apply_override(roots: dict[str, Any], override: str) -> None:
  if "=" not in override:
    raise ValueError(f"Invalid override '{override}', expected key=value format.")
  path, raw_value = override.split("=", 1)
  parts = path.split(".")
  if len(parts) < 2:
    raise ValueError(f"Override '{override}' must start with env. or agent.")

  root_name = parts[0]
  if root_name not in roots:
    raise KeyError(
      f"Unknown override root '{root_name}'. Expected one of: {sorted(roots)}"
    )

  current = roots[root_name]
  _assert_supported_root(current)
  for part in parts[1:-1]:
    current = _get_child(current, part)

  _set_child(current, parts[-1], parse_override_value(raw_value))


def apply_overrides(roots: dict[str, Any], overrides: list[str]) -> None:
  for override in overrides:
    apply_override(roots, override)
