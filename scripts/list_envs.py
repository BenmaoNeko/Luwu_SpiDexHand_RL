# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""List registered Wuji MJLab tasks."""

from __future__ import annotations

from prettytable import PrettyTable

import wuji_mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks


def main() -> int:
  table = PrettyTable(["#", "Task ID"])
  table.title = "Available Wuji MJLab Tasks"
  table.align["Task ID"] = "l"

  for idx, task_id in enumerate(list_tasks(), start=1):
    table.add_row([idx, task_id])

  print(table)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
