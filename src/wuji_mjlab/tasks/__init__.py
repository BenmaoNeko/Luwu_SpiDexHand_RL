# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = [
  "common",
  ".mdp",
  ".rl",
]

import_packages(__name__, _BLACKLIST_PKGS)
