# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch

from wuji_mjlab.tasks.reorient.mdp.contact_task_state import box_surface_distance


def test_box_surface_distance_supports_multiple_points_per_batched_box():
  site_pos_w = torch.tensor(
    [
      [[0.08, 0.0, 0.0], [0.0, 0.08, 0.0]],
      [[0.08, 0.0, 0.0], [0.0, 0.08, 0.0]],
    ]
  )
  box_pos_w = torch.zeros((2, 3))
  box_quat_w = torch.tensor(
    [
      [1.0, 0.0, 0.0, 0.0],
      [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
    ]
  )
  half_size = torch.tensor([[0.05, 0.03, 0.02], [0.05, 0.03, 0.02]])

  distance = box_surface_distance(site_pos_w, box_pos_w, box_quat_w, half_size)

  assert torch.allclose(
    distance,
    torch.tensor([[0.03, 0.05], [0.05, 0.03]]),
    atol=1e-6,
  )
