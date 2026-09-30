# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer for AMP policy transitions."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str) -> None:
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            head_len = self.buffer_size - self.step
            self.states[self.step : self.buffer_size] = states[:head_len]
            self.next_states[self.step : self.buffer_size] = next_states[:head_len]
            self.states[: end_idx - self.buffer_size] = states[head_len:]
            self.next_states[: end_idx - self.buffer_size] = next_states[head_len:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield self.states[sample_idxs], self.next_states[sample_idxs]
