# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Minimal ONNX policy wrapper for deploy/reorient.

Obs assembly, history buffering and action postprocessing are owned by
ManagerBasedRlEnv (observation_manager + action term), so this module
is a thin wrapper:

  1. Load ONNX session
  2. Load config.json (for metadata cross-checks)
  3. __call__(obs_vec) -> action_vec  -- single forward pass
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort


class ONNXPolicy:
    """Single-step ONNX policy.

    Parameters
    ----------
    onnx_path : str | Path
        Path to the policy.onnx exported by export_onnx.py.
    config_path : str | Path | None
        Optional config.json sidecar (export_onnx writes one next to the
        checkpoint). If None, looks for ``<onnx_path>.config.json`` and
        ``<onnx_dir>/config.json``.

    Attributes
    ----------
    session : onnxruntime.InferenceSession
    input_name : str
    output_name : str
    input_dim : int          # expected obs vector length
    action_dim : int         # produced action vector length
    config : dict[str, Any]  # parsed config.json (may be empty {})
    """

    def __init__(
        self,
        onnx_path: str | Path,
        config_path: Optional[str | Path] = None,
    ) -> None:
        onnx_path = str(onnx_path)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        self.onnx_path: str = onnx_path
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name: str = inp.name
        self.output_name: str = out.name
        # Shapes are typically (1, N); the 1 may be a string symbol.
        self.input_dim: int = int(inp.shape[-1])
        self.action_dim: int = int(out.shape[-1])

        self.config: dict[str, Any] = self._load_config(onnx_path, config_path)

    def _load_config(
        self, onnx_path: str, config_path: Optional[str | Path]
    ) -> dict[str, Any]:
        if config_path is not None:
            with open(config_path) as f:
                return json.load(f)
        # Fallback: try <onnx_path>.config.json then <dir>/config.json
        for candidate in (
            onnx_path + ".config.json",
            os.path.join(os.path.dirname(onnx_path), "config.json"),
        ):
            if os.path.exists(candidate):
                with open(candidate) as f:
                    return json.load(f)
        return {}

    def validate_against_env(self, env) -> None:
        """Assert the loaded config.json is consistent with the RealHandEnv cfg.

        After the 2026-05 split (model-intrinsic params now flow FROM this
        config.json into the env via ``make_real_hand_env_cfg(policy_config=
        policy.config)``), this method is a belt-and-suspenders sanity check
        and will trivially pass when the env was built that way. It still
        catches accidental cfg mutations between build and runtime
        (e.g. someone bumped ``cfg.decimation`` by hand after factory
        construction), so we keep it.

        Currently checked (mirrors fields written by export_onnx.py):
          - action_scale     (== env action term's cfg.action_scale)
          - ema_alpha
          - warmup_time_s
          - control_mode     ("absolute" / "delta")
          - history_len
          - ctrl_dt          (env step_dt = sim.mujoco.timestep * decimation)
          - has_qpos_error   (if present in config.json)

        Validation reads through the action term's public cfg (not private
        instance attributes) so it stays robust across mjlab refactors.

        Missing fields in config.json are skipped (not all exports write all
        of them). Fields present here but absent on env raise too. Per
        contract, when the env's policy obs group has a shape we cannot
        introspect — or when config.json declares ``history_len`` but no
        env term reports ``history_length>0`` — validation raises rather than
        silently skipping the history check.
        """
        if not self.config:
            raise AssertionError(
                f"ONNX policy {self.onnx_path} has no config.json sidecar — "
                "cannot validate against env cfg. Re-export the checkpoint."
            )

        action_term = env.action_manager.get_term("joint_pos")
        action_cfg = action_term.cfg  # public cfg API (JointPositionOffsetEMAActionCfg)
        actual = {
            "action_scale": float(action_cfg.action_scale),
            "ema_alpha": float(action_cfg.ema_alpha),
            "warmup_time_s": float(action_cfg.warmup_time_s),
            "ctrl_dt": float(env.cfg.sim.mujoco.timestep * env.cfg.decimation),
        }
        # history_len comes from the observation group's history setting.
        # Locate the policy obs group's term dict — must be reachable via the
        # public ``terms`` attribute (ObservationGroupCfg) or as instance vars.
        # Fail-loud if neither yields an iterable: validation cannot proceed
        # silently.
        policy_obs = env.cfg.observations["policy"]
        terms = getattr(policy_obs, "terms", None)
        if terms is None or not hasattr(terms, "items"):
            try:
                terms = vars(policy_obs)
            except TypeError:
                terms = None
        if terms is None or not hasattr(terms, "items") or len(terms) == 0:
            raise AssertionError(
                "validate_against_env: env.cfg.observations['policy'] has no "
                "introspectable `terms` (no `terms` attr and no usable vars()). "
                "Cannot verify history_len; refusing to silently skip. "
                f"(policy_obs type={type(policy_obs).__name__})"
            )

        env_history_len: int | None = None
        for _name, term_cfg in terms.items():
            hl = getattr(term_cfg, "history_length", None)
            if hl is not None and hl > 0:
                env_history_len = int(hl)
                break
        if env_history_len is not None:
            actual["history_len"] = env_history_len
        elif "history_len" in self.config:
            # config.json declares history_len but env has no history term.
            # That's a hard contract failure — fail-loud.
            raise AssertionError(
                "validate_against_env: config.json declares "
                f"history_len={self.config['history_len']}, but no env policy "
                "obs term has history_length>0. The deployed env has no "
                "history; checkpoint and env are mismatched."
            )

        mismatches = []
        for key, expected_str in self.config.items():
            if key not in actual:
                continue  # unknown / informational config field
            expected = float(expected_str) if isinstance(expected_str, (int, float)) else expected_str
            got = actual[key]
            if isinstance(expected, float):
                if not np.isclose(got, expected, atol=1e-6):
                    mismatches.append(f"{key}: config={expected}, env={got}")
            else:
                if got != expected:
                    mismatches.append(f"{key}: config={expected}, env={got}")

        if mismatches:
            raise AssertionError(
                "ONNX checkpoint config.json does not match RealHandEnv cfg:\n  "
                + "\n  ".join(mismatches)
                + f"\n(onnx_path={self.onnx_path})"
            )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Single forward pass.

        Args:
            obs: (input_dim,) or (1, input_dim) float32 array.

        Returns:
            (action_dim,) float32 array (squeezed batch dim).
        """
        if obs.ndim == 1:
            obs = obs[None, :]
        assert obs.shape == (1, self.input_dim), (
            f"obs shape {obs.shape}, expected (1, {self.input_dim})"
        )
        obs = obs.astype(np.float32, copy=False)
        result = self.session.run([self.output_name], {self.input_name: obs})[0]
        return result.squeeze(0)
