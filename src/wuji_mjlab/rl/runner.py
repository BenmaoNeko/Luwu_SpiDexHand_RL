# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji-specific extension of mjlab's on-policy runner.

Restricts W&B artifact uploads to the final-iteration checkpoint only.
Local ``.pt`` files are still written on every ``save_interval`` boundary
(disk-only, useful for crash recovery); W&B receives just the terminal
``model_{N-1}.pt``.

Subclasses that emit their own artifacts (e.g. an ``.onnx`` export
alongside the ``.pt``) must call :meth:`WujiOnPolicyRunner.should_upload_artifact`
before invoking ``self.logger.save_model`` so they honor the same gate.
"""

from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner

from wuji_mjlab.rl.transfer import ActorTransferMonitor


class WujiOnPolicyRunner(MjlabOnPolicyRunner):
  def enable_transfer_monitor(
    self,
    probe_observations,
    *,
    legacy_prefix_dim: int,
    window_iterations: int = 100,
  ) -> dict[str, object]:
    if window_iterations <= 0:
      raise ValueError("window_iterations must be positive")
    self._actor_transfer_monitor = ActorTransferMonitor(
      self.alg.get_policy(),
      probe_observations,
      legacy_prefix_dim=legacy_prefix_dim,
    )
    self._actor_transfer_monitor_window = window_iterations
    return {
      "enabled": True,
      "probe_count": int(probe_observations.batch_size[0]),
      "legacy_prefix_dim": legacy_prefix_dim,
      "window_iterations": window_iterations,
      "scalars": [
        "probe_action_delta_rms",
        "legacy_prefix_mean_delta_rms",
        "legacy_prefix_std_delta_rms",
        "rubik_channels_mean_rms",
        "rubik_channels_std_rms",
        "normalizer_count",
      ],
    }

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    self._final_iteration = (
      self.current_learning_iteration + num_learning_iterations - 1
    )
    monitor = getattr(self, "_actor_transfer_monitor", None)
    if monitor is None:
      super().learn(num_learning_iterations, init_at_random_ep_len)
      return

    original_log = self.logger.log
    monitor_start = self.current_learning_iteration
    monitor_window = self._actor_transfer_monitor_window

    def log_with_transfer_metrics(*args, **kwargs):
      original_log(*args, **kwargs)
      iteration = kwargs.get("it", args[0] if args else monitor_start)
      if iteration - monitor_start >= monitor_window:
        return
      writer = self.logger.writer
      add_scalar = getattr(writer, "add_scalar", None)
      if not callable(add_scalar):
        return
      for name, value in monitor.snapshot().items():
        add_scalar(f"Transfer/{name}", value, iteration)

    self.logger.log = log_with_transfer_metrics
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.logger.log = original_log

  def save(self, path: str, infos=None) -> None:
    # Temporarily flip cfg["upload_model"] so the parent's save() handles
    # the W&B upload conditionally. This avoids re-implementing the parent's
    # local-checkpoint logic, which is fragile against upstream mjlab changes.
    original = self.cfg.get("upload_model")
    controller = getattr(self.env.unwrapped, '_smart_cube_curriculum', None)
    if controller is not None:
      infos = {**(infos or {}), 'smart_cube_curriculum': {
        k: v for k, v in vars(controller).items() if k != 'cfg'
      }}
    from wuji_mjlab.tasks.reorient.mdp.gentle_flip import gentle_checkpoint_metadata
    infos = {**(infos or {}), **gentle_checkpoint_metadata(self.env.unwrapped)}
    self.cfg["upload_model"] = self.should_upload_artifact()
    try:
      super().save(path, infos)
    finally:
      self.cfg["upload_model"] = original

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    if load_cfg is None or load_cfg.get('iteration', False):
      # Saved iteration is the last completed update, not the next one.
      self.current_learning_iteration += 1
      controller = getattr(self.env.unwrapped, '_smart_cube_curriculum', None)
      state = (infos or {}).get('smart_cube_curriculum')
      if controller is not None and state is not None:
        for name, value in state.items():
          if name in vars(controller) and name != 'cfg':
            setattr(controller, name, value)
        if hasattr(controller, 'joint_dr_last_step') and 'joint_dr_progress' not in state:
          # A legacy run has no joint-error curriculum. Start at zero using
          # restored cumulative counters as the baseline, never past outcomes.
          controller.joint_dr_last_step = self.env.unwrapped.common_step_counter
          controller.joint_dr_last_totals = (
            controller.low_friction_skill_attempts, controller.low_friction_skill_successes)
      from wuji_mjlab.tasks.reorient.mdp.gentle_flip import restore_gentle_checkpoint
      restore_gentle_checkpoint(self.env.unwrapped, infos)
    return infos

  def should_upload_artifact(self) -> bool:
    """Whether the current iteration is eligible for a W&B artifact upload.

    True iff ``upload_model`` is set in the cfg, ``learn()`` has been
    called (so ``_final_iteration`` is known), and the current iteration
    has reached the final one of the training session. The comparison
    uses ``>=`` (not ``==``) so a one-off counter shift in upstream rsl-rl
    — e.g. the loop incrementing ``current_learning_iteration`` past
    ``N-1`` before the final ``save()`` — surfaces as an extra upload
    rather than a silently-dropped final ckpt.

    Subclasses that emit their own artifacts (e.g. ONNX exports) should
    gate ``self.logger.save_model`` calls on this predicate.
    """
    if not self.cfg.get("upload_model"):
      return False
    final = getattr(self, "_final_iteration", None)
    if final is None:
      return False
    return self.current_learning_iteration >= final
