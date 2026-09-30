# SPDX-License-Identifier: Apache-2.0
"""From-scratch flip curriculum for the Luwu hand.

The curriculum changes only task difficulty and reset randomization.  It does
not change the observation groups or action interface, so a policy can learn
one stable network contract from the first rollout through the final stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from wuji_mjlab.tasks.reorient.mdp.event_utils import resolve_env_ids


@dataclass(frozen=True)
class FlipCurriculumStage:
  """Immutable parameters for one training stage."""

  stage: int
  task_mode: str
  target_sampling_mode: str
  adjacent_probability: float
  yaw_goal_probability: float
  continuous: bool
  flip_terminates: bool
  success_threshold_rad: float
  hold_steps: int
  object_com_range: tuple[float, float]
  object_size_range: tuple[float, float]
  object_mass_range: tuple[float, float]
  robot_friction_range: tuple[float, float]
  pd_kp_range: tuple[float, float]
  pd_kd_range: tuple[float, float]
  encoder_bias_range: tuple[float, float]
  contact_solref_timeconst_range: tuple[float, float]
  contact_solref_dampratio_range: tuple[float, float]
  contact_solimp_width_range: tuple[float, float]
  contact_solimp_dmin_range: tuple[float, float]
  dof_damping_range: tuple[float, float]
  dof_armature_range: tuple[float, float]
  dof_frictionloss_range: tuple[float, float]
  link_inertia_range: tuple[float, float]
  link_mass_range: tuple[float, float]
  action_smoothness_weight: float
  finger_collision_weight: float
  command_delay_steps: int
  command_noise_std: float


_STAGES = (
  FlipCurriculumStage(
    stage=0,
    task_mode="adjacent_continuous",
    target_sampling_mode="adjacent_only",
    adjacent_probability=1.0,
    yaw_goal_probability=0.30,
    continuous=True,
    flip_terminates=False,
    success_threshold_rad=math.radians(8.0),
    hold_steps=5,
    object_com_range=(0.0, 0.0),
    object_size_range=(1.0, 1.0),
    object_mass_range=(1.0, 1.0),
    robot_friction_range=(1.0, 1.0),
    pd_kp_range=(1.0, 1.0),
    pd_kd_range=(1.0, 1.0),
    encoder_bias_range=(0.0, 0.0),
    contact_solref_timeconst_range=(1.0, 1.0),
    contact_solref_dampratio_range=(1.0, 1.0),
    contact_solimp_width_range=(1.0, 1.0),
    contact_solimp_dmin_range=(1.0, 1.0),
    dof_damping_range=(1.0, 1.0),
    dof_armature_range=(1.0, 1.0),
    dof_frictionloss_range=(1.0, 1.0),
    link_inertia_range=(1.0, 1.0),
    link_mass_range=(1.0, 1.0),
    action_smoothness_weight=-0.04,
    finger_collision_weight=-0.08,
    command_delay_steps=0,
    command_noise_std=0.0,
  ),
  FlipCurriculumStage(
    stage=1,
    task_mode="adjacent_continuous",
    target_sampling_mode="adjacent_only",
    adjacent_probability=1.0,
    yaw_goal_probability=0.30,
    continuous=True,
    flip_terminates=False,
    success_threshold_rad=math.radians(6.0),
    hold_steps=8,
    object_com_range=(0.0, 0.0),
    object_size_range=(1.0, 1.0),
    object_mass_range=(1.0, 1.0),
    robot_friction_range=(1.0, 1.0),
    pd_kp_range=(1.0, 1.0),
    pd_kd_range=(1.0, 1.0),
    encoder_bias_range=(0.0, 0.0),
    contact_solref_timeconst_range=(0.9, 1.1),
    contact_solref_dampratio_range=(0.9, 1.1),
    contact_solimp_width_range=(0.9, 1.1),
    contact_solimp_dmin_range=(0.9, 1.0),
    dof_damping_range=(0.85, 1.15),
    dof_armature_range=(0.9, 1.1),
    dof_frictionloss_range=(0.8, 1.2),
    link_inertia_range=(0.95, 1.05),
    link_mass_range=(0.95, 1.05),
    action_smoothness_weight=-0.06,
    finger_collision_weight=-0.12,
    command_delay_steps=1,
    command_noise_std=0.002,
  ),
  FlipCurriculumStage(
    stage=2,
    task_mode="all_faces_continuous",
    target_sampling_mode="all_other",
    adjacent_probability=0.0,
    yaw_goal_probability=0.30,
    continuous=True,
    flip_terminates=False,
    success_threshold_rad=math.radians(6.0),
    hold_steps=8,
    object_com_range=(0.0, 0.0),
    object_size_range=(1.0, 1.0),
    object_mass_range=(1.0, 1.0),
    robot_friction_range=(1.0, 1.0),
    pd_kp_range=(1.0, 1.0),
    pd_kd_range=(1.0, 1.0),
    encoder_bias_range=(0.0, 0.0),
    contact_solref_timeconst_range=(0.8, 1.25),
    contact_solref_dampratio_range=(0.8, 1.25),
    contact_solimp_width_range=(0.8, 1.5),
    contact_solimp_dmin_range=(0.7, 1.0),
    dof_damping_range=(0.7, 1.4),
    dof_armature_range=(0.8, 1.2),
    dof_frictionloss_range=(0.7, 1.4),
    link_inertia_range=(0.9, 1.1),
    link_mass_range=(0.9, 1.1),
    action_smoothness_weight=-0.08,
    finger_collision_weight=-0.18,
    command_delay_steps=1,
    command_noise_std=0.005,
  ),
  FlipCurriculumStage(
    stage=3,
    task_mode="all_faces_continuous",
    target_sampling_mode="all_other",
    adjacent_probability=0.0,
    yaw_goal_probability=0.30,
    continuous=True,
    flip_terminates=False,
    success_threshold_rad=math.radians(4.0),
    hold_steps=10,
    object_com_range=(-0.003, 0.003),
    object_size_range=(0.97, 1.03),
    object_mass_range=(0.85, 1.15),
    robot_friction_range=(0.5, 1.5),
    pd_kp_range=(0.75, 1.25),
    pd_kd_range=(0.75, 1.25),
    encoder_bias_range=(-0.02, 0.02),
    contact_solref_timeconst_range=(0.7, 1.4),
    contact_solref_dampratio_range=(0.7, 1.4),
    contact_solimp_width_range=(0.8, 2.5),
    contact_solimp_dmin_range=(0.5, 1.0),
    dof_damping_range=(0.6, 1.6),
    dof_armature_range=(0.75, 1.25),
    dof_frictionloss_range=(0.6, 1.6),
    link_inertia_range=(0.85, 1.15),
    link_mass_range=(0.85, 1.15),
    action_smoothness_weight=-0.10,
    finger_collision_weight=-0.25,
    command_delay_steps=2,
    command_noise_std=0.01,
  ),
)


def stage_parameters(stage: int) -> FlipCurriculumStage:
  """Return the validated immutable parameters for ``stage``."""
  if stage not in range(len(_STAGES)):
    raise ValueError(f"stage must be in [0, {len(_STAGES) - 1}], got {stage}")
  return _STAGES[stage]


def promote_flip_curriculum_stage(
  current_stage: int,
  *,
  observations: int,
  flip_success_rate: float,
  mean_flip_chain: float,
  mean_face_coverage: float,
  cage_drop_rate: float,
  min_observations: int = 256,
) -> int:
  """Apply conservative promotion gates using flip-only statistics.

  Rotation successes are deliberately absent from this gate.  A policy must
  demonstrate actual face changes and a bounded drop rate before the task
  becomes harder. Single-flip stages do not require a chain metric because a
  successful episode intentionally ends after one flip.
  """
  stage = min(max(int(current_stage), 0), len(_STAGES) - 1)
  if observations < min_observations or stage == len(_STAGES) - 1:
    return stage
  if stage == 0:
    if flip_success_rate >= 0.25 and cage_drop_rate <= 0.20:
      return 1
    return 0
  if stage == 1:
    if flip_success_rate >= 0.25 and cage_drop_rate <= 0.20:
      return 2
    return 1
  if (
    flip_success_rate >= 0.20
    and mean_flip_chain >= 1.50
    and mean_face_coverage >= 2.50
    and cage_drop_rate <= 0.20
  ):
    return 3
  return 2


def apply_stage_parameters(cfg, stage: FlipCurriculumStage) -> None:
  """Apply staged DR and reward parameters to an env config."""
  events = getattr(cfg, "events", None)
  if events is None:
    return
  if "object_com" in events:
    events["object_com"].params["ranges"] = stage.object_com_range
  if "object_size" in events:
    events["object_size"].params["scale_range"] = stage.object_size_range
  if "object_mass" in events:
    events["object_mass"].params["scale_range"] = stage.object_mass_range
  if "robot_friction" in events:
    events["robot_friction"].params["ranges"] = stage.robot_friction_range
  if "pd_gains" in events:
    events["pd_gains"].params.update(
      kp_range=stage.pd_kp_range,
      kd_range=stage.pd_kd_range,
    )
  if "encoder_bias" in events:
    events["encoder_bias"].params["bias_range"] = stage.encoder_bias_range
  for name in ("contact_params_palm_thumb", "contact_params_fingers"):
    if name in events:
      events[name].params.update(
        solref_timeconst_range=stage.contact_solref_timeconst_range,
        solref_dampratio_range=stage.contact_solref_dampratio_range,
        solimp_width_range=stage.contact_solimp_width_range,
        solimp_dmin_range=stage.contact_solimp_dmin_range,
      )
  if "robot_dof_damping" in events:
    events["robot_dof_damping"].params["ranges"] = stage.dof_damping_range
  if "robot_dof_armature" in events:
    events["robot_dof_armature"].params["ranges"] = stage.dof_armature_range
  if "robot_dof_frictionloss" in events:
    events["robot_dof_frictionloss"].params["ranges"] = stage.dof_frictionloss_range
  if "robot_link_inertia" in events:
    events["robot_link_inertia"].params["scale_range"] = stage.link_inertia_range
  if "robot_link_mass" in events:
    events["robot_link_mass"].params["ranges"] = stage.link_mass_range

  rewards = getattr(cfg, "rewards", None)
  if rewards is not None:
    if "action_rate" in rewards:
      rewards["action_rate"].weight = stage.action_smoothness_weight
    if "finger_collision" in rewards:
      rewards["finger_collision"].weight = stage.finger_collision_weight
  actions = getattr(cfg, "actions", None)
  if actions is not None and "joint_pos" in actions:
    actions["joint_pos"].command_delay_steps = stage.command_delay_steps
    actions["joint_pos"].command_noise_std = stage.command_noise_std


def _set_stage_event_ranges(env, stage: FlipCurriculumStage) -> None:
  """Update startup DR ranges before the event manager applies a reset."""
  cfg = getattr(env, "cfg", None)
  if cfg is not None:
    apply_stage_parameters(cfg, stage)


def flip_curriculum_stage(
  env,
  env_ids,
  command_name: str = "reorient_command",
  min_observations: int = 256,
) -> dict[str, float]:
  """Advance the global curriculum stage from completed episode statistics.

  The manager calls curriculum terms before command/event reset.  We therefore
  read the just-finished episodes, update a bounded cumulative gate state, and
  write the selected stage before the next goals and DR are sampled.
  """
  env_ids = resolve_env_ids(env, env_ids)
  current_stage = int(getattr(env, "_flip_curriculum_stage", 0))
  stage = stage_parameters(current_stage)
  # The first manager reset is initialization, not a completed episode.  Do
  # not seed the promotion window with zero-valued pseudo-statistics.
  if int(getattr(env, "common_step_counter", 0)) == 0 and not getattr(
    env, "_flip_curriculum_initialized", False
  ):
    env._flip_curriculum_initialized = True
    _set_stage_event_ranges(env, stage)
    return {
      "stage": float(current_stage),
      "observations": 0.0,
      "flip_episode_success_rate": 0.0,
      "mean_consecutive_flip_count": 0.0,
      "mean_face_coverage_count": 0.0,
      "cage_drop_rate": 0.0,
      "promoted": 0.0,
    }
  if env_ids.numel() == 0:
    _set_stage_event_ranges(env, stage)
    return {"stage": float(current_stage), "observations": 0.0}

  command = env.command_manager.get_term(command_name)
  metrics = command.metrics
  flip_count = metrics["flip_success_count"][env_ids].float()
  chain = metrics["consecutive_flip_count"][env_ids].float()
  coverage = metrics["face_coverage_count"][env_ids].float()
  try:
    dropped = env.termination_manager.get_term("cage_drop")[env_ids].float()
  except (AttributeError, KeyError, ValueError):
    dropped = torch.zeros_like(flip_count)

  # Keep a device-side ring buffer so promotion reflects the latest episode
  # window rather than an ever-growing average.  This avoids changing the
  # observation/action contract and avoids a Python loop over reset IDs.
  window_size = max(int(min_observations), 256)
  rows = torch.stack(
    ((flip_count > 0).float(), chain, coverage, dropped), dim=-1
  )
  history = getattr(env, "_flip_curriculum_history", None)
  if history is None or history.shape != (window_size, 4):
    history = torch.zeros(
      (window_size, 4), device=rows.device, dtype=rows.dtype
    )
    env._flip_curriculum_history = history
    env._flip_curriculum_history_ptr = 0
    env._flip_curriculum_history_filled = 0

  batch_rows = rows[-window_size:]
  batch_count = batch_rows.shape[0]
  if batch_count == window_size:
    history[:] = batch_rows
    env._flip_curriculum_history_ptr = 0
    env._flip_curriculum_history_filled = window_size
  else:
    pointer = int(env._flip_curriculum_history_ptr)
    first = min(batch_count, window_size - pointer)
    history[pointer : pointer + first] = batch_rows[:first]
    if first < batch_count:
      history[: batch_count - first] = batch_rows[first:]
    env._flip_curriculum_history_ptr = (pointer + batch_count) % window_size
    env._flip_curriculum_history_filled = min(
      window_size,
      int(env._flip_curriculum_history_filled) + batch_count,
    )

  previous_count = int(getattr(env, "_flip_curriculum_observations", 0))
  count = previous_count + int(env_ids.numel())
  env._flip_curriculum_observations = count
  filled = int(env._flip_curriculum_history_filled)
  summary = history[:filled]
  success_rate = float(summary[:, 0].mean().item())
  mean_chain = float(summary[:, 1].mean().item())
  mean_coverage = float(summary[:, 2].mean().item())
  drop_rate = float(summary[:, 3].mean().item())
  next_stage = promote_flip_curriculum_stage(
    current_stage,
    observations=filled,
    flip_success_rate=success_rate,
    mean_flip_chain=mean_chain,
    mean_face_coverage=mean_coverage,
    cage_drop_rate=drop_rate,
    min_observations=min_observations,
  )
  env._flip_curriculum_stage = next_stage
  selected = stage_parameters(next_stage)
  _set_stage_event_ranges(env, selected)
  return {
    "stage": float(next_stage),
    "observations": float(count),
    "flip_episode_success_rate": float(success_rate),
    "mean_consecutive_flip_count": float(mean_chain),
    "mean_face_coverage_count": float(mean_coverage),
    "cage_drop_rate": float(drop_rate),
    "promoted": float(next_stage != current_stage),
  }


def curriculum_stage_metric(env, command_name: str = "reorient_command") -> torch.Tensor:
  """Per-environment stage metric for TensorBoard diagnostics."""
  del command_name
  stage = int(getattr(env, "_flip_curriculum_stage", 0))
  return torch.full((env.num_envs,), float(stage), device=env.device)


def curriculum_flip_success_rate_metric(
  env, command_name: str = "reorient_command"
) -> torch.Tensor:
  """Broadcast the current cumulative flip success rate for diagnostics."""
  del command_name
  history = getattr(env, "_flip_curriculum_history", None)
  filled = int(getattr(env, "_flip_curriculum_history_filled", 0))
  value = float(history[:filled, 0].mean().item()) if filled else 0.0
  return torch.full((env.num_envs,), value, device=env.device)


def curriculum_flip_chain_metric(env) -> torch.Tensor:
  """Broadcast rolling mean consecutive flips for diagnostics."""
  history = getattr(env, "_flip_curriculum_history", None)
  filled = int(getattr(env, "_flip_curriculum_history_filled", 0))
  value = float(history[:filled, 1].mean().item()) if filled else 0.0
  return torch.full((env.num_envs,), value, device=env.device)


def curriculum_face_coverage_metric(env) -> torch.Tensor:
  """Broadcast rolling mean unique-face coverage for diagnostics."""
  history = getattr(env, "_flip_curriculum_history", None)
  filled = int(getattr(env, "_flip_curriculum_history_filled", 0))
  value = float(history[:filled, 2].mean().item()) if filled else 0.0
  return torch.full((env.num_envs,), value, device=env.device)


def curriculum_cage_drop_rate_metric(env) -> torch.Tensor:
  """Broadcast rolling cage-drop rate for diagnostics."""
  history = getattr(env, "_flip_curriculum_history", None)
  filled = int(getattr(env, "_flip_curriculum_history_filled", 0))
  value = float(history[:filled, 3].mean().item()) if filled else 0.0
  return torch.full((env.num_envs,), value, device=env.device)
