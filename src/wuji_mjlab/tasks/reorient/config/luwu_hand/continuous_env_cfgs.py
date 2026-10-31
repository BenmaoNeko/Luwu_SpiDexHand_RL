# SPDX-License-Identifier: Apache-2.0
"""Transfer-training configuration for continuous Luwu cube manipulation."""

from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.mdp.flip_curriculum import (
  apply_stage_parameters,
  stage_parameters,
)
from wuji_mjlab.tasks.reorient.reorient_terms import build_reorient_events

from .env_cfgs import luwu_hand_reorient_env_cfg


def luwu_hand_continuous_faces_env_cfg(
  play: bool = False,
  num_envs: int = 4096,
  *,
  enable_visual_pose_dr: bool = False,
):
  """Create the independent continuous-face transfer environment.

  The first successful Luwu task remains available through
  :func:`luwu_hand_reorient_env_cfg`; this function only overrides the
  transfer-specific command, rewards, curriculum, and approved DR ranges.
  """
  cfg = luwu_hand_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    enable_visual_pose_dr=enable_visual_pose_dr,
  )
  if play:
    cfg.scene.num_envs = 8

  command_cfg = cfg.commands["reorient_command"]
  command_cfg.goal_sampling_mode = "continuous_yaw_or_random_face"
  command_cfg.yaw_goal_probability = 0.50
  command_cfg.success_hold_steps = 10
  command_cfg.goal_switch_delay = 0
  command_cfg.require_tip_contact = True
  command_cfg.min_contact_fingertips = 1
  command_cfg.synchronized_contact_task = True

  cfg.terminations.pop("flip_completed", None)
  cfg.curriculum.pop("adaptive_episode", None)
  if "success_curriculum" in cfg.curriculum:
    cfg.curriculum["success_curriculum"].params.update(
      task_aware=True,
      count_threshold=3,
      flip_count_threshold=2,
    )

  cfg.rewards["action_rate"].weight = -0.10
  cfg.rewards["finger_collision"].weight = -0.25
  cfg.rewards["flip_success"].weight = 30.0
  cfg.metrics.update(
    {
      "consecutive_flip_count": MetricsTermCfg(
        func=mdp.consecutive_flip_count,
        params={"command_name": "reorient_command"},
      ),
      "face_coverage_count": MetricsTermCfg(
        func=mdp.face_coverage_count,
        params={"command_name": "reorient_command"},
      ),
      "adjacent_flip_count": MetricsTermCfg(
        func=mdp.adjacent_flip_count,
        params={"command_name": "reorient_command"},
      ),
      "opposite_flip_count": MetricsTermCfg(
        func=mdp.opposite_flip_count,
        params={"command_name": "reorient_command"},
      ),
      "flip_target_angle": MetricsTermCfg(
        func=mdp.flip_target_angle,
        params={"command_name": "reorient_command"},
      ),
    }
  )

  # The Luwu binding intentionally keeps the full URDF joint limits as the
  # action domain.  This is repeated here as a contract for transfer runs.
  cfg.actions["joint_pos"].action_scale = 1.0
  cfg.actions["joint_pos"].use_joint_limits = True

  if not play:
    # The baseline disables object DR for diagnosis. Transfer training
    # restores only the approved narrow object and actuator uncertainty bands.
    generic_events = build_reorient_events()
    for name in ("object_com", "object_size", "object_mass"):
      cfg.events[name] = generic_events[name]
    cfg.events["object_com"].params["ranges"] = (-0.001, 0.001)
    cfg.events["object_size"].params["scale_range"] = (0.98, 1.02)
    cfg.events["object_mass"].params["scale_range"] = (0.9, 1.1)
    cfg.events["robot_friction"].params["ranges"] = (0.10, 1.50)
    cfg.events["pd_gains"].params.update(
      kp_range=(0.75, 1.25),
      kd_range=(0.75, 1.25),
    )
    cfg.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)

    # Keep object-only event selectors explicit after restoring the generic
    # definitions, so the transfer task cannot accidentally touch the robot.
    cfg.events["object_com"].params["asset_cfg"] = SceneEntityCfg(
      "object", body_names=("cube",)
    )
    cfg.events["object_size"].params["asset_cfg"] = SceneEntityCfg(
      "object", geom_names=("cube",)
    )
    cfg.events["object_mass"].params["asset_cfg"] = SceneEntityCfg(
      "object", body_names=("cube",)
    )

  return cfg


def luwu_hand_from_scratch_curriculum_env_cfg(
  play: bool = False, num_envs: int = 4096
):
  """Create the fresh-policy, staged-flip Luwu training environment.

  This is intentionally a new configuration entry point.  It shares the
  verified observation/action/reward interface with the continuous-face task,
  but starts task difficulty at stage 0 and promotes only from flip metrics.
  ``play=True`` disables curriculum updates and previews the final stage.
  """
  cfg = luwu_hand_continuous_faces_env_cfg(
    play=play,
    num_envs=num_envs,
    # Keep metadata terms present in play mode, but only corrupt the training
    # pose stream. This fixes the actor contract at 166 dimensions.
    enable_visual_pose_dr=not play,
  )
  command_cfg = cfg.commands["reorient_command"]
  command_cfg.curriculum_enabled = not play
  command_cfg.curriculum_stage = 3 if play else 0
  command_cfg.goal_sampling_mode = "curriculum_flip"
  selected_stage = stage_parameters(command_cfg.curriculum_stage)
  command_cfg.yaw_goal_probability = selected_stage.yaw_goal_probability
  command_cfg.success_threshold = selected_stage.success_threshold_rad
  command_cfg.success_hold_steps = selected_stage.hold_steps
  command_cfg.synchronized_contact_task = True

  # Keep this termination term in the config for interface compatibility.
  # All curriculum stages are continuous now, so the command never raises
  # ``flip_completed`` after a successful flip.
  cfg.terminations["flip_completed"] = TerminationTermCfg(
    func=mdp.flip_completed,
    params={"command_name": "reorient_command"},
    time_out=False,
  )
  cfg.rewards["action_rate"].weight = -0.04
  cfg.rewards["finger_collision"].weight = -0.08
  cfg.rewards["rotation_success"].weight = 5.0
  cfg.rewards["flip_success"].weight = 40.0

  # Training starts with nominal physics.  The curriculum function updates
  # these same event ranges before subsequent resets as stages are promoted.
  if not play:
    stage = stage_parameters(0)
    apply_stage_parameters(cfg, stage)

    cfg.curriculum.clear()
    cfg.curriculum["flip_curriculum"] = CurriculumTermCfg(
      func=mdp.flip_curriculum_stage,
      params={
        "command_name": "reorient_command",
        "min_observations": 256,
      },
    )

  cfg.metrics.update(
    {
      "curriculum_stage": MetricsTermCfg(func=mdp.curriculum_stage_metric),
      "curriculum_flip_success_rate": MetricsTermCfg(
        func=mdp.curriculum_flip_success_rate_metric,
      ),
      "curriculum_flip_chain": MetricsTermCfg(
        func=mdp.curriculum_flip_chain_metric,
      ),
      "curriculum_face_coverage": MetricsTermCfg(
        func=mdp.curriculum_face_coverage_metric,
      ),
      "curriculum_cage_drop_rate": MetricsTermCfg(
        func=mdp.curriculum_cage_drop_rate_metric,
      ),
    }
  )
  return cfg


def luwu_hand_80g_visual_transfer_env_cfg(
  play: bool = False, num_envs: int = 4096
):
  """Create the model-2999 actor-transfer task for the 80 g cube.

  This task fixes the manipulation curriculum at its final all-face stage and
  changes only the cube's nominal mass and the approved actor-side visual DR
  envelope. The policy observation/action contract remains 162/16 so the
  source actor can be loaded strictly; the privileged critic is freshly built.
  """
  cfg = luwu_hand_from_scratch_curriculum_env_cfg(
    play=play,
    num_envs=num_envs,
  )
  if play:
    cfg.scene.num_envs = 8

  object_cfg = get_inhand_object_cfg(mass_kg=0.080)
  object_cfg.init_state = cfg.scene.entities["object"].init_state
  cfg.scene.entities["object"] = object_cfg

  final_stage = stage_parameters(3)
  command_cfg = cfg.commands["reorient_command"]
  command_cfg.curriculum_enabled = False
  command_cfg.curriculum_stage = 3
  command_cfg.goal_sampling_mode = "curriculum_flip"
  command_cfg.yaw_goal_probability = final_stage.yaw_goal_probability
  command_cfg.success_threshold = final_stage.success_threshold_rad
  command_cfg.success_hold_steps = final_stage.hold_steps
  command_cfg.synchronized_contact_task = True

  # Start directly at the already-trained final task/DR level; there is no
  # curriculum promotion or learning-curve reset in this transfer run.
  cfg.curriculum.clear()
  for metric_name in (
    "curriculum_stage",
    "curriculum_flip_success_rate",
    "curriculum_flip_chain",
    "curriculum_face_coverage",
    "curriculum_cage_drop_rate",
  ):
    cfg.metrics.pop(metric_name, None)
  if not play:
    apply_stage_parameters(cfg, final_stage)

  # The source actor predates the four metadata channels. Keep the corrupted
  # pose signal, but remove only those metadata terms to preserve 162 inputs.
  policy_terms = cfg.observations["policy"].terms
  for term_name in (
    "vision_valid",
    "vision_age",
    "vision_confidence",
    "vision_innovation",
  ):
    policy_terms.pop(term_name, None)
  orientation_term = policy_terms["cube_ori_error"]
  orientation_term.params["vision_stage_override"] = 4

  return cfg
