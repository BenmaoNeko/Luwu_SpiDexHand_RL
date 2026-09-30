# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Reorient environment configuration for Wuji Hand.

This module is a thin assembler: ``make_reorient_env_cfg`` calls the
per-group ``build_*`` helpers in :mod:`wuji_mjlab.tasks.reorient.reorient_terms`
and wires them into a single :class:`ManagerBasedRlEnvCfg`. The task design
(sensors, rewards, DR events, play-mode overrides) lives in
``reorient_terms``; robot-specific entity bindings live in
``config/<robot>/env_cfgs.py``.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sensor import ContactMatch
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from wuji_mjlab.tasks.reorient.reorient_terms import (
  build_reorient_actions,
  build_reorient_commands,
  build_reorient_curriculum,
  build_reorient_events,
  build_reorient_metrics,
  build_reorient_observations,
  build_reorient_rewards,
  build_reorient_sensors,
  build_reorient_terminations,
)

# Obs history window for all history-enabled observation terms.
_HISTORY_LENGTH = 3

TIP_SITE_NAMES = (".*_finger[1-5]_tip",)

TIP_BODY_NAMES = (".*_finger[1-5]_link4",)

TIP_COLLISION_GEOMS = (".*_finger[1-5]_link4_col",)

UNDESIRED_OBJECT_CONTACT_BODIES = (
  ".*_palm_link",
  ".*_finger1_link1",
  ".*_finger2_link1",
  ".*_finger2_link2",
  ".*_finger2_link3",
  ".*_finger3_link1",
  ".*_finger3_link2",
  ".*_finger3_link3",
  ".*_finger4_link1",
  ".*_finger4_link2",
  ".*_finger4_link3",
  ".*_finger5_link1",
  ".*_finger5_link2",
)

# Events removed in play (evaluation) mode so the policy is exercised against
# nominal physics rather than the training-time randomization / disturbance
# schedule. Names not present in the current cfg are tolerated via
# ``pop(name, None)``.
_PLAY_DISABLED_EVENTS = frozenset(
  {
    "object_velocity_disturbance",
    "object_disturbance_force",
    "reset_object_disturbance_force",
    "object_com",
    "robot_friction",
    "friction_curriculum_event",
    "robot_geom_size",
    "geom_size_curriculum_event",
    "contact_params_palm_thumb",
    "contact_params_fingers",
    "object_size",
    "object_mass",
    "pd_gains",
    "robot_dof_damping",
    "robot_dof_armature",
    "robot_dof_frictionloss",
    "encoder_bias",
    "robot_link_inertia",
    "robot_link_mass",
  }
)


def make_reorient_env_cfg(
  play: bool = False,
  num_envs: int = 8192,
  *,
  tip_site_names: tuple[str, ...] = TIP_SITE_NAMES,
  tip_body_names: tuple[str, ...] = TIP_BODY_NAMES,
  tip_collision_geoms: tuple[str, ...] = TIP_COLLISION_GEOMS,
  undesired_object_contact_bodies: tuple[str, ...] = UNDESIRED_OBJECT_CONTACT_BODIES,
  include_cube_position_in_policy: bool = True,
  enable_visual_pose_dr: bool = False,
  object_body_pattern: str = "cube",
  object_geom_pattern: str = "cube",
  tip_object_secondary: ContactMatch | None = None,
) -> ManagerBasedRlEnvCfg:
  """Create Reorient task configuration.

  Args:
    play: If True, switch to evaluation defaults (small num_envs, longer
      episode, debug viz, no policy obs noise, all startup DR / interval
      disturbance events stripped, curriculum cleared).
    num_envs: Parallel-env count for the training scene. Ignored when
      ``play=True`` (play mode forces a tiny scene).
  """
  observations = build_reorient_observations(
    history_length=_HISTORY_LENGTH,
    tip_body_names=tip_body_names,
    include_cube_position_in_policy=include_cube_position_in_policy,
    enable_visual_pose_dr=enable_visual_pose_dr,
    object_body_pattern=object_body_pattern,
    object_geom_pattern=object_geom_pattern,
  )
  actions = build_reorient_actions()
  commands = build_reorient_commands()
  events = build_reorient_events()
  rewards = build_reorient_rewards(tip_site_names=tip_site_names)
  terminations = build_reorient_terminations()
  curriculum = build_reorient_curriculum()
  metrics = build_reorient_metrics()
  sensors = build_reorient_sensors(
    tip_collision_geoms=tip_collision_geoms,
    tip_body_names=tip_body_names,
    undesired_object_contact_bodies=undesired_object_contact_bodies,
    object_body_pattern=object_body_pattern,
    object_geom_pattern=object_geom_pattern,
    tip_object_secondary=tip_object_secondary,
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
      env_spacing=0.75,
      extent=0.8,
      sensors=sensors,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=0.5,
      elevation=-18.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=180,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.01,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=5,
    episode_length_s=50.0,
  )

  if play:
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 0.5
    cfg.episode_length_s = 60.0
    cfg.commands["reorient_command"].debug_vis = True
    cfg.observations["policy"].enable_corruption = False
    for key in _PLAY_DISABLED_EVENTS:
      cfg.events.pop(key, None)
    cfg.curriculum.clear()

  return cfg
