# SPDX-License-Identifier: Apache-2.0
"""Stage-one Luwu task: continuous 45/45/10 Rubik manipulation."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch

from wuji_mjlab.assets.objects.rubik.rubik_cfg import (
  RUBIK_INIT_STATE,
  get_rubik_cfg,
  get_rubik_target_cfg,
)
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.mdp.rubik_command import RubikUFaceCommandCfg
from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import (
  RubikFullCubeCommandCfg,
)
from wuji_mjlab.tasks.reorient.mdp.smart_cube_curriculum import (
  SmartCubeCurriculumCfg,
  randomize_smart_cube_face_resistance,
  record_reset_skill_failures,
  update_smart_cube_curriculum,
)
from wuji_mjlab.tasks.reorient.mdp.smart_cube_observations import (
  SmartCubeProfile,
  reset_smart_cube_observation_state,
)

from .env_cfgs import luwu_hand_reorient_env_cfg


def luwu_hand_rubik_u_face_env_cfg(
  play: bool = False,
  num_envs: int = 256,
):
  """Build the constrained full-cube task with a fresh actor/critic policy."""
  cfg = luwu_hand_reorient_env_cfg(
    play=play,
    num_envs=num_envs,
    enable_visual_pose_dr=True,
    object_body_pattern=r"cubelet:.*",
    object_geom_pattern=r"cubelet:.*",
    tip_object_secondary=ContactMatch(
      mode="subtree", pattern="middle", entity="object"
    ),
  )
  if play:
    cfg.scene.num_envs = 1

  cfg.scene.entities["object"] = get_rubik_cfg()
  cfg.scene.entities["object"].init_state = RUBIK_INIT_STATE
  if play:
    cfg.scene.entities["target"] = get_rubik_target_cfg()
  cfg.commands["reorient_command"] = RubikUFaceCommandCfg(
    entity_name="object",
    target_entity_name="target" if play else "",
    resampling_time_range=(1.0e6, 1.0e6),
    upright_tolerance_rad=0.20,
    upright_hold_steps=8,
    flip_angle_tolerance_rad=0.1745329252,
    turn_tolerance_rad=0.0872664626,
    turn_hold_steps=8,
    max_u_face_speed=0.45,
    combine_after_steps=40_000,  # Iteration 1000 of a 3000-iteration run.
  )
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=mdp.reset_rubik_u_face,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("object"),
      "robot_cfg": SceneEntityCfg("robot", body_names=("right_palm_link",)),
      "palm_clearance_m": 0.006,
    },
  )
  if not play:
    cfg.events["rubik_mass"] = EventTermCfg(
      func=mdp.randomize_body_mass_and_inertia,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("object", body_names=("middle", "cubelet:.*")),
        "scale_range": (0.7875, 1.0),
        "shared_scale_per_env": True,
      },
    )

  policy = cfg.observations["policy"].terms
  policy.pop("cube_ori_error", None)
  policy["rubik_cube_orientation"] = ObservationTermCfg(
    func=mdp.rubik_cube_orientation_6d_in_tag,
    params={"vision_dr": True},
    history_length=3,
  )
  policy["rubik_u_face_angle"] = ObservationTermCfg(
    func=mdp.rubik_u_face_angle_sincos,
    params={"observation_noise_rad": 0.05},
    history_length=3,
  )
  policy["rubik_u_face_goal"] = ObservationTermCfg(
    func=mdp.rubik_u_face_goal_sincos,
    history_length=0,
  )
  policy["rubik_face_goal"] = ObservationTermCfg(
    func=mdp.rubik_face_goal_local,
    history_length=0,
  )
  policy["rubik_root_goal"] = ObservationTermCfg(
    func=mdp.rubik_root_goal_6d_in_tag,
    history_length=0,
  )
  policy["rubik_task_phase"] = ObservationTermCfg(
    func=mdp.rubik_task_phase,
    history_length=0,
  )

  critic = cfg.observations["critic"].terms
  for name in (
    "cube_ori_error",
    "cube_ori_error_clean",
    "state_progress",
  ):
    critic.pop(name, None)
  critic["rubik_state_privileged"] = ObservationTermCfg(
    func=mdp.rubik_state_privileged,
    history_length=0,
  )

  for name in (
    "goal_progress",
    "rotation_success",
    "flip_success",
  ):
    cfg.rewards.pop(name, None)
  cfg.rewards["approach_progress"].params.update(
    object_cfg=SceneEntityCfg("object"),
    box_half_size_m=0.02855,  # Robogym cubelet envelope: 57.1 mm edge.
  )
  cfg.rewards.update(
    {
      "rubik_task_progress": RewardTermCfg(
        func=mdp.rubik_task_progress_reward,
        weight=10.0,
      ),
      "rubik_flip_to_top": RewardTermCfg(
        func=mdp.rubik_flip_to_top_reward,
        weight=5.0,
      ),
      "rubik_u_turn_success": RewardTermCfg(
        func=mdp.rubik_u_turn_success_reward,
        weight=10.0,
      ),
      "rubik_cube_yaw_success": RewardTermCfg(
        func=mdp.rubik_cube_yaw_success_reward,
        weight=1.0,
      ),
    }
  )
  cfg.rewards["finger_collision"].weight = -0.15

  cfg.terminations.pop("flip_completed", None)
  cfg.curriculum.clear()

  for name in (
    "success_interval",
    "goal_reach_count",
    "rotation_goal_reach_count",
    "flip_success_count",
  ):
    cfg.metrics.pop(name, None)
  cfg.metrics.update(
    {
      "rubik_task_phase": MetricsTermCfg(func=mdp.rubik_phase_metric),
      "rubik_turn_task_fraction": MetricsTermCfg(
        func=mdp.rubik_turn_task_fraction,
      ),
      "rubik_yaw_task_fraction": MetricsTermCfg(func=mdp.rubik_yaw_task_fraction),
      "rubik_upright_dot": MetricsTermCfg(func=mdp.rubik_upright_dot),
      "rubik_u_face_angle_error": MetricsTermCfg(
        func=mdp.rubik_u_face_angle_error,
      ),
      "rubik_u_face_speed": MetricsTermCfg(func=mdp.rubik_u_face_speed),
      "rubik_flip_to_top_count": MetricsTermCfg(
        func=mdp.rubik_flip_to_top_count,
      ),
      "rubik_turn_success_count": MetricsTermCfg(
        func=mdp.rubik_turn_success_count,
      ),
      "rubik_yaw_success_count": MetricsTermCfg(func=mdp.rubik_yaw_success_count),
      "rubik_goal_reach_count": MetricsTermCfg(func=mdp.rubik_goal_reach_count),
      "rubik_flip_stage": MetricsTermCfg(func=mdp.rubik_flip_stage),
      "rubik_turn_stage": MetricsTermCfg(func=mdp.rubik_turn_stage),
      "rubik_side_success_rate": MetricsTermCfg(func=mdp.rubik_side_success_rate),
      "rubik_bottom_success_rate": MetricsTermCfg(func=mdp.rubik_bottom_success_rate),
      "rubik_turn90_success_rate": MetricsTermCfg(func=mdp.rubik_turn90_success_rate),
      "rubik_turn180_success_rate": MetricsTermCfg(func=mdp.rubik_turn180_success_rate),
      "rubik_yaw_success_rate": MetricsTermCfg(func=mdp.rubik_yaw_success_rate),
    }
  )
  cfg.episode_length_s = 50.0
  return cfg


def luwu_hand_rubik_smart_cube_transfer_env_cfg(
  play: bool = False, num_envs: int = 256,
  *, sensor_profile: SmartCubeProfile | None = None, plan_pool_path: str | None = None,
  play_alpha: float = .7, curriculum_cfg: SmartCubeCurriculumCfg | None = None,
):
  """204-D smart-cube actor / unchanged 470-D continuous privileged critic."""
  from dataclasses import replace
  controller_cfg = replace(curriculum_cfg or SmartCubeCurriculumCfg(), play=play, play_alpha=play_alpha, force_discrete=True)
  profile = sensor_profile or SmartCubeProfile()
  cfg = luwu_hand_rubik_full_cube_solver_transfer_env_cfg(
    play=play, num_envs=num_envs, plan_pool_path=plan_pool_path,
  )
  terms = cfg.observations['policy'].terms
  for name in ('rubik_cube_orientation', 'rubik_compat_cube_ori_error'):
    terms[name].params = {'vision_dr': True, 'smart_cube_profile': profile}
  terms['rubik_u_face_angle'].func = mdp.rubik_smart_cube_angle_sincos
  terms['rubik_u_face_angle'].params = {'smart_cube_profile': profile}
  cfg.events['reset_smart_cube_observation_state'] = EventTermCfg(
    func=reset_smart_cube_observation_state, mode='reset', params={'profile': profile},
  )
  # Intermediate facelet mismatches are diagnostic only so solver inference
  # can continue; the final move remains physically validated.
  cfg.commands['reorient_command'].verify_physical_facelets = False
  cfg.commands['reorient_command'].verify_final_physical_facelets = True
  cfg.commands['reorient_command'].smart_cube_curriculum = controller_cfg
  cfg.commands['reorient_command'].task_fractions = (.1, .8, .1)
  cfg.commands['reorient_command'].partial_turn_start_probability = .5
  cfg.commands['reorient_command'].turn_180_probability = 0.2
  cfg.commands['reorient_command'].turn_episode_seconds = 120.0
  cfg.commands['reorient_command'].turn_specialization = True
  cfg.actions['joint_pos'].smart_cube_dr = True
  cfg.observations["critic"].terms["rubik_smart_move_residual"] = ObservationTermCfg(
    func=mdp.rubik_smart_move_residual_privileged,
    history_length=0,
  )
  cfg.rewards["rubik_smart_move_residual"] = RewardTermCfg(
    func=mdp.rubik_smart_move_residual_reward,
    weight=0.5,
    params={"threshold_rad": 0.0872664626, "penalty_window_rad": 0.3490658504,
            "specialized": True, "positive_progress_only": True},
  )
  cfg.rewards["rubik_u_turn_success"].weight = 5.0
  cfg.rewards["rubik_u_turn_progress"].weight = 1.0
  cfg.rewards["rubik_turn_streak"] = RewardTermCfg(
    func=mdp.rubik_turn_streak_reward, weight=10.0,
  )
  cfg.curriculum['smart_cube_skill_outcomes'] = CurriculumTermCfg(func=record_reset_skill_failures)
  cfg.events['smart_cube_curriculum_step'] = EventTermCfg(func=update_smart_cube_curriculum, mode='step')
  cfg.events['smart_cube_face_resistance'] = EventTermCfg(
    func=randomize_smart_cube_face_resistance, mode='reset',
    params={'asset_cfg': SceneEntityCfg('object', joint_names=r'cubelet:driver:.*')},
  )
  return cfg


def luwu_hand_rubik_flip_specialist_env_cfg(
  play: bool = False, num_envs: int = 256, *, plan_pool_path: str | None = None,
):
  """Continuous clean flips; retain the deployed SmartCube actor contract."""
  from dataclasses import replace

  from wuji_mjlab.tasks.reorient.mdp.flip_specialization import (
    flip_preservation_privileged,
    flip_preservation_reward,
  )
  from wuji_mjlab.tasks.reorient.mdp.smart_cube_curriculum import (
    randomize_smart_cube_hand_friction,
  )
  cfg = luwu_hand_rubik_smart_cube_transfer_env_cfg(
    play=play, num_envs=num_envs, plan_pool_path=plan_pool_path,
  )
  if play:
    cfg.scene.num_envs = 1
  command = cfg.commands['reorient_command']
  command.smart_cube_curriculum = replace(
    command.smart_cube_curriculum, imu_noise_max_degrees=5., irregular_imu_latency=True,
  )
  cfg.events.pop('robot_friction', None)
  cfg.events['smart_cube_hand_friction'] = EventTermCfg(
    func=randomize_smart_cube_hand_friction, mode='reset',
    params={'asset_cfg': SceneEntityCfg(
      'robot', geom_names=('.*palm_.*', '.*finger.*_col'),
    )},
  )
  cfg.events['smart_cube_face_resistance'].params['scale_range'] = (.1, 1.)
  command.task_fractions = (1., 0., 0.)
  command.flip_specialization = True
  command.turn_specialization = False
  command.partial_turn_start_probability = 0.
  command.base_episode_seconds = 120.
  command.verify_final_physical_facelets = False
  for name in ('rubik_u_turn_success', 'rubik_u_turn_progress',
               'rubik_smart_move_residual', 'rubik_turn_streak',
               'rubik_solved_bonus', 'rubik_failure_penalty'):
    cfg.rewards.pop(name, None)
  cfg.rewards['rubik_flip_to_top'] = RewardTermCfg(
    func=flip_preservation_reward, weight=20., params={'component': 'arrival'},
  )
  for term in cfg.rewards.values():
    if term.func == mdp.rubik_task_progress_reward:
      term.weight = 2.
  # Repairing an intentionally introduced error must never pay more than its
  # arrival penalty; prefer a clean flip over creating and fixing deformation.
  for component, weight in (('boundary', -2.), ('improvement', 1.),
                            ('move', -2.), ('streak', 5.)):
    cfg.rewards[f'flip_align_{component}'] = RewardTermCfg(
      func=flip_preservation_reward, weight=weight,
      params={'component': component},
    )
  cfg.observations['critic'].terms['flip_preservation'] = ObservationTermCfg(
    func=flip_preservation_privileged, history_length=0,
  )
  return cfg


def luwu_hand_rubik_flip_low_friction_env_cfg(play=False, num_envs=256, *, plan_pool_path=None):
  """Model900 flip rewards with measured-scale, step-driven transfer DR."""
  from dataclasses import replace

  from wuji_mjlab.tasks.reorient.mdp.smart_cube_curriculum import (
    randomize_realistic_friction_surfaces,
  )
  cfg = luwu_hand_rubik_flip_specialist_env_cfg(
    play=play, num_envs=num_envs, plan_pool_path=plan_pool_path,
  )
  command = cfg.commands['reorient_command']
  command.smart_cube_curriculum = replace(
    command.smart_cube_curriculum, low_friction_transition_steps=0,
    low_friction_require_bottom_contact=False,
    joint_dr_transition_steps=0,
    dr_success_gated=True,
    realistic_dr=True,
    realistic_success_gated=True,
    realistic_initial_progress=.20,
    realistic_success_threshold=.78,
    realistic_progress_increment=.05,
    imu_noise_max_degrees=.1,
  )
  cfg.actions['joint_pos'].ema_alpha = .75
  cfg.actions['joint_pos'].realistic_dr = True
  cfg.actions['joint_pos'].smart_cube_dr = False
  from wuji_mjlab.tasks.reorient.mdp.observations import (
    joint_pos_limit_normalized_with_dr,
    joint_pos_target_error_with_dr,
  )
  joint_obs = cfg.observations['policy'].terms['noisy_joint_angles']
  joint_obs.func, joint_obs.noise = joint_pos_limit_normalized_with_dr, None
  cfg.observations['policy'].terms['qpos_error'].func = joint_pos_target_error_with_dr
  cfg.events['low_friction_surfaces'] = EventTermCfg(
    func=randomize_realistic_friction_surfaces, mode='reset', params={
      'finger_cfg': SceneEntityCfg('robot', geom_names=r'.*finger.*_col'),
      'cube_cfg': SceneEntityCfg('object', geom_names=r'cubelet:.*'),
      'palm_cfg': SceneEntityCfg('robot', geom_names=r'.*palm_.*collision'),
    },
  )
  cfg.events.pop('smart_cube_hand_friction', None)
  # Keep the first transfer isolated to measured-scale friction, action bias
  # and sensor error; the legacy broad dynamics randomizers are uncalibrated.
  for name in (
    'robot_geom_size', 'contact_params_palm_thumb', 'contact_params_fingers',
    'object_size', 'object_mass', 'object_com', 'pd_gains',
    'robot_dof_damping', 'robot_dof_armature', 'robot_dof_frictionloss',
    'encoder_bias', 'robot_link_inertia', 'robot_link_mass',
  ):
    cfg.events.pop(name, None)
  cfg.sim.mujoco.timestep, cfg.decimation = .005, 10
  # Very low sliding friction destabilizes pyramidal constraints in GPU probes.
  # Elliptic avoids that regime; CG avoids the expensive elliptic Newton Hessian.
  # Keep the control period and convergence budgets unchanged.
  cfg.sim.mujoco.cone, cfg.sim.mujoco.solver = 'elliptic', 'cg'
  cfg.sim.nconmax, cfg.sim.njmax, cfg.sim.contact_sensor_maxmatch = 200, 4096, 128
  from wuji_mjlab.rl.physics_failure_guard import guarded_cage_drop
  cage = cfg.terminations['cage_drop']
  cfg.terminations['cage_drop'] = TerminationTermCfg(
    func=guarded_cage_drop, params={'original': cage.func, **cage.params},
  )
  return cfg


def configure_gentle_calibration(cfg, calibration_path):
  """Load frozen per-joint calibration; never guess a nonzero reward weight."""
  import json
  import math
  from dataclasses import replace
  from pathlib import Path

  from wuji_mjlab.tasks.reorient.mdp.gentle_flip import speed_parameter

  try:
    calibration = json.loads(Path(calibration_path).read_text())
    names = tuple(calibration['joint_names'])
    reference = speed_parameter(calibration['v_ref'], 16, positive=True, name='v_ref')
    soft = speed_parameter(calibration['v_soft'], 16, positive=False, name='v_soft')
    coefficient = float(calibration['lambda_target'])
    initial = float(calibration['lambda_initial'])
    approved = float(calibration.get('approved_max_penalty', 1.))
    if (calibration.get('schema_version') != 1 or len(names) != 16
        or len(set(names)) != 16 or not math.isfinite(coefficient)
        or not math.isfinite(initial) or not 0 < initial <= coefficient
        or not math.isfinite(approved) or not 1. <= approved <= 2.5):
      raise ValueError('Invalid calibration values')
  except (OSError, KeyError, TypeError, ValueError) as exc:
    raise ValueError(f'Invalid gentle flip calibration: {calibration_path}') from exc
  params = cfg.rewards['gentle_joint_velocity'].params
  params.update(v_ref=reference, v_soft=soft, coefficient=coefficient, joint_names=names)
  event = cfg.events['gentle_flip_schedule'].params
  event['coefficient'] = coefficient
  # Only the initial ~1-point penalty is approved by default. Raising the cap
  # requires a separate fixed-policy evaluation, not success rate alone.
  event['schedule'] = replace(event['schedule'], speed_increment=initial / coefficient,
    max_speed_scale=min(1., approved / 2.5))


def luwu_hand_rubik_flip_gentle_env_cfg(
  play=False, num_envs=256, *, plan_pool_path=None, calibration_path=None,
):
  """Full source DR, gradual stronger EMA, calibrated soft velocity penalty."""
  from dataclasses import replace

  from wuji_mjlab.tasks.reorient.mdp.gentle_flip import (
    GentleFlipSchedule,
    update_gentle_flip,
  )
  cfg = luwu_hand_rubik_flip_low_friction_env_cfg(
    play=play, num_envs=num_envs, plan_pool_path=plan_pool_path)
  command = cfg.commands['reorient_command']
  command.smart_cube_curriculum = replace(
    command.smart_cube_curriculum, realistic_initial_progress=1., play_alpha=0.)
  cfg.actions['joint_pos'].ema_alpha = .5 if play else .75
  cfg.rewards['gentle_joint_velocity'] = RewardTermCfg(
    func=mdp.gentle_joint_velocity_penalty, weight=-1.,
    params={'v_ref': 1., 'v_soft': 1., 'coefficient': 0., 'joint_names': ()})
  cfg.events['gentle_flip_schedule'] = EventTermCfg(
    func=update_gentle_flip, mode='step',
    params={'schedule': GentleFlipSchedule(frozen=play, max_speed_scale=0.), 'coefficient': 0.})
  if calibration_path is not None:
    configure_gentle_calibration(cfg, calibration_path)
  return cfg


def luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(play=False, num_envs=256, *, plan_pool_path=None):
  """Instrumented source-policy baseline; reward transition awaits contact probes."""
  from mjlab.sensor import ContactSensorCfg

  from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip_runtime import (
    bottom_tip_metric,
    sample_bottom_tip,
  )
  cfg = luwu_hand_rubik_flip_specialist_env_cfg(play=play,num_envs=num_envs,plan_pool_path=plan_pool_path)
  cfg.commands['reorient_command'].bottom_tip_observer = True
  sensors = list(cfg.scene.sensors)
  for i in range(1,5):
    sensors.append(ContactSensorCfg(
      name=f'bottom_tip_{i}',
      primary=ContactMatch(mode='geom',entity='robot',pattern=f'right_finger{i}_link4_cap[0-3]_col'),
      secondary=ContactMatch(mode='subtree',entity='object',pattern='middle'),
      secondary_policy='any', fields=('found','force','pos','normal'),
      reduce='none',num_slots=16,
    ))
  cfg.scene.sensors = tuple(sensors)
  cfg.metrics['bottom_tip_substep'] = MetricsTermCfg(func=sample_bottom_tip,per_substep=True)
  for key in ('requests','successes','contacted_requests','supported_successes',
              'success_rate','success_after_bottom_contact','bottom_contact_given_success',
              'supported_success_rate','geometry_valid','contact_overflow'):
    cfg.metrics[f'bottom_tip/{key}'] = MetricsTermCfg(
      func=bottom_tip_metric,params={'key':key},reduce='last')
  return cfg


def luwu_hand_rubik_flip_bottom_tip_train_env_cfg(play=False, num_envs=256, *, plan_pool_path=None):
  """Fixed -1/30 success rewards with a first40 contact window per90."""
  from dataclasses import replace

  from wuji_mjlab.tasks.reorient.mdp.flip_bottom_tip_runtime import (
    bottom_tip_budget_reward,
    bottom_tip_goal_reward,
    bottom_tip_privileged,
    bottom_tip_stability_reward,
  )
  from wuji_mjlab.tasks.reorient.mdp.smart_cube_curriculum import (
    randomize_low_friction_surfaces,
  )
  cfg = luwu_hand_rubik_flip_bottom_tip_transfer_env_cfg(
    play=play, num_envs=num_envs, plan_pool_path=plan_pool_path,
  )
  cfg.commands['reorient_command'].bottom_tip_early_per_leg = True
  cfg.commands['reorient_command'].bottom_tip_compact_contact = True
  cfg.commands['reorient_command'].bottom_tip_debug_diagnostics = False
  cfg.commands['reorient_command'].bottom_tip_contact_window_deg = 40.
  cfg.commands['reorient_command'].bottom_tip_success_rewards = (-1.,30.)
  command = cfg.commands['reorient_command']
  command.smart_cube_curriculum = replace(
    command.smart_cube_curriculum, low_friction_transition_steps=20000,
  )
  # Runs after legacy hand DR: anneal fingers/cube, halve legacy palm friction.
  cfg.events['low_friction_surfaces'] = EventTermCfg(
    func=randomize_low_friction_surfaces, mode='reset', params={
      'finger_cfg': SceneEntityCfg('robot', geom_names=r'.*finger.*_col'),
      'cube_cfg': SceneEntityCfg('object', geom_names=r'cubelet:.*'),
      'palm_cfg': SceneEntityCfg('robot', geom_names=r'.*palm_.*collision'),
    },
  )
  cfg.rewards.pop('rubik_flip_to_top')
  cfg.rewards.pop('flip_align_streak')
  # Keep shape repair secondary to flipping; do not weaken deformation penalties.
  cfg.rewards['flip_align_improvement'].weight = .25
  cfg.rewards['bottom_tip_goal'] = RewardTermCfg(func=bottom_tip_goal_reward, weight=1.)
  for name, limit in (('rubik_task_progress', 2.), ('approach_progress', .25),
                      ('contact_milestones', .25)):
    term = cfg.rewards[name]
    cfg.rewards[name] = RewardTermCfg(func=bottom_tip_budget_reward, weight=1., params={
      **term.params, 'original': term.func, 'original_weight': term.weight,
      'budget_name': name, 'limit': limit,
    })
  cfg.rewards['rubik_cube_position_stability'].func = bottom_tip_stability_reward
  cfg.observations['critic'].terms['bottom_tip_request'] = ObservationTermCfg(
    func=bottom_tip_privileged, history_length=0,
  )
  from wuji_mjlab.rl.physics_failure_guard import (
    guarded_cage_drop,
    guarded_reward,
    physics_failure,
    physics_failure_penalty,
  )
  # Paired model450 GPU probe supports5ms as the faster training compromise;
  # keep the deployed50ms control period and the30ms contact requirement.
  cfg.sim.mujoco.timestep = .005
  cfg.decimation = 10
  cage = cfg.terminations['cage_drop']
  cfg.terminations['cage_drop'] = TerminationTermCfg(
    func=guarded_cage_drop, params={'original': cage.func, **cage.params},
  )
  cfg.terminations['physics_failure'] = TerminationTermCfg(func=physics_failure)
  for term in cfg.rewards.values():
    if isinstance(term.func, type):
      continue  # ManagerTerm classes must be instantiated by RewardManager first.
    term.params = {'reward_func': term.func, **term.params}
    term.func = guarded_reward
  cfg.rewards['physics_failure'] = RewardTermCfg(
    func=physics_failure_penalty, weight=-10.,
  )
  return cfg


def luwu_hand_rubik_u_face_transfer_env_cfg(
  play: bool = False,
  num_envs: int = 256,
):
  """Build a Rubik environment with the successful cube actor input as prefix."""
  cfg = luwu_hand_rubik_u_face_env_cfg(play=play, num_envs=num_envs)
  # Transfer-specific balance: strengthen all three task completions and
  # soften the self-collision cost while adapting the pretrained actor.
  cfg.rewards["rubik_flip_to_top"].weight = 10.0
  cfg.rewards["rubik_u_turn_success"].weight = 20.0
  cfg.rewards["rubik_cube_yaw_success"].weight = 3.0
  cfg.rewards["finger_collision"].weight = -0.07

  terms = cfg.observations["policy"].terms
  legacy_joint_angles = terms["noisy_joint_angles"]
  legacy_qpos_error = terms["qpos_error"]
  legacy_action_history = terms["action_history"]
  compat_orientation = ObservationTermCfg(
    func=mdp.rubik_compat_cube_orientation_error_6d,
    params={"vision_dr": True},
    history_length=3,
  )
  extra_order = (
    "rubik_cube_orientation",
    "vision_valid",
    "vision_age",
    "vision_confidence",
    "vision_innovation",
    "rubik_u_face_angle",
    "rubik_u_face_goal",
    "rubik_face_goal",
    "rubik_root_goal",
    "rubik_task_phase",
  )
  missing = [name for name in extra_order if name not in terms]
  if missing:
    raise ValueError(f"Rubik transfer observation terms are missing: {missing}")
  extras = {name: terms[name] for name in extra_order}
  terms.clear()
  terms.update(
    {
      "noisy_joint_angles": legacy_joint_angles,
      "qpos_error": legacy_qpos_error,
      "rubik_compat_cube_ori_error": compat_orientation,
      "action_history": legacy_action_history,
      **extras,
    }
  )
  return cfg


def luwu_hand_rubik_u_face_continuous_transfer_env_cfg(
  play: bool = False,
  num_envs: int = 256,
):
  """Initialize Rubik control from the six-face continuous cube actor."""
  cfg = luwu_hand_rubik_u_face_transfer_env_cfg(play=play, num_envs=num_envs)
  cfg.rewards["rubik_u_turn_progress"] = RewardTermCfg(
    func=mdp.rubik_u_turn_progress_reward,
    weight=5.0,
  )
  return cfg


def luwu_hand_rubik_full_cube_solver_transfer_env_cfg(
  play: bool = False,
  num_envs: int = 256,
  plan_pool_path: str | None = None,
):
  """Full-unlocked mixed flip/turn/Node-solver transfer task."""
  import os
  from dataclasses import replace
  from pathlib import Path

  cfg = luwu_hand_rubik_u_face_continuous_transfer_env_cfg(play=play, num_envs=num_envs)
  if play:
    # Keep all three task groups represented in the native viewer.
    cfg.scene.num_envs = 4
  if play:
    cfg.scene.entities["target"] = get_rubik_target_cfg(full_cube=True)
    x, y, z = RUBIK_INIT_STATE.pos
    cfg.scene.entities["target"].init_state = replace(
      cfg.scene.entities["target"].init_state,
      pos=(x + 0.18, y, z + 0.04),
    )
  else:
    cfg.scene.entities.pop("target", None)
  cfg.scene.entities["object"] = get_rubik_cfg(full_cube=True, dynamic_layers=False)
  cfg.scene.entities["object"].init_state = RUBIK_INIT_STATE

  default_pool = (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "objects"
    / "rubik"
    / "full_cube_solver_plan_pool.json"
  )
  pool_path = plan_pool_path or os.environ.get(
    "LUWU_RUBIK_SOLVER_PLAN_POOL", str(default_pool)
  )
  cfg.commands["reorient_command"] = RubikFullCubeCommandCfg(
    entity_name="object",
    target_entity_name="target" if play else "",
    target_fixed_pose=play,
    resampling_time_range=(1.0e6, 1.0e6),
    upright_tolerance_rad=0.20,
    upright_hold_steps=8,
    flip_angle_tolerance_rad=0.1745329252,
    turn_tolerance_rad=0.0872664626,
    turn_hold_steps=8,
    max_u_face_speed=0.45,
    solver_mode=True,
    plan_pool_path=pool_path,
  )
  cfg.events["reset_object_pose"] = EventTermCfg(
    func=mdp.reset_rubik_full_cube,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("object"),
      "robot_cfg": SceneEntityCfg("robot", body_names=("right_palm_link",)),
      "palm_clearance_m": 0.0005,
    },
  )

  cfg.observations["critic"].terms["rubik_internal_state_privileged"] = (
    ObservationTermCfg(
      func=mdp.rubik_full_cube_internal_privileged,
      history_length=0,
    )
  )
  cfg.observations["critic"].terms["rubik_cube_position_stability"] = (
    ObservationTermCfg(
      func=mdp.rubik_cube_position_stability_privileged,
      history_length=0,
    )
  )
  cfg.rewards["rubik_cube_position_stability"] = RewardTermCfg(
    func=mdp.rubik_cube_position_stability_reward,
    weight=1.0,
    params={
      "task_kind": mdp.FULL_CUBE_FLIP,
      "stable_delta_m": 5.0e-3,
      "unstable_delta_m": 1.0e-2,
    },
  )
  cfg.rewards["rubik_flip_to_top"].weight = 20.0
  cfg.rewards["rubik_u_turn_progress"] = RewardTermCfg(
    func=mdp.rubik_u_turn_progress_reward,
    weight=5.0,
  )
  cfg.rewards["rubik_solved_bonus"] = RewardTermCfg(
    func=mdp.rubik_full_cube_solved_reward,
    weight=200.0,
  )
  cfg.rewards["rubik_failure_penalty"] = RewardTermCfg(
    func=mdp.rubik_full_cube_failure_reward,
    weight=-10.0,
  )
  cfg.rewards.pop("rubik_cube_yaw_success", None)
  cfg.terminations["rubik_full_cube_solved"] = TerminationTermCfg(
    func=mdp.rubik_full_cube_solved
  )
  cfg.terminations["rubik_full_cube_failed"] = TerminationTermCfg(
    func=mdp.rubik_full_cube_failed
  )
  cfg.terminations["rubik_full_cube_timeout"] = TerminationTermCfg(
    func=mdp.rubik_full_cube_timeout,
    time_out=True,
  )
  cfg.terminations["rubik_full_cube_stage_timeout"] = TerminationTermCfg(
    func=mdp.rubik_full_cube_stage_timeout
  )
  cfg.metrics.update(
    {
      "rubik_flip_task_fraction": MetricsTermCfg(
        func=mdp.rubik_task_fraction_metric,
        params={"task_kind": mdp.FULL_CUBE_FLIP},
      ),
      "rubik_turn_task_fraction": MetricsTermCfg(
        func=mdp.rubik_task_fraction_metric,
        params={"task_kind": mdp.FULL_CUBE_TURN},
      ),
      "rubik_solve_task_fraction": MetricsTermCfg(
        func=mdp.rubik_task_fraction_metric,
        params={"task_kind": mdp.FULL_CUBE_SOLVE},
      ),
      "rubik_solver_plan_length": MetricsTermCfg(
        func=mdp.rubik_solver_plan_length_metric
      ),
      "rubik_solver_moves_completed": MetricsTermCfg(
        func=mdp.rubik_solver_moves_completed_metric, reduce="last"
      ),
      "rubik_solver_attempt": MetricsTermCfg(
        func=mdp.rubik_solver_attempt_metric, reduce="last"
      ),
      "rubik_solver_success": MetricsTermCfg(
        func=mdp.rubik_solver_success_metric, reduce="last"
      ),
      "rubik_failure_reason": MetricsTermCfg(
        func=mdp.rubik_failure_reason_metric, reduce="last"
      ),
      "rubik_facelet_check_mismatch": MetricsTermCfg(
        func=mdp.rubik_facelet_check_mismatch_metric, reduce="last"
      ),
      "rubik_failure_phase": MetricsTermCfg(
        func=mdp.rubik_failure_phase_metric, reduce="last"
      ),
      "rubik_failure_move": MetricsTermCfg(
        func=mdp.rubik_failure_move_metric, reduce="last"
      ),
      "rubik_solved": MetricsTermCfg(func=mdp.rubik_solved_metric, reduce="last"),
      "rubik_internal_error_rad": MetricsTermCfg(func=mdp.rubik_internal_error_metric),
    }
  )
  for phase_name, phase in (("flip", 0), ("turn", 1)):
    for face_id, face in enumerate(("R", "L", "B", "F", "U", "D")):
      for kind, success in (("attempts", False), ("successes", True)):
        name = f"rubik_{phase_name}_{face}_{kind}"
        cfg.metrics[name] = MetricsTermCfg(
          func=mdp.rubik_goal_count_metric,
          params={"face_id": face_id, "phase": phase, "success": success},
          reduce="last",
        )
  for name in (
    "rubik_yaw_task_fraction",
    "rubik_flip_stage",
    "rubik_turn_stage",
    "rubik_side_success_rate",
    "rubik_bottom_success_rate",
    "rubik_turn90_success_rate",
    "rubik_turn180_success_rate",
    "rubik_yaw_success_rate",
  ):
    cfg.metrics.pop(name, None)
  cfg.episode_length_s = 600.0
  cfg.sim.nconmax = max(cfg.sim.nconmax, 600)
  cfg.sim.njmax = max(cfg.sim.njmax, 6000)
  return cfg
