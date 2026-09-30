# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wuji_mjlab.tasks.reorient.mdp.curriculums import get_curriculum_value
from wuji_mjlab.tasks.reorient.mdp.event_impl.state import get_reorient_event_state
from wuji_mjlab.tasks.reorient.mdp.event_utils import resolve_env_ids

# `rl_cfg` is imported lazily here to avoid importing the full task config
# graph whenever the public facade module is imported from tests.
_TOTAL_TRAINING_STEPS_CACHE: int | None = None


def _get_total_training_steps() -> int:
  global _TOTAL_TRAINING_STEPS_CACHE
  if _TOTAL_TRAINING_STEPS_CACHE is None:
    from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
      wuji_hand_reorient_ppo_runner_cfg,
    )

    runner_cfg = wuji_hand_reorient_ppo_runner_cfg()
    _TOTAL_TRAINING_STEPS_CACHE = int(
      runner_cfg.max_iterations * runner_cfg.num_steps_per_env
    )
  assert _TOTAL_TRAINING_STEPS_CACHE is not None
  return _TOTAL_TRAINING_STEPS_CACHE


@requires_model_fields("geom_friction")
def apply_friction_curriculum(
  env,
  env_ids,
  curriculum_term: str = "friction_curriculum",
  robot_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", geom_names=(".*palm_.*", ".*finger.*_col")
  ),
  object_cfg: SceneEntityCfg = SceneEntityCfg("object", geom_names=("cube",)),
) -> None:
  """Apply friction curriculum: scale per-env startup DR friction by curriculum."""
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  friction_scale = get_curriculum_value(env, curriculum_term, 1.0)
  state = get_reorient_event_state(env)

  if not state.friction_dr_snapshot:
    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    model_friction = env.sim.model.geom_friction
    for asset_cfg in (robot_cfg, object_cfg):
      asset = env.scene[asset_cfg.name]
      geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].long()
      env_grid, geom_grid = torch.meshgrid(all_env_ids, geom_ids, indexing="ij")
      state.friction_dr_snapshot[asset_cfg.name] = model_friction[
        env_grid, geom_grid, 0
      ].clone()

  model_friction = env.sim.model.geom_friction
  for asset_cfg in (robot_cfg, object_cfg):
    asset = env.scene[asset_cfg.name]
    geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].long()
    snapshot = state.friction_dr_snapshot[asset_cfg.name][env_ids]
    new_sliding = snapshot * friction_scale

    env_grid, geom_grid = torch.meshgrid(env_ids.long(), geom_ids, indexing="ij")
    model_friction[env_grid, geom_grid, 0] = new_sliding


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def apply_geom_size_curriculum(
  env,
  env_ids,
  curriculum_term: str = "geom_size_curriculum",
  robot_cfg: SceneEntityCfg = SceneEntityCfg(
    "robot", geom_names=(".*palm_.*", ".*finger.*_col")
  ),
  dr_range: tuple[float, float] | None = None,
) -> None:
  """Apply geom size curriculum on top of startup-randomized geom sizes."""
  del dr_range

  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  geom_scale = get_curriculum_value(env, curriculum_term, 1.0)
  asset = env.scene[robot_cfg.name]
  geom_ids = asset.indexing.geom_ids[robot_cfg.geom_ids].long()
  state = get_reorient_event_state(env)

  if state.geom_size_dr_snapshot is None:
    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    env_grid, geom_grid = torch.meshgrid(all_env_ids, geom_ids, indexing="ij")
    state.geom_size_dr_snapshot = env.sim.model.geom_size[env_grid, geom_grid].clone()

  assert state.geom_size_dr_snapshot is not None
  snapshot = state.geom_size_dr_snapshot[env_ids]
  new_size = snapshot * geom_scale

  env_grid, geom_grid = torch.meshgrid(env_ids.long(), geom_ids.long(), indexing="ij")
  env.sim.model.geom_size[env_grid, geom_grid] = new_size
  _recompute_geom_bounds(env, env_ids.int(), robot_cfg)
