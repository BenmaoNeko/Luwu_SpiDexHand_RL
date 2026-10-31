"""Full-cube resets must use the current palm, with no inherited arm offset."""
import torch
from wuji_mjlab.tasks.reorient.mdp.rubik_full_cube import palm_centered_cube_position


def test_reset_centers_cube_on_horizontal_palm_without_drop_gap():
  palm = torch.tensor([
    [-0.0177281978, -0.000000374254, 0.281801055, 1., 0., 0., 0.],
    [2., 3., 0.5, 1., 0., 0., 0.],
  ], dtype=torch.float64)
  position = palm_centered_cube_position(palm)
  torch.testing.assert_close(position[:, :2], palm[:, :2])
  # Cylinder top Z=96 mm plus a 0.5 mm fixed support pad.
  # Cube half extent: 28.55 mm, measured from transformed mesh vertices.
  bottom = position[:, 2] - 0.02855
  torch.testing.assert_close(bottom, palm[:, 2] + 0.0965)


def test_support_offset_rotates_with_palm_and_respects_explicit_clearance():
  q = 2.**-0.5
  palm = torch.tensor([[1., 2., 3., q, 0., q, 0.]], dtype=torch.float64)
  position = palm_centered_cube_position(palm, palm_clearance_m=0.001)
  torch.testing.assert_close(position, torch.tensor([[1.12605, 2., 3.]], dtype=torch.float64))


def test_multi_environment_reset_uses_updated_mocap_palm():
  """Opt-in CUDA regression for stale palm poses on the very first reset."""
  import os

  import pytest

  if os.environ.get("LUWU_RUN_GPU_ASSET_TESTS") != "1":
    pytest.skip("Set LUWU_RUN_GPU_ASSET_TESTS=1 for the CUDA reset regression")
  from mjlab.envs import ManagerBasedRlEnv
  from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs

  cfg, _ = prepare_task_cfgs("LuwuHand_Rubik_FlipSpecialist", [], play=False)
  cfg.scene.num_envs = 64
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 2048
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    for _ in range(3):
      obs, _ = env.reset()
      robot = env.scene["robot"]
      ids, _ = robot.find_bodies(("right_palm_link",))
      palm = robot.data.body_link_pose_w[:, ids[0]]
      cube = env.scene["object"].data.root_link_pose_w[:, :3]
      clearance = cfg.events["reset_object_pose"].params["palm_clearance_m"]
      expected = palm_centered_cube_position(palm, clearance)
      torch.testing.assert_close(cube, expected, atol=1e-6, rtol=0)
      assert obs["policy"].shape == (64, 204)
      assert obs["critic"].shape == (64, 481)
      assert env.scene["tip_object_contact"].data.found.shape == (64, 4)
  finally:
    env.close()
