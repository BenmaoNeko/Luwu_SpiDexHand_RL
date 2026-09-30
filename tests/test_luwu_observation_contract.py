import ast
from pathlib import Path


REORIENT_TERMS = (
  Path(__file__).resolve().parents[1]
  / "reference"
  / "wuji-mjlab-main"
  / "src"
  / "wuji_mjlab"
  / "tasks"
  / "reorient"
  / "reorient_terms.py"
)
LUWU_ENV_CFG = (
  Path(__file__).resolve().parents[1]
  / "reference"
  / "wuji-mjlab-main"
  / "src"
  / "wuji_mjlab"
  / "tasks"
  / "reorient"
  / "config"
  / "luwu_hand"
  / "env_cfgs.py"
)
REAL_HAND_ENV_CFG = (
  Path(__file__).resolve().parents[1]
  / "reference"
  / "wuji-mjlab-main"
  / "deploy"
  / "reorient"
  / "lib"
  / "real_hand_env_cfg.py"
)
PLAY_REAL = (
  Path(__file__).resolve().parents[1]
  / "reference"
  / "wuji-mjlab-main"
  / "deploy"
  / "reorient"
  / "scripts"
  / "play_real.py"
)


def _observation_term_names(source: str, assignment_name: str) -> set[str]:
  tree = ast.parse(source)
  for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
      continue
    if not any(
      isinstance(target, ast.Name) and target.id == assignment_name
      for target in node.targets
    ):
      continue
    assert isinstance(node.value, ast.Dict)
    return {
      key.value
      for key in node.value.keys
      if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
  raise AssertionError(f"assignment {assignment_name!r} not found")


def test_actor_does_not_require_cube_position_but_critic_keeps_privileged_copy():
  source = REORIENT_TERMS.read_text(encoding="utf-8")
  policy_terms = _observation_term_names(source, "policy_terms")
  critic_terms = _observation_term_names(source, "critic_terms")

  assert "include_cube_position_in_policy: bool = True" in source
  assert 'policy_terms["cube_pos_in_tag"]' in source
  assert '"cube_ori_error"' in source
  assert '"action_history"' in source
  assert "cube_pos_in_tag" in critic_terms
  assert "cube_pos_in_tag_clean" in critic_terms
  assert "include_cube_position_in_policy=False" in LUWU_ENV_CFG.read_text(encoding="utf-8")


def test_deploy_observation_swap_keeps_legacy_wuji_and_orientation_paths():
  source = REAL_HAND_ENV_CFG.read_text(encoding="utf-8")

  assert "goal_rot_err_6d_from_zmq" in source
  assert "cube_pos_in_tag_from_zmq" in source


def test_real_runner_does_not_hard_code_old_policy_observation_size():
  source = PLAY_REAL.read_text(encoding="utf-8")

  assert "policy.input_dim == 207" not in source
  assert "env.action_manager.total_action_dim" in source
