from pathlib import Path


CONTACT_TEST = Path(__file__).resolve().parents[1] / "tools" / "contact_test.py"


def test_contact_probe_is_a_standalone_non_ppo_entrypoint():
  assert CONTACT_TEST.exists()
  source = CONTACT_TEST.read_text(encoding="utf-8")

  assert "ManagerBasedRlEnv" in source
  assert "LuwuHand_Reorient" in source
  assert "fingertip_contact_count" in source
  assert "tip_object_contact" in source
  assert "data.ncon" in source
  assert "mj_contactForce" in source
  assert "cube" in source
  assert "PPO" not in source
  assert "checkpoint" not in source.lower()
  assert "RslRlVecEnvWrapper" not in source


def test_contact_probe_has_explicit_chain_failure_paths():
  assert CONTACT_TEST.exists()
  source = CONTACT_TEST.read_text(encoding="utf-8")

  assert "raw contact" in source.lower()
  assert "sensor" in source.lower()
  assert "metric" in source.lower()
  assert "raise ContactTestFailure" in source
  assert "return 1" in source
