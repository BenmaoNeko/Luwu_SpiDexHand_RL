"""Collision geometry in the specified Spidex asset is the hand source of truth."""

import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from wuji_mjlab.assets.robots.luwu_hand.luwu_hand_cfg import LUWU_HAND_XML
from wuji_mjlab.tasks.reorient.config.luwu_hand.env_cfgs import LUWU_TIP_COLLISION_GEOMS

SOURCE = Path(__file__).resolve().parents[5] / "URDF" / "spidex--准确" / "hand.xml"


def _body_map():
  return {
    "base_link": "right_palm_link",
    **{
      f"finger{f}_{s}": f"right_finger{f}_link{s + 1}"
      for f in range(1, 5)
      for s in range(4)
    },
  }


def test_hand_collision_volumes_match_spidex_source():
  source = ET.parse(SOURCE).getroot()
  target = ET.parse(LUWU_HAND_XML).getroot()
  expected_counts = Counter()
  for source_name, target_name in _body_map().items():
    src_body = source.find(f'.//body[@name="{source_name}"]')
    dst_body = target.find(f'.//body[@name="{target_name}"]')
    src = [g for g in src_body.findall("geom") if g.get("class") == "collision"]
    dst = [g for g in dst_body.findall("geom")
           if g.get("group") == "2" and g.get("name") != "right_palm_support_collision"]
    assert len(dst) == len(src), target_name
    for actual, expected in zip(dst, src, strict=True):
      assert actual.get("type") == expected.get("type"), actual.get("name")
      assert actual.get("mesh") is None
      for attr, default in [("pos", "0 0 0"), ("size", ""), ("quat", "1 0 0 0")]:
        np.testing.assert_allclose(
          np.fromstring(actual.get(attr, default), sep=" "),
          np.fromstring(expected.get(attr, default), sep=" "),
          atol=1e-9,
        )
      assert actual.get("contype") == "1"
      assert actual.get("conaffinity") == "1"
      expected_counts[actual.get("type")] += 1
  assert expected_counts == {"cylinder": 1, "box": 20, "capsule": 16}


def test_tip_matching_includes_every_fingertip_primitive():
  root = ET.parse(LUWU_HAND_XML).getroot()
  names = [g.get("name") for g in root.iter("geom")]
  matches = [
    n for n in names if n and any(re.fullmatch(p, n) for p in LUWU_TIP_COLLISION_GEOMS)
  ]
  assert len(matches) == 24
  assert all("link4" in n for n in matches)


def test_collision_replacement_preserves_actuators_and_joint_limits():
  model = mujoco.MjModel.from_xml_path(str(LUWU_HAND_XML))
  assert model.nu == 16
  for f in range(1, 5):
    for j in range(1, 5):
      name = f"right_finger{f}_joint{j}"
      assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0


def test_all_fingers_have_matching_joint_and_actuator_limits():
  model = mujoco.MjModel.from_xml_path(str(LUWU_HAND_XML))
  limits = ((-.8, 1.3), (-1.5, .5), (-1.5, 1.5), (-1., 1.))
  for finger in range(1, 5):
    for joint, expected in enumerate(limits, start=1):
      name = f"right_finger{finger}_joint{joint}"
      joint_id = model.joint(name).id
      actuator_id = model.actuator(f"{name}_position").id
      assert model.jnt_limited[joint_id]
      assert model.actuator_ctrllimited[actuator_id]
      np.testing.assert_allclose(model.jnt_range[joint_id], expected)
      np.testing.assert_allclose(model.actuator_ctrlrange[actuator_id], expected)


def test_tip_contact_sensor_reduces_per_finger_body():
  from wuji_mjlab.tasks.reorient.config.luwu_hand.env_cfgs import (
    luwu_hand_reorient_env_cfg,
  )

  cfg = luwu_hand_reorient_env_cfg(num_envs=1)
  sensor = next(s for s in cfg.scene.sensors if s.name == "tip_object_contact")
  assert sensor.primary.mode == "body"
  assert sensor.primary.pattern == (r".*_finger[1-4]_link4",)


def test_palm_is_fixed_horizontal_without_arm_geometry():
  model = mujoco.MjModel.from_xml_path(str(LUWU_HAND_XML))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  palm = model.body("right_palm_link").id
  np.testing.assert_allclose(
    data.xmat[palm].reshape(3, 3)[:, 2], [0.0, 0.0, 1.0], atol=1e-12
  )
  assert model.body_dofnum[palm] == 0
  body_names = {model.body(i).name for i in range(model.nbody)}
  assert body_names == {
    "world", "base_link", "right_palm_link",
    *(f"right_finger{f}_link{j}" for f in range(1, 5) for j in range(1, 5)),
  }
  collisions = [i for i in range(model.ngeom) if model.geom_group[i] == 2]
  assert len(collisions) == 38
  assert model.njnt == model.nu == 16


def test_palm_pad_is_fixed_flat_and_above_original_cylinder():
  root = ET.parse(LUWU_HAND_XML).getroot()
  palm = root.find('.//body[@name="right_palm_link"]')
  pad = palm.find('geom[@name="right_palm_support_collision"]')
  assert pad is not None
  assert pad.get("type") == "cylinder"
  np.testing.assert_allclose(np.fromstring(pad.get("size"), sep=" "), [.03, .00025])
  np.testing.assert_allclose(np.fromstring(pad.get("pos"), sep=" "), [0., 0., .09625])
  assert pad.get("density") == "0"
  assert pad.get("priority") == "1"
  np.testing.assert_allclose(np.fromstring(pad.get("solref"), sep=" "), [.04, 1.])
  assert pad.get("contype") == pad.get("conaffinity") == "1"
