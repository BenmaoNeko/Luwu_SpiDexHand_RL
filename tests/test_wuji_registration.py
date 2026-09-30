from pathlib import Path

import mujoco
import numpy as np

from tools.convex_preview import HAND_JOINT_NAMES, parse_urdf
from tools.build_wuji_asset import build_wuji_robot_asset


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCENE = ROOT / "generated" / "luwu_arm_hand_002_convex" / "scene.xml"
SOURCE_URDF = next(
  (ROOT / "URDF").rglob("机械臂-灵巧爪-总装v3.0-完整版URDF.SLDASM.urdf")
)
CUBE_XML = (
    ROOT
    / "reference"
    / "wuji-mjlab-main"
    / "src"
    / "wuji_mjlab"
    / "assets"
    / "objects"
    / "inhand_object"
    / "xmls"
    / "cube.xml"
)
LUWU_TASK_INIT = (
    ROOT
    / "reference"
    / "wuji-mjlab-main"
    / "src"
    / "wuji_mjlab"
    / "tasks"
    / "reorient"
    / "config"
    / "luwu_hand"
    / "__init__.py"
)
LUWU_ENV_CFG = LUWU_TASK_INIT.parent / "env_cfgs.py"
ROBOT_CFGS = (
    ROOT
    / "reference"
    / "wuji-mjlab-main"
    / "src"
    / "wuji_mjlab"
    / "assets"
    / "robots"
    / "luwu_hand"
    / "luwu_hand_cfg.py",
    ROOT
    / "reference"
    / "wuji-mjlab-main"
    / "src"
    / "wuji_mjlab"
    / "assets"
    / "robots"
    / "wuji_hand"
    / "wuji_hand_cfg.py",
)


def test_wuji_robot_asset_is_fixed_four_finger_hand(tmp_path):
    xml_path = build_wuji_robot_asset(SOURCE_SCENE, tmp_path / "luwu_hand")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    joint_names = [model.joint(i).name for i in range(model.njnt)]
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    site_names = [model.site(i).name for i in range(model.nsite)]

    assert model.njnt == 16
    assert model.nu == 16
    assert model.nsite == 4
    assert "preview_cube" not in body_names
    assert "right_palm_link" in body_names
    assert site_names == [f"right_finger{i}_tip" for i in range(1, 5)]
    assert joint_names == [
        f"right_finger{finger}_joint{joint}"
        for finger in range(1, 5)
        for joint in range(1, 5)
    ]
    assert all("joint7" not in name and "joint6" not in name for name in actuator_names)
    assert np.allclose(model.actuator_forcerange, (-0.5, 0.5))


def test_wuji_asset_preserves_urdf_joint_limits(tmp_path):
    xml_path = build_wuji_robot_asset(SOURCE_SCENE, tmp_path / "luwu_hand")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    urdf = parse_urdf(SOURCE_URDF)
    limits = {
        joint.name: (joint.lower, joint.upper)
        for joint in urdf.joints
        if joint.name in HAND_JOINT_NAMES
    }

    for index, joint_name in enumerate(HAND_JOINT_NAMES):
        finger = index // 4 + 1
        joint = index % 4 + 1
        model_joint = model.joint(f"right_finger{finger}_joint{joint}")
        assert np.allclose(model.jnt_range[model_joint.id], limits[joint_name])


def test_wuji_asset_has_no_arm_actuators_or_free_base(tmp_path):
    xml_path = build_wuji_robot_asset(SOURCE_SCENE, tmp_path / "luwu_hand")
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    assert model.nq == 16
    assert model.nv == 16
    assert all(model.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE for i in range(model.njnt))
    assert all(
        model.actuator(i).trnid[0] == model.joint(i).id
        for i in range(model.nu)
    )


def test_wuji_asset_composes_with_dynamic_cube(tmp_path):
    xml_path = build_wuji_robot_asset(SOURCE_SCENE, tmp_path / "luwu_hand")
    robot_spec = mujoco.MjSpec.from_file(str(xml_path))
    cube_spec = mujoco.MjSpec.from_file(str(CUBE_XML))
    scene = mujoco.MjSpec()
    scene.attach(robot_spec, prefix="robot/", frame=scene.worldbody.add_frame())
    scene.attach(cube_spec, prefix="object/", frame=scene.worldbody.add_frame())
    model = scene.compile()
    data = mujoco.MjData(model)

    for index in range(16):
        joint_name = f"robot/right_finger{index // 4 + 1}_joint{index % 4 + 1}"
        joint_id = model.joint(joint_name).id
        data.qpos[model.jnt_qposadr[joint_id]] = 0.610865238 if index % 4 == 0 else 0.0
    cube_joint_id = model.joint("object/cube_freejoint").id
    cube_qpos = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos : cube_qpos + 7] = (
        -0.0264848718,
        8.59749199e-08,
        0.40649396,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    mujoco.mj_forward(model, data)

    assert model.nq == 23  # 16 hand joints + cube freejoint
    assert model.nu == 16
    assert model.joint("object/cube_freejoint").type == mujoco.mjtJoint.mjJNT_FREE
    assert data.ncon == 0

    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()


def test_luwu_task_registration_contract_is_present():
    registration = LUWU_TASK_INIT.read_text(encoding="utf-8")
    env_cfg = LUWU_ENV_CFG.read_text(encoding="utf-8")

    assert 'task_id="LuwuHand_Reorient"' in registration
    assert 'task_id="LuwuHand_Reorient_Light"' in registration
    assert "get_luwu_hand_cfg()" in env_cfg
    assert "LUWU_TIP_SITE_NAMES" in env_cfg
    assert "LUWU_TIP_BODY_NAMES" in env_cfg
    assert 'cfg.events.pop("reset_robot_pose", None)' in env_cfg
    assert 'cfg.events.pop("reset_robot_joints", None)' in env_cfg
    assert 'cfg.events.pop("contact_params_palm_thumb", None)' in env_cfg
    assert 'cfg.events.pop("robot_geom_size", None)' in env_cfg


def test_robot_configs_support_mjlab_12_and_13_actuator_names():
    for cfg_path in ROBOT_CFGS:
        source = cfg_path.read_text(encoding="utf-8")
        assert "try:" in source
        assert "from mjlab.actuator import XmlActuatorCfg" in source
        assert "XmlPositionActuatorCfg as XmlActuatorCfg" in source
