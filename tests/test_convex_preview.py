from pathlib import Path

import mujoco
import numpy as np

from tools.convex_preview import (
    ARM_FIXED_Q,
    HAND_HOME_Q,
    HAND_JOINT_NAMES,
    build_mjcf,
    generate_hulls,
    parse_urdf,
    render_preview,
    reset_to_home,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF = ROOT / "URDF/open_arm_002/luwu_arm_hand_002/机械臂-灵巧爪-总装v3.0-完整版URDF.SLDASM/机械臂-灵巧爪-总装v3.0-完整版URDF.SLDASM.urdf"


def test_fixed_arm_pose_is_home_with_j4_at_upper_limit():
    assert np.allclose(ARM_FIXED_Q, [0.0, 0.0, 0.0, 1.5, 0.0, 0.0])


def test_source_has_all_current_links_and_hulls(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    assert len(robot.links) == 23

    manifest = generate_hulls(robot, tmp_path)

    assert len(manifest["hulls"]) == len(robot.mesh_links)
    assert all(Path(item["path"]).exists() for item in manifest["hulls"])
    assert all(item["vertices"] >= 4 for item in manifest["hulls"])
    assert all(item["extents"][i] > 0 for item in manifest["hulls"] for i in range(3))


def test_mjcf_loads_with_fixed_arm_and_sixteen_hand_joints(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    generate_hulls(robot, tmp_path)
    xml_path = build_mjcf(robot, tmp_path, ARM_FIXED_Q)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    joint_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)}

    assert joint_names == {f"joint{i}" for i in range(7, 23)}
    assert model.nu == 16

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    link6_z = data.xmat[link6_id].reshape(3, 3)[:, 2]
    assert float(np.dot(link6_z, [0.0, 0.0, 1.0])) > 0.99


def test_mjcf_has_sixteen_torque_limited_position_actuators(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    generate_hulls(robot, tmp_path)
    xml_path = build_mjcf(robot, tmp_path, ARM_FIXED_Q)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nu == 16

    actuator_joint_names = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joint_names.append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        )
        assert np.allclose(model.actuator_forcerange[actuator_id], [-0.5, 0.5])
        assert np.isclose(model.actuator_gainprm[actuator_id, 0], 4.0)
        assert np.isclose(model.actuator_biasprm[actuator_id, 2], -0.15)

    assert actuator_joint_names == [f"joint{i}" for i in range(7, 23)]

    urdf_limits = {
        joint.name: (joint.lower, joint.upper)
        for joint in robot.joints
        if joint.name in HAND_JOINT_NAMES
    }
    for actuator_id, joint_name in enumerate(actuator_joint_names):
        assert np.allclose(model.actuator_ctrlrange[actuator_id], urdf_limits[joint_name])


def test_home_keyframe_uses_calibrated_cross_pose(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    generate_hulls(robot, tmp_path)
    xml_path = build_mjcf(robot, tmp_path, ARM_FIXED_Q)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert home_id >= 0
    assert np.allclose(model.key_qpos[home_id], HAND_HOME_Q)
    assert np.allclose(model.key_ctrl[home_id], HAND_HOME_Q)

    data = mujoco.MjData(model)
    reset_to_home(model, data)
    assert np.allclose(data.qpos, HAND_HOME_Q)
    assert np.allclose(data.ctrl, HAND_HOME_Q)


def test_home_preview_excludes_structural_self_contacts(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    generate_hulls(robot, tmp_path)
    xml_path = build_mjcf(robot, tmp_path, ARM_FIXED_Q)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert data.ncon == 0


def test_render_preview_writes_png(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    generate_hulls(robot, tmp_path)
    xml_path = build_mjcf(robot, tmp_path, ARM_FIXED_Q)

    image_path = render_preview(xml_path, tmp_path / "preview.png")

    assert image_path.exists()
    assert image_path.stat().st_size > 0


def test_render_collision_only_preview_writes_png(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    generate_hulls(robot, tmp_path)
    xml_path = build_mjcf(robot, tmp_path, ARM_FIXED_Q)

    image_path = render_preview(xml_path, tmp_path / "collision_preview.png", collision_only=True)

    assert image_path.exists()
    assert image_path.stat().st_size > 0
