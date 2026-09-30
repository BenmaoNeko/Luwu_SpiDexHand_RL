from pathlib import Path

import mujoco
import numpy as np

from tools.render_wuji_preview import (
    CUBE_INIT_POS,
    CUBE_INIT_QUAT,
    HAND_HOME_FIRST_JOINT_RAD,
    build_preview_model,
    render_wuji_preview,
    set_visual_only_option,
)


ROOT = Path(__file__).resolve().parents[1]


def _qpos_index(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert joint_id >= 0, joint_name
    return int(model.jnt_qposadr[joint_id])


def test_preview_matches_training_home_and_cube_initial_state():
    model, data = build_preview_model()

    assert model.nu == 16
    assert model.nq == 23
    for finger in range(1, 5):
        assert np.isclose(
            data.qpos[_qpos_index(model, f"robot/right_finger{finger}_joint1")],
            HAND_HOME_FIRST_JOINT_RAD,
        )
        for joint in range(2, 5):
            assert np.isclose(
                data.qpos[_qpos_index(model, f"robot/right_finger{finger}_joint{joint}")],
                0.0,
            )

    cube_qpos = _qpos_index(model, "object/cube_freejoint")
    assert np.allclose(data.qpos[cube_qpos : cube_qpos + 3], CUBE_INIT_POS)
    assert np.allclose(data.qpos[cube_qpos + 3 : cube_qpos + 7], CUBE_INIT_QUAT)
    assert data.ncon == 0


def test_preview_disables_hull_collision_but_keeps_visual_stl():
    model, _ = build_preview_model()

    visual_meshes = []
    collision_geoms = []
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[geom_id] == 1:
            visual_meshes.append(geom_name)
        if geom_name.endswith("_col") or geom_name == "cube":
            collision_geoms.append(geom_id)

    assert len(visual_meshes) >= 23
    assert collision_geoms
    assert all(model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0 for i in collision_geoms)


def test_render_wuji_preview_writes_high_resolution_png(tmp_path: Path):
    image_path = render_wuji_preview(tmp_path / "luwu_wuji_preview.png", width=640, height=480)

    assert image_path.exists()
    assert image_path.stat().st_size > 0


def test_interactive_viewer_hides_convex_hull_group():
    option = mujoco.MjvOption()

    set_visual_only_option(option)

    assert np.array_equal(option.geomgroup, [0, 1, 0, 1, 0, 0])
