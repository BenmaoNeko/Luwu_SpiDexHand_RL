"""Preview the Wuji training initial state with full-resolution URDF visuals.

The training asset keeps convex-hull collision meshes. This derived preview
scene deliberately disables those collision geoms and renders the original
URDF STL meshes instead. It is static: the viewer never advances MuJoCo time,
so the cube remains at the exact training reset pose.
"""

from __future__ import annotations

import argparse
import copy
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = (
    ROOT
    / "reference/wuji-mjlab-main/src/wuji_mjlab/assets/robots/luwu_hand/mjcf/luwu_mjlab.xml"
)
CUBE_XML = (
    ROOT
    / "reference/wuji-mjlab-main/src/wuji_mjlab/assets/objects/inhand_object/xmls/cube.xml"
)

HAND_HOME_FIRST_JOINT_RAD = 0.610865238
CUBE_INIT_POS = (-0.0264848718, 8.59749199e-08, 0.40649396)
CUBE_INIT_QUAT = (1.0, 0.0, 0.0, 0.0)


def _absolute_asset_file(value: str, *, base_dir: Path, root: ET.Element) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)

    compiler = root.find("./compiler")
    asset_dir = compiler.get("assetdir") if compiler is not None else None
    mesh_dir = compiler.get("meshdir") if compiler is not None else None
    relative_dir = asset_dir or mesh_dir or "."
    return str((base_dir / relative_dir / path).resolve())


def _make_absolute_assets(root: ET.Element, source_xml: Path) -> None:
    for asset in root.findall("./asset/*"):
        for attribute, file_value in list(asset.attrib.items()):
            if attribute.startswith("file") and file_value:
                asset.set(
                    attribute,
                    _absolute_asset_file(file_value, base_dir=source_xml.parent, root=root),
                )
    compiler = root.find("./compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("assetdir", None)


def _prefix_robot_names(root: ET.Element) -> None:
    named_elements = root.findall(".//worldbody//*[@name]")
    for element in named_elements:
        element.set("name", f"robot/{element.get('name')}")

    for element in root.findall(".//actuator/*"):
        joint = element.get("joint")
        if joint:
            element.set("joint", f"robot/{joint}")
    for element in root.findall(".//contact/exclude"):
        for attr in ("body1", "body2"):
            body = element.get(attr)
            if body:
                element.set(attr, f"robot/{body}")


def _prefix_cube_names(root: ET.Element) -> None:
    asset_names: dict[str, str] = {}
    for element in root.findall("./asset/*[@name]"):
        old_name = element.get("name")
        if old_name:
            new_name = f"object/{old_name}"
            asset_names[old_name] = new_name
            element.set("name", new_name)

    for element in root.findall("./asset/*"):
        for attr in ("mesh", "material", "texture"):
            value = element.get(attr)
            if value in asset_names:
                element.set(attr, asset_names[value])

    for element in root.findall(".//worldbody//*[@name]"):
        element.set("name", f"object/{element.get('name')}")
    for element in root.iter():
        if element.tag not in {"geom", "site"}:
            continue
        for attr in ("mesh", "material"):
            value = element.get(attr)
            if value in asset_names:
                element.set(attr, asset_names[value])
        if element.tag == "geom" and element.get("mesh") == "object/cube_mesh":
            element.set("group", "1")


def _disable_collision_geoms(root: ET.Element) -> None:
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if name.endswith("_col") or name == "object/cube":
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "0")


def _set_cube_initial_pose(root: ET.Element) -> None:
    cube = root.find("./worldbody/body[@name='object/cube']")
    if cube is None:
        raise ValueError("Preview scene is missing the object/cube body")
    cube.set("pos", " ".join(f"{value:.9g}" for value in CUBE_INIT_POS))
    cube.set("quat", " ".join(f"{value:.9g}" for value in CUBE_INIT_QUAT))


def _add_grid(root: ET.Element, *, extent: float = 0.45, spacing: float = 0.05) -> None:
    worldbody = root.find("./worldbody")
    if worldbody is None:
        raise ValueError("Preview scene is missing worldbody")
    grid = ET.SubElement(worldbody, "body", {"name": "preview_grid"})
    grid_z = -0.135
    line_width = 0.0007
    line_height = 0.0006
    count = int(round(extent / spacing))
    positions = [index * spacing for index in range(-count, count + 1)]

    def add_box(name: str, pos: tuple[float, float, float], size: tuple[float, float, float], rgba: str) -> None:
        ET.SubElement(
            grid,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": " ".join(f"{value:.9g}" for value in pos),
                "size": " ".join(f"{value:.9g}" for value in size),
                "rgba": rgba,
                "group": "3",
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
            },
        )

    for index, x in enumerate(positions):
        add_box(f"grid_x_{index}", (x, 0.0, grid_z), (line_width, extent, line_height), "0.28 0.31 0.36 1")
    for index, y in enumerate(positions):
        add_box(f"grid_y_{index}", (0.0, y, grid_z + line_height), (extent, line_width, line_height), "0.28 0.31 0.36 1")
    add_box("axis_x", (extent / 2.0, 0.0, grid_z + 0.0012), (extent / 2.0, 0.002, 0.0012), "0.86 0.18 0.16 1")
    add_box("axis_y", (0.0, extent / 2.0, grid_z + 0.0014), (0.002, extent / 2.0, 0.0012), "0.18 0.66 0.30 1")
    add_box("axis_z", (0.0, 0.0, grid_z + 0.10), (0.002, 0.002, 0.10), "0.16 0.38 0.86 1")


def build_preview_xml() -> str:
    """Compose a self-contained preview XML from the registered Wuji assets."""
    robot_root = ET.parse(ROBOT_XML).getroot()
    cube_root = ET.parse(CUBE_XML).getroot()
    _make_absolute_assets(robot_root, ROBOT_XML)
    _make_absolute_assets(cube_root, CUBE_XML)
    _prefix_robot_names(robot_root)
    _prefix_cube_names(cube_root)
    _disable_collision_geoms(robot_root)
    _disable_collision_geoms(cube_root)
    _set_cube_initial_pose(cube_root)

    compiler = robot_root.find("./compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("assetdir", None)

    robot_asset = robot_root.find("./asset")
    cube_asset = cube_root.find("./asset")
    if robot_asset is None or cube_asset is None:
        raise ValueError("Both preview sources must define an asset section")
    for child in list(cube_asset):
        robot_asset.append(copy.deepcopy(child))

    robot_default = robot_root.find("./default")
    cube_default = cube_root.find("./default/default[@class='cube']")
    if robot_default is None or cube_default is None:
        raise ValueError("Both preview sources must define the cube default class")
    robot_default.append(copy.deepcopy(cube_default))

    worldbody = robot_root.find("./worldbody")
    cube_worldbody = cube_root.find("./worldbody")
    if worldbody is None or cube_worldbody is None:
        raise ValueError("Both preview sources must define worldbody")
    cube_body = cube_worldbody.find("./body[@name='object/cube']")
    if cube_body is None:
        raise ValueError("Preview cube body was not found after name prefixing")
    worldbody.append(copy.deepcopy(cube_body))
    _add_grid(robot_root)

    ET.indent(robot_root, space="  ")
    return ET.tostring(robot_root, encoding="unicode") + "\n"


def write_preview_xml(output_path: str | Path) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_preview_xml(), encoding="utf-8")
    return output_path


def _set_initial_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for finger in range(1, 5):
        joint_name = f"robot/right_finger{finger}_joint1"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = HAND_HOME_FIRST_JOINT_RAD
        for joint in range(2, 5):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"robot/right_finger{finger}_joint{joint}",
            )
            data.qpos[model.jnt_qposadr[joint_id]] = 0.0

    cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object/cube_freejoint")
    cube_qpos = model.jnt_qposadr[cube_joint_id]
    data.qpos[cube_qpos : cube_qpos + 3] = CUBE_INIT_POS
    data.qpos[cube_qpos + 3 : cube_qpos + 7] = CUBE_INIT_QUAT
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    mujoco.mj_forward(model, data)


def build_preview_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(build_preview_xml())
    data = mujoco.MjData(model)
    _set_initial_state(model, data)
    return model, data


def set_visual_only_option(scene_option: mujoco.MjvOption) -> None:
    """Show only high-resolution visual meshes and the presentation grid."""
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[1] = 1  # URDF visual STL meshes and cube visual
    scene_option.geomgroup[3] = 1  # presentation grid and axes


def render_wuji_preview(
    image_path: str | Path,
    *,
    width: int = 1600,
    height: int = 1200,
) -> Path:
    """Render a static full-resolution visual-STL snapshot."""
    model, data = build_preview_model()
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.045, 0.0, 0.16)
    camera.distance = 0.78
    camera.azimuth = 137.0
    camera.elevation = -24.0

    scene_option = mujoco.MjvOption()
    set_visual_only_option(scene_option)
    renderer.update_scene(data, camera, scene_option=scene_option)
    pixels = renderer.render()
    renderer.close()

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-specific fallback
        raise RuntimeError("Pillow is required to write the preview PNG") from exc
    image_path = Path(image_path).resolve()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels)).save(image_path)
    return image_path


def launch_static_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Open a viewer without stepping, so the initial state stays frozen."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_visual_only_option(viewer.opt)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("generated/luwu_wuji_preview/scene.xml"),
        help="Where to write the composed static preview XML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/luwu_wuji_preview/preview.png"),
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--viewer", action="store_true", help="Also launch the frozen interactive viewer.")
    args = parser.parse_args()

    scene_path = write_preview_xml(args.scene)
    image_path = render_wuji_preview(args.output, width=args.width, height=args.height)
    print(f"preview scene: {scene_path}")
    print(f"visual STL preview: {image_path}")
    print("state: Home, cube at training reset pose, static/no mj_step")
    if args.viewer:
        model, data = build_preview_model()
        launch_static_viewer(model, data)


if __name__ == "__main__":
    main()
