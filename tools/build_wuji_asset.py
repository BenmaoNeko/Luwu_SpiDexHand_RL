"""Build the package-local Wuji MJLab asset from the validated Luwu preview."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


HAND_JOINT_RENAME = {
    f"joint{7 + index}": (
        f"right_finger{index // 4 + 1}_joint{index % 4 + 1}"
    )
    for index in range(16)
}

BODY_RENAME = {
    "link6": "right_palm_link",
    **{
        f"finger{finger}_{link_index}": (
            f"right_finger{finger}_link{link_index + 1}"
        )
        for finger in range(1, 5)
        for link_index in range(4)
    },
}


def _indent(element: ET.Element) -> None:
    ET.indent(element, space="  ")


def _tip_position(mesh_path: Path) -> tuple[float, float, float]:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    vertices = np.asarray(mesh.vertices, dtype=float)
    x_min = float(vertices[:, 0].min())
    x_span = float(vertices[:, 0].max() - x_min)
    near_tip = vertices[:, 0] <= x_min + max(0.05 * x_span, 1.0e-6)
    return tuple(float(value) for value in vertices[near_tip].mean(axis=0))


def _copy_meshes(source_root: Path, output_root: Path) -> None:
    source_meshes = source_root / "meshes"
    output_meshes = output_root / "meshes"
    if output_meshes.exists():
        shutil.rmtree(output_meshes)
    shutil.copytree(source_meshes, output_meshes)


def build_wuji_robot_asset(source_scene: str | Path, output_root: str | Path) -> Path:
    """Transform the standalone Luwu preview into a fixed Wuji robot asset.

    The source preview already contains the baked arm pose, convex collision
    meshes, hand PD actuators, and parent-child contact exclusions. This
    function only changes the package layout and public names needed by the
    Wuji/MJLab task while leaving the source URDF and generated preview intact.
    """

    source_scene = Path(source_scene).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_meshes(source_scene.parent, output_root)

    root = ET.parse(source_scene).getroot()
    root.set("model", "luwu-arm-hand-002-mjlab")
    compiler = root.find("./compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", "../meshes")

    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file", "")
        if filename.startswith("meshes/"):
            mesh.set("file", filename.removeprefix("meshes/"))

    worldbody = root.find("./worldbody")
    if worldbody is None:
        raise ValueError("Source scene has no worldbody")
    preview_cube = worldbody.find("./body[@name='preview_cube']")
    if preview_cube is not None:
        worldbody.remove(preview_cube)

    keyframe = root.find("./keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    for body in root.iter("body"):
        old_body_name = body.get("name", "")
        new_body_name = BODY_RENAME.get(old_body_name, old_body_name)
        body.set("name", new_body_name)

        for joint in body.findall("./joint"):
            old_joint_name = joint.get("name", "")
            new_joint_name = HAND_JOINT_RENAME.get(old_joint_name, old_joint_name)
            joint.set("name", new_joint_name)

        for geom in body.findall("./geom"):
            old_geom_name = geom.get("name", "")
            if old_geom_name.endswith("_collision_collision"):
                geom.set(
                    "name",
                    "right_palm_collision"
                    if new_body_name == "right_palm_link"
                    else f"{new_body_name}_col",
                )
            elif old_geom_name.endswith("_visual_visual"):
                geom.set("name", f"{new_body_name}_visual")

        if new_body_name.startswith("right_finger") and new_body_name.endswith("_link4"):
            finger = int(new_body_name.split("finger", 1)[1].split("_", 1)[0])
            source_mesh = source_scene.parent / "meshes" / "visual" / f"finger{finger}_3_visual.stl"
            if not source_mesh.exists():
                raise FileNotFoundError(source_mesh)
            ET.SubElement(
                body,
                "site",
                {
                    "name": f"right_finger{finger}_tip",
                    "pos": " ".join(f"{value:.9g}" for value in _tip_position(source_mesh)),
                    "group": "4",
                },
            )

    contact = root.find("./contact")
    if contact is not None:
        for exclude in contact.findall("./exclude"):
            exclude.set("body1", BODY_RENAME.get(exclude.get("body1", ""), exclude.get("body1", "")))
            exclude.set("body2", BODY_RENAME.get(exclude.get("body2", ""), exclude.get("body2", "")))

    actuator = root.find("./actuator")
    if actuator is None:
        raise ValueError("Source scene has no actuator section")
    for position in actuator.findall("./position"):
        target = position.get("joint", "")
        renamed_target = HAND_JOINT_RENAME.get(target, target)
        position.set("joint", renamed_target)
        position.set("name", f"{renamed_target}_position")

    _indent(root)
    output_xml = output_root / "mjcf" / "luwu_mjlab.xml"
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_wuji_robot_asset(args.source_scene, args.output_root))


if __name__ == "__main__":
    main()
