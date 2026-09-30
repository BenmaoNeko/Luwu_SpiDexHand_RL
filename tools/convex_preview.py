"""Generate convex collision proxies and a fixed-arm MuJoCo preview.

The source URDF is the authority for topology, frames, axes, and limits. This
module deliberately writes only derived assets; it never edits the source
URDF or source STL files.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import mujoco
import mujoco.viewer
import numpy as np
import trimesh


ARM_FIXED_Q = (
    0.0,
    0.0,
    0.0,
    1.5,
    0.0,
    0.0,
)
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
HAND_JOINT_NAMES = tuple(f"joint{i}" for i in range(7, 23))
HAND_ACTUATOR_KP = 4.0
HAND_ACTUATOR_KD = 0.15
HAND_MAX_TORQUE = 0.5
HAND_HOME_FIRST_JOINT_RAD = math.radians(35.0)
HAND_HOME_Q = tuple(
    HAND_HOME_FIRST_JOINT_RAD if index % 4 == 0 else 0.0
    for index in range(len(HAND_JOINT_NAMES))
)


@dataclass(frozen=True)
class MeshSpec:
    path: Path
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


@dataclass(frozen=True)
class InertialSpec:
    pos: tuple[float, float, float]
    rpy: tuple[float, float, float]
    mass: float
    diagonal_inertia: tuple[float, float, float]


@dataclass(frozen=True)
class LinkSpec:
    name: str
    visual: MeshSpec | None
    collision: MeshSpec | None
    inertial: InertialSpec | None


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass
class RobotSpec:
    source_urdf: Path
    links: dict[str, LinkSpec]
    joints: list[JointSpec]
    children: dict[str, list[JointSpec]] = field(default_factory=dict)

    @property
    def mesh_links(self) -> list[LinkSpec]:
        return [link for link in self.links.values() if link.collision or link.visual]

    @property
    def root_link(self) -> str:
        child_names = {joint.child for joint in self.joints}
        roots = [name for name in self.links if name not in child_names]
        if len(roots) != 1:
            raise ValueError(f"Expected one root link, found {roots}")
        return roots[0]


def _floats(value: str | None, count: int) -> tuple[float, ...]:
    if value is None:
        return tuple(0.0 for _ in range(count))
    values = tuple(float(item) for item in value.split())
    if len(values) != count:
        raise ValueError(f"Expected {count} floats, got {value!r}")
    return values


def _mesh_path(source_urdf: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise ValueError(f"Package URI is not supported by the standalone converter: {filename}")
    path = Path(filename)
    return path if path.is_absolute() else (source_urdf.parent / path).resolve()


def _rgba(link_element: ET.Element) -> tuple[float, float, float, float]:
    color = link_element.find("./visual/material/color")
    if color is None:
        return (0.7, 0.7, 0.7, 1.0)
    return _floats(color.get("rgba"), 4)  # type: ignore[return-value]


def _mesh_spec(source_urdf: Path, element: ET.Element | None, rgba: tuple[float, ...]) -> MeshSpec | None:
    if element is None:
        return None
    mesh = element.find("./geometry/mesh")
    if mesh is None or mesh.get("filename") is None:
        return None
    origin = element.find("./origin")
    return MeshSpec(
        path=_mesh_path(source_urdf, mesh.get("filename", "")),
        origin_xyz=_floats(origin.get("xyz") if origin is not None else None, 3),
        origin_rpy=_floats(origin.get("rpy") if origin is not None else None, 3),
        rgba=rgba,  # type: ignore[arg-type]
    )


def _inertial_spec(element: ET.Element | None) -> InertialSpec | None:
    if element is None:
        return None
    origin = element.find("./origin")
    mass_element = element.find("./mass")
    inertia = element.find("./inertia")
    if mass_element is None or inertia is None:
        return None
    return InertialSpec(
        pos=_floats(origin.get("xyz") if origin is not None else None, 3),  # type: ignore[arg-type]
        rpy=_floats(origin.get("rpy") if origin is not None else None, 3),  # type: ignore[arg-type]
        mass=float(mass_element.get("value", "0")),
        diagonal_inertia=(
            float(inertia.get("ixx", "1e-6")),
            float(inertia.get("iyy", "1e-6")),
            float(inertia.get("izz", "1e-6")),
        ),
    )


def parse_urdf(source_urdf: str | Path) -> RobotSpec:
    source_urdf = Path(source_urdf).resolve()
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    links: dict[str, LinkSpec] = {}
    for link_element in root.findall("./link"):
        name = link_element.get("name")
        if not name:
            raise ValueError("URDF link is missing a name")
        rgba = _rgba(link_element)
        links[name] = LinkSpec(
            name=name,
            visual=_mesh_spec(source_urdf, link_element.find("./visual"), rgba),
            collision=_mesh_spec(source_urdf, link_element.find("./collision"), rgba),
            inertial=_inertial_spec(link_element.find("./inertial")),
        )

    joints: list[JointSpec] = []
    for joint_element in root.findall("./joint"):
        name = joint_element.get("name")
        parent = joint_element.find("./parent")
        child = joint_element.find("./child")
        if not name or parent is None or child is None:
            raise ValueError("URDF joint is missing name, parent, or child")
        origin = joint_element.find("./origin")
        axis = joint_element.find("./axis")
        limit = joint_element.find("./limit")
        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_element.get("type", "fixed"),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin_xyz=_floats(origin.get("xyz") if origin is not None else None, 3),  # type: ignore[arg-type]
                origin_rpy=_floats(origin.get("rpy") if origin is not None else None, 3),  # type: ignore[arg-type]
                axis=_floats(axis.get("xyz") if axis is not None else "0 0 1", 3),  # type: ignore[arg-type]
                lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
                upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
            )
        )

    robot = RobotSpec(source_urdf=source_urdf, links=links, joints=joints)
    robot.children = {name: [] for name in links}
    for joint in joints:
        if joint.parent not in robot.children:
            raise ValueError(f"Joint {joint.name} references unknown parent {joint.parent}")
        robot.children[joint.parent].append(joint)
    return robot


def _rpy_to_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _axis_angle(axis: Iterable[float], angle: float) -> np.ndarray:
    axis_array = np.asarray(tuple(axis), dtype=float)
    norm = np.linalg.norm(axis_array)
    if norm == 0:
        raise ValueError("Joint axis cannot be zero")
    axis_array /= norm
    x, y, z = axis_array
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(axis_array, axis_array)
        + math.sin(angle) * skew
    )


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray((w, x, y, z), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _format(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset a generated scene to the calibrated hand Home keyframe."""
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise ValueError("MuJoCo model is missing the calibrated 'home' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(mesh).__name__}")
    return mesh


def generate_hulls(robot: RobotSpec, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir).resolve()
    visual_dir = output_dir / "meshes" / "visual"
    collision_dir = output_dir / "meshes" / "collision"
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)

    hull_manifest: list[dict] = []
    for link in robot.mesh_links:
        source_visual = link.visual.path if link.visual else link.collision.path  # type: ignore[union-attr]
        source_collision = link.collision.path if link.collision else source_visual
        if not source_visual.exists():
            raise FileNotFoundError(source_visual)
        if not source_collision.exists():
            raise FileNotFoundError(source_collision)

        visual_target = visual_dir / f"{link.name}_visual{source_visual.suffix.lower()}"
        shutil.copy2(source_visual, visual_target)

        mesh = _load_mesh(source_collision)
        hull = mesh.convex_hull
        if len(hull.vertices) < 4 or not np.isfinite(hull.vertices).all():
            raise ValueError(f"Invalid convex hull for {link.name}")
        extents = np.asarray(hull.extents, dtype=float)
        if np.any(extents <= 0.0):
            raise ValueError(f"Degenerate convex hull for {link.name}: {extents}")
        hull_target = collision_dir / f"{link.name}_collision_hull.stl"
        hull.export(hull_target)
        hull_manifest.append(
            {
                "link": link.name,
                "source": str(source_collision),
                "path": str(hull_target),
                "vertices": int(len(hull.vertices)),
                "faces": int(len(hull.faces)),
                "extents": [float(value) for value in extents],
            }
        )

    manifest = {
        "source_urdf": str(robot.source_urdf),
        "link_count": len(robot.links),
        "mesh_link_count": len(robot.mesh_links),
        "hulls": hull_manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _add_geom(parent: ET.Element, mesh: MeshSpec, mesh_name: str, collision: bool) -> None:
    origin_rotation = _rpy_to_matrix(mesh.origin_rpy)
    attributes = {
        "name": f"{mesh_name}_{'collision' if collision else 'visual'}",
        "type": "mesh",
        "mesh": mesh_name,
        "pos": _format(mesh.origin_xyz),
        "quat": _format(_matrix_to_quat_wxyz(origin_rotation)),
        "rgba": _format(mesh.rgba),
    }
    if collision:
        attributes.update({"group": "2", "contype": "1", "conaffinity": "1", "friction": "0.8 0.05 0.01"})
    else:
        attributes.update({"group": "1", "contype": "0", "conaffinity": "0", "density": "0"})
    ET.SubElement(parent, "geom", attributes)


def _add_inertial(parent: ET.Element, inertial: InertialSpec | None) -> None:
    if inertial is None:
        return
    attributes = {
        "pos": _format(inertial.pos),
        "quat": _format(_matrix_to_quat_wxyz(_rpy_to_matrix(inertial.rpy))),
        "mass": f"{max(inertial.mass, 1e-7):.9g}",
        "diaginertia": _format(max(value, 1e-9) for value in inertial.diagonal_inertia),
    }
    ET.SubElement(parent, "inertial", attributes)


def _world_transforms(robot: RobotSpec, arm_q: tuple[float, ...]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arm_values = dict(zip(ARM_JOINT_NAMES, arm_q))
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {
        robot.root_link: (np.eye(3), np.zeros(3))
    }

    def visit(parent: str) -> None:
        parent_rotation, parent_position = result[parent]
        for joint in robot.children.get(parent, []):
            local_rotation = _rpy_to_matrix(joint.origin_rpy)
            if joint.name in arm_values:
                local_rotation = local_rotation @ _axis_angle(joint.axis, arm_values[joint.name])
            rotation = parent_rotation @ local_rotation
            position = parent_position + parent_rotation @ np.asarray(joint.origin_xyz, dtype=float)
            result[joint.child] = (rotation, position)
            visit(joint.child)

    visit(robot.root_link)
    return result


def build_mjcf(robot: RobotSpec, output_dir: str | Path, arm_q: tuple[float, ...] = ARM_FIXED_Q) -> Path:
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run generate_hulls first: {manifest_path}")
    if len(arm_q) != 6:
        raise ValueError("arm_q must contain exactly six values")

    root = ET.Element("mujoco", {"model": "luwu_arm_hand_002_convex"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "."})
    ET.SubElement(root, "option", {"timestep": "0.005", "iterations": "50", "integrator": "implicitfast"})
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", {"solref": "0.02 1", "solimp": "0.9 0.95 0.001"})
    ET.SubElement(default, "joint", {"damping": "0.02", "armature": "0.0002"})

    asset = ET.SubElement(root, "asset")
    for link in robot.mesh_links:
        visual_file = f"meshes/visual/{link.name}_visual{(link.visual or link.collision).path.suffix.lower()}"  # type: ignore[union-attr]
        collision_file = f"meshes/collision/{link.name}_collision_hull.stl"
        ET.SubElement(asset, "mesh", {"name": f"{link.name}_visual", "file": visual_file})
        ET.SubElement(asset, "mesh", {"name": f"{link.name}_collision", "file": collision_file})

    worldbody = ET.SubElement(root, "worldbody")
    arm_values = dict(zip(ARM_JOINT_NAMES, arm_q))
    joint_by_child = {joint.child: joint for joint in robot.joints}

    def add_link(parent_element: ET.Element, link_name: str, incoming: JointSpec | None) -> None:
        link = robot.links[link_name]
        body_attributes = {"name": link.name}
        if incoming is not None:
            local_rotation = _rpy_to_matrix(incoming.origin_rpy)
            if incoming.name in arm_values:
                local_rotation = local_rotation @ _axis_angle(incoming.axis, arm_values[incoming.name])
            body_attributes.update(
                {
                    "pos": _format(incoming.origin_xyz),
                    "quat": _format(_matrix_to_quat_wxyz(local_rotation)),
                }
            )
        else:
            body_attributes.update({"pos": "0 0 0", "quat": "1 0 0 0"})
        body = ET.SubElement(parent_element, "body", body_attributes)
        _add_inertial(body, link.inertial)
        if link.visual:
            _add_geom(body, link.visual, f"{link.name}_visual", collision=False)
        if link.collision:
            collision_mesh = MeshSpec(
                path=output_dir / "meshes" / "collision" / f"{link.name}_collision_hull.stl",
                origin_xyz=link.collision.origin_xyz,
                origin_rpy=link.collision.origin_rpy,
                rgba=link.collision.rgba,
            )
            _add_geom(body, collision_mesh, f"{link.name}_collision", collision=True)
        if incoming is not None and incoming.name not in arm_values and incoming.joint_type == "revolute":
            joint_attributes = {
                "name": incoming.name,
                "type": "hinge",
                "axis": _format(incoming.axis),
                "limited": "true" if incoming.lower is not None and incoming.upper is not None else "false",
                "damping": "0.02",
            }
            if incoming.lower is not None and incoming.upper is not None:
                joint_attributes["range"] = f"{incoming.lower:.9g} {incoming.upper:.9g}"
            ET.SubElement(body, "joint", joint_attributes)
        for child_joint in robot.children.get(link_name, []):
            add_link(body, child_joint.child, child_joint)

    add_link(worldbody, robot.root_link, None)

    transforms = _world_transforms(robot, arm_q)
    link6_rotation, link6_position = transforms["link6"]
    cube_position = link6_position + link6_rotation @ np.array([0.0, 0.0, 0.125])
    cube_body = ET.SubElement(worldbody, "body", {"name": "preview_cube", "pos": _format(cube_position)})
    ET.SubElement(
        cube_body,
        "geom",
        {
            "name": "preview_cube_collision",
            "type": "box",
            "size": "0.025 0.025 0.025",
            "rgba": "0.95 0.45 0.1 1",
            "group": "2",
            "contype": "1",
            "conaffinity": "1",
            "friction": "0.8 0.05 0.01",
        },
    )

    contact = ET.SubElement(root, "contact")
    for joint in robot.joints:
        ET.SubElement(
            contact,
            "exclude",
            {"body1": joint.parent, "body2": joint.child},
        )

    hand_joints = {joint.name: joint for joint in robot.joints if joint.name in HAND_JOINT_NAMES}
    missing_joints = [name for name in HAND_JOINT_NAMES if name not in hand_joints]
    if missing_joints:
        raise ValueError(f"Missing hand joints in URDF: {missing_joints}")

    actuator = ET.SubElement(root, "actuator")
    for joint_name in HAND_JOINT_NAMES:
        joint = hand_joints[joint_name]
        if joint.lower is None or joint.upper is None:
            raise ValueError(f"Hand joint {joint.name} must have URDF lower and upper limits")
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint.name}_position",
                "joint": joint.name,
                "kp": f"{HAND_ACTUATOR_KP:.9g}",
                "kv": f"{HAND_ACTUATOR_KD:.9g}",
                "ctrllimited": "true",
                "ctrlrange": f"{joint.lower:.9g} {joint.upper:.9g}",
                "forcelimited": "true",
                "forcerange": f"{-HAND_MAX_TORQUE:.9g} {HAND_MAX_TORQUE:.9g}",
            },
        )

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": _format(HAND_HOME_Q),
            "ctrl": _format(HAND_HOME_Q),
        },
    )
    ET.indent(root, space="  ")
    xml_path = output_dir / "scene.xml"
    xml_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return xml_path


def render_preview(
    xml_path: str | Path,
    image_path: str | Path,
    *,
    collision_only: bool = False,
    closeup: bool = False,
) -> Path:
    xml_path = Path(xml_path).resolve()
    image_path = Path(image_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if closeup:
        link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        camera.lookat[:] = data.xpos[link6_id]
        camera.distance = 0.50
    else:
        camera.lookat[:] = (0.0, 0.0, 0.35)
        camera.distance = 1.10
    camera.azimuth = 135.0
    camera.elevation = -20.0
    scene_option = None
    if collision_only:
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[2] = 1
        for geom_id in range(model.ngeom):
            if model.geom_group[geom_id] == 2:
                model.geom_rgba[geom_id] = (0.95, 0.12, 0.08, 1.0)
    renderer.update_scene(data, camera, scene_option=scene_option)
    pixels = renderer.render()
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment-specific fallback
        raise RuntimeError("Pillow is required to write the MuJoCo preview PNG") from exc
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(image_path)
    renderer.close()
    return image_path


def launch_viewer(xml_path: str | Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", type=Path, default=Path("URDF/open_arm_002/luwu_arm_hand_002/机械臂-灵巧爪-总装v3.0-完整版URDF.SLDASM/机械臂-灵巧爪-总装v3.0-完整版URDF.SLDASM.urdf"))
    parser.add_argument("--output", type=Path, default=Path("generated/luwu_arm_hand_002_convex"))
    parser.add_argument("--render", action="store_true", help="write preview.png with MuJoCo's offscreen renderer")
    parser.add_argument("--viewer", action="store_true", help="open MuJoCo's interactive viewer")
    args = parser.parse_args()

    robot = parse_urdf(args.source_urdf)
    manifest = generate_hulls(robot, args.output)
    xml_path = build_mjcf(robot, args.output, ARM_FIXED_Q)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(f"generated {len(manifest['hulls'])} convex hulls")
    print(f"MuJoCo model: bodies={model.nbody}, joints={model.njnt}, actuators={model.nu}")
    if args.render:
        image_path = render_preview(xml_path, args.output / "preview.png")
        print(f"preview image: {image_path}")
        collision_image_path = render_preview(
            xml_path,
            args.output / "preview_collision.png",
            collision_only=True,
        )
        print(f"collision-only preview image: {collision_image_path}")
        hand_collision_image_path = render_preview(
            xml_path,
            args.output / "preview_hand_collision.png",
            collision_only=True,
            closeup=True,
        )
        print(f"hand collision close-up: {hand_collision_image_path}")
    if args.viewer:
        launch_viewer(xml_path)


if __name__ == "__main__":
    main()
