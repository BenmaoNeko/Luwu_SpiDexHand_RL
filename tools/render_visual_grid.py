"""Render the full-resolution URDF visual meshes with a presentation grid.

This creates a derived visualization scene only. It does not change the
training scene, collision meshes, or source URDF.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

try:
    from tools.convex_preview import reset_to_home
except ModuleNotFoundError:  # pragma: no cover - direct script entry point
    from convex_preview import reset_to_home


GRID_GROUP = "3"
GRID_RGBA = "0.28 0.31 0.36 1"
AXIS_X_RGBA = "0.86 0.18 0.16 1"
AXIS_Y_RGBA = "0.18 0.66 0.30 1"
AXIS_Z_RGBA = "0.16 0.38 0.86 1"


def _add_grid_geom(
    parent: ET.Element,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: str,
) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": " ".join(f"{value:.9g}" for value in pos),
            "size": " ".join(f"{value:.9g}" for value in size),
            "rgba": rgba,
            "group": GRID_GROUP,
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
        },
    )


def build_grid_scene(
    source_scene: str | Path,
    output_scene: str | Path,
    *,
    grid_extent: float = 0.45,
    grid_spacing: float = 0.05,
    grid_z: float = -0.135,
) -> Path:
    """Add a non-physical floor grid and coordinate axes to a MuJoCo scene."""
    if grid_extent <= 0 or grid_spacing <= 0:
        raise ValueError("grid_extent and grid_spacing must be positive")

    source_scene = Path(source_scene).resolve()
    output_scene = Path(output_scene).resolve()
    root = ET.parse(source_scene).getroot()
    worldbody = root.find("./worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo scene is missing worldbody")

    old_grid = worldbody.find("./body[@name='preview_grid']")
    if old_grid is not None:
        worldbody.remove(old_grid)
    grid_body = ET.SubElement(worldbody, "body", {"name": "preview_grid"})

    line_width = 0.0007
    line_height = 0.0006
    count = int(round(grid_extent / grid_spacing))
    positions = [index * grid_spacing for index in range(-count, count + 1)]
    for index, x in enumerate(positions):
        _add_grid_geom(
            grid_body,
            f"grid_x_{index}",
            (x, 0.0, grid_z),
            (line_width, grid_extent, line_height),
            GRID_RGBA,
        )
    for index, y in enumerate(positions):
        _add_grid_geom(
            grid_body,
            f"grid_y_{index}",
            (0.0, y, grid_z + line_height),
            (grid_extent, line_width, line_height),
            GRID_RGBA,
        )

    axis_height = 0.0012
    axis_width = 0.0020
    _add_grid_geom(
        grid_body,
        "axis_x",
        (grid_extent / 2.0, 0.0, grid_z + axis_height),
        (grid_extent / 2.0, axis_width, axis_height),
        AXIS_X_RGBA,
    )
    _add_grid_geom(
        grid_body,
        "axis_y",
        (0.0, grid_extent / 2.0, grid_z + axis_height * 1.2),
        (axis_width, grid_extent / 2.0, axis_height),
        AXIS_Y_RGBA,
    )
    _add_grid_geom(
        grid_body,
        "axis_z",
        (0.0, 0.0, grid_z + 0.10),
        (axis_width, axis_width, 0.10),
        AXIS_Z_RGBA,
    )

    ET.indent(root, space="  ")
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    output_scene.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return output_scene


def render_visual_grid(
    scene_xml: str | Path,
    image_path: str | Path,
    *,
    width: int = 1280,
    height: int = 960,
) -> Path:
    """Render visual STL geoms and grid geoms while hiding collision hulls."""
    model = mujoco.MjModel.from_xml_path(str(Path(scene_xml).resolve()))
    data = mujoco.MjData(model)
    reset_to_home(model, data)

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
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[1] = 1  # full-resolution visual STL meshes
    scene_option.geomgroup[3] = 1  # presentation grid and axes
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("generated/luwu_arm_hand_002_convex/scene.xml"),
    )
    parser.add_argument(
        "--grid-scene",
        type=Path,
        default=Path("generated/luwu_arm_hand_002_convex/visual_grid_scene.xml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/luwu_arm_hand_002_convex/visual_grid_preview.png"),
    )
    parser.add_argument("--grid-extent", type=float, default=0.45)
    parser.add_argument("--grid-spacing", type=float, default=0.05)
    parser.add_argument("--grid-z", type=float, default=-0.135)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()

    grid_scene = build_grid_scene(
        args.scene,
        args.grid_scene,
        grid_extent=args.grid_extent,
        grid_spacing=args.grid_spacing,
        grid_z=args.grid_z,
    )
    image = render_visual_grid(
        grid_scene,
        args.output,
        width=args.width,
        height=args.height,
    )
    print(f"grid scene: {grid_scene}")
    print(f"visual STL preview: {image}")


if __name__ == "__main__":
    main()
