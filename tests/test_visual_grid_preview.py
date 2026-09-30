from pathlib import Path
import xml.etree.ElementTree as ET

from tools.convex_preview import ARM_FIXED_Q, build_mjcf, generate_hulls, parse_urdf
from tools.render_visual_grid import build_grid_scene


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF = ROOT / "URDF/open_arm_002/luwu_arm_hand_002/机械臂-灵巧爪-总装v3.0-完整版URDF.SLDASM/机械臂-灵巧爪-总装v3.0-完整版URDF.SLDASM.urdf"


def test_grid_scene_adds_visual_only_grid(tmp_path: Path):
    robot = parse_urdf(SOURCE_URDF)
    generate_hulls(robot, tmp_path)
    source_scene = build_mjcf(robot, tmp_path, ARM_FIXED_Q)
    grid_scene = build_grid_scene(source_scene, tmp_path / "visual_grid_scene.xml")

    root = ET.parse(grid_scene).getroot()
    grid_geoms = [
        geom
        for geom in root.findall("./worldbody/body[@name='preview_grid']/geom")
    ]

    assert len(grid_geoms) > 10
    assert all(geom.get("group") == "3" for geom in grid_geoms)
    assert all(geom.get("contype") == "0" for geom in grid_geoms)
    assert all(geom.get("conaffinity") == "0" for geom in grid_geoms)
    assert root.findall(".//geom[@group='1']")
