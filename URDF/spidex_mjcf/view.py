from pathlib import Path
import mujoco
import mujoco.viewer

xml_path = str(Path(__file__).resolve().parent / "xmls" / "scene_spidex.xml")

print(f"加载: {xml_path}")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("右键面板拖动滑块控制关节 | 按数字键2、3检查连杆视觉/碰撞体积")
mujoco.viewer.launch(model, data, show_left_ui=True, show_right_ui=True)
