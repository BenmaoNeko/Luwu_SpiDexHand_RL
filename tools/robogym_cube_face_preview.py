"""Local native MuJoCo viewer and desktop face-turn panel for Robogym's cube."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOGYM_ROOT = PROJECT_ROOT / "URDF" / "robogym-master"
CUBE_XML = ROBOGYM_ROOT / "robogym" / "assets" / "xmls" / "rubik" / "rubik_perpendicular.xml"
MESH_DIR = ROBOGYM_ROOT / "robogym" / "assets" / "stls"
TEXTURE_DIR = ROBOGYM_ROOT / "robogym" / "assets" / "textures"
CUBE_CENTER_HEIGHT = 0.029
DEFAULT_TORQUE_PER_CUBIE_NM = 0.005
MAX_TORQUE_PER_CUBIE_NM = 0.01

if str(ROBOGYM_ROOT) not in sys.path:
  sys.path.insert(0, str(ROBOGYM_ROOT))

from robogym.utils.rotation import euler2mat, mat2euler  # noqa: E402


# side is 0 for the negative axis and 1 for the positive axis. Angles use the
# right-hand convention, so clockwise is negative about the outward normal.
FACE_AXIS_SIDE = {
  "L": (0, 0),
  "R": (0, 1),
  "F": (1, 0),
  "B": (1, 1),
  "D": (2, 0),
  "U": (2, 1),
}
AXIS_NAMES = ("x", "y", "z")


def face_turn_parameters(face: str, direction: str) -> tuple[int, int, float]:
  """Return (axis, side, signed quarter-turn radians) for a Rubik face move."""
  face = face.upper()
  direction = direction.lower()
  if face not in FACE_AXIS_SIDE:
    raise ValueError(f"Unknown face {face!r}; use one of {tuple(FACE_AXIS_SIDE)}")
  if direction not in {"cw", "ccw"}:
    raise ValueError("direction must be 'cw' or 'ccw'")

  axis, side = FACE_AXIS_SIDE[face]
  outward_sign = 1 if side else -1
  turn_sign = -1 if direction == "cw" else 1
  return axis, side, turn_sign * outward_sign * (math.pi / 2.0)


def build_preview_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
  """Load the Robogym cube, adding preview-only assets and a ground plane."""
  if not CUBE_XML.is_file():
    raise FileNotFoundError(f"Missing Robogym cube MJCF: {CUBE_XML}")

  root = ET.parse(CUBE_XML).getroot()
  root.insert(
    0,
    ET.Element(
      "compiler",
      {
        "angle": "radian",
        "meshdir": str(MESH_DIR),
        "texturedir": str(TEXTURE_DIR),
      },
    ),
  )
  worldbody = root.find("worldbody")
  if worldbody is None:
    raise ValueError("Robogym cube MJCF has no worldbody")
  worldbody.append(
    ET.Element(
      "geom",
      {
        "name": "preview_ground",
        "type": "plane",
        "pos": "0 0 0",
        "size": "0.2 0.2 0.1",
        "rgba": "0.62 0.65 0.68 1",
        "condim": "6",
        "friction": "0.8 0.005 0.001",
      },
    )
  )

  model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
  model.opt.timestep = 0.005
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  data = mujoco.MjData(model)
  reset_cube(model, data)
  return model, data


def _joint_id(model: mujoco.MjModel, name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if joint_id < 0:
    raise KeyError(f"Required Robogym cube joint not found: {name}")
  return joint_id


def _coordinate_name(value: int, axis: str) -> str:
  return f"{'pos' if value > 0 else 'neg'}_{axis}"


def _piece_name(coords: tuple[int, int, int]) -> str:
  return "_".join(
    _coordinate_name(value, axis)
    for axis, value in zip(AXIS_NAMES, coords)
    if value != 0
  )


def _selected_face_pieces(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  face: str,
  axis: int,
  side: int,
) -> list[str]:
  """Return the nine cubies currently on a selected face."""
  side_sign = 1 if side else -1
  qpos_before_turn = data.qpos.copy()
  selected = []

  for coords in itertools.product((-1, 0, 1), repeat=3):
    piece_name = _piece_name(coords)
    if not piece_name:
      continue

    if sum(value != 0 for value in coords) == 1:
      if coords[axis] * side_sign > 0.5:
        selected.append(piece_name)
      continue

    joint_ids = [
      _joint_id(model, f"cubelet:rot{joint_axis}:{piece_name}")
      for joint_axis in AXIS_NAMES
    ]
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    euler = qpos_before_turn[qpos_addresses]
    rotation = euler2mat(euler)
    current_coords = rotation @ np.asarray(coords, dtype=float)
    if current_coords[axis] * side_sign > 0.5:
      selected.append(piece_name)

  return selected


def select_face_cubies(
  model: mujoco.MjModel, data: mujoco.MjData, face: str
) -> tuple[str, ...]:
  """Snapshot the cubies belonging to the selected face when a hold begins."""
  axis, side, _ = face_turn_parameters(face, "cw")
  return tuple(_selected_face_pieces(model, data, face, axis, side))


def apply_face_torque(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  face: str,
  direction: str,
  torque_per_cubie_nm: float,
  *,
  selected_cubies: tuple[str, ...] | None = None,
) -> int:
  """Apply a sustained generalized torque to the selected face's nine cubies.

  The caller should call this once per simulation step while the panel button is
  held. It never writes qpos; MuJoCo integrates the resulting dynamics.
  """
  if not math.isfinite(torque_per_cubie_nm) or torque_per_cubie_nm < 0.0:
    raise ValueError("torque_per_cubie_nm must be a finite, non-negative value")
  mujoco.mj_forward(model, data)
  axis, side, signed_angle = face_turn_parameters(face, direction)
  selected = (
    tuple(selected_cubies)
    if selected_cubies is not None
    else select_face_cubies(model, data, face)
  )

  middle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "middle")
  if middle_id < 0:
    raise KeyError("Robogym root body 'middle' was not found")
  cube_axis_world = data.xmat[middle_id].reshape(3, 3)[:, axis]
  desired_torque_world = cube_axis_world * math.copysign(
    torque_per_cubie_nm, signed_angle
  )

  data.qfrc_applied[:] = 0.0
  for piece_name in selected:
    if sum(axis_name in piece_name for axis_name in AXIS_NAMES) == 1:
      joint_names = (f"cubelet:driver:{piece_name}",)
    else:
      joint_names = tuple(
        f"cubelet:rot{joint_axis}:{piece_name}" for joint_axis in AXIS_NAMES
      )
    for joint_name in joint_names:
      joint_id = _joint_id(model, joint_name)
      dof_address = int(model.jnt_dofadr[joint_id])
      joint_axis_world = data.xaxis[joint_id]
      data.qfrc_applied[dof_address] += float(
        np.dot(desired_torque_world, joint_axis_world)
      )
  return len(selected)


def reset_cube(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Restore solved face/cubie angles while keeping the cube above the floor."""
  for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
    if (
      name.startswith("cubelet:driver:")
      or name.startswith("cubelet:rot")
      or name.startswith("cubelet:spring:")
    ):
      data.qpos[model.jnt_qposadr[joint_id]] = 0.0

  for name, value in (
    ("cube:tx", 0.0),
    ("cube:ty", 0.0),
    ("cube:tz", CUBE_CENTER_HEIGHT),
  ):
    joint_id = _joint_id(model, name)
    data.qpos[model.jnt_qposadr[joint_id]] = value

  rotation_joint = _joint_id(model, "cube:rot")
  rotation_qpos = int(model.jnt_qposadr[rotation_joint])
  data.qpos[rotation_qpos : rotation_qpos + 4] = (1.0, 0.0, 0.0, 0.0)
  data.qvel[:] = 0.0
  data.qfrc_applied[:] = 0.0
  mujoco.mj_forward(model, data)


def reset_viewer_camera(camera: mujoco.MjvCamera) -> None:
  """Restore a three-quarter view that exposes more than one colored face."""
  camera.lookat[:] = (0.0, 0.0, CUBE_CENTER_HEIGHT)
  camera.distance = 0.23
  camera.azimuth = 135.0
  camera.elevation = -22.0


def launch_preview() -> None:
  """Open the native MuJoCo viewer and a hold-to-torque Tk panel."""
  model, data = build_preview_model()
  viewer = mujoco.viewer.launch_passive(model, data)
  with viewer.lock():
    reset_viewer_camera(viewer.cam)
  viewer.sync()

  commands: queue.Queue[tuple[str, ...]] = queue.Queue()
  stop = threading.Event()

  def simulate() -> None:
    active_drive: tuple[str, str, float, tuple[str, ...]] | None = None
    try:
      while viewer.is_running() and not stop.is_set():
        started = time.perf_counter()
        with viewer.lock():
          while True:
            try:
              command = commands.get_nowait()
            except queue.Empty:
              break
            if command[0] == "reset":
              active_drive = None
              reset_cube(model, data)
              reset_viewer_camera(viewer.cam)
            elif command[0] == "start":
              _, face, direction, torque = command
              selected = select_face_cubies(model, data, face)
              active_drive = (face, direction, float(torque), selected)
            elif command[0] == "stop":
              active_drive = None
            elif command[0] == "torque" and active_drive is not None:
              active_drive = (
                active_drive[0],
                active_drive[1],
                float(command[1]),
                active_drive[3],
              )
          data.qfrc_applied[:] = 0.0
          if active_drive is not None:
            face, direction, torque, selected_cubies = active_drive
            apply_face_torque(
              model,
              data,
              face,
              direction,
              torque,
              selected_cubies=selected_cubies,
            )
          mujoco.mj_step(model, data, nstep=2)
        viewer.sync()
        delay = model.opt.timestep * 2 - (time.perf_counter() - started)
        if delay > 0:
          time.sleep(delay)
    finally:
      stop.set()

  simulation_thread = threading.Thread(
    target=simulate, name="robogym-mujoco-loop", daemon=True
  )
  simulation_thread.start()

  root = tk.Tk()
  root.title("Robogym Rubik - Dynamic Face Torque")
  root.resizable(False, False)
  style = ttk.Style(root)
  preview_font = ("DejaVu Sans", 10)
  style.configure("TLabel", font=preview_font)
  style.configure("TLabelframe.Label", font=preview_font)
  style.configure("TButton", font=preview_font, padding=4)
  root.geometry("+1950+180")
  panel = ttk.Frame(root, padding=12)
  panel.grid(row=0, column=0, sticky="nsew")
  ttk.Label(
    panel,
    text="Hold CW/CCW to apply torque; release to stop. No hand in this scene.",
    wraplength=360,
  ).grid(row=0, column=0, columnspan=4, pady=(0, 6), sticky="w")

  torque_per_cubie = tk.DoubleVar(value=DEFAULT_TORQUE_PER_CUBIE_NM)
  torque_label = tk.StringVar(
    value=f"Torque per cubie: {DEFAULT_TORQUE_PER_CUBIE_NM:.3f} N*m"
  )
  ttk.Label(panel, textvariable=torque_label).grid(
    row=1, column=0, columnspan=4, sticky="w"
  )

  held_button: list[tuple[str, str] | None] = [None]
  status = tk.StringVar(value="Ready. Hold a face button to rotate it.")
  status_label = ttk.Label(panel, textvariable=status, wraplength=360)
  status_label.grid(row=7, column=0, columnspan=4, pady=(10, 0), sticky="w")

  def update_torque(value: str) -> None:
    torque = float(value)
    torque_label.set(f"Torque per cubie: {torque:.3f} N*m")
    if held_button[0] is not None:
      commands.put(("torque", f"{torque:.6f}"))

  tk.Scale(
    panel,
    from_=0.0,
    to=MAX_TORQUE_PER_CUBIE_NM,
    resolution=0.001,
    orient="horizontal",
    variable=torque_per_cubie,
    showvalue=False,
    length=340,
    command=update_torque,
  ).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 8))

  def press_face(face: str, direction: str):
    if held_button[0] is not None:
      commands.put(("stop",))
    held_button[0] = (face, direction)
    torque = float(torque_per_cubie.get())
    commands.put(("start", face, direction, f"{torque:.6f}"))
    status.set(f"Holding: {face} {direction.upper()} at {torque:.3f} N*m/cubie.")
    return "break"

  def release_face(_event=None):
    if held_button[0] is not None:
      face, direction = held_button[0]
      held_button[0] = None
      commands.put(("stop",))
      status.set(f"Released: {face} {direction.upper()}.")
    return "break"

  locations = {
    "U": ("Up", 1, 0),
    "L": ("Left", 0, 1),
    "F": ("Front", 1, 1),
    "R": ("Right", 2, 1),
    "B": ("Back", 3, 1),
    "D": ("Down", 1, 2),
  }
  for face, (face_name, column, row) in locations.items():
    group = ttk.LabelFrame(panel, text=f"{face} - {face_name}", padding=5)
    group.grid(row=row + 3, column=column, padx=3, pady=3, sticky="nsew")
    for button_row, direction, label in (
      (0, "cw", "Hold CW"),
      (1, "ccw", "Hold CCW"),
    ):
      button = ttk.Button(group, text=label)
      button.grid(row=button_row, column=0, sticky="ew", pady=2)
      button.bind(
        "<ButtonPress-1>",
        lambda _event, selected=face, turn=direction: press_face(selected, turn),
      )
      button.bind("<ButtonRelease-1>", release_face, add="+")

  ttk.Button(
    panel,
    text="RESET CUBE + VIEW",
    command=lambda: (
      release_face(),
      commands.put(("reset",)),
      status.set("Queued: reset to solved."),
    ),
  ).grid(row=6, column=0, columnspan=4, pady=(8, 0), sticky="ew")
  root.bind_all("<ButtonRelease-1>", release_face, add="+")
  root.bind("<FocusOut>", release_face, add="+")

  closing = False

  def close_panel() -> None:
    nonlocal closing
    if closing:
      return
    closing = True
    stop.set()
    viewer.close()
    root.destroy()

  def poll_viewer() -> None:
    if stop.is_set() or not viewer.is_running():
      close_panel()
    else:
      root.after(200, poll_viewer)

  root.protocol("WM_DELETE_WINDOW", close_panel)
  root.after(200, poll_viewer)
  try:
    root.mainloop()
  finally:
    stop.set()
    if viewer.is_running():
      viewer.close()
    simulation_thread.join(timeout=1.0)


if __name__ == "__main__":
  launch_preview()
