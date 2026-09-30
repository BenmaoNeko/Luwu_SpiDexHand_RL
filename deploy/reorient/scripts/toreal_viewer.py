#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""ToReal Viewer - Real Hand Deployment Visualization.

Visualizes ToReal deployment: shows goal orientation and observed cube pose.
Use this alongside play_real.py and cube_world_observer.py to monitor real hand performance.

Features:
    1. Publish goal orientation (ZMQ:5556) for play_real.py
    2. Receive and display observed cube pose (ZMQ:5555) from cube_world_observer.py
    3. Real-time error visualization (goal vs observed)

NO hand model, NO ONNX policy - this is purely for visualization during real deployment.

Usage:
    # Terminal 1: Camera observation
    pixi run -e deploy vision

    # Terminal 2: Mirror viewer (this script)
    pixi run -e deploy python deploy/reorient/scripts/toreal_viewer.py

    # Terminal 3: Closed-loop control
    pixi run -e deploy play-real --ckpt <path-to.onnx>

Controls:
    - Double-click goal cube (orange) to select, drag to rotate
    - Green cube shows observed pose from camera
    - ESC to quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # deploy/reorient
DEPLOY_REPO_ROOT = os.path.dirname(os.path.dirname(ROOT_DIR))  # worktree root
sys.path.insert(0, DEPLOY_REPO_ROOT)

from deploy.reorient.lib.config_loader import goal_port, cube_port
from deploy.reorient.lib.zmq_bridge import GoalPublisher, CubeReceiver


# Viewer scene is built programmatically from training-side cube assets so that
# cube geometry (mesh, texture, scale, mass) tracks the training mjcf without
# any vendoring. Cube parameters come from the SINGLE source of truth:
# src/wuji_mjlab/assets/objects/inhand_object/.
_INHAND_PKG = "wuji_mjlab.assets.objects.inhand_object"


def build_viewer_scene_xml() -> str:
    """Generate viewer scene XML string referencing training cube assets.

    Returns:
        MuJoCo XML string. Two mocap bodies: ``goal`` (orange, user-draggable)
        and ``observed_cube`` (textured, mirrors live cube pose from ZMQ).
        Mesh scale and texture both come from training-side cube.xml asset paths,
        so any cube model change on the training side propagates here.
    """
    import importlib.resources

    inhand = importlib.resources.files(_INHAND_PKG)
    mesh_path = str(inhand / "meshes" / "dex_cube.obj")
    tex_path = str(inhand / "textures" / "dex_cube.png")

    # Mesh scale must match training cube.xml mesh scale.
    # SoT: src/wuji_mjlab/assets/objects/inhand_object/xmls/cube.xml `<mesh scale=...>`.
    # We read it at runtime instead of hard-coding to stay self-syncing.
    import xml.etree.ElementTree as ET

    cube_xml_path = inhand / "xmls" / "cube.xml"
    cube_tree = ET.parse(cube_xml_path)
    mesh_el = cube_tree.getroot().find(".//mesh[@name='cube_mesh']")
    if mesh_el is None or "scale" not in mesh_el.attrib:
        raise RuntimeError(
            f"Could not parse mesh scale from training cube.xml at {cube_xml_path}"
        )
    cube_mesh_scale = mesh_el.attrib["scale"]  # e.g. "0.027 0.027 0.027"

    return f"""<mujoco model="cube_viewer">
  <compiler angle="radian"/>
  <asset>
    <texture name="dexcube" type="2d" file="{tex_path}"/>
    <material name="dexcube" texture="dexcube"/>
    <mesh name="cube_mesh" file="{mesh_path}" scale="{cube_mesh_scale}"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.9" rgb2="0.9 0.95 1"
      width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <statistic center="0 0 0.1" extent="0.4" meansize="0.02"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="8192"/>
  </visual>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 .05" type="plane" material="groundplane"/>
    <!-- Goal cube: orange, mocap-controlled. User drags to set orientation. -->
    <body name="goal" pos="-0.15 0 0.1" mocap="true">
      <geom type="mesh" mesh="cube_mesh" rgba="1 0.5 0 0.7" contype="0" conaffinity="0"/>
    </body>
    <!-- Observed cube: textured, mocap; reflects live cube pose from ZMQ:5555. -->
    <body name="observed_cube" pos="0.15 0 0.1" mocap="true">
      <geom type="mesh" mesh="cube_mesh" material="dexcube" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Global state
_goal_publisher = None
_cube_receiver = None
_last_print_time = 0
_step_count = 0


def control_callback(model: mujoco.MjModel, data: mujoco.MjData):
    """Update observed cube pose from ZMQ and publish goal."""
    global _goal_publisher, _cube_receiver, _last_print_time, _step_count

    _step_count += 1

    # Publish goal orientation (mocap 0)
    if _goal_publisher is not None and model.nmocap > 0:
        goal_quat = data.mocap_quat[0].copy()
        _goal_publisher.publish(goal_quat)

    # Update observed cube from ZMQ (mocap 1)
    if _cube_receiver is not None:
        obs_quat, obs_pos, valid = _cube_receiver.get_pose()
        if valid:
            # observed_cube is mocap index 1
            data.mocap_pos[1] = obs_pos
            data.mocap_quat[1] = obs_quat

    # Print status every second
    now = time.time()
    if now - _last_print_time > 1.0:
        goal_quat = data.mocap_quat[0] if model.nmocap > 0 else [1, 0, 0, 0]
        obs_quat = data.mocap_quat[1] if model.nmocap > 1 else [1, 0, 0, 0]
        obs_pos = data.mocap_pos[1] if model.nmocap > 1 else [0, 0, 0]

        # Compute angle error between goal and observed
        goal_inv = np.zeros(4)
        mujoco.mju_negQuat(goal_inv, goal_quat)
        qd = np.zeros(4)
        mujoco.mju_mulQuat(qd, obs_quat, goal_inv)
        angle_err = 2 * np.arcsin(np.clip(np.linalg.norm(qd[1:]), 0, 1)) * 180 / np.pi

        valid_str = "OK" if _cube_receiver and _cube_receiver.cube_count > 0 else "waiting..."
        print(f"[{_step_count:5d}] Goal: ({goal_quat[0]:+.2f},{goal_quat[1]:+.2f},{goal_quat[2]:+.2f},{goal_quat[3]:+.2f}) | "
              f"Obs: ({obs_quat[0]:+.2f},{obs_quat[1]:+.2f},{obs_quat[2]:+.2f},{obs_quat[3]:+.2f}) | "
              f"Err: {angle_err:5.1f}deg | {valid_str}")
        _last_print_time = now


def load_callback(model=None, data=None):
    """MuJoCo viewer load callback."""
    mujoco.set_mjcb_control(None)

    print("=" * 60)
    print("ToReal Viewer - Deployment Visualization")
    print("=" * 60)

    scene_xml = build_viewer_scene_xml()
    print("\nLoading viewer scene (cube assets from training side)...")
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data = mujoco.MjData(model)

    # Reset to home keyframe
    try:
        mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    except KeyError:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    model.opt.timestep = 0.01  # 100 Hz

    # Set control callback
    mujoco.set_mjcb_control(control_callback)

    print("\n" + "=" * 60)
    print("Data Flow:")
    print(f"  - Publishing goal -> ZMQ:{goal_port()}")
    print(f"  - Receiving cube <- ZMQ:{cube_port()}")
    print("")
    print("Controls:")
    print("  - Double-click goal cube (orange) to select")
    print("  - Drag to rotate goal orientation")
    print("  - Green cube shows observed pose from camera")
    print("  - ESC to quit")
    print("=" * 60 + "\n")

    return model, data


def main():
    global _goal_publisher, _cube_receiver

    parser = argparse.ArgumentParser(
        description='ToReal Viewer - Real deployment visualization')
    parser.add_argument('--goal-port', type=int, default=goal_port(),
                        help=f'ZMQ port for goal publishing (default: {goal_port()})')
    parser.add_argument('--cube-port', type=int, default=cube_port(),
                        help=f'ZMQ port for cube observation (default: {cube_port()})')
    parser.add_argument('--no-publish', action='store_true',
                        help='Disable goal publishing (view only)')
    args = parser.parse_args()

    # Start ZMQ publisher (enabled by default)
    if not args.no_publish:
        _goal_publisher = GoalPublisher(port=args.goal_port)
        print(f"Publishing goal on ZMQ:{args.goal_port}")
    else:
        print("Goal publishing disabled (--no-publish)")

    # Start ZMQ receiver
    _cube_receiver = CubeReceiver(port=args.cube_port)
    print(f"Receiving cube pose from ZMQ:{args.cube_port}")

    try:
        mujoco.viewer.launch(loader=load_callback)
    finally:
        if _goal_publisher is not None:
            _goal_publisher.close()
        if _cube_receiver is not None:
            _cube_receiver.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
