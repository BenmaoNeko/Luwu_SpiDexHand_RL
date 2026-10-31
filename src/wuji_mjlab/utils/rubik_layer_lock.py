"""Switch joint-coupled Rubik layers with minimal cubelet pose disturbance."""

import mujoco
import numpy as np
import torch

from wuji_mjlab.tasks.reorient.mdp.rubik_solver import face_layer_joint_names


class RubikLayerLock:
  def __init__(self, sim, joint_names):
    self.sim = sim
    self.joint_names = tuple(joint_names)
    self.model = sim.mj_model
    self.scratch = mujoco.MjData(self.model)
    self.ids = {}
    for index in range(self.model.neq):
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, index) or ""
      # Entity attachment prefixes names (e.g. object/); resolve actual model names.
      short = name.rsplit("/", 1)[-1]
      if short.startswith(("dynamic_lock_", "dynamic_follow_")):
        if short in self.ids:
          raise ValueError(f"Ambiguous Rubik equality: {short}")
        self.ids[short] = index
    for name in self.joint_names:
      if f"dynamic_lock_{name}" not in self.ids:
        raise ValueError(f"Missing Rubik dynamic lock for {name}")
    self.active_face = None

  def _following_hinge(self, piece, driver_id):
    """Find a single hinge parallel to the driver, including Euler singularities."""
    joint_ids = [
      self.model.eq_obj1id[self.ids[f"dynamic_lock_cubelet:rot{axis}:{piece}"]]
      for axis in "xyz"
    ]
    addresses = self.model.jnt_qposadr[joint_ids]
    original = self.scratch.qpos[addresses].copy()
    body = self.model.jnt_bodyid[joint_ids[0]]
    pose = self.scratch.xmat[body].copy()
    driver_axis = self.scratch.xaxis[driver_id].copy()
    dots = self.scratch.xaxis[joint_ids] @ driver_axis
    if np.max(np.abs(dots)) < 0.999:
      # Near the Euler singularity, try both gauges and use the one that
      # exposes a parallel hinge with the smallest physical pose change.
      best = None
      for dx, dz in ((np.pi / 2, np.pi / 2), (np.pi / 2, -np.pi / 2)):
        self.scratch.qpos[addresses] = original + [dx, 0, dz]
        mujoco.mj_forward(self.model, self.scratch)
        candidate_dots = self.scratch.xaxis[joint_ids] @ driver_axis
        if np.max(np.abs(candidate_dots)) < 0.999:
          continue
        pose_error = np.max(np.abs(self.scratch.xmat[body] - pose))
        if best is None or pose_error < best[0]:
          best = (pose_error, self.scratch.qpos[addresses].copy())
      if best is None:
        raise ValueError(f"Cannot find a single-axis layer for {piece}")
      self.scratch.qpos[addresses] = best[1]
      mujoco.mj_forward(self.model, self.scratch)
      dots = self.scratch.xaxis[joint_ids] @ driver_axis
    index = int(np.argmax(np.abs(dots)))
    return "xyz"[index], float(np.sign(dots[index]))

  def select(self, state, face_id=None):
    """None freezes all internal joints; a face frees one coherent physical layer."""
    self.scratch.qpos[:] = self.sim.data.qpos[0].detach().cpu().numpy()
    original_qpos = self.scratch.qpos.copy()
    self.scratch.eq_active[:] = False
    mujoco.mj_forward(self.model, self.scratch)
    original_axes = self.scratch.xaxis.copy()
    active = np.zeros(self.model.neq, dtype=bool)
    released = set()
    if face_id is not None:
      layer = face_layer_joint_names(face_id, state, self.joint_names)
      driver = layer[0]
      face = driver.split(":", 2)[2]
      driver_id = self.model.eq_obj1id[self.ids[f"dynamic_lock_{driver}"]]
      released.add(driver)
      for hinge in layer[1:]:
        piece = hinge.split(":", 2)[2]
        axis, sign = self._following_hinge(piece, driver_id)
        released.add(f"cubelet:rot{axis}:{piece}")
        eq_id = self.ids[f"dynamic_follow_{face}:{axis}:{piece}"]
        first, second = self.model.eq_obj1id[eq_id], self.model.eq_obj2id[eq_id]
        a, b = self.model.jnt_qposadr[[first, second]]
        offset = (self.scratch.qpos[a] - self.model.qpos0[a]) - sign * (
          self.scratch.qpos[b] - self.model.qpos0[b]
        )
        self.model.eq_data[eq_id, :5] = [offset, sign, 0, 0, 0]
        active[eq_id] = True
    for name in self.joint_names:
      eq_id = self.ids[f"dynamic_lock_{name}"]
      joint_id = self.model.eq_obj1id[eq_id]
      self.model.eq_data[eq_id, :5] = [
        self.scratch.qpos[self.model.jnt_qposadr[joint_id]]
        - self.model.qpos0[self.model.jnt_qposadr[joint_id]],
        0,
        0,
        0,
        0,
      ]
      active[eq_id] = name not in released
    # In-place writes preserve CUDA graph references. Only this object's
    # equalities are touched, never unrelated environment constraints.
    indices = list(self.ids.values())
    device = self.sim.data.qpos.device
    changed = np.flatnonzero(self.scratch.qpos != original_qpos)
    if len(changed):
      # Map angular rates into the equivalent chart as well. At a singularity
      # use the minimum-norm representation, not stale Euler-coordinate rates.
      velocities = self.sim.data.qvel[0].detach().cpu().numpy().copy()
      for name in self.joint_names:
        if not name.startswith("cubelet:rotx:"):
          continue
        piece = name.split(":", 2)[2]
        joints = [
          self.model.eq_obj1id[self.ids[f"dynamic_lock_cubelet:rot{a}:{piece}"]]
          for a in "xyz"
        ]
        if not np.intersect1d(self.model.jnt_qposadr[joints], changed).size:
          continue
        dofs = self.model.jnt_dofadr[joints]
        omega = original_axes[joints].T @ velocities[dofs]
        rates = np.linalg.lstsq(self.scratch.xaxis[joints].T, omega, rcond=1e-6)[0]
        if np.linalg.norm(self.scratch.xaxis[joints].T @ rates - omega) > 1e-5:
          raise ValueError(
            "Wait for the internal layer to settle before changing Euler chart"
          )
        velocities[dofs] = rates
      self.sim.data.qvel[0] = torch.as_tensor(
        velocities, device=device, dtype=self.sim.data.qvel.dtype
      )
      self.sim.mj_data.qvel[:] = velocities
      # Keep the chosen Euler chart coordinates synchronized with the simulator.
      self.sim.data.qpos[0, changed] = torch.as_tensor(
        self.scratch.qpos[changed], device=device, dtype=self.sim.data.qpos.dtype
      )
      self.sim.mj_data.qpos[changed] = self.scratch.qpos[changed]
      for name in self.joint_names:
        joint = self.model.eq_obj1id[self.ids[f"dynamic_lock_{name}"]]
        state[name] = float(self.scratch.qpos[self.model.jnt_qposadr[joint]])
    self.sim.model.eq_data[0, indices] = torch.as_tensor(
      self.model.eq_data[indices], device=device, dtype=self.sim.model.eq_data.dtype
    )
    self.sim.data.eq_active[0, indices] = torch.as_tensor(
      active[indices], device=device, dtype=self.sim.data.eq_active.dtype
    )
    self.sim.mj_data.eq_active[indices] = active[indices]
    self.active_face = face_id
    if face_id is None:
      print(
        "[Rubik lock] all internal layers locked at current angles; whole-cube flip allowed"
      )
    else:
      print(
        f"[Rubik lock] top operation layer {face} unlocked; 8 pieces follow its driver"
      )
