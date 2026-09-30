#!/usr/bin/env python3
"""Deterministic fingertip-to-cube contact-chain probe.

This probe constructs the registered Luwu task directly, manually writes one
finger's joint positions, and compares contacts from a standard MuJoCo data
mirror with the live MJWarp contact sensor and metric.  It intentionally does
not construct an RL runner or load a model.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

__test__ = False  # pytest must not collect this executable probe as a test module.


TASK_ID = "LuwuHand_Reorient"
ENV_ID = 0
NUM_FINGERS = 4
JOINTS_PER_FINGER = 4
TIP_GEOM_TEMPLATE = "robot/right_finger{finger}_link4_col"
TIP_SITE_TEMPLATE = "robot/right_finger{finger}_tip"
CUBE_GEOM_NAME = "object/cube"


class ContactTestFailure(RuntimeError):
  """Raised when one boundary in the contact chain fails."""


@dataclass(frozen=True)
class RawContact:
  """One standard MuJoCo raw contact involving the cube."""

  geom1: str
  geom2: str
  distance: float
  normal_force: float
  total_force: float


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--finger",
    type=int,
    choices=range(1, NUM_FINGERS + 1),
    default=1,
    help="finger to drive (1-4), default: 1",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=16,
    help="number of deterministic interpolation poses, default: 16",
  )
  parser.add_argument(
    "--step-size",
    type=float,
    default=None,
    help="unused compatibility option; physics uses the task timestep",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="MJLab device, default: cpu",
  )
  return parser.parse_args()


def _geom_name(model, geom_id: int) -> str:
  import mujoco

  return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or (
    f"geom{geom_id}"
  )


def _raw_cube_contacts(
  model,
  raw_data,
  cube_geom_id: int,
) -> list[RawContact]:
  import mujoco

  contacts: list[RawContact] = []
  force = np.zeros(6, dtype=np.float64)
  for contact_id in range(raw_data.ncon):
    contact = raw_data.contact[contact_id]
    if cube_geom_id not in (contact.geom1, contact.geom2):
      continue
    mujoco.mj_contactForce(model, raw_data, contact_id, force)
    contacts.append(
      RawContact(
        geom1=_geom_name(model, contact.geom1),
        geom2=_geom_name(model, contact.geom2),
        distance=float(contact.dist),
        normal_force=float(force[0]),
        total_force=float((force[:3] ** 2).sum() ** 0.5),
      )
    )
  return contacts


def _target_raw_contacts(
  contacts: list[RawContact], target_geom_name: str
) -> list[RawContact]:
  return [
    contact
    for contact in contacts
    if target_geom_name in (contact.geom1, contact.geom2)
  ]


def _warp_target_contacts(env, target_geom_id: int, cube_geom_id: int):
  """Read actual MJWarp contact slots for environment 0."""
  ncon = int(env.sim.data.nacon[ENV_ID].item())
  contacts = []
  geom = env.sim.data.contact.geom
  dist = env.sim.data.contact.dist
  for contact_id in range(ncon):
    geom_pair = [int(value) for value in geom[contact_id].tolist()]
    if cube_geom_id in geom_pair and target_geom_id in geom_pair:
      contacts.append(
        {
          "geom": geom_pair,
          "distance": float(dist[contact_id].item()),
        }
      )
  return ncon, contacts


def _format_raw_contacts(contacts: list[RawContact]) -> str:
  if not contacts:
    return "[]"
  return "[" + "; ".join(
    f"{c.geom1}<->{c.geom2} dist={c.distance:+.6f} "
    f"normal_force={c.normal_force:+.6f} total_force={c.total_force:+.6f}"
    for c in contacts
  ) + "]"


def _make_env(device: str):
  import wuji_mjlab.tasks  # noqa: F401  # register LuwuHand_Reorient
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 1
  cfg.scene.env_spacing = 0.0
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def run_contact_test(
  *, finger: int = 1, steps: int = 16, device: str = "cpu"
) -> int:
  import mujoco
  import torch

  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from wuji_mjlab.tasks.reorient.mdp.metrics import fingertip_contact_count

  if steps < 2:
    raise ContactTestFailure("--steps must be at least 2")

  env = _make_env(device)
  try:
    env.reset(seed=20260920)
    model = env.sim.mj_model
    robot = env.scene["robot"]
    sensor = env.scene["tip_object_contact"]
    metric_cfg = SceneEntityCfg("tip_object_contact")

    target_geom_name = TIP_GEOM_TEMPLATE.format(finger=finger)
    target_site_name = TIP_SITE_TEMPLATE.format(finger=finger)
    cube_geom_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM_NAME
    )
    target_geom_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name
    )
    target_site_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_SITE, target_site_name
    )
    if cube_geom_id < 0 or target_geom_id < 0 or target_site_id < 0:
      raise ContactTestFailure(
        "could not resolve target geometry/site: "
        f"cube={cube_geom_id}, tip_geom={target_geom_id}, tip_site={target_site_id}"
      )

    # The mirror uses the exact compiled MuJoCo model and the same qpos/qvel
    # as env 0, so raw_data.ncon is the standard MuJoCo contact view for the
    # state consumed by the live MJWarp sensor.
    raw_data = mujoco.MjData(model)
    base_position = robot.data.joint_pos[ENV_ID].detach().clone()
    limits = robot.data.joint_pos_limits[ENV_ID].detach().clone()
    finger_slice = slice(
      (finger - 1) * JOINTS_PER_FINGER,
      finger * JOINTS_PER_FINGER,
    )
    target_position = (limits[:, 0] + limits[:, 1]) * 0.5
    target_finger_position = target_position[finger_slice]
    print(
      f"[contact-test] task={TASK_ID} envs=1 finger={finger} "
      f"tip_geom={target_geom_name} cube_geom={CUBE_GEOM_NAME}"
    )
    print(
      "[contact-test] deterministic sweep: home -> per-joint URDF-limit midpoint; "
      "direct qpos writes only"
    )

    first_raw_step: int | None = None
    first_sensor_step: int | None = None
    first_metric_step: int | None = None
    last_trace: str = ""

    for step in range(steps):
      fraction = step / (steps - 1)
      position = base_position.clone()
      position[finger_slice] = (
        base_position[finger_slice]
        + fraction * (target_finger_position - base_position[finger_slice])
      )
      robot.write_joint_state_to_sim(
        position.unsqueeze(0),
        torch.zeros_like(position).unsqueeze(0),
      )
      env.sim.forward()
      env.scene.update(0.0)

      raw_data.qpos[:] = env.sim.data.qpos[ENV_ID].detach().cpu().numpy()
      raw_data.qvel[:] = env.sim.data.qvel[ENV_ID].detach().cpu().numpy()
      mujoco.mj_forward(model, raw_data)
      raw_contacts = _raw_cube_contacts(
        model, raw_data, cube_geom_id
      )
      target_contacts = _target_raw_contacts(raw_contacts, target_geom_name)
      warp_ncon, warp_target_contacts = _warp_target_contacts(
        env, target_geom_id, cube_geom_id
      )

      found = sensor.data.found[ENV_ID].detach().cpu().tolist()
      force = sensor.data.force[ENV_ID].detach().cpu().tolist()
      metric = float(fingertip_contact_count(env, metric_cfg)[ENV_ID].item())
      tip_position = raw_data.site_xpos[target_site_id]
      cube_position = raw_data.geom_xpos[cube_geom_id]
      tip_distance = float(((tip_position - cube_position) ** 2).sum() ** 0.5)
      has_raw = bool(target_contacts)
      has_sensor = found[finger - 1] > 0.0
      has_metric = metric >= 1.0
      if has_raw and first_raw_step is None:
        first_raw_step = step
      if has_sensor and first_sensor_step is None:
        first_sensor_step = step
      if has_metric and first_metric_step is None:
        first_metric_step = step
      last_trace = (
        f"step={step:02d} fraction={fraction:.3f} "
        f"tip_to_cube={tip_distance:.6f} raw_ncon={raw_data.ncon} "
        f"warp_ncon={warp_ncon} sensor_found={found} "
        f"sensor_force={force} metric_fingertip_contact_count={metric:.1f} "
        f"raw_cube_contacts={_format_raw_contacts(raw_contacts)} "
        f"warp_target_contacts={warp_target_contacts}"
      )
      print(f"[contact-test] {last_trace}")

      if has_raw:
        if not has_sensor:
          raise ContactTestFailure(
            "raw MuJoCo tip/cube contact exists but sensor found is zero; "
            f"{last_trace}"
          )
        if not has_metric:
          raise ContactTestFailure(
            "sensor found is positive but fingertip_contact_count is zero; "
            f"{last_trace}"
          )
        if not warp_target_contacts:
          raise ContactTestFailure(
            "standard MuJoCo raw contact exists but live MJWarp contact list "
            f"does not contain the target pair; {last_trace}"
          )
        print(
          "[contact-test] PASS: raw contact -> live MJWarp contact -> "
          f"sensor found=1 -> metric={metric:.1f} at step {step}"
        )
        return 0

    raise ContactTestFailure(
      "no target fingertip/cube raw contact occurred during the sweep; "
      f"first_raw_step={first_raw_step}, first_sensor_step={first_sensor_step}, "
      f"first_metric_step={first_metric_step}; last={last_trace}"
    )
  finally:
    env.close()


def main() -> int:
  args = _parse_args()
  del args.step_size
  try:
    return run_contact_test(
      finger=args.finger,
      steps=args.steps,
      device=args.device,
    )
  except ContactTestFailure as exc:
    print(f"[contact-test] FAIL: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
