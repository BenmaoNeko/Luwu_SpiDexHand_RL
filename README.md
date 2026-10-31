# Luwu SpiDexHand RL

Reinforcement-learning and simulation workspace for the Luwu SpiDex dexterous hand. The training stack is based on [Wuji MJLab](https://github.com/wuji-technology/wuji-mjlab), with Luwu hand assets, cube reorientation, and Rubik's Cube task configurations.

## Repository contents

- `src/` — MuJoCo/MJLab task code, hand assets, observations, actions, rewards, and training configuration.
- `scripts/` — training, playback, and environment utilities.
- `deploy/` — sim-to-real vision and hand-control integration.
- `URDF/` — robot and Rubik's Cube source assets used by the project.
- `Vision/` — cube pose estimation and the Node.js Rubik solver.
- `tests/` — regression tests for task configuration, rewards, and transfer.
- `docs/sim2real/` — deployment setup instructions.

Training outputs, checkpoints, local reference archives, dated experiment notes, and standalone debug/preview tools are excluded from this handoff. Checkpoints must be supplied separately; see `.gitignore`.

## Environment and commands

The checked-in Pixi environment is the reproducible setup for the MJLab code. On a compatible Linux/CUDA host with Pixi installed:

```bash
pixi install
pixi run list-envs
pixi run train --task LuwuHand_Rubik_UFace_Stage1
pixi run play --task LuwuHand_Rubik_UFace_Stage1 --checkpoint-file <checkpoint.pt>
```

Full-cube tasks also require Node.js >=22.12 and the solver dependencies:

```bash
cd Vision/xgo_spidex_cube-master/robot_solver
npm ci
cd ../../..
```

Training options can be passed after the task, for example `--env.scene.num-envs 4096` or `--agent.max-iterations 3000`.

## Upstream

This project reuses and adapts the [Wuji MJLab](https://github.com/wuji-technology/wuji-mjlab) training framework. Refer to the upstream project for its license and upstream components; task usage is described in `src/wuji_mjlab/tasks/reorient/README.md`, and deployment setup is in `docs/sim2real/`.
