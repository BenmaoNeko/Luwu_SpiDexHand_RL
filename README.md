# Luwu SpiDexHand RL

Reinforcement-learning and simulation workspace for the Luwu SpiDex dexterous hand. The training stack is based on [Wuji MJLab](https://github.com/wuji-technology/wuji-mjlab), with Luwu hand assets, cube reorientation, and Rubik's Cube task configurations.

## Repository contents

- `src/` — MuJoCo/MJLab task code, hand assets, observations, actions, rewards, and training configuration.
- `scripts/` — training, playback, and environment utilities.
- `deploy/` — sim-to-real vision and hand-control integration.
- `URDF/` — robot and Rubik's Cube source assets used by the project.
- `docs/` — training baselines, runbooks, task designs, and sim-to-real notes.

Training outputs, checkpoints, and local reference archives are intentionally excluded from Git; see `.gitignore`.

## Environment and commands

The checked-in Pixi environment is the reproducible setup for the MJLab code. On a compatible Linux/CUDA host with Pixi installed:

```bash
pixi install
pixi run list-envs
pixi run train --task LuwuHand_Rubik_UFace_Stage1
pixi run play --task LuwuHand_Rubik_UFace_Stage1 --checkpoint-file <checkpoint.pt>
```

Training options can be passed after the task, for example `--env.scene.num-envs 4096` or `--agent.max-iterations 3000`.

## Upstream

This project reuses and adapts the [Wuji MJLab](https://github.com/wuji-technology/wuji-mjlab) training framework. Refer to the upstream project for its license and upstream components; project-specific behavior and procedures are documented under `docs/`.
