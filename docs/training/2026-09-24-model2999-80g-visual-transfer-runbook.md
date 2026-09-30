# model_2999 → 80 g Cube / Visual-Robust Transfer Runbook

Date: 2026-09-24  
Task: `LuwuHand_Reorient_80g_VisualTransfer`  
Source run: `Luwu_FromScratch_Curriculum`, final checkpoint `model_2999.pt`

## Purpose and scope

This transfer isolates two sim-to-real differences requested for the physical
80 g cube: lower cube mass and noisier/delayed visual orientation measurements.
The already learned all-face continuous task remains at its final stage. The
fixed arm, hand model, goals, reward scales, and action interface are not
otherwise redesigned. The existing 120 g tasks remain unchanged.

The transfer loads only the actor and its observation normalizer. Critic,
optimizer, environment state, curriculum state, and iteration are fresh. Do
not pass `--resume` together with `--transfer-checkpoint-file`.

## Source checkpoint and interface

Local checkpoint:

`/home/benmao/桌面/Luwu_Hand_Wuji/logs/rsl_rl/wuji_reorient/remote-2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/model_2999.pt`

Server checkpoint used by the startup smoke test:

`/root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main/logs/rsl_rl/wuji_reorient/2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/model_2999.pt`

- SHA-256: `399b3876a714fc9ed19e9305742942f6c7595a10672d11b9ce46bb955d888313`
- Iteration: `2999` (the final checkpoint of the 3000-iteration run)
- Actor observation: 162
- Critic observation: 344 (not transferred)
- Action: 16

The task keeps the policy inputs that make up the source actor's 162-wide
input: three frames each of normalized joint angles (48), joint-target error
(48), cube orientation error (18), and previous action (48). The four newer
vision metadata channels are omitted because they would change the actor input
to 166 and fail the transfer loader's strict shape check. The cube orientation
error is still corrupted by the visual measurement model. The critic remains
privileged and is initialized from scratch.

## Cube mass

- Nominal side length: 54 mm (unchanged).
- Nominal mass: **0.080 kg (80 g)**.
- Stage 3 body-mass/inertia multiplier: **0.85–1.15**.
- Effective randomized mass: **0.068–0.092 kg (68–92 g)**.

Mass is overridden in the new task's MuJoCo spec, so the generic mass/inertia
randomizer scales from 80 g. No shared XML or existing task default is
changed; the original cube remains 120 g in the old tasks. The body mass and
inertia use the existing physically consistent shared multiplier.

## Task and reward contract

- Start directly at Stage 3: all six cube faces, continuous successful goals,
  70% face-flip targets / 30% yaw-rotation targets.
- Flip targets sample any of the other five faces; yaw targets retain the
  Stage 3 small-angle rotation task.
- Success tolerance: 4°; hold: 10 policy steps; a successful target does not
  end the episode.
- Rotation-success reward: `5`; flip-success reward: `40`.
- Action-smoothness weight: `-0.10`; finger-collision weight: `-0.25`.
- The other Stage 3 reward terms, contact requirements, 0.5 N·m torque limit,
  and non-visual physical DR ranges are retained from the source task.
- Full URDF joint limits and the 16-dimensional normalized position-target
  action remain unchanged.

## Visual measurement DR

The new visual profile is transfer-task-only; profiles 0–3 are untouched.
The corrupted signal is the cube's orientation in the wrist-Tag frame; the
critic and physics state remain clean.

| Component | Transfer range | Interpretation |
| --- | ---: | --- |
| Per-axis orientation noise | uniform ±7° | Rotation-vector component bounds; combined magnitude can reach about 12.1° |
| Slowly varying bias | ±5° | Unchanged from Stage 3 |
| Bias random-walk step | 0.3° | Unchanged from Stage 3 |
| Calibration orientation error | ±5° | Unchanged from Stage 3 |
| Measurement delay | 1–8 policy steps | With 50 ms control steps: 50–400 ms |
| Packet dropout | 8% per policy step | Reuse the previous measured pose while dropped |
| Maximum stale hold | 6 policy steps | At most 300 ms at 50 ms control steps |
| Outlier injection | 2% | Unchanged from policy observation config |

The play configuration uses the 80 g nominal cube but disables this synthetic
visual DR, matching the existing preview convention.

## Training command

Run from the repository root on the training host:

```bash
cd /root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/src/wuji_rl_libs/rsl_rl"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/lib:${LD_LIBRARY_PATH:-}"
export TRANSFER_CHECKPOINT=/root/gpufree-data/Luwu_Hand_Wuji/reference/wuji-mjlab-main/logs/rsl_rl/wuji_reorient/2026-09-23_17-52-30_Luwu_FromScratch_Curriculum/model_2999.pt

/root/gpufree-data/mojuco_offload/.luwu_mjlab_repro/bin/python \
  scripts/train/train_rsl_rl.py \
  --task LuwuHand_Reorient_80g_VisualTransfer \
  --transfer-checkpoint-file "$TRANSFER_CHECKPOINT" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --agent.logger tensorboard \
  --gpu-ids '[0]'
```

This starts a new 3000-iteration actor-transfer run. It is not a full resume.
The run should log a `transfer_source.json` confirming the checkpoint hash and
that only `actor` / `actor_normalizer` were loaded.

## Preflight checklist

Before running PPO, confirm:

1. The task registry lists `LuwuHand_Reorient_80g_VisualTransfer`.
2. Its built policy observation shape is 162 and action shape is 16.
3. The MuJoCo cube body mass before DR is 0.080 kg; mass DR yields 0.068–0.092 kg.
4. Training visual state allocates at least nine quaternion history slots
   (current sample plus up to eight delayed policy steps).
5. The selected checkpoint SHA-256 matches the value above.

Do not start training from a 166-input checkpoint/task pairing; the transfer
loader intentionally rejects actor shape mismatches.
