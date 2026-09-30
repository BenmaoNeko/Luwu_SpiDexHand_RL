# From-scratch staged continuous reorientation curriculum

This is the landed training curriculum for `LuwuHand_Reorient_FromScratch_Curriculum`.
The observation groups and full-URDF-limit action interface are unchanged.

| Stage | Task | Goal mix | Target faces | Episode behavior | Physics |
|---:|---|---:|---|---|---|
| 0 | adjacent continuous | 70% flip / 30% +15° rotation | one of four adjacent faces | continue after every success | nominal |
| 1 | adjacent continuous | 70% flip / 30% +15° rotation | one of four adjacent faces | continue after every success | light contact/dynamics DR |
| 2 | all-face continuous | 70% flip / 30% +15° rotation | any of five non-current faces | continue after every success | medium contact/dynamics DR |
| 3 | all-face continuous | 70% flip / 30% +15° rotation | any of five non-current faces | continue after every success | full approved task DR |

The 70/30 ratio is applied whenever a new goal is sampled. A flip goal is
sampled from an adjacent face in stages 0–1, and from any of the five other
faces in stages 2–3. Rotations advance in approximately 15° increments.
All stages keep the episode alive after a successful goal and proceed to the
next goal.

Stage-specific success thresholds and valid-contact hold steps are:

| Stage | Error tolerance | Contact hold |
|---:|---:|---:|
| 0 | 8° | 5 steps |
| 1 | 6° | 8 steps |
| 2 | 6° | 8 steps |
| 3 | 4° | 10 steps |

Promotion uses a rolling window of at least 256 completed environments. Stages
0 and 1 require flip success ≥ 25% and cage-drop rate ≤ 20%. Promotion from
stage 2 additionally requires flip success ≥ 20%, mean consecutive flip count
≥ 1.5, face coverage ≥ 2.5, and cage-drop rate ≤ 20%. Rotation successes do
not promote the curriculum.

Action-smoothness and finger self-collision penalties increase by stage:
`(-0.04, -0.08)`, `(-0.06, -0.12)`, `(-0.08, -0.18)`, and `(-0.10, -0.25)`.
The from-scratch curriculum uses success reward weights of `5` for rotation
and `40` for flipping at every stage.
The final stage is selected in play mode, but the curriculum manager and
training randomization are disabled there.

## Verification record

The previous 50/50 revision passed 158 reorient tests. The current 70/30 and
5/40 reward update has not yet been tested.
