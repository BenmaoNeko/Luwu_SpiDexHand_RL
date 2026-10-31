#!/usr/bin/env python3
"""Generate the offline Node-solver pool used by full-cube transfer training."""

from __future__ import annotations

import argparse
from pathlib import Path

from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool


def main() -> None:
  default_output = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "wuji_mjlab"
    / "assets"
    / "objects"
    / "rubik"
    / "full_cube_solver_plan_pool.json"
  )
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=default_output)
  parser.add_argument("--count", type=int, default=256)
  parser.add_argument("--seed", type=int, default=20261008)
  parser.add_argument("--min-scramble-moves", type=int, default=2)
  parser.add_argument("--max-scramble-moves", type=int, default=6)
  args = parser.parse_args()

  pool = generate_solver_plan_pool(
    args.output,
    count=args.count,
    seed=args.seed,
    min_scramble_moves=args.min_scramble_moves,
    max_scramble_moves=args.max_scramble_moves,
  )
  print(
    f"Wrote {len(pool['plans'])} unique plans to {args.output.resolve()} "
    f"(seed={pool['seed']}, solver={pool['solver_fingerprint']})"
  )


if __name__ == "__main__":
  main()
