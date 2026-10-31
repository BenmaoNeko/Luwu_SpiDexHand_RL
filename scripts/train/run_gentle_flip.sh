#!/usr/bin/env bash
# Explicit two-stage launcher; does not overwrite the source checkpoint/run.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/wuji-mpl}"
gentle_python="${GENTLE_PYTHON:-$project_dir/.pixi/envs/default/bin/python}"
gentle_checkpoint="${GENTLE_CHECKPOINT:-$project_dir/logs/rsl_rl/wuji_reorient/remote_20261027_SuccessGated20_FlowFix_EMA075/model_2999.pt}"
gentle_mode="${1:-}"
case "$gentle_mode" in
  ema-only)
    gentle_args=(--gentle-ema-only)
    shift
    ;;
  velocity)
    if [[ $# -lt 2 ]]; then
      echo 'Usage: bash scripts/train/run_gentle_flip.sh velocity calibration.json [training overrides]' >&2
      exit 2
    fi
    gentle_args=(--gentle-calibration "$2")
    shift 2
    ;;
  *)
    echo 'Usage: bash scripts/train/run_gentle_flip.sh {ema-only|velocity calibration.json} [training overrides]' >&2
    exit 2
    ;;
esac
"$gentle_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA required; refusing silent CPU long training"'
if [[ ! -f "$gentle_checkpoint" ]]; then
  echo "Checkpoint missing: $gentle_checkpoint" >&2
  exit 2
fi
gentle_pool="${GENTLE_PLAN_POOL:-$project_dir/logs/gentle_flip/solver_pool_20261028.json}"
if [[ ! -f "$gentle_pool" ]]; then
  "$gentle_python" scripts/generate_rubik_plan_pool.py --output "$gentle_pool" --count 256
fi
# Fingerprint validation is mandatory, including for an existing pool.
"$gentle_python" -c 'import sys; from wuji_mjlab.tasks.reorient.mdp.rubik_solver import load_solver_plan_pool; load_solver_plan_pool(sys.argv[1])' "$gentle_pool"
exec "$gentle_python" scripts/train/train_rsl_rl.py \
  --task LuwuHand_Rubik_FlipGentleTransfer \
  --transfer-checkpoint-file "$gentle_checkpoint" \
  "${gentle_args[@]}" \
  "env.scene.num_envs=4096" \
  "env.commands.reorient_command.plan_pool_path=$gentle_pool" \
  "agent.max_iterations=3000" \
  "agent.run_name=Luwu_GentleFlip_${gentle_mode}_EMA050" \
  "$@"
