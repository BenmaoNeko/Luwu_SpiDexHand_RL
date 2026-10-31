"""Frozen-policy, first-episode friction ablation; never changes training."""
import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import requires_model_fields
from replay_flip_bottom_tip_real import load_actor
from tensordict import TensorDict
from wuji_mjlab.tasks.reorient.config.luwu_hand.rubik_env_cfg import (
    luwu_hand_rubik_flip_bottom_tip_train_env_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.rubik_solver import generate_solver_plan_pool


@requires_model_fields("geom_friction")
def fixed_friction(env, env_ids, asset_cfg, scale_range):
    dr.geom_friction(env, env_ids, ranges=scale_range, asset_cfg=asset_cfg,
                     operation="scale", axes=[0])


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def summarize(rows):
    n = len(rows)
    s = sum(row[1] for row in rows)
    h = sum(row[2] for row in rows)
    sh = sum(row[1] and row[2] for row in rows)
    return dict(legs=n, successes=s, contacted=h, supported_successes=sh,
                success_rate=s/n if n else None,
                success_after_contact=sh/h if h else None,
                contact_given_success=sh/s if s else None,
                geometry_invalid=sum(not row[3] for row in rows),
                overflow=sum(row[4] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--friction", choices=("high", "low"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scale = (.7, 1.1) if args.friction == "high" else (.05, .1)
    torch.manual_seed(args.seed)
    actor = load_actor(args.checkpoint).to(args.device)
    with tempfile.TemporaryDirectory(prefix="friction-ablation-") as directory:
        pool = Path(directory)/"pool.json"
        generate_solver_plan_pool(pool, count=2, seed=args.seed,
                                  min_scramble_moves=2, max_scramble_moves=2)
        cfg = luwu_hand_rubik_flip_bottom_tip_train_env_cfg(
            num_envs=args.num_envs, plan_pool_path=str(pool))
        cfg.seed = args.seed
        cfg.sim.njmax = 4096
        command_cfg = cfg.commands["reorient_command"]
        # Same full-strength non-friction DR in both arms, frozen throughout.
        command_cfg.smart_cube_curriculum = replace(
            command_cfg.smart_cube_curriculum, play=True, play_alpha=1.)
        term = cfg.events["smart_cube_hand_friction"]
        term.func = fixed_friction
        term.params["scale_range"] = scale
        env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
        try:
            obs, _ = env.reset(seed=args.seed)
            command = env.command_manager.get_term("reorient_command")
            tracker = command.bottom_tip_runtime
            initial = dict(policy=digest(obs["policy"]),
                           robot=digest(env.scene["robot"].data.joint_pos),
                           cube=digest(command.object.data.root_link_pose_w),
                           cube_joints=digest(command.object.data.joint_pos),
                           requested_face=digest(command.requested_face))
            asset_cfg = env.event_manager.get_term_cfg("smart_cube_hand_friction").params["asset_cfg"]
            ids = env.scene["robot"].indexing.geom_ids[asset_cfg.geom_ids].long()
            actual = env.sim.model.geom_friction[:, ids, 0]
            default = env.sim.get_default_field("geom_friction")[ids, 0]
            ratios = actual/default[None, :]
            measured = dict(min=ratios.min().item(), max=ratios.max().item(),
                            mean=ratios.mean().item(), geoms=len(ids))
            assert measured["min"] >= scale[0]-1e-6
            assert measured["max"] <= scale[1]+1e-6
            alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
            rows = []
            original_finish = tracker.requests.finish

            def finish(ids, success):
                selected = tracker.requests.active[ids] & alive[ids]
                chosen = ids[selected]
                if chosen.numel():
                    values = torch.stack((chosen, success[selected].long(),
                        tracker.requests.supported[chosen].long(),
                        tracker.geometry_valid[chosen].long(),
                        tracker.overflow[chosen].long()), dim=-1).cpu().tolist()
                    rows.extend(values)
                original_finish(ids, success)

            tracker.requests.finish = finish
            ends = {}
            for step in range(args.steps):
                with torch.inference_mode():
                    actions = actor(TensorDict({"policy": obs["policy"]},
                                              batch_size=[args.num_envs]))
                obs, _, terminated, truncated, _ = env.step(actions)
                done = terminated | truncated
                for env_id in (done & alive).nonzero().flatten().cpu().tolist():
                    ends[env_id] = step+1
                alive &= ~done
                if (step+1) % 200 == 0:
                    print(f"PROGRESS {args.friction} seed={args.seed} steps={step+1} "
                          f"first_episodes_alive={int(alive.sum())} legs={len(rows)}", flush=True)
                if not alive.any():
                    break
            per_env = []
            for env_id in range(args.num_envs):
                own = [row for row in rows if row[0] == env_id]
                per_env.append(dict(env_id=env_id, steps=ends.get(env_id, step+1),
                                    horizon_censored=env_id not in ends, **summarize(own)))
            result = dict(checkpoint=str(args.checkpoint),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                seed=args.seed, friction=scale, nonfriction_dr_alpha=1.,
                deterministic_actor=True, initial_hashes=initial,
                measured_friction_multiplier=measured, num_envs=args.num_envs,
                max_steps=args.steps, actual_steps=step+1,
                unit="settled 90-degree legs in each environment's first episode only",
                contact_window_deg=40, success_rewards=[-1,30],
                summary=summarize(rows),
                envs_with_success=sum(r["successes"] > 0 for r in per_env),
                mean_successes_per_env=sum(r["successes"] for r in per_env)/args.num_envs,
                horizon_censored_envs=int(alive.sum()),
                mean_first_episode_steps=sum(r["steps"] for r in per_env)/args.num_envs,
                per_env=per_env, rows=rows,
                limitations="Finite horizon; unfinished legs excluded; simulation not real robot.")
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            print("FRICTION_RESULT "+json.dumps({k:v for k,v in result.items()
                                                if k not in ("rows", "per_env")}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
