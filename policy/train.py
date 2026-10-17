"""PPO training of one E2 run: ``python train.py --memory gru --seed 0 --out RUN_DIR``.

Owns the run directory (DESIGN.md §4): ``config.json``, ``log.csv``, ``val.csv``,
``ckpt_last.pt`` and ``ckpt_best.pt``, and the iteration loop that ties
``rollout`` and ``ppo`` together.

Invariants:
- Progress is counted in alive agent transitions ("agent-steps").
- ``ckpt_last.pt`` holds everything needed to continue: weights, optimiser,
  reward scaler (statistics and per-row return carry), iteration, agent-steps,
  elapsed seconds, torch RNG state and the config. After ``--resume`` the
  per-iteration generators (minibatch order, worker scenes and sampling) are
  derived from ``(seed, iteration)``, so they do not depend on history; env
  state is *not* restored, the workers start fresh scenes.
- ``log.csv`` / ``val.csv`` are trimmed to the checkpoint's iteration on resume,
  so each iteration appears exactly once.
- A finished run (agent-steps >= budget) exits at once on ``--resume``.
- Ctrl-C saves ``ckpt_last.pt``, shuts the workers down and exits with 130.
"""
from __future__ import annotations

import _paths  # noqa: F401

import argparse
import csv
import json
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from env import obs_dim
from evaluate import ROW_METRICS
from nets import ActorCritic
from policyconfig import (LAYOUTS_E2, EnvConfig, PPOConfig, RunConfig, sim_config,
                          sim_config_from_dict)
from ppo import RewardScaler, gae, replay, update
from rollout import WorkerPool

LOG_COLS = ["iteration", "agent_steps", "seconds", "agent_steps_per_s", "episodes",
            "success_rate", "collision_rate", "timeout_rate", "mean_return", "policy_loss",
            "value_loss", "entropy", "approx_kl", "clip_frac", "explained_variance", "lr",
            "log_std_turn", "log_std_speed"]
VAL_METRICS = ROW_METRICS                      # the overall metrics of §5.3


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# ------------------------------------------------------------------ csv

def _read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def trim_csv(path: Path, max_iteration: int):
    """Drop rows written after the checkpoint we resume from."""
    rows = _read_rows(path)
    if not rows:
        return
    keep = [r for r in rows if int(r["iteration"]) <= max_iteration]
    if len(keep) != len(rows):
        write_csv(path, list(rows[0].keys()), keep)


def write_csv(path: Path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def append_csv(path: Path, cols, row: dict):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------ checkpoints

def save_ckpt(path: Path, model, opt, scaler, state: dict, cfg: RunConfig, sim_cfg):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimiser": opt.state_dict(), "scaler": scaler.state_dict(),
                "iteration": state["iteration"], "agent_steps": state["agent_steps"],
                "seconds": state["seconds"], "rng": torch.get_rng_state(),
                "config": cfg.to_dict(sim_cfg)}, tmp)
    tmp.replace(path)                                     # atomic on both OSes


def load_ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_best(path: Path, model, cfg: RunConfig, sim_cfg, iteration, agent_steps, val: dict):
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": cfg.to_dict(sim_cfg), "iteration": iteration,
                "agent_steps": agent_steps, "val": val}, path)


def restore(path: Path, model, opt, scaler, state: dict):
    """Load ``ckpt_last.pt`` into the live objects (weights, optimiser, scaler, counters, RNG)."""
    ck = load_ckpt(path)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimiser"])
    scaler.load_state_dict(ck["scaler"])
    state.update(iteration=ck["iteration"], agent_steps=ck["agent_steps"], seconds=ck["seconds"])
    torch.set_rng_state(ck["rng"])


def better(new: dict, best: dict | None) -> bool:
    if best is None:
        return True
    return (new["success_rate"], -new["collision_rate"]) > (best["success_rate"],
                                                            -best["collision_rate"])


# ------------------------------------------------------------------ main

def episode_stats(records: list) -> dict:
    n = sum(r["n_agents"] for r in records)
    if n == 0:
        return {"episodes": 0, "success_rate": float("nan"), "collision_rate": float("nan"),
                "timeout_rate": float("nan"), "mean_return": float("nan")}
    return {"episodes": n, "success_rate": sum(r["goal"] for r in records) / n,
            "collision_rate": sum(r["collision"] for r in records) / n,
            "timeout_rate": sum(r["timeout"] for r in records) / n,
            "mean_return": sum(r["return_sum"] for r in records) / n}


def run(cfg: RunConfig, sim_cfg, out: Path, val_suite: Path | None = None, resume=False,
        val_workers: int = 0, check_replay_every: int = 25, quiet=False,
        stop_after: int | None = None) -> dict:
    """Train until the budget is spent (or ``stop_after`` iterations, for tests)."""
    out.mkdir(parents=True, exist_ok=True)
    p = cfg.ppo
    device = resolve_device(p.device)
    torch.manual_seed(cfg.seed)
    model = ActorCritic(obs_dim(cfg.env), cfg.net).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p.lr, eps=1e-5)
    scaler = RewardScaler(p.gamma)
    state = {"iteration": 0, "agent_steps": 0, "seconds": 0.0}
    last, best_path = out / "ckpt_last.pt", out / "ckpt_best.pt"

    if resume and last.exists():
        restore(last, model, opt, scaler, state)
        trim_csv(out / "log.csv", state["iteration"])
        trim_csv(out / "val.csv", state["iteration"])
        if not quiet:
            print(f"resumed at iteration {state['iteration']}, "
                  f"{state['agent_steps']:,} agent-steps", flush=True)
        if state["agent_steps"] >= p.total_steps:
            if not quiet:
                print("run already finished", flush=True)
            return state
    else:
        for f in ("log.csv", "val.csv"):                   # a fresh run starts clean
            (out / f).unlink(missing_ok=True)
    cfg.save(out / "config.json", sim_cfg)

    validator = None
    if val_suite is not None:
        from evaluate import Validator
        validator = Validator(val_suite, sim_cfg, val_workers)
    best = None
    if best_path.exists():
        best = load_ckpt(best_path).get("val")

    pool = WorkerPool(cfg, sim_cfg, epoch=state["iteration"] if state["iteration"] else None)
    interrupted = False
    t_start = time.time() - state["seconds"]
    rows_done = 0
    try:
        while state["agent_steps"] < p.total_steps:
            t0 = time.time()
            it = state["iteration"] + 1
            lr = p.lr * max(0.0, 1.0 - state["agent_steps"] / p.total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = pool.collect(model.get_weights())
            alive = batch["alive"]
            done = batch["terminal"] | batch["truncated"]
            scaler.update(batch["reward"].astype(np.float64), done, alive)
            adv, ret = gae(batch, p.gamma, p.lam, reward=scaler.scale(batch["reward"]))
            if check_replay_every and (it == 1 or it % check_replay_every == 0):
                lp, v, _ = replay(model, batch, p.L)           # §3.4: the bug detector
                err = max(np.abs(lp - batch["logp"]).max(), np.abs(v - batch["value"]).max())
                if err > (1e-4 if device.type == "cpu" else 1e-2):
                    print(f"WARNING: replay differs from rollout by {err:.2e}", flush=True)
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, it, 777]))
            model.train()
            stats = update(model, opt, batch, adv, ret, p, device, rng)
            n_alive = int(alive.sum())
            state["iteration"] = it
            state["agent_steps"] += n_alive
            state["seconds"] = time.time() - t_start
            dt = time.time() - t0
            ep = episode_stats(batch["records"])
            row = {"iteration": it, "agent_steps": state["agent_steps"],
                   "seconds": round(state["seconds"], 1),
                   "agent_steps_per_s": round(n_alive / dt, 1), **ep, **stats, "lr": lr,
                   "log_std_turn": float(model.log_std[0].detach()), "log_std_speed": float(model.log_std[1].detach())}
            append_csv(out / "log.csv", LOG_COLS, row)
            rows_done += 1
            if not quiet:
                print(f"it {it:4d}  steps {state['agent_steps']:>11,}  "
                      f"{n_alive / dt:7.0f} st/s  succ {ep['success_rate']:.3f}  "
                      f"coll {ep['collision_rate']:.3f}  ret {ep['mean_return']:8.2f}  "
                      f"kl {stats['approx_kl']:.4f}  ev {stats['explained_variance']:.2f}",
                      flush=True)

            finished = state["agent_steps"] >= p.total_steps or (
                stop_after is not None and rows_done >= stop_after)
            if validator is not None and (it % cfg.val_every == 0 or finished):
                val = validator.run(cfg, model.get_weights())
                append_csv(out / "val.csv", val_cols(val), {"iteration": it,
                           "agent_steps": state["agent_steps"], **val})
                if not quiet:
                    print(f"  val  success {val['success_rate']:.3f}  "
                          f"collision {val['collision_rate']:.3f}", flush=True)
                if better(val, best):
                    best = val
                    save_best(best_path, model, cfg, sim_cfg, it, state["agent_steps"], val)
            if it % cfg.ckpt_every == 0 or finished:
                save_ckpt(last, model, opt, scaler, state, cfg, sim_cfg)
            if finished:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted: saving ckpt_last.pt", flush=True)
        save_ckpt(last, model, opt, scaler, state, cfg, sim_cfg)
    finally:
        pool.close()
        if validator is not None:
            validator.close()
    if interrupted:
        raise SystemExit(130)
    return state


def val_cols(val: dict) -> list:
    layouts = sorted(k for k in val if k.startswith("success_") and k != "success_rate")
    return ["iteration", "agent_steps"] + VAL_METRICS + layouts


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--memory", choices=["mlp", "stack", "gru"], default="gru")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-suite", type=Path, default=None)
    ap.add_argument("--val-workers", type=int, default=0, help="processes for validation")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--layouts", default=None, help="comma list, default all ten")
    ap.add_argument("--agents", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--goal-obs", choices=["bearing", "bearing_distance"], default=None)
    ap.add_argument("--heading", choices=["random", "generator"], default=None)
    ap.add_argument("--stack-k", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--check-replay-every", type=int, default=25)
    ap.add_argument("--stop-after", type=int, default=None, help="iterations, for smoke runs")
    for f in fields(PPOConfig):
        flag = "--" + f.name.replace("_", "-")
        ap.add_argument(flag, type=type(getattr(PPOConfig(), f.name)), default=None)
    ap.add_argument("--workers", type=int, default=None, dest="W", help="alias of --W")
    ap.add_argument("--envs", type=int, default=None, dest="E", help="alias of --E")
    return ap


def config_from_args(a) -> RunConfig:
    cfg = RunConfig.make(a.memory, a.seed)
    ppo = {f.name: getattr(a, f.name) for f in fields(PPOConfig)
           if getattr(a, f.name, None) is not None}
    env = {}
    if a.layouts:
        env["layouts"] = tuple(a.layouts.split(","))
    if a.agents:
        env["agents"] = tuple(a.agents)
    for k in ("goal_obs", "heading", "stack_k"):
        if getattr(a, k) is not None:
            env[k] = getattr(a, k)
    cfg = replace(cfg, ppo=replace(cfg.ppo, **ppo), env=replace(cfg.env, **env))
    if a.val_every:
        cfg = replace(cfg, val_every=a.val_every)
    if a.ckpt_every:
        cfg = replace(cfg, ckpt_every=a.ckpt_every)
    return cfg


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    sim_cfg = sim_config()
    if a.resume and (a.out / "ckpt_last.pt").exists():
        ck = load_ckpt(a.out / "ckpt_last.pt")
        saved = RunConfig.from_dict(ck["config"])
        sim_cfg = sim_config_from_dict(ck["config"]["sim"])
        # the run is defined by its checkpoint; only the budget, the machine
        # (workers, device) and the validation cadence may change on resume
        over = {k: getattr(a, k) for k in ("total_steps", "W", "device")
                if getattr(a, k, None) is not None}
        cfg = replace(saved, ppo=replace(saved.ppo, **over))
    run(cfg, sim_cfg, a.out, a.val_suite, a.resume, a.val_workers,
        a.check_replay_every, stop_after=a.stop_after)


if __name__ == "__main__":
    main()
