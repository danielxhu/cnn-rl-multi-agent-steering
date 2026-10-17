"""Rollout collection: inline and in worker processes (DESIGN.md §3.1).

Owns: ``RolloutWorker`` (E envs, one CPU copy of the network, recurrent state
carried across iterations), ``collect`` (the stepping loop), ``WorkerPool``
(W spawned processes, or the same workers inline when W = 0) and the
``Batch`` layout.

Batch layout -- a dict of numpy arrays, time first, then one *row* per
(env, slot) in the order ``env * max_agents + slot`` (workers concatenated):

    obs (T, N, D)  action (T, N, 2)  logp, value (T, N)
    reward, bootstrap (T, N) float32     alive, terminal, truncated, first (T, N) bool
    last_value (N,)         V of the state after the last step
    h_actor, h_critic (T // L, N, H)    recurrent state at the start of each chunk
    records                 list of finished-scene dicts (not an array)

Row ``t`` is one transition: ``obs[t]`` (seen when ``alive[t]``, new episode if
``first[t]``), the sampled ``action[t]``, and what the env returned for it:
``reward[t]``, ``terminal[t]`` / ``truncated[t]`` and, for a timeout,
``bootstrap[t] = V(final_obs)``.

Invariants:
- Rollouts never read global random state: sampling uses a per-worker
  ``torch.Generator``, scenes use per-env numpy generators, so a pool of W
  workers and ``W`` inline workers give identical batches (test 9).
- Workers use one torch thread, ignore SIGINT (the learner owns shutdown) and
  are daemons.
"""
from __future__ import annotations

import _paths  # noqa: F401

import multiprocessing as mp
import signal

import numpy as np
import torch

from env import MultiAgentEnv, make_env_rng, obs_dim
from nets import ActorCritic

ARRAY_KEYS = ("obs", "action", "logp", "value", "reward", "bootstrap", "alive",
              "terminal", "truncated", "first")


def collect(model, envs, state, T, L, gen):
    """Step ``envs`` for ``T`` steps with ``model`` (CPU, no grad); return a Batch.

    ``state`` is a dict (obs, alive, first, h) updated in place so the next
    call continues where this one stopped.
    """
    assert T % L == 0, "T must be a multiple of L"
    E, S = len(envs), envs[0].S
    N, D = E * S, envs[0].D
    out = {k: np.zeros((T, N) + shp, dt) for k, shp, dt in (
        ("obs", (D,), np.float32), ("action", (2,), np.float32), ("logp", (), np.float32),
        ("value", (), np.float32), ("reward", (), np.float32), ("bootstrap", (), np.float32),
        ("alive", (), bool), ("terminal", (), bool), ("truncated", (), bool),
        ("first", (), bool))}
    H = model.H
    h_actor = np.zeros((T // L, N, H), np.float32)
    h_critic = np.zeros((T // L, N, H), np.float32)

    for t in range(T):
        if t % L == 0:
            h_actor[t // L] = state["h"][0][0].numpy()
            h_critic[t // L] = state["h"][1][0].numpy()
        obs = torch.from_numpy(state["obs"])
        first = torch.from_numpy(state["first"])
        with torch.no_grad():
            dist, value, h = model.step(obs, state["h"], first)
            a = dist.mean + dist.stddev * torch.randn(dist.mean.shape, generator=gen)
            logp = dist.log_prob(a).sum(-1)
        out["obs"][t] = state["obs"]
        out["alive"][t] = state["alive"]
        out["first"][t] = state["first"]
        out["action"][t] = a.numpy()
        out["logp"][t] = logp.numpy()
        out["value"][t] = value.numpy()

        a_np = a.numpy().reshape(E, S, 2)
        nxt_obs = np.zeros((E, S, D), np.float32)
        nxt_alive = np.zeros((E, S), bool)
        nxt_first = np.zeros((E, S), bool)
        final = np.zeros((E, S, D), np.float32)
        for e, env in enumerate(envs):
            r = env.step(a_np[e])
            sl = slice(e * S, (e + 1) * S)
            out["reward"][t, sl] = r["reward"]
            out["terminal"][t, sl] = r["terminal"]
            out["truncated"][t, sl] = r["truncated"]
            nxt_obs[e], nxt_alive[e], nxt_first[e] = r["obs"], r["alive"], r["first"]
            final[e] = r["final_obs"]
        trunc = out["truncated"][t]
        if trunc.any():
            with torch.no_grad():
                v = model.value_only(torch.from_numpy(final.reshape(N, D)), h,
                                     torch.zeros(N, dtype=torch.bool))
            out["bootstrap"][t] = np.where(trunc, v.numpy(), 0.0)
        state["obs"] = nxt_obs.reshape(N, D)
        state["alive"] = nxt_alive.reshape(N)
        state["first"] = nxt_first.reshape(N)
        state["h"] = h

    with torch.no_grad():
        last = model.value_only(torch.from_numpy(state["obs"]), state["h"],
                                torch.from_numpy(state["first"]))
    out["last_value"] = last.numpy().astype(np.float32)
    out["h_actor"], out["h_critic"] = h_actor, h_critic
    out["records"] = [rec for env in envs for rec in env.episode_records()]
    return out


class RolloutWorker:
    """E envs + a CPU network + recurrent state; lives in a process or inline."""

    def __init__(self, run_cfg, sim_cfg, worker: int, epoch: int | None = None):
        torch.set_num_threads(1)
        self.cfg, self.worker = run_cfg, worker
        ec = run_cfg.env
        self.envs = [MultiAgentEnv(ec, sim_cfg, make_env_rng(run_cfg.seed, worker, e, epoch))
                     for e in range(run_cfg.ppo.E)]
        self.model = ActorCritic(obs_dim(ec), run_cfg.net)
        self.model.eval()
        ent = [run_cfg.seed, worker, 10_000] + ([] if epoch is None else [epoch])
        seed = int(np.random.SeedSequence(ent).generate_state(1)[0])
        self.gen = torch.Generator().manual_seed(seed)
        S, D = ec.max_agents, obs_dim(ec)
        N = len(self.envs) * S
        rs = [env.reset() for env in self.envs]
        self.state = {
            "obs": np.concatenate([r["obs"] for r in rs]).astype(np.float32),
            "alive": np.concatenate([r["alive"] for r in rs]),
            "first": np.concatenate([r["first"] for r in rs]),
            "h": self.model.initial_state(N),
        }

    def collect(self, weights, T=None, L=None):
        self.model.set_weights(weights)
        return collect(self.model, self.envs, self.state, T or self.cfg.ppo.T,
                       L or self.cfg.ppo.L, self.gen)


def _worker_main(conn, run_cfg, sim_cfg, worker, epoch):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        w = RolloutWorker(run_cfg, sim_cfg, worker, epoch)
        conn.send(("ready", None))
        while True:
            msg, payload = conn.recv()
            if msg == "close":
                break
            conn.send(("batch", w.collect(payload)))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception as exc:                    # surface the failure in the learner
        import traceback
        conn.send(("error", traceback.format_exc() + repr(exc)))
    finally:
        conn.close()


def merge(batches):
    """Concatenate per-worker batches along the row axis."""
    if len(batches) == 1:
        return batches[0]
    out = {}
    for k in ARRAY_KEYS:
        out[k] = np.concatenate([b[k] for b in batches], 1)
    out["last_value"] = np.concatenate([b["last_value"] for b in batches])
    out["h_actor"] = np.concatenate([b["h_actor"] for b in batches], 1)
    out["h_critic"] = np.concatenate([b["h_critic"] for b in batches], 1)
    out["records"] = [r for b in batches for r in b["records"]]
    return out


class WorkerPool:
    """W spawned worker processes; W = 0 runs ``inline_workers`` workers inline."""

    def __init__(self, run_cfg, sim_cfg, epoch: int | None = None, inline_workers: int = 1):
        self.cfg, self.W = run_cfg, run_cfg.ppo.W
        self.inline = self.W == 0
        self.procs, self.conns, self.workers = [], [], []
        if self.inline:
            self.workers = [RolloutWorker(run_cfg, sim_cfg, w, epoch)
                            for w in range(inline_workers)]
            return
        ctx = mp.get_context("spawn")
        for w in range(self.W):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(child, run_cfg, sim_cfg, w, epoch),
                            daemon=True)
            p.start()
            child.close()
            self.procs.append(p)
            self.conns.append(parent)
        for c in self.conns:
            self._recv(c, "ready")

    @staticmethod
    def _recv(conn, expect):
        msg, payload = conn.recv()
        if msg == "error":
            raise RuntimeError("rollout worker failed:\n" + payload)
        assert msg == expect, msg
        return payload

    def collect(self, weights) -> dict:
        if self.inline:
            return merge([w.collect(weights) for w in self.workers])
        for c in self.conns:
            c.send(("collect", weights))
        return merge([self._recv(c, "batch") for c in self.conns])

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (OSError, BrokenPipeError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.procs, self.conns = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
