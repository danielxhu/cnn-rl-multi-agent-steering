"""Frontier arm (DESIGN_FRONTIER.md §2): IPPO + SFL curriculum on the JAX simulator.

    python jaxsim/train_ippo.py --bank data/jaxbank/train.npz --val-bank data/jaxbank/val.npz \\
        --out runs/frontier_seed0 --seed 0 --total-steps 1e9
    python jaxsim/train_ippo.py ... --resume              # continue from <out>/state.pkl

Same learning rule as ``ppo.py`` where it matters for the comparison: one
policy shared by all agents (each agent its own stream, masked when not
acting), rewards scaled by the running std of the discounted return,
GAE(0.99, 0.95), clipped surrogate 0.2, advantages normalised per minibatch,
value loss 0.5 (v - R)^2 unclipped, Adam 3e-4 annealed linearly to 0 over
``--total-steps`` (as ``train.py``), grad norm 0.5. Goal,
collision and (timeout_penalty > 0) timeout are terminal. The network is
``nets.ActorCritic`` (gru) at ``--hidden`` (jaxsim/net.py).

Curriculum, SFL (Rutherford et al. 2024): every ``--sfl-every`` updates,
``--sfl-candidates`` bank levels are rolled out ``--sfl-rollouts`` times
with the current (stochastic) policy; p = share of agents that reach the
goal; the ``--sfl-buffer`` levels with the highest p (1 - p) form the
buffer. New episodes draw from the buffer with probability ``--sfl-rho``,
else uniformly from the bank. ``--sfl-rho 0`` = uniform (an ablation).

Outputs in ``--out``: ``log.csv`` per update, ``val.csv``, ``params_best.npz``
/ ``params_last.npz`` (torch-layout weights; ``export_ckpt.py`` turns them
into an ``evaluate.py`` checkpoint), ``state.pkl`` for ``--resume``.
Observation: the simulator's 51 numbers without the goal distance (50), as
``env.to_policy_obs`` with ``goal_obs = "bearing"``.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
sys.path.insert(0, str(POLICY / "jaxsim"))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from functools import partial  # noqa: E402
from typing import NamedTuple  # noqa: E402

import numpy as np  # noqa: E402

GOAL_DIST_COL = 48


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--val-bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=float, default=1e9, help="agent-steps")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--log-std-init", type=float, default=-0.5)
    ap.add_argument("--sfl-every", type=int, default=20)
    ap.add_argument("--sfl-candidates", type=int, default=2048)
    ap.add_argument("--sfl-rollouts", type=int, default=2)
    ap.add_argument("--sfl-buffer", type=int, default=1000)
    ap.add_argument("--sfl-rho", type=float, default=0.5)
    ap.add_argument("--val-every", type=int, default=25, help="updates")
    ap.add_argument("--save-at", default="5e7,2e8,5e8",
                    help="agent-steps at which params are also kept (params_<n>.npz)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--f64", action="store_true", help="debug: float64 everywhere")
    return ap.parse_args(argv)


def load_bank(path, jnp, dtype):
    z = np.load(path)
    lv = {k: z[k] for k in z.files if k not in ("layout", "index")}
    for k, v in lv.items():
        if v.dtype == np.float64 or v.dtype == np.float32:
            lv[k] = v.astype(dtype)
    lv["level"] = np.arange(len(lv["valid"]), dtype=np.int32)
    geo = jnp.asarray(lv.pop("geo"))
    return {k: jnp.asarray(v) for k, v in lv.items()}, geo, z["layout"]


class Carry(NamedTuple):
    idx: object        # (E,) bank index of each env's level
    st: object         # batched sim.State
    obs: object        # (E, N, 51)
    h: object          # ((E*N, H), (E*N, H))
    first: object      # (E, N)
    ret: object        # (E, N) running discounted return (reward scaler)


def main(argv=None):
    a = parse(argv)
    import jax
    if a.f64:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax

    import net
    import sim as J
    from config import Config

    dt = jnp.float64 if a.f64 else jnp.float32
    cfg = replace(Config(), progress_metric="geodesic", timeout_penalty=25.0)
    P = J.params(cfg)
    bank, geo, _ = load_bank(a.bank, jnp, np.float64 if a.f64 else np.float32)
    vbank, vgeo, _ = load_bank(a.val_bank, jnp, np.float64 if a.f64 else np.float32)
    L = int(bank["valid"].shape[0])
    N = int(bank["valid"].shape[1])
    E, T, H = a.envs, a.T, a.hidden
    D = 50
    a.out.mkdir(parents=True, exist_ok=True)

    take = lambda b, i: jax.tree.map(lambda x: x[i], b)          # noqa: E731
    v_reset = jax.vmap(J.reset)
    v_obs = jax.vmap(lambda lv, st: J.observe(P, lv, st))
    v_step = jax.vmap(lambda lv, st, act, g: J.step(P, lv, st, act, g), (0, 0, 0, None))

    def pobs(o51):
        return jnp.delete(o51, GOAL_DIST_COL, axis=-1)

    # ------------------------------------------------------------- acting
    def act(params, obs51, h, first, key, det):
        x = pobs(obs51).reshape(-1, D)
        mean, log_std, v, h = net.step(params, x, h, first.reshape(-1))
        a_ = jnp.where(det, mean, mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape, dt))
        return a_, net.log_prob(mean, log_std, a_), v, h

    def sample_levels(key, probs, n):
        return jax.random.categorical(key, jnp.log(probs + 1e-30), shape=(n,))

    # ------------------------------------------------------------ rollout
    def env_step(params, probs, c: Carry, key):
        k1, k2 = jax.random.split(key)
        lv = take(bank, c.idx)
        alive = c.st.active
        a_, logp, v, h = act(params, c.obs, c.h, c.first, k1, False)
        act_sim = net.actions_to_sim(a_).reshape(E, N, 2)
        st, obs, rew = v_step(lv, c.st, act_sim, geo)
        terminal = alive & ~st.active
        newly = jnp.stack([((st.outcome == k) & terminal).sum() for k in (1, 2, 3)])
        env_done = ~st.active.any(1)
        nidx = jnp.where(env_done, sample_levels(k2, probs, E), c.idx)
        nlv = take(bank, nidx)
        rst = v_reset(nlv)
        st = jax.tree.map(lambda x, y: jnp.where(env_done.reshape((-1,) + (1,) * (x.ndim - 1)), x, y),
                          rst, st)
        obs = jnp.where(env_done[:, None, None], v_obs(nlv, rst), obs)
        first = jnp.broadcast_to(env_done[:, None], (E, N))
        ret = c.ret * a.gamma + rew
        out = dict(obs=c.obs, first=c.first, alive=alive, action=a_.reshape(E, N, 2),
                   logp=logp.reshape(E, N), value=v.reshape(E, N), reward=rew,
                   terminal=terminal, ret=ret)
        ret = jnp.where(terminal, 0.0, ret)
        return Carry(nidx, st, obs, h, first, ret), (out, newly)

    @jax.jit
    def rollout(params, probs, c: Carry, key):
        h0 = c.h
        c, (traj, newly) = jax.lax.scan(lambda cc, k: env_step(params, probs, cc, k), c,
                                        jax.random.split(key, T))
        _, _, last_v, _ = act(params, c.obs, c.h, c.first, key, True)
        return c, traj, newly.sum(0), last_v.reshape(E, N), h0

    # ---------------------------------------------------------------- GAE
    def gae(traj, last_v, scale):
        r = traj["reward"] / scale
        V, done, alive = traj["value"], traj["terminal"], traj["alive"]
        nv = jnp.concatenate([V[1:], last_v[None]], 0)
        # a new episode in the slot (first at t + 1) only follows a done slot
        delta = r + a.gamma * nv * ~done - V

        def back(A, x):
            d, nd, al = x
            A = jnp.where(al, d + a.gamma * a.lam * nd * A, 0.0)
            return A, A
        _, adv = jax.lax.scan(back, jnp.zeros_like(last_v), (delta, ~done, alive), reverse=True)
        return adv, adv + V

    # -------------------------------------------------------------- update
    opt = optax.chain(optax.clip_by_global_norm(a.max_grad_norm),
                      optax.inject_hyperparams(optax.adam)(learning_rate=a.lr, eps=1e-5))

    def set_lr(opt_state, lr):
        hp = dict(opt_state[1].hyperparams, learning_rate=jnp.asarray(lr, dt))
        return (opt_state[0], opt_state[1]._replace(hyperparams=hp))

    def loss_fn(params, mb):
        h = (mb["h0a"], mb["h0c"])

        def f(h, x):
            o, fi = x
            mean, log_std, v, h = net.step(params, pobs(o).reshape(-1, D), h, fi.reshape(-1))
            return h, (mean, v)
        n_e = mb["obs"].shape[1]
        _, (mean, v) = jax.lax.scan(f, h, (mb["obs"], mb["first"]))
        mean = mean.reshape(T, n_e, N, 2)
        v = v.reshape(T, n_e, N)
        logp = net.log_prob(mean, params["log_std"], mb["action"])
        m = mb["alive"]
        cnt = jnp.maximum(m.sum(), 1)
        A = mb["adv"]
        am = (A * m).sum() / cnt
        astd = jnp.sqrt((((A - am) ** 2) * m).sum() / jnp.maximum(cnt - 1, 1)) + 1e-8
        A = (A - am) / astd
        ratio = jnp.exp(logp - mb["logp"])
        pg = jnp.maximum(-A * ratio, -A * jnp.clip(ratio, 1 - a.clip, 1 + a.clip))
        vl = 0.5 * (v - mb["ret"]) ** 2
        ent = net.entropy(params["log_std"])
        mean_m = lambda x: (x * m).sum() / cnt                    # noqa: E731
        loss = mean_m(pg) + a.vf_coef * mean_m(vl) - a.ent_coef * ent
        kl = mean_m((ratio - 1) - (logp - mb["logp"]))
        cf = mean_m((jnp.abs(ratio - 1) > a.clip).astype(dt))
        return loss, (mean_m(pg), mean_m(vl), kl, cf)

    @jax.jit
    def update(params, opt_state, traj, adv, ret, h0, key):
        batch = dict(obs=traj["obs"], first=traj["first"], alive=traj["alive"],
                     action=traj["action"], logp=traj["logp"], adv=adv, ret=ret,
                     h0a=h0[0].reshape(E, N, H), h0c=h0[1].reshape(E, N, H))
        M = a.minibatches
        per = E // M

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, E)

            def mbstep(carry, j):
                params, opt_state = carry
                ids = jax.lax.dynamic_slice(perm, (j * per,), (per,))
                mb = {kk: (vv[ids].reshape(-1, H) if kk in ("h0a", "h0c") else vv[:, ids])
                      for kk, vv in batch.items()}
                (_, stats), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), jnp.stack(stats)
            (params, opt_state), st = jax.lax.scan(mbstep, (params, opt_state), jnp.arange(M))
            return (params, opt_state), st
        (params, opt_state), st = jax.lax.scan(epoch, (params, opt_state),
                                               jax.random.split(key, a.epochs))
        return params, opt_state, st.reshape(-1, 4).mean(0)

    # ----------------------------------------------------- whole episodes
    def run_levels(params, lvb, gbank, idx, key, det):
        """Episodes on levels ``idx`` to the end -> outcome (K, N), valid (K, N)."""
        K = idx.shape[0]
        lv = take(lvb, idx)
        st = v_reset(lv)
        obs = v_obs(lv, st)
        h = net.initial_state(K * N, H)
        first = jnp.ones((K, N), bool)

        def f(c, k):
            st, obs, h, first = c
            a_, _, _, h = act(params, obs, h, first, k, det)
            st, obs, _ = v_step(lv, st, net.actions_to_sim(a_).reshape(K, N, 2), gbank)
            return (st, obs, h, jnp.zeros_like(first)), None
        (st, _, _, _), _ = jax.lax.scan(f, (st, obs, h, first), jax.random.split(key, P.max_steps))
        return st.outcome, lv["valid"]

    run_train = jax.jit(lambda p, idx, k: run_levels(p, bank, geo, idx, k, False))
    run_val = jax.jit(lambda p, idx, k: run_levels(p, vbank, vgeo, idx, k, True))

    def sfl_probs(params, key):
        k1, k2 = jax.random.split(key)
        cand = jax.random.choice(k1, L, (min(a.sfl_candidates, L),), replace=False)
        succ = []
        for k in jax.random.split(k2, a.sfl_rollouts):
            out, valid = run_train(params, cand, k)
            succ.append(((out == J.GOAL) & valid).sum(1) / valid.sum(1))
        p = jnp.mean(jnp.stack(succ), 0)
        score = p * (1 - p)
        top = cand[jnp.argsort(-score)[:a.sfl_buffer]]
        probs = jnp.full(L, (1 - a.sfl_rho) / L, dt)
        probs = probs.at[top].add(a.sfl_rho / top.shape[0])
        return probs, float(score[jnp.argsort(-score)[:a.sfl_buffer]].mean()), float(p.mean())

    def validate(params):
        nv = int(vbank["valid"].shape[0])
        out, valid = run_val(params, jnp.arange(nv), jax.random.PRNGKey(0))
        n = valid.sum()
        return {k: float(((out == c) & valid).sum() / n)
                for k, c in (("success_rate", J.GOAL), ("collision_rate", J.COLLISION),
                             ("timeout_rate", J.TIMEOUT))}

    # ------------------------------------------------------------- state
    key = jax.random.PRNGKey(a.seed)
    state_path = a.out / "state.pkl"
    if a.resume and state_path.exists():
        S = pickle.loads(state_path.read_bytes())
        params = jax.tree.map(jnp.asarray, S["params"])
        opt_state = jax.tree.map(jnp.asarray, S["opt_state"])
        carry = Carry(*jax.tree.map(jnp.asarray, S["carry"]))
        probs = jnp.asarray(S["probs"])
        key = jnp.asarray(S["key"])
        it, steps, best = S["it"], S["steps"], S["best"]
        sc_mean, sc_var, sc_count = S["scaler"]
        saved = set(S["saved"])
        print(f"resumed at update {it}, {steps:.3g} agent-steps")
    else:
        key, k0, k1 = jax.random.split(key, 3)
        params = net.init(k0, D, H, a.log_std_init)
        params = jax.tree.map(lambda x: x.astype(dt), params)
        opt_state = opt.init(params)
        probs = jnp.full(L, 1.0 / L, dt)
        idx = sample_levels(k1, probs, E)
        lv = take(bank, idx)
        st = v_reset(lv)
        carry = Carry(idx, st, v_obs(lv, st), net.initial_state(E * N, H),
                      jnp.ones((E, N), bool), jnp.zeros((E, N), dt))
        it, steps, best = 0, 0.0, -1.0
        sc_mean, sc_var, sc_count = 0.0, 1.0, 1e-4
        saved = set()
        (a.out / "args.json").write_text(json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
            | {"sim": asdict(cfg)}, indent=1))
    save_at = sorted(float(x) for x in a.save_at.split(",") if x)

    def save_params(p, name):
        np.savez(a.out / name, **net.to_torch_state(p))

    def save_state():
        S = dict(params=jax.tree.map(np.asarray, params), opt_state=jax.tree.map(np.asarray, opt_state),
                 carry=tuple(jax.tree.map(np.asarray, carry)), probs=np.asarray(probs),
                 key=np.asarray(key), it=it, steps=steps, best=best,
                 scaler=(sc_mean, sc_var, sc_count), saved=sorted(saved))
        tmp = state_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(S))
        tmp.replace(state_path)

    logf = open(a.out / "log.csv", "a", newline="")
    log = csv.writer(logf)
    if it == 0:
        log.writerow(["update", "agent_steps", "seconds", "agent_steps_per_s", "goals", "collisions",
                      "timeouts", "success_rate", "policy_loss", "value_loss", "approx_kl",
                      "clip_frac", "reward_std", "sfl_score", "sfl_p", "log_std_turn",
                      "log_std_speed"])
    sfl_score = sfl_p = float("nan")
    t_start = time.time()
    while steps < a.total_steps:
        key, kr, ku, ks = jax.random.split(key, 4)
        if a.sfl_rho > 0 and it % a.sfl_every == 0:
            probs, sfl_score, sfl_p = sfl_probs(params, ks)
        t0 = time.time()
        carry, traj, newly, last_v, h0 = rollout(params, probs, carry, kr)
        # reward scaler (ppo.RewardScaler): statistics of the running return
        alive = np.asarray(traj["alive"])
        x = np.asarray(traj["ret"])[alive]
        if len(x):
            bm, bv, bc = float(x.mean()), float(x.var()), len(x)
            d = bm - sc_mean
            tot = sc_count + bc
            sc_mean += d * bc / tot
            sc_var = (sc_var * sc_count + bv * bc + d * d * sc_count * bc / tot) / tot
            sc_count = tot
        scale = float(np.sqrt(sc_var + 1e-8))
        adv, ret = gae(traj, last_v, scale)
        opt_state = set_lr(opt_state, a.lr * max(0.0, 1.0 - steps / a.total_steps))
        params, opt_state, stats = update(params, opt_state, traj, adv, ret, h0, ku)
        n_alive = int(alive.sum())
        steps += n_alive
        it += 1
        el = time.time() - t0
        g, c_, to = (int(v) for v in np.asarray(newly))
        ls = np.asarray(params["log_std"])
        log.writerow([it, int(steps), round(time.time() - t_start, 1), round(n_alive / el, 1), g, c_,
                      to, round(g / max(1, g + c_ + to), 4)] +
                     [round(float(v), 6) for v in np.asarray(stats)] +
                     [round(scale, 4), sfl_score, sfl_p, float(ls[0]), float(ls[1])])
        logf.flush()
        if it % a.val_every == 0 or steps >= a.total_steps:
            v = validate(params)
            new = not (a.out / "val.csv").exists()
            with open(a.out / "val.csv", "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["update", "agent_steps"] + list(v))
                w.writerow([it, int(steps)] + [round(x, 4) for x in v.values()])
            if v["success_rate"] > best:
                best = v["success_rate"]
                save_params(params, "params_best.npz")
            save_params(params, "params_last.npz")
            for s_ in save_at:
                if steps >= s_ and s_ not in saved:
                    save_params(params, f"params_{int(s_):d}.npz")
                    saved.add(s_)
            save_state()
            print(f"update {it}  {steps:.3g} steps  {n_alive / el:.0f} steps/s  "
                  f"val success {v['success_rate']:.3f} (best {best:.3f})", flush=True)
    logf.close()


if __name__ == "__main__":
    main()
