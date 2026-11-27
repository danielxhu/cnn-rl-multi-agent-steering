"""Step-by-step check of ``jaxsim.sim`` against ``sim/physics.py``.

    python jaxsim/port_check.py --n 300            # float64: must match to rounding
    python jaxsim/port_check.py --n 300 --f32      # float32: how often outcomes differ

Scenes: training-distribution scenes (``env.sample_scene``, 2-8 agents, every
training layout, multi-exit included). Reward: the e3 / e7 training reward
(geodesic progress, timeout penalty 25). Actions: random, seeded, the same
for both simulators; an agent's action is drawn biased toward its goal
bearing so that episodes reach goals as well as walls.

Per step: max |obs diff|, max |reward diff|, max |position diff|, and equal
active flags; per episode: equal outcomes. Prints the worst values and the
number of mismatching episodes; exits non-zero on a float64 mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
sys.path.insert(0, str(POLICY / "jaxsim"))

import _paths  # noqa: F401,E402

import argparse  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402

TOL64 = 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f32", action="store_true")
    a = ap.parse_args(argv)
    import jax
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import bank
    import sim as J
    from config import Config
    from env import sample_scene
    from physics import Simulator
    from policyconfig import EnvConfig

    cfg = replace(Config(), progress_metric="geodesic", timeout_penalty=25.0)
    P = J.params(cfg)
    stepj = jax.jit(lambda lv, st, act: J.step(P, lv, st, act))
    obsj = jax.jit(lambda lv, st: J.observe(P, lv, st))
    rng = np.random.default_rng(a.seed)
    ec = EnvConfig()
    worst = dict(obs=0.0, rew=0.0, pos=0.0)
    bad_active = bad_outcome = steps = 0
    big = cells = 0                                   # obs entries off by > 1e-3
    codes = {"goal": J.GOAL, "collision": J.COLLISION, "timeout": J.TIMEOUT}
    t0 = time.time()
    for k in range(a.n):
        sc = sample_scene(ec, cfg, rng)
        n = len(sc.agents)
        s = Simulator(sc, cfg)
        lv = {key: jnp.asarray(v) for key, v in bank.pack(sc, cfg).items()}
        st = J.reset(lv)
        o_np = s.observe()
        o_j = np.asarray(obsj(lv, st))[:n]
        worst["obs"] = max(worst["obs"], float(np.abs(o_np - o_j).max()))
        while not s.done:
            act = np.zeros((bank.N, 2))
            b = o_np[:, 49] * np.pi / cfg.max_turn
            act[:n, 0] = np.clip(b + rng.normal(0, 0.8, n), -1, 1)
            act[:n, 1] = rng.uniform(0.2, 1.0, n)
            o_np, r_np, _ = s.step(act[:n])
            st, o_j, r_j = stepj(lv, st, jnp.asarray(act))
            o_j, r_j = np.asarray(o_j)[:n], np.asarray(r_j)[:n]
            steps += 1
            diff = np.abs(o_np - o_j)
            worst["obs"] = max(worst["obs"], float(diff.max()))
            big += int((diff > 1e-3).sum())
            cells += diff.size
            worst["rew"] = max(worst["rew"], float(np.abs(r_np - r_j).max()))
            worst["pos"] = max(worst["pos"], float(np.abs(s.pos - np.asarray(st.pos)[:n]).max()))
            if not np.array_equal(s.active, np.asarray(st.active)[:n]):
                bad_active += 1
                break                       # diverged: the rest of the episode differs
        want = [codes[o] for o in s.outcome]
        got = [int(x) for x in np.asarray(st.outcome)[:n]]
        bad_outcome += want != got
    dt = time.time() - t0
    print(f"{'float32' if a.f32 else 'float64'}: {a.n} scenes, {steps} steps, {dt:.0f} s")
    print(f"  worst |obs| {worst['obs']:.2e}  |reward| {worst['rew']:.2e}  |pos| {worst['pos']:.2e}")
    print(f"  obs entries off by > 1e-3: {big} of {cells} ({100 * big / max(1, cells):.4f} %)")
    print(f"  episodes with an active-flag mismatch {bad_active}, outcome mismatch {bad_outcome}")
    if not a.f32 and (bad_active or bad_outcome or max(worst.values()) > TOL64):
        sys.exit(1)


if __name__ == "__main__":
    main()
