"""Self-contained test suite (no pytest needed):  python tests/run_tests.py

CPU only, a few tens of seconds. Same harness as ``sim/tests/run_tests.py``;
the numbering follows DESIGN.md §9.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: F401,E402

import numpy as np  # noqa: E402

from config import Config  # noqa: E402
from layouts import random_scene  # noqa: E402
from physics import Simulator  # noqa: E402

import env as envmod  # noqa: E402
from policyconfig import EnvConfig, LAYOUTS_E2  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def small_env_cfg(**kw):
    return EnvConfig(**kw)


# ------------------------------------------------------------ observation

@test
def test_1_obs_is_51_minus_distance():
    cfg = Config()
    rng = np.random.default_rng(0)
    scene = random_scene(rng, cfg, layout="room", n_agents=4, n_obstacles=2)
    sim = Simulator(scene, cfg)
    o51 = sim.observe()
    assert o51.shape == (4, 51)
    ec = EnvConfig()
    o50 = envmod.to_policy_obs(o51, ec)
    assert o50.shape == (4, 50) and envmod.obs_dim(ec) == 50
    assert np.array_equal(o50, np.delete(o51, 48, axis=1))
    full = envmod.to_policy_obs(o51, EnvConfig(goal_obs="bearing_distance"))
    assert np.array_equal(full, o51) and envmod.obs_dim(EnvConfig(goal_obs="bearing_distance")) == 51
    assert envmod.obs_dim(EnvConfig(memory="stack", stack_k=4)) == 200


@test
def test_2_heading_randomised_only():
    cfg = Config()
    rng = np.random.default_rng(1)
    scene = random_scene(rng, cfg, layout="open", n_agents=8, n_obstacles=3)
    a = envmod.randomise_headings(scene, np.random.default_rng(5))
    b = envmod.randomise_headings(scene, np.random.default_rng(5))
    assert [x.heading for x in a.agents] == [x.heading for x in b.agents]
    assert [x.heading for x in a.agents] != [x.heading for x in scene.agents]
    assert a.walls == scene.walls and a.obstacles == scene.obstacles
    assert a.groups == scene.groups and a.goal_region == scene.goal_region
    assert [x.pos for x in a.agents] == [x.pos for x in scene.agents]
    hs = []
    r = np.random.default_rng(2)
    for _ in range(40):
        hs += [x.heading for x in envmod.randomise_headings(scene, r).agents]
    hs = np.array(hs)
    assert hs.min() >= -math.pi and hs.max() < math.pi
    assert hs.min() < -2.5 and hs.max() > 2.5 and abs(hs.mean()) < 0.4, "spread over the circle"


@test
def test_3_slots_and_auto_reset():
    cfg = Config()
    e = envmod.MultiAgentEnv(EnvConfig(), cfg, np.random.default_rng(3))
    r = e.reset()
    n = e.n
    assert r["alive"].sum() == n and not r["alive"][n:].any() and r["first"].sum() == n
    assert r["obs"].shape == (8, 50) and not r["obs"][n:].any()
    ever_dead = np.zeros(8, bool)
    resets = 0
    policy = np.random.default_rng(4)
    for _ in range(900):
        prev_n = e.n
        r = e.step(policy.uniform(-1, 1, (8, 2)))
        assert not r["alive"][e.n:].any(), "empty slots never alive"
        if r["first"].any():
            resets += 1
            assert r["first"].sum() == e.n and r["alive"].sum() == e.n
            ever_dead[:] = False
        else:
            assert not (r["alive"] & ever_dead).any(), "a finished agent never comes back"
            ever_dead |= (~r["alive"])[:]
            ever_dead[e.n:] = False
        assert not (r["terminal"] & r["truncated"]).any()
        assert not r["obs"][~r["alive"]].any()
    assert resets >= 2, "scenes ended and were replaced"
    assert len(e.episode_records()) == resets and e.episode_records() == []


@test
def test_4_truncation_vs_termination():
    # timeouts: a tiny step budget and agents that stand still
    cfg = replace(Config(), max_steps=3)
    still = np.full((8, 2), -1.0)          # speed -> 0
    e = envmod.MultiAgentEnv(EnvConfig(), cfg, np.random.default_rng(6))
    e.reset()
    n0 = e.n
    for t in range(3):
        r = e.step(still)
    assert r["truncated"][:n0].all() and not r["terminal"].any()
    assert (r["final_obs"][:n0] != 0).any(axis=1).all(), "final_obs for truncated slots"
    assert not r["final_obs"][n0:].any()
    assert r["first"].any(), "auto-reset happened on the same call"

    # terminals: full speed ahead with the greedy controller reaches goals; drive blind -> collisions
    cfg = Config()
    e = envmod.MultiAgentEnv(EnvConfig(layouts=("room",), agents=(4, 4)), cfg,
                             np.random.default_rng(7))
    r = e.reset()
    saw_term = False
    for _ in range(400):
        # raw action: turn 0, speed 1 (a1 = 1) -> straight line, ignoring everything
        a = np.zeros((8, 2))
        a[:, 1] = 1.0
        r = e.step(a)
        if r["terminal"].any():
            saw_term = True
            assert not r["truncated"][r["terminal"]].any()
            assert (r["final_obs"][r["terminal"]] == 0).all()
            break
    assert saw_term


@test
def test_8_stack_padding():
    ec = EnvConfig(memory="stack", stack_k=3, layouts=("open",), agents=(3, 3))
    e = envmod.MultiAgentEnv(ec, Config(), np.random.default_rng(8))
    assert e.D == 150
    r = e.reset()
    n = e.n
    o = r["obs"]
    assert o.shape == (8, 150)
    assert not o[:n, :100].any(), "first k-1 frames are zero"
    assert o[:n, 100:].any()
    frame0 = o[:n, 100:].copy()
    r = e.step(np.zeros((8, 2)))
    o = r["obs"]
    assert not r["first"].any()
    assert not o[:n, :50].any() and np.array_equal(o[:n, 50:100], frame0)
    # plain-mode frame equals the stack's newest frame
    e2 = envmod.MultiAgentEnv(EnvConfig(layouts=("open",), agents=(3, 3)), Config(),
                              np.random.default_rng(8))
    p = e2.reset()["obs"]
    assert np.array_equal(p[:n], frame0)
    # a scene change clears the stack
    for _ in range(400):
        r = e.step(np.zeros((8, 2)) + np.array([0, 1.0]))
        if r["first"].any():
            assert not r["obs"][:e.n, :100].any()
            break
    else:
        raise AssertionError("no scene change")


# ------------------------------------------------------------------ nets

@test
def test_6_sequence_equals_steps():
    import torch
    from nets import ActorCritic
    from policyconfig import NetConfig
    torch.manual_seed(0)
    for mem, D in (("mlp", 50), ("stack", 200), ("gru", 50)):
        net = ActorCritic(D, NetConfig(memory=mem))
        L, n = 12, 5
        obs = torch.randn(L, n, D)
        first = torch.rand(L, n) < 0.25
        first[0] = True
        state0 = tuple(torch.randn_like(s) for s in net.initial_state(n))
        dist, val = net.sequence(obs, state0, first)
        state = state0
        for t in range(L):
            d, v, state = net.step(obs[t], state, first[t])
            assert torch.allclose(d.mean, dist.mean[t], atol=1e-6), mem
            assert torch.allclose(v, val[t], atol=1e-6), mem
    # the recurrent net really uses its state, the others ignore it
    net = ActorCritic(50, NetConfig(memory="gru"))
    x = torch.randn(3, 50)
    f = torch.zeros(3, dtype=torch.bool)
    a, _, s1 = net.step(x, net.initial_state(3), f)
    b, _, _ = net.step(x, s1, f)
    assert not torch.allclose(a.mean, b.mean)
    c, _, _ = net.step(x, s1, ~f)
    assert torch.allclose(a.mean, c.mean, atol=1e-6), "first=True zeroes the state"


# ------------------------------------------------------------ ppo / rollout

def tiny_run_cfg(memory="gru", E=2, T=64, L=32, W=0, seed=0, **env_kw):
    from policyconfig import NetConfig, PPOConfig, RunConfig
    return RunConfig(env=EnvConfig(memory=memory, **env_kw), net=NetConfig(memory=memory),
                     ppo=PPOConfig(E=E, T=T, L=L, W=W), seed=seed)


def gae_reference(b, gamma, lam):
    """Episode by episode, straight from the definition (no backward recursion)."""
    T, N = b["value"].shape
    adv = np.zeros((T, N))
    for n in range(N):
        t = 0
        while t < T:
            if not b["alive"][t, n]:
                t += 1
                continue
            end = t
            while end + 1 < T and b["alive"][end + 1, n] and not b["first"][end + 1, n] \
                    and not (b["terminal"][end, n] or b["truncated"][end, n]):
                end += 1
            deltas = []
            for u in range(t, end + 1):
                done = b["terminal"][u, n] or b["truncated"][u, n]
                if u < T - 1 and not done:
                    nv = b["value"][u + 1, n]
                elif not done:
                    nv = b["last_value"][n]
                else:
                    nv = 0.0
                boot = b["bootstrap"][u, n] if b["truncated"][u, n] else 0.0
                deltas.append(b["reward"][u, n] + gamma * nv + gamma * boot - b["value"][u, n])
            for u in range(t, end + 1):
                adv[u, n] = sum((gamma * lam) ** k * deltas[u - t + k]
                                for k in range(end - u + 1))
            t = end + 1
    return adv


@test
def test_5_gae_reference():
    from ppo import gae
    rng = np.random.default_rng(0)
    T, N = 12, 4
    b = {k: np.zeros((T, N), np.float32) for k in ("reward", "value", "bootstrap")}
    for k in ("alive", "terminal", "truncated", "first"):
        b[k] = np.zeros((T, N), bool)
    b["reward"][:] = rng.normal(size=(T, N))
    b["value"][:] = rng.normal(size=(T, N))
    b["last_value"] = rng.normal(size=N).astype(np.float32)
    # row 0: one episode, ends in a terminal at t=7, empty slot afterwards, new episode at t=10
    b["alive"][:8, 0] = True; b["terminal"][7, 0] = True
    b["alive"][10:, 0] = True; b["first"][10, 0] = True
    # row 1: episode ending in a truncation at t=5 with a bootstrap value, a new one right after
    b["alive"][:6, 1] = True; b["truncated"][5, 1] = True; b["bootstrap"][5, 1] = 0.7
    b["alive"][6:, 1] = True; b["first"][6, 1] = True
    # row 2: alive the whole batch, cut by the end (bootstrap from last_value)
    b["alive"][:, 2] = True
    # row 3: never alive
    got, ret = gae(b, 0.99, 0.95)
    want = gae_reference(b, 0.99, 0.95)
    assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()
    assert np.allclose(ret, got + b["value"])
    assert not got[~b["alive"]].any()


@test
def test_7_replay_matches_rollout():
    import torch
    from ppo import replay
    from rollout import RolloutWorker
    for mem in ("mlp", "gru"):
        cfg = tiny_run_cfg(mem, E=3, T=64, L=32)
        w = RolloutWorker(cfg, replace(Config(), max_steps=20), 0)   # resets inside sequences
        torch.manual_seed(1)
        for p in w.model.parameters():              # make the policy non-trivial
            p.data += 0.1 * torch.randn_like(p)
        weights = w.model.get_weights()
        for it in range(3):                          # state is carried across calls
            b = w.collect(weights)
            logp, val, _ = replay(w.model, b, cfg.ppo.L)
            assert np.abs(logp - b["logp"]).max() < 1e-5, (mem, it, np.abs(logp - b["logp"]).max())
            assert np.abs(val - b["value"]).max() < 1e-5, (mem, it)
            assert b["alive"].any() and b["first"][1:].any(), "episode starts inside the batch"
        if mem == "gru":
            assert np.abs(b["h_actor"][1:]).max() > 0, "recurrent state was stored"


@test
def test_9_workers_match_inline():
    import torch
    from nets import ActorCritic
    from rollout import WorkerPool
    cfg = tiny_run_cfg("gru", E=2, T=32, L=32, W=2)
    torch.manual_seed(2)
    net = ActorCritic(50, cfg.net)
    weights = net.get_weights()
    inline_cfg = tiny_run_cfg("gru", E=2, T=32, L=32, W=0)
    ref = WorkerPool(inline_cfg, Config(), inline_workers=2)
    with WorkerPool(cfg, Config()) as pool:
        for _ in range(2):
            a, b = pool.collect(weights), ref.collect(weights)
            for k in ("obs", "action", "logp", "value", "reward", "bootstrap", "alive",
                      "terminal", "truncated", "first", "last_value", "h_actor", "h_critic"):
                assert np.array_equal(a[k], b[k]), k
            assert a["records"] == b["records"]
    assert a["obs"].shape[1] == 2 * 2 * 8


@test
def test_10_reward_scaler_reference():
    from ppo import RewardScaler
    rng = np.random.default_rng(3)
    T, N = 40, 5
    sc = RewardScaler(gamma=0.99)
    mean, var, cnt = 0.0, 1.0, 1e-4
    ret = np.zeros(N)
    for it in range(3):
        r = rng.normal(0, 10, (T, N))
        done = rng.random((T, N)) < 0.05
        alive = rng.random((T, N)) < 0.8
        r = r * alive
        sc.update(r, done, alive)
        vals = []
        for t in range(T):                           # per-sample Welford with the same prior
            ret = ret * 0.99 + r[t]
            vals += list(ret[alive[t]])
            ret = np.where(done[t], 0, ret)
        for x in vals:
            cnt += 1
            d = x - mean
            mean += d / cnt
            var += (d * (x - mean) - var) / cnt
    assert abs(sc.mean - mean) < 1e-6 and abs(sc.var - var) < 1e-5 * max(1, var), (sc.var, var)
    assert np.allclose(sc.scale(np.array([sc.std])), 1.0)
    sc2 = RewardScaler()
    sc2.load_state_dict(sc.state_dict())
    assert sc2.std == sc.std and np.array_equal(sc2.ret, sc.ret)


@test
def test_11_checkpoint_resume():
    import csv
    import torch
    from dataclasses import replace as rep
    import train
    cfg = tiny_run_cfg("gru", E=2, T=32, L=16, W=0, layouts=("open",), agents=(2, 3))
    cfg = rep(cfg, ckpt_every=100)                       # only the interrupt / finish checkpoints
    sim_cfg = Config()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        s2 = train.run(cfg, sim_cfg, out, quiet=True, stop_after=2)
        assert s2["iteration"] == 2
        ck = train.load_ckpt(out / "ckpt_last.pt")
        assert ck["iteration"] == 2 and ck["agent_steps"] == s2["agent_steps"]
        # resume: every piece of state comes back
        s3 = train.run(cfg, sim_cfg, out, resume=True, quiet=True, stop_after=1)
        assert s3["iteration"] == 3 and s3["agent_steps"] > s2["agent_steps"]
        ck3 = train.load_ckpt(out / "ckpt_last.pt")
        assert ck3["iteration"] == 3
        # restoring the last checkpoint gives back exactly what was saved
        ck_path = Path(d) / "ck2.pt"
        model = train.ActorCritic(50, cfg.net)
        opt = torch.optim.Adam(model.parameters(), eps=1e-5)
        sc, st = train.RewardScaler(), {}
        import shutil
        shutil.copy(out / "ckpt_last.pt", ck_path)       # a copy, restored into fresh objects
        train.restore(ck_path, model, opt, sc, st)
        assert st["iteration"] == ck3["iteration"] and st["agent_steps"] == ck3["agent_steps"]
        assert all(torch.equal(model.state_dict()[k], v) for k, v in ck3["model"].items())
        o_new, o_old = opt.state_dict()["state"], ck3["optimiser"]["state"]
        assert o_new.keys() == o_old.keys() and len(o_new) > 0
        assert all(torch.equal(o_new[i]["exp_avg"], o_old[i]["exp_avg"]) for i in o_old)
        assert sc.state_dict()["count"] == ck3["scaler"]["count"] > 1
        assert np.array_equal(sc.ret, ck3["scaler"]["ret"]) and sc.mean == ck3["scaler"]["mean"]
        # weights moved between iteration 2 and 3
        assert any(not torch.equal(ck["model"][k], ck3["model"][k]) for k in ck["model"])
        rows = list(csv.DictReader(open(out / "log.csv")))
        assert [int(r["iteration"]) for r in rows] == [1, 2, 3], rows
        # a finished run exits without training
        done = rep(cfg, ppo=rep(cfg.ppo, total_steps=s3["agent_steps"]))
        s4 = train.run(done, sim_cfg, out, resume=True, quiet=True)
        assert s4["iteration"] == 3
        assert len(list(csv.DictReader(open(out / "log.csv")))) == 3
        assert (out / "config.json").exists()
        # rows written after the last checkpoint are dropped on resume
        with open(out / "log.csv", "a") as fh:
            fh.write(",".join(["9"] + ["0"] * 17) + "\n")
        train.trim_csv(out / "log.csv", 3)
        assert len(list(csv.DictReader(open(out / "log.csv")))) == 3


# ------------------------------------------------------- suites / evaluation

@test
def test_12_suite_deterministic():
    import json
    from suites import make_suite, load_suite
    from layouts import random_scene as rs
    with tempfile.TemporaryDirectory() as d:
        kw = dict(layouts=("open", "room"), per_layout=2, agents=(3, 4), obstacles=(0, 2),
                  seed=42, sim_cfg=Config())
        make_suite(Path(d) / "a", name="x", **kw)
        make_suite(Path(d) / "b", name="x", **kw)
        fa = sorted(p.name for p in (Path(d) / "a").iterdir())
        assert fa == sorted(p.name for p in (Path(d) / "b").iterdir()) and len(fa) == 5
        for f in fa:
            assert (Path(d) / "a" / f).read_text() == (Path(d) / "b" / f).read_text(), f
        scenes = load_suite(Path(d) / "a")
        assert [n for n, _ in scenes][0] == "scene_00000" and len(scenes) == 4
        # headings are not the generator's (which faces the goal +-26 degrees)
        for _, sc in scenes:
            goal = sc.goal_region.center
            off = [abs((math.atan2(goal[1] - a.pos[1], goal[0] - a.pos[0]) - a.heading
                        + math.pi) % (2 * math.pi) - math.pi) for a in sc.agents]
            assert max(off) > math.radians(30), off


@test
def test_13_greedy_episode_matches_simulator():
    import evaluate
    cfg = Config()
    rng = np.random.default_rng(13)
    for layout in ("open", "doorway", "boxrooms"):
        scene = random_scene(rng, cfg, layout=layout, n_agents=5, n_obstacles=3)
        scene = envmod.randomise_headings(scene, rng)
        row = evaluate.run_episode(scene, evaluate.GreedyController(), cfg)
        from physics import greedy_action
        sim = Simulator(scene, cfg)
        done = sim.done
        while not done:
            _, _, done = sim.step(np.stack([greedy_action(sim, i) for i in range(5)]))
        st = sim.stats()
        for k in ("success_rate", "collision_rate", "timeout_rate", "collision_violations",
                  "finish_time", "remaining_agents", "mean_path_length"):
            assert abs(row[k] - st[k]) < 1e-9, (layout, k, row[k], st[k])
        assert abs(row["static_collision_rate"] + row["agent_collision_rate"]
                   - row["collision_rate"]) < 1e-9


@test
def test_14_smoke_train_eval_aggregate():
    import json
    import evaluate
    import aggregate
    import train
    from dataclasses import replace as rep
    from suites import make_suite
    sim_cfg = Config()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name in ("val", "test4", "test_mixed"):
            make_suite(d / "data" / name, ("open", "doorway"), 3, (2, 4), (0, 2), 5, sim_cfg,
                       name=name)
        runs = []
        for mem in ("mlp", "gru"):
            cfg = rep(tiny_run_cfg(mem, E=2, T=16, L=16, W=0), val_every=2, ckpt_every=2)
            out = d / "runs" / f"e2_{mem}_seed0"
            train.run(cfg, sim_cfg, out, val_suite=d / "data" / "val", quiet=True, stop_after=3)
            for f in ("config.json", "log.csv", "val.csv", "ckpt_last.pt", "ckpt_best.pt"):
                assert (out / f).exists(), f
            import csv
            log = list(csv.DictReader(open(out / "log.csv")))
            assert list(log[0].keys()) == train.LOG_COLS and len(log) == 3
            val = list(csv.DictReader(open(out / "val.csv")))
            assert "success_open" in val[0] and "success_doorway" in val[0] and len(val) == 2
            ev = d / "eval" / out.name
            evaluate.main(["--ckpt", str(out / "ckpt_best.pt"), "--suites", "test4,test_mixed",
                           "--data-root", str(d / "data"), "--out", str(ev)])
            runs.append(ev)
        gr = d / "eval" / "greedy"
        evaluate.main(["--greedy", "--suites", "test4,test_mixed", "--data-root", str(d / "data"),
                       "--out", str(gr)])
        for ev in runs + [gr]:
            for f in ("per_scene.csv", "metrics.json"):
                assert (ev / f).exists()
            assert len(list((ev / "trajectories").glob("*.json"))) == 4   # 2 layouts x 2 suites
        m = json.loads((runs[0] / "metrics.json").read_text())
        assert {"overall", "per_layout", "per_bottleneck"} <= set(m["suites"]["test4"])
        res = d / "res"
        aggregate.main(["--evals", str(d / "eval"), "--out", str(res), "--runs", str(d / "runs")])
        for f in ("summary.csv", "summary_per_layout.csv", "table_e2.md", "table_e2_mixed.md",
                  "fig_training_curves.png", "fig_per_layout.png", "fig_bottleneck.png",
                  "fig_paths.png"):
            assert (res / f).exists() and (res / f).stat().st_size > 0, f


@test
def test_15_update_is_finite_with_tiny_minibatches():
    """Regression: a minibatch with one alive entry made the advantage std NaN."""
    import torch
    from nets import ActorCritic
    from policyconfig import NetConfig, PPOConfig
    import ppo
    cfg = PPOConfig(L=4, T=4, n_minibatches=4, epochs=2)
    N, T = 6, 4
    rng = np.random.default_rng(0)
    b = {"obs": rng.normal(size=(T, N, 50)).astype(np.float32),
         "action": rng.normal(size=(T, N, 2)).astype(np.float32),
         "logp": np.full((T, N), -2.0, np.float32), "value": np.zeros((T, N), np.float32),
         "reward": rng.normal(size=(T, N)).astype(np.float32),
         "bootstrap": np.zeros((T, N), np.float32), "last_value": np.zeros(N, np.float32),
         "alive": np.zeros((T, N), bool), "terminal": np.zeros((T, N), bool),
         "truncated": np.zeros((T, N), bool), "first": np.zeros((T, N), bool),
         "h_actor": np.zeros((1, N, 8), np.float32), "h_critic": np.zeros((1, N, 8), np.float32)}
    b["alive"][0, :4] = True                                  # four sequences with one alive step
    net = ActorCritic(50, NetConfig(memory="gru", hidden=8))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    adv, ret = ppo.gae(b, 0.99, 0.95)
    st = ppo.update(net, opt, b, adv, ret, cfg, torch.device("cpu"), np.random.default_rng(0))
    assert all(np.isfinite(v) for v in st.values()), st
    assert all(torch.isfinite(p).all() for p in net.parameters())


# ------------------------------------------------- E3: reward (DESIGN_E3.md §5)

@test
def test_e3_6_timeout_terminal_with_penalty():
    still = np.full((8, 2), -1.0)                          # speed -> 0: everyone times out
    for pen, want in ((0.0, "truncated"), (25.0, "terminal")):
        cfg = replace(Config(), max_steps=3, timeout_penalty=pen)
        e = envmod.MultiAgentEnv(EnvConfig(), cfg, np.random.default_rng(6))
        e.reset()
        n0 = e.n
        for _ in range(3):
            r = e.step(still)
        other = "terminal" if want == "truncated" else "truncated"
        assert r[want][:n0].all() and not r[other].any(), (pen, r[want], r[other])
        if want == "terminal":
            assert not r["final_obs"].any(), "no bootstrap for a penalised timeout"
            assert np.allclose(r["reward"][:n0], -0.01 - pen)
        else:
            assert (r["final_obs"][:n0] != 0).any(axis=1).all()


@test
def test_e3_7_reward_config_roundtrip():
    import torch  # noqa: F401
    import train
    from policyconfig import RunConfig, sim_config
    cfg = RunConfig.make("gru", 0, reward={"timeout_penalty": 25.0, "progress_metric": "geodesic"})
    sc = sim_config(cfg.reward)
    assert sc.timeout_penalty == 25.0 and sc.progress_metric == "geodesic"
    try:
        sim_config({"max_speed": 9.0})
        raise AssertionError("a physics field must not pass as a reward override")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg.save(d / "config.json", sc)
        back = RunConfig.load(d / "config.json")
        assert back.reward == cfg.reward and back == cfg
        out = d / "run"
        common = ["--memory", "gru", "--out", str(out), "--layouts", "open", "--agents", "2", "3",
                  "--W", "0", "--E", "2", "--T", "16", "--L", "16", "--total-steps", "1000000",
                  "--stop-after", "1"]
        train.main(common + ["--timeout-penalty", "25", "--progress-metric", "geodesic"])
        ck = train.load_ckpt(out / "ckpt_last.pt")
        assert ck["config"]["reward"] == {"timeout_penalty": 25.0, "progress_metric": "geodesic"}
        assert ck["config"]["sim"]["timeout_penalty"] == 25.0
        assert ck["config"]["sim"]["progress_metric"] == "geodesic"
        # --resume without the reward flags keeps the checkpoint's reward
        train.main(common + ["--resume"])
        ck = train.load_ckpt(out / "ckpt_last.pt")
        assert ck["iteration"] == 2 and ck["config"]["reward"]["timeout_penalty"] == 25.0
        assert ck["config"]["sim"]["progress_metric"] == "geodesic"
        assert RunConfig.load(out / "config.json").reward == cfg.reward


@test
def test_e3_8_variant_grouping():
    import contextlib
    import io
    import json
    import aggregate
    import evaluate
    rng = np.random.default_rng(8)

    def fake_rows():
        return [{"layout": l, "bottleneck": 8.0, **{k: float(rng.random()) for k in evaluate.ROW_METRICS}}
                for l in ("open", "doorway") for _ in range(3)]

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, ident in (("greedy", {"controller": "greedy", "variant": "greedy", "seed": None}),
                            ("e3_geo_seed0", {"controller": "gru", "variant": "e3_geo", "seed": 0}),
                            ("e3_geo_seed1", {"controller": "gru", "variant": "e3_geo", "seed": 1}),
                            ("e2_gru_seed0", {"controller": "gru", "seed": 0})):   # pre-E3: no variant
            run = d / "runs" / name
            ckpt = None
            if name != "greedy":
                run.mkdir(parents=True)
                ckpt = str(run / "ckpt_best.pt")
                (run / "val.csv").write_text(
                    "iteration,agent_steps,success_rate,collision_rate\n"
                    f"25,1000,{0.5 + 0.2 * ident['seed']},0.3\n50,2000,0.6,0.2\n")
            (d / "eval" / name).mkdir(parents=True)
            metrics = {"identity": {**ident, "ckpt": ckpt},
                       "suites": {"test4": evaluate.metrics_for(fake_rows())}}
            (d / "eval" / name / "metrics.json").write_text(json.dumps(metrics))
        res = d / "res"
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate.main(["--evals", str(d / "eval"), "--out", str(res), "--runs", str(d / "runs"),
                            "--group", "variant", "--variants", "greedy", "e2_gru", "e3_geo"])
        text = (res / "table_e3.md").read_text(encoding="utf-8")
        header = [l for l in text.splitlines() if l.startswith("| layout")][0]
        assert header == "| layout | greedy | e2_gru | e3_geo |", header
        assert "e3_geo 2" in text and "Timeout rate" in text
        win = json.loads((res / "winner.json").read_text())
        assert win["variant"] == "e3_geo" and win["seeds"] == [0, 1], win
        assert abs(win["val_success_mean"] - 0.65) < 1e-9         # best rows 0.6 and 0.7
        # the default grouping still gives the E2 columns
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate.main(["--evals", str(d / "eval"), "--out", str(d / "res2")])
        header = [l for l in (d / "res2" / "table_e2.md").read_text(encoding="utf-8").splitlines()
                  if l.startswith("| layout")][0]
        assert header == "| layout | greedy | gru |", header


# ------------------------------------------------------ ORCA (DESIGN_ORCA.md §5)

def _grid_best(lines, radius, v_pref, step=0.005):
    """Dense grid over the speed disk: (nearest feasible point or None, smallest max-violation)."""
    g = np.arange(-radius, radius + step, step)
    X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    pts = pts[(pts ** 2).sum(1) <= radius * radius]
    viol = np.full(len(pts), -np.inf)
    for (px, py), (dx, dy) in lines:
        viol = np.maximum(viol, dx * (py - pts[:, 1]) - dy * (px - pts[:, 0]))
    feas = viol <= 0
    near = None
    if feas.any():
        f = pts[feas]
        near = f[np.argmin(((f - v_pref) ** 2).sum(1))]
        if step > 0.0005:                                # refine round the coarse answer
            loc = near + np.stack(np.meshgrid(*[np.arange(-0.03, 0.0301, 0.0005)] * 2), -1).reshape(-1, 2)
            loc = loc[(loc ** 2).sum(1) <= radius * radius]
            v = np.full(len(loc), -np.inf)
            for (px, py), (dx, dy) in lines:
                v = np.maximum(v, dx * (py - loc[:, 1]) - dy * (px - loc[:, 0]))
            f = loc[v <= 0]
            near = f[np.argmin(((f - v_pref) ** 2).sum(1))]
    return near, viol.min()


@test
def test_orca_lp_matches_grid_search():
    from orca import linear_program, max_violation
    rng = np.random.default_rng(0)
    R = 2.5
    n_feas = n_infeas = 0
    while n_feas < 25 or n_infeas < 10:
        k = int(rng.integers(1, 9))
        lines = []
        for _ in range(k):
            ang = rng.uniform(-math.pi, math.pi)
            pt = rng.uniform(-2.5, 2.5, 2)
            lines.append(((pt[0], pt[1]), (math.cos(ang), math.sin(ang))))
        v_pref = rng.uniform(-3.5, 3.5, 2)
        near, best_viol = _grid_best(lines, R, v_pref)
        x = linear_program(lines, R, v_pref)
        assert np.hypot(*x) <= R + 1e-6
        if best_viol <= -0.05 and n_feas < 25:          # clearly feasible
            n_feas += 1
            assert max_violation(lines, x) <= 1e-6, (lines, x)
            assert np.hypot(*(x - near)) <= 0.02, (x, near)
        elif best_viol >= 0.05 and n_infeas < 10:       # clearly infeasible: 3-D fallback
            n_infeas += 1
            assert abs(max_violation(lines, x) - best_viol) <= 0.02, (max_violation(lines, x), best_viol)


def _orca_run(scene, mode="straight", cfg=None):
    import evaluate
    from orca import ORCAController
    cfg = cfg or Config()
    sim = Simulator(scene, cfg)
    ctrl = ORCAController(mode)
    ctrl.reset(sim)
    obs = sim.observe()
    done = sim.done
    while not done:
        obs, _, done = sim.step(ctrl.act(sim, obs))
    return sim


def _open_scene(agents_goals, obstacles=()):
    from scene import Agent, Obstacle, Region, Scene, boundary_walls
    agents = [Agent(tuple(p), h, goal_region=Region(*g)) for p, h, g in agents_goals]
    return Scene(walls=boundary_walls(100.0), obstacles=[Obstacle(c, r) for c, r in obstacles],
                 agents=agents, goal_region=agents[0].goal_region,
                 start_region=Region(1, 1, 99, 99))


@test
def test_orca_head_on_swap():
    scene = _open_scene([((40.0, 50.0), 0.0, (58, 46, 66, 54)),
                         ((60.0, 50.0), math.pi, (34, 46, 42, 54))])
    sim = _orca_run(scene)
    assert sim.outcome == ["goal", "goal"], sim.outcome
    assert sim.collisions.sum() == 0


@test
def test_orca_four_way_swap():
    c, d = 50.0, 14.0
    spec = []
    for k in range(4):
        a = k * math.pi / 2
        p = (c + d * math.cos(a), c + d * math.sin(a))
        q = (c - d * math.cos(a), c - d * math.sin(a))
        spec.append((p, a + math.pi, (q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4)))
    sim = _orca_run(_open_scene(spec))
    assert sim.outcome == ["goal"] * 4, sim.outcome
    assert sim.collisions.sum() == 0


@test
def test_orca_avoids_static_circle():
    scene = _open_scene([((20.0, 50.0), 0.0, (80, 44, 90, 56))], obstacles=[((50.0, 50.0), 8.0)])
    sim = _orca_run(scene)
    assert sim.outcome == ["goal"] and sim.collisions.sum() == 0
    traj = np.array(sim.trajectories[0])
    near = traj[np.abs(traj[:, 0] - 50) < 3]
    assert (np.abs(near[:, 1] - 50) > 8 + 2).all(), "went round, not through"


@test
def test_orca_path_heads_for_the_gap():
    from orca import ORCAController
    from scene import Agent, Region, Scene, boundary_walls
    # wall at x = 50 with a gap 40..50 (centre lines), goal behind it; agent below the gap's axis
    walls = boundary_walls(100.0) + [((50.0, 0.0), (50.0, 40.0)), ((50.0, 50.0), (50.0, 100.0))]
    goal = Region(80.0, 10.0, 95.0, 30.0)
    scene = Scene(walls=walls, goal_region=goal, start_region=Region(5, 5, 20, 95),
                  agents=[Agent((35.0, 20.0), 0.0)])
    cfg = Config()
    sim = Simulator(scene, cfg)
    path, straight = ORCAController("path"), ORCAController("straight")
    path.reset(sim)
    straight.reset(sim)
    p = sim.pos[0]
    vp = path._path(p, goal, cfg)
    vs = straight._straight(p, goal, cfg)
    assert abs(np.hypot(*vp) - cfg.max_speed) < 1e-9
    assert vs[1] < 0.1 * cfg.max_speed, "straight bearing points at the wall (east)"
    assert vp[1] > 0.5 * cfg.max_speed, ("path bearing climbs toward the gap at y 40..50", vp)
    # and the agent actually gets through
    s = _orca_run(scene, "path")
    assert s.outcome == ["goal"], s.outcome


@test
def test_orca_action_conversion():
    from orca import velocity_to_action
    cfg = Config()
    a = velocity_to_action(np.array([2.5, 0.0]), 0.0, np.array([2.5, 0.0]), cfg)
    assert np.allclose(a, [0.0, 1.0])
    v = 2.5 * np.array([math.cos(math.radians(120)), math.sin(math.radians(120))])
    a = velocity_to_action(v, 0.0, v, cfg)
    assert np.allclose(a, [1.0, 0.0]), a
    w = 2.5 * np.array([math.cos(math.radians(-60)), math.sin(math.radians(-60))])
    a = velocity_to_action(w, 0.0, w, cfg)                         # 60 deg to the right
    assert a[0] == -1.0 and abs(a[1] - 0.5) < 1e-9, a


@test
def test_orca_deterministic():
    cfg = Config()
    scene = random_scene(np.random.default_rng(21), cfg, layout="crossing", n_agents=6,
                         n_obstacles=3)
    scene = envmod.randomise_headings(scene, np.random.default_rng(22))
    for mode in ("straight", "path"):
        a, b = _orca_run(scene, mode), _orca_run(scene, mode)
        assert a.trajectories == b.trajectories and a.outcome == b.outcome, mode


@test
def test_orca_eval_and_aggregate():
    import contextlib
    import io
    import json
    import aggregate
    import evaluate
    from suites import make_suite
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_suite(d / "data" / "test4", ("open", "doorway"), 5, (4, 4), (0, 2), 31, Config(),
                   name="test4")
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.main(["--orca", "straight", "--suites", "test4", "--data-root", str(d / "data"),
                           "--out", str(d / "eval" / "orca_straight")])
            evaluate.main(["--greedy", "--suites", "test4", "--data-root", str(d / "data"),
                           "--out", str(d / "eval" / "greedy")])
        ev = d / "eval" / "orca_straight"
        for f in ("per_scene.csv", "metrics.json"):
            assert (ev / f).exists(), f
        assert len(list((ev / "trajectories").glob("*.json"))) == 2
        m = json.loads((ev / "metrics.json").read_text())
        assert m["identity"]["controller"] == "orca" and m["identity"]["variant"] == "orca_straight"
        assert m["identity"]["seed"] is None and m["suites"]["test4"]["overall"]["scenes"] == 10
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate.main(["--evals", str(d / "eval"), "--out", str(d / "res")])
        text = (d / "res" / "table_e2.md").read_text(encoding="utf-8")
        header = [l for l in text.splitlines() if l.startswith("| layout")][0]
        assert header == "| layout | greedy | orca_straight |", header
        rows = [l for l in text.splitlines() if l.startswith("| open")]
        assert rows and "±" not in rows[0], rows                 # one run: no std
        assert "orca_straight 1" in text
        for f in ("fig_per_layout.png", "fig_bottleneck.png", "fig_paths.png"):
            assert (d / "res" / f).exists(), f


# ----------------------------------------- E4: layout hold-out (DESIGN_E4.md §7)

def _run_e4():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_e4.py"
    spec = importlib.util.spec_from_file_location("run_e4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frame_only(scene, cfg):
    from layouts import boundary_walls
    return len(scene.walls) == len(boundary_walls(cfg.world))


@test
def test_e4_1_sampling_never_draws_the_held_out_layout():
    from policyconfig import sim_config
    run_e4 = _run_e4()
    cfg = sim_config()
    for i, held in enumerate(LAYOUTS_E2):
        env_cfg = EnvConfig(layouts=tuple(run_e4.train_layouts(held)))
        rng = np.random.default_rng(100 + i)
        seen = {envmod.sample_scene(env_cfg, cfg, rng).layout for _ in range(50)}
        assert held not in seen and seen == set(env_cfg.layouts), (held, seen)


@test
def test_e4_2_open_fold_rejects_open_scenes_under_other_labels():
    """walls_multi / rooms_multi fall back to the open generator and keep their label."""
    from policyconfig import sim_config
    cfg = sim_config()
    multi = ("walls_multi", "rooms_multi")
    rng = np.random.default_rng(3)
    drawn = [envmod.sample_scene(EnvConfig(layouts=("open",) + multi), cfg, rng)
             for _ in range(120)]
    assert any(s.layout in multi and _frame_only(s, cfg) for s in drawn), "test is vacuous"
    rng = np.random.default_rng(3)
    for _ in range(120):
        s = envmod.sample_scene(EnvConfig(layouts=multi), cfg, rng)
        assert s.layout in multi and not _frame_only(s, cfg)
    # the same rule at the scene level, whatever the generator does
    real = envmod.random_scene
    wall_free = real(np.random.default_rng(0), cfg, layout="open", n_agents=2, n_obstacles=0)
    fake = replace(wall_free, layout="rooms_multi")
    calls = []

    def first_fake(*a, **k):
        calls.append(k["layout"])
        return fake if len(calls) == 1 else real(*a, **k)

    envmod.random_scene = first_fake
    try:
        s = envmod.sample_scene(EnvConfig(layouts=multi), cfg, np.random.default_rng(1))
    finally:
        envmod.random_scene = real
    assert len(calls) == 2 and not _frame_only(s, cfg)
    assert envmod.in_training_layouts(fake, ("open",) + multi, cfg)    # E2 / E3 keep such scenes
    assert not envmod.in_training_layouts(fake, multi, cfg)
    # and validation: an open fold cannot choose its checkpoint on such scenes either
    import evaluate
    from scene import Scene
    from suites import make_suite
    with tempfile.TemporaryDirectory() as d:
        make_suite(Path(d) / "val", multi, 40, (2, 4), (0, 2), 13, cfg, name="val")
        every = evaluate.Validator(Path(d) / "val", cfg)
        fold = evaluate.Validator(Path(d) / "val", cfg, layouts=multi)
        free = sum(_frame_only(Scene.from_dict(s), cfg) for s in every.scene_dicts)
        assert free > 0, "test is vacuous"
        assert len(fold.scene_dicts) == len(every.scene_dicts) - free
        assert not any(_frame_only(Scene.from_dict(s), cfg) for s in fold.scene_dicts)


@test
def test_e4_3_all_ten_layouts_unchanged():
    """E2 and E3: the sampled stream and the validation set are what they were."""
    import evaluate
    from policyconfig import sim_config
    from suites import make_suite
    cfg = sim_config()

    def old_sample(ec, rng):                               # sample_scene before E4
        while True:
            layout = str(rng.choice(ec.layouts))
            n = int(rng.integers(ec.agents[0], ec.agents[1] + 1))
            m = int(rng.integers(ec.obstacles[0], ec.obstacles[1] + 1))
            scene = random_scene(rng, cfg, layout=layout, n_agents=n, n_obstacles=m)
            if scene is None or not 1 <= len(scene.agents) <= ec.max_agents:
                continue
            return envmod.randomise_headings(scene, rng)

    a, b = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(40):
        x, y = envmod.sample_scene(EnvConfig(), cfg, a), old_sample(EnvConfig(), b)
        assert x.as_dict() == y.as_dict()
    with tempfile.TemporaryDirectory() as d:
        make_suite(Path(d) / "val", LAYOUTS_E2, 2, (2, 4), (0, 2), 9, cfg, name="val")
        plain = evaluate.Validator(Path(d) / "val", cfg)
        full = evaluate.Validator(Path(d) / "val", cfg, layouts=LAYOUTS_E2)
        assert full.names == plain.names and full.scene_dicts == plain.scene_dicts
        assert len(plain.names) == 2 * len(LAYOUTS_E2)
        some = evaluate.Validator(Path(d) / "val", cfg, layouts=("open", "doorway"))
        assert {s["layout"] for s in some.scene_dicts} == {"open", "doorway"}
        make_suite(Path(d) / "other", ("corridor",), 1, (2, 4), (0, 2), 9, cfg, name="other")
        try:
            evaluate.Validator(Path(d) / "other", cfg, layouts=("open",))
            raise AssertionError("an empty validation set must be refused")
        except ValueError:
            pass


@test
def test_e4_4_held_out_layout_absent_from_a_real_fold():
    """Training scenes, validation scenes, selection metric and saved config, with resume."""
    import csv
    import contextlib
    import io
    import evaluate
    import train
    from policyconfig import RunConfig, sim_config
    from suites import make_suite
    run_e4 = _run_e4()
    held = "doorway"
    others = tuple(run_e4.train_layouts(held))
    sim = sim_config(run_e4.REWARD)
    drawn, validated = [], []
    real_scene, real_episode = envmod.random_scene, evaluate.run_episode

    def spy_scene(*a, **k):
        drawn.append(k["layout"])
        return real_scene(*a, **k)

    def spy_episode(scene, *a, **k):
        validated.append(scene.layout)
        return real_episode(scene, *a, **k)

    envmod.random_scene, evaluate.run_episode = spy_scene, spy_episode
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_suite(d / "val", LAYOUTS_E2, 1, (2, 3), (0, 1), 7, sim_config(), name="val")
            out = d / "run"
            cfg = replace(tiny_run_cfg("gru", E=2, T=16, L=16, W=0, layouts=others),
                          val_every=1, ckpt_every=1, reward=dict(run_e4.REWARD))
            train.run(cfg, sim, out, val_suite=d / "val", quiet=True, stop_after=2)
            with contextlib.redirect_stdout(io.StringIO()):       # no --layouts: from the checkpoint
                train.main(["--out", str(out), "--resume", "--val-suite", str(d / "val"),
                            "--W", "0", "--stop-after", "1"])
            header = next(csv.reader(open(out / "val.csv")))
            n_val = len(list(csv.DictReader(open(out / "val.csv"))))
            saved = [RunConfig.load(out / "config.json").env.layouts]
            for name in ("ckpt_last.pt", "ckpt_best.pt"):
                ck = train.load_ckpt(out / name)
                saved.append(tuple(ck["config"]["env"]["layouts"]))
            best_val = train.load_ckpt(out / "ckpt_best.pt")["val"]
            last = train.load_ckpt(out / "ckpt_last.pt")
    finally:
        envmod.random_scene, evaluate.run_episode = real_scene, real_episode
    assert drawn and held not in drawn, drawn
    assert set(validated) == set(others), validated
    assert len(validated) == n_val * len(others), (len(validated), n_val)
    assert f"success_{held}" not in header
    assert all(f"success_{l}" in header for l in others)
    assert all(tuple(s) == others for s in saved), saved
    assert f"success_{held}" not in best_val
    assert last["iteration"] == 3 and last["config"]["reward"] == run_e4.REWARD


@test
def test_e4_5_worker_processes_keep_to_the_fold():
    from dataclasses import replace as rep
    from policyconfig import sim_config
    from nets import ActorCritic
    from rollout import WorkerPool
    run_e4 = _run_e4()
    held = "cul_de_sac"
    cfg = tiny_run_cfg("gru", E=2, T=32, L=16, W=2, layouts=tuple(run_e4.train_layouts(held)))
    sim = rep(sim_config(), max_steps=8)                    # scenes end inside one iteration
    model = ActorCritic(envmod.obs_dim(cfg.env), cfg.net)
    pool = WorkerPool(cfg, sim)
    try:
        records = [r for _ in range(3) for r in pool.collect(model.get_weights())["records"]]
    finally:
        pool.close()
    seen = {r["layout"] for r in records}
    assert seen and held not in seen and seen <= set(cfg.env.layouts), seen


@test
def test_e4_6_dry_run_prints_the_windows_commands():
    import contextlib
    import importlib.util
    import io
    import json
    run_e4 = _run_e4()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_e4.main(["--dry-run"])
    text = buf.getvalue()
    trains = [l for l in text.splitlines() if " train.py " in l]
    evals = [l for l in text.splitlines() if " evaluate.py " in l]
    assert len(trains) == len(evals) == 10
    # the reward is the E3 winner's, flag for flag
    win = json.loads((Path(run_e4.__file__).parent.parent / "results/e3/winner.json").read_text())
    spec = importlib.util.spec_from_file_location("run_e3", Path(run_e4.__file__).parent / "run_e3.py")
    run_e3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_e3)
    flags = " ".join(run_e3.VARIANTS[win["variant"]])
    for held, line in zip(LAYOUTS_E2, trains):
        layouts = line.split(" --layouts ")[1].split()[0].split(",")
        assert held not in layouts and len(layouts) == 9 and set(layouts) < set(LAYOUTS_E2)
        assert f"--out E:\\policy-runs\\e4_holdout_{held}_seed0 " in line, line
        assert "--val-suite E:\\policy-data\\val " in line and flags in line, line
        assert "--seed 0 " in line and "--resume" not in line
    assert f"--out E:\\policy-eval\\e4\\e4_holdout_open_seed0 " in evals[0], evals[0]
    assert "--ckpt E:\\policy-runs\\e4_holdout_open_seed0\\ckpt_best.pt " in evals[0]
    assert "aggregate_e4.py --evals E:\\policy-eval --runs E:\\policy-runs --out results/e4" in text
    assert 'git commit -m "E4: leave-one-layout-out results"' in text
    assert "git push" in text and "git pull --rebase --autostash" in text
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_e4.main(["--dry-run", "--no-push", "--seeds", "1", "2", "--layouts", "room", "open"])
    text = buf.getvalue()
    assert "git push" not in text and "git pull" not in text
    trains = [l for l in text.splitlines() if " train.py " in l]
    names = [l.split("--out ")[1].split()[0].rsplit("\\", 1)[1] for l in trains]
    assert names == ["e4_holdout_room_seed1", "e4_holdout_open_seed1",
                     "e4_holdout_room_seed2", "e4_holdout_open_seed2"], names


@test
def test_e4_7_driver_resumes_and_skips():
    import contextlib
    import io
    import json
    import torch
    run_e4 = _run_e4()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = run_e4.parse_args(["--runs", str(d / "runs"), "--eval", str(d / "eval"),
                               "--data-root", str(d / "data")])
        held, seed = "room", 0
        name = run_e4.run_name(held, seed)
        rd = d / "runs" / name
        rd.mkdir(parents=True)
        calls = []
        run_e4.run = lambda cmd, dry, cwd=None: calls.append([str(c) for c in cmd])

        def save(steps, layouts=None, reward=None):
            torch.save({"agent_steps": steps, "config": {
                "env": {"layouts": layouts or run_e4.train_layouts(held)},
                "reward": reward or dict(run_e4.REWARD), "seed": seed,
                "ppo": {"total_steps": 100}}}, rd / "ckpt_last.pt")

        def fold():
            calls.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                run_e4.run_fold(a, held, seed, dry=False)
            return [c[1] for c in calls], list(calls)

        assert fold()[0] == ["train.py", "evaluate.py"]           # new: no --resume
        assert "--resume" not in calls[0]
        save(50)                                                  # interrupted
        steps, cmds = fold()
        assert steps == ["train.py", "evaluate.py"] and "--resume" in cmds[0]
        save(100)                                                 # finished, not evaluated
        assert fold()[0] == ["evaluate.py"]
        ev = d / "eval" / "e4" / name
        ev.mkdir(parents=True)
        (ev / "metrics.json").write_text("{\"suites\": {\"test4\"")   # a write that was cut off
        assert fold()[0] == ["evaluate.py"]
        per = {held: {"success_rate": 0.5, "collision_rate": 0.2, "timeout_rate": 0.3}}
        (ev / "metrics.json").write_text(json.dumps({"suites": {
            "test4": {"per_layout": per}, "test_mixed": {"per_layout": per}}}))
        (rd / "ckpt_best.pt").write_text("x")
        os.utime(rd / "ckpt_best.pt", (1000, 1000))
        assert fold()[0] == []                                    # finished fold: nothing runs
        os.utime(rd / "ckpt_best.pt", None)                       # the run was extended later
        os.utime(ev / "metrics.json", (1000, 1000))
        assert fold()[0] == ["evaluate.py"]
        for bad in ({"layouts": list(LAYOUTS_E2)}, {"reward": {"timeout_penalty": 25.0}}):
            save(50, **bad)                                       # not this fold: never resumed
            try:
                fold()
                raise AssertionError(f"{bad} must stop the driver")
            except SystemExit as e:
                assert "is not fold" in str(e)


def _fake_eval(root, name, ident, success_by_layout, bins):
    """An evaluation dir; every layout has 4 scenes, success as given, a bottleneck from ``bins``."""
    import csv
    import json
    import evaluate
    rows = []
    for layout, ok in success_by_layout.items():
        for i in range(4):
            r = {k: 0.0 for k in evaluate.ROW_METRICS}
            r.update(layout=layout, suite="test4", scene=f"{layout}{i}", n_agents=4,
                     bottleneck=bins[i % len(bins)], success_rate=ok, collision_rate=1 - ok)
            rows.append(r)
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(
        {"identity": ident, "suites": {"test4": evaluate.metrics_for(rows)}}))
    cols = ["suite", "scene", "layout", "n_agents", "bottleneck"] + evaluate.ROW_METRICS
    with open(d / "per_scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@test
def test_e4_8_aggregate_reports_held_out_scenes_only():
    import contextlib
    import csv
    import io
    import json
    import aggregate_e4
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ev = d / "eval"
        # baseline: 0.9 / 0.8 on the two layouts with folds, 0.1 on one without
        for seed, shift in ((0, 0.0), (1, 0.02)):
            _fake_eval(ev, f"e3_both_seed{seed}",
                       {"controller": "gru", "variant": "e3_both", "seed": seed, "ckpt": None},
                       {"open": 0.9 + shift, "doorway": 0.8 + shift, "crossing": 0.1}, [6.0, 8.0])
        # E2 and an E3 variant in the same root must not count as baseline or fold
        _fake_eval(ev, "e2_gru_seed0", {"controller": "gru", "seed": 0, "ckpt": None},
                   {"open": 0.0, "doorway": 0.0}, [6.0])
        # fold: bad on its held-out layout, good on the trained ones, narrow scenes differ
        for held, low in (("open", 0.5), ("doorway", 0.3)):
            name = f"e4_holdout_{held}_seed0"
            run = d / "runs" / name
            run.mkdir(parents=True)
            layouts = [l for l in LAYOUTS_E2 if l != held]
            (run / "config.json").write_text(json.dumps({"env": {"layouts": layouts}}))
            (run / "val.csv").write_text("iteration,agent_steps,success_rate,collision_rate\n"
                                         "25,1000,0.5,0.3\n50,2000,0.6,0.2\n")
            trained = "doorway" if held == "open" else "open"
            _fake_eval(ev / "e4", name,
                       {"controller": "gru", "variant": f"e4_holdout_{held}", "seed": 0,
                        "ckpt": str(run / "ckpt_best.pt")},
                       {held: low, trained: 1.0, "crossing": 1.0}, [6.0, 20.0])
        res = d / "res"
        args = ["--evals", str(ev), "--runs", str(d / "runs"), "--out", str(res)]
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate_e4.main(args)
        text = (res / "table_e4.md").read_text(encoding="utf-8")
        row = [l for l in text.splitlines() if l.startswith("| open |")][0].split(" | ")
        assert row[1] == "1" and row[2] == "0.910 ± 0.010" and row[3] == "0.500", row
        assert row[4] == "+0.410", row
        row = [l for l in text.splitlines() if l.startswith("| doorway |")][0].split(" | ")
        assert row[4] == "+0.510", row
        # mean over folds: only the two folds, crossing's 0.1 stays out
        row = [l for l in text.splitlines() if l.startswith("| **mean over folds**")][0].split(" | ")
        assert row[2] == "0.860" and row[3] == "0.400" and row[4] == "+0.460", row
        with open(res / "summary_bottleneck.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        got = {(r["layout"], r["bin"], r["series"]): r for r in rows}
        # the fold's 1.0 on trained layouts must not reach the held-out split
        h = got[("open", "[5.2,7)", "held-out")]
        assert float(h["success"]) == 0.5 and h["scenes"] == "2" and h["runs"] == "1", h
        assert got[("open", "[15,inf)", "held-out")]["success"] == "0.5"
        a = got[("ALL", "[5.2,7)", "e3_both")]                    # baseline restricted to the folds
        assert abs(float(a["success"]) - 0.86) < 1e-9 and a["scenes"] == "4", a
        # the control column averages the trained layouts of the fold, not its held-out one
        fold = aggregate_e4.load_folds(ev)["open"][0]
        assert aggregate_e4.trained_mean(fold, "test4", ["doorway", "crossing"]) == 1.0
        assert aggregate_e4.trained_mean(fold, "test4", ["doorway", "room"]) is None
        for f in ("summary.csv", "summary_per_layout.csv", "fig_per_layout.png",
                  "fig_bottleneck.png", "fig_training_curves.png"):
            assert (res / f).stat().st_size > 0, f
        assert not (res / "table_e4_mixed.md").exists()           # suite absent: skipped
        # a mislabelled fold stops the report
        run = d / "runs" / "e4_holdout_open_seed0"
        (run / "config.json").write_text(json.dumps({"env": {"layouts": list(LAYOUTS_E2)}}))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                aggregate_e4.main(args)
            raise AssertionError("a fold that trained on its held-out layout must stop the report")
        except SystemExit as e:
            assert "trained on open" in str(e)
        # the default E2 / E3 report never sees the folds
        import aggregate
        assert not [e for e in aggregate.load_evals(ev) if e["name"].startswith("e4_")]

# ------------------------------------------------------------ behaviour gap

GAP_CKPT = Path(__file__).resolve().parents[1] / "models" / "e3_both_seed0" / "ckpt_best.pt"


def gap_fixture(root: Path, n=6, set_name="set_a"):
    """A set whose parses copy the true scene, but scene 0 failed and scene 1 has no exit."""
    import csv
    import json
    import gap as gapmod
    cfg = Config()
    rng = np.random.default_rng(5)
    data, parsed, perc = root / "data" / set_name, root / "parsed" / set_name, root / "perc" / set_name
    for d in (data, parsed, perc):
        d.mkdir(parents=True)
    (data / "dataset.json").write_text(json.dumps({"config": {"world": cfg.world}}))
    rows = []
    for i in range(n):
        stem = f"scene_{i:05d}"
        scene = random_scene(rng, cfg, layout="room", n_agents=3, n_obstacles=1)
        d = scene.as_dict(cfg, stem)
        (data / f"{stem}.json").write_text(json.dumps(d))
        if i == 0:
            rec = {"id": stem, "parsed": False, "parse_fail": True}
        else:
            rec = dict(d, layout=None, bottleneck=None, parsed=True)
            if i == 1:
                rec.update(goal_region=None, exits=None, groups=[])
        (parsed / f"{stem}.json").write_text(json.dumps(rec))
        rows.append({"set": set_name, "scene": stem, "parse_fail": float(i == 0),
                     "n_pred_agents": 0 if i == 0 else 3,
                     **{k: 0.9 + 0.01 * i for k in gapmod.PERCEPTION}})
    with open(perc / "per_scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (perc / "metrics.json").write_text(json.dumps(
        {"_meta": {"ckpt": "models\\sketch7\\ckpt_best.pt", "extract": {"heat_thresh": 0.4}}}))
    return ["--ckpt", str(GAP_CKPT), "--data-root", str(root / "data"), "--parsed", str(root / "parsed"),
            "--perception", str(root / "perc"), "--sets", set_name, "--out", str(root / "out")]


@test
def test_gap_block_keeps_failed_parses_out_of_the_gap():
    from gap import block
    nan = float("nan")
    ok = {"parse_ok": 1.0, "true_success": 1.0, "parsed_success": 0.5, "gap_success": -0.5,
          "true_collision": 0.0, "parsed_collision": 0.5, "gap_collision": 0.5,
          "true_timeout": 0.0, "parsed_timeout": 0.0, "gap_timeout": 0.0}
    plus = dict(ok, true_success=0.5, parsed_success=1.0, gap_success=0.5, true_collision=0.5,
                parsed_collision=0.0, gap_collision=-0.5)
    bad = {k: nan for k in ok}
    bad.update(parse_ok=0.0, true_success=0.0, true_collision=1.0, true_timeout=0.0)
    b = block([ok, plus, bad])
    assert (b["scenes"], b["parsed_scenes"], b["failed_parses"]) == (3, 2, 1)
    assert b["true_success"] == 0.75                       # the scenes with a parse only
    assert abs(b["true_all_success"] - 0.5) < 1e-12        # all three
    assert b["gap_success"] == 0.0 and b["abs_gap_success"] == 0.5   # opposite errors cancel
    assert b["end_to_end_success"] == 0.5                  # (0.5 + 1.0 + 0) / 3
    assert b["true_success_on_failed"] == 0.0


@test
def test_gap_identical_parse_has_no_gap_and_failures_are_reported():
    import gap as gapmod
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = gap_fixture(root)
        gapmod.main(args)
        rows = gapmod.numeric(gapmod.read_rows(root / "out" / "per_scene.csv"))
        assert len(rows) == 6
        assert rows[0]["parse_ok"] == 0 and "not parsed" in rows[0]["parse_reason"]
        assert rows[1]["parse_ok"] == 0 and "no exit" in rows[1]["parse_reason"]
        for r in rows[:2]:                                 # the true episode still ran
            assert not math.isnan(r["true_success"]) and math.isnan(r["parsed_success"])
        for r in rows[2:]:                                 # same scene, deterministic policy
            assert r["parse_ok"] == 1 and r["gap_success"] == r["gap_collision"] == r["gap_timeout"] == 0
        assert sum(r["parse_differs"] for r in rows) == 0
        text = (root / "out" / "gap.md").read_text(encoding="utf-8")
        assert "## Failed parses" in text and "| set_a | 6 | 2 | 0.333 |" in text
        assert "no exit" in text
        assert (root / "out" / "fig_gap.png").stat().st_size > 0
        # another parse must not extend these rows
        meta = root / "perc" / "set_a" / "metrics.json"
        meta.write_text(meta.read_text().replace("0.4", "0.5"))
        try:
            gapmod.main(args)
        except SystemExit as e:
            assert "another policy or parse" in str(e)
        else:
            raise AssertionError("a run of another parse was accepted")


@test
def test_gap_refuses_a_set_with_other_physics():
    import json
    import gap as gapmod
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = gap_fixture(root, n=3)
        (root / "data" / "set_a" / "dataset.json").write_text(json.dumps({"config": {"max_speed": 9.0}}))
        try:
            gapmod.main(args)
        except SystemExit as e:
            assert "max_speed" in str(e)
        else:
            raise AssertionError("other physics were accepted")


@test
def test_gap_resumes_after_a_kill():
    import subprocess
    import time
    import gap as gapmod
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = gap_fixture(root, n=16)
        csv_path = root / "out" / "per_scene.csv"
        here = Path(__file__).resolve().parents[1]
        proc = subprocess.Popen([sys.executable, "gap.py", *args], cwd=here,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < 120:
            if csv_path.exists() and len(csv_path.read_text().splitlines()) >= 4:
                proc.kill()                                # this test's own child, no one else's
                break
            time.sleep(0.02)
        proc.wait()
        before = len(gapmod.read_rows(csv_path))
        assert before >= 3
        with open(csv_path, "a") as fh:                    # as if killed in the middle of a line
            fh.write("set_a,scene_99999,room,0")
        gapmod.main(args)
        rows = gapmod.read_rows(csv_path)
        assert [r["scene"] for r in rows] == [f"scene_{i:05d}" for i in range(16)]
        fresh = root / "fresh"
        gapmod.main(args[:-1] + [str(fresh)])              # a clean run gives the same rows
        assert gapmod.read_rows(fresh / "per_scene.csv") == rows


def _orca_map_scene():
    from scene import Agent, Obstacle, Region, Scene, boundary_walls
    return Scene(walls=boundary_walls(100.0) + [((50.0, 20.0), (50.0, 80.0))],
                 obstacles=[Obstacle((40.0, 42.0), 3.0)],
                 agents=[Agent((30.0, 50.0), 0.0, goal_region=Region(70.0, 46.0, 78.0, 54.0)),
                         Agent((30.0, 35.0), 0.3, goal_region=Region(70.0, 30.0, 78.0, 38.0)),
                         Agent((70.0, 65.0), math.pi, goal_region=Region(22.0, 60.0, 30.0, 68.0))],
                 goal_region=Region(70.0, 46.0, 78.0, 54.0), start_region=Region(1, 1, 99, 99))


def _orca_map_run(scene, mode="path", **kw):
    from orca import ORCAController
    sim = Simulator(scene, Config())
    ctrl = ORCAController(mode, **kw)
    ctrl.reset(sim)
    obs = sim.observe()
    while not sim.done:
        obs, _, _ = sim.step(ctrl.act(sim, obs))
    return sim, ctrl


@test
def test_orca_default_behaviour_matches_the_recorded_baseline():
    """Numbers recorded from orca.py before ``map_scene`` existed (commit eacc889)."""
    for mode, steps, outcome, final in (
            ("straight", 11, ["collision"] * 3,
             [[46.701, 43.892], [46.673, 28.89], [53.368, 70.956]]),
            ("path", 34, ["goal"] * 3,
             [[70.62, 53.075], [71.625, 30.908], [29.185, 67.162]])):
        sim, _ = _orca_map_run(_orca_map_scene(), mode)
        assert sim.steps == steps and sim.outcome == outcome, (mode, sim.steps, sim.outcome)
        assert np.allclose(sim.pos, final, atol=2e-3), (mode, sim.pos)


@test
def test_orca_map_scene_equal_to_the_world_changes_nothing():
    scene = _orca_map_scene()
    a, _ = _orca_map_run(scene)
    b, _ = _orca_map_run(scene, map_scene=_orca_map_scene())
    assert a.trajectories == b.trajectories and a.outcome == b.outcome


@test
def test_orca_wrong_map_keeps_the_true_world_and_goals():
    from dataclasses import replace as dc_replace
    from scene import boundary_walls
    scene = _orca_map_scene()
    # the map misses the wall: agents head straight through it and hit the real one
    sim, ctrl = _orca_map_run(scene, map_scene=dc_replace(scene, walls=boundary_walls(100.0)))
    assert sim.outcome == ["collision"] * 3, sim.outcome
    assert all(sim._hits_static(i) for i in range(3))
    assert ctrl.no_path_steps == 0
    # the map closes the wall to the world's edges: no route, so the straight bearing
    closed = dc_replace(scene, walls=scene.walls + [((50.0, 0.0), (50.0, 100.0))])
    sim, ctrl = _orca_map_run(scene, map_scene=closed)
    assert ctrl.no_path_steps > 0
    assert ctrl.goals == [a.goal_region for a in scene.agents]    # goals stay the world's
    # the map invents an obstacle: planning avoids it, physics does not know it
    from scene import Obstacle
    ghost = dc_replace(scene, obstacles=scene.obstacles + [Obstacle((60.0, 50.0), 4.0)])
    sim, ctrl = _orca_map_run(scene, map_scene=ghost)
    assert len(sim._obs_c) == 1 and len(ctrl.obs_c) == 2


@test
def test_gap_orca_arms_on_an_identical_map_have_no_gap():
    import gap as gapmod
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gapmod.main(gap_fixture(root))
        rows = gapmod.numeric(gapmod.read_rows(root / "out" / "per_scene.csv"))
        for r in rows[:2]:                                 # failed parse: only the true ORCA run
            assert not math.isnan(r["orca_true_success"])
            assert math.isnan(r["orca_parsed_success"]) and math.isnan(r["orca_wrongmap_success"])
            assert math.isnan(r["map_gap_success"])
        for r in rows[2:]:
            for m in gapmod.GAP_METRICS:
                assert r[f"orca_gap_{m}"] == 0 and r[f"map_gap_{m}"] == 0, (r["scene"], m)
            assert r["wrongmap_no_path_agents"] == 0 and r["orca_error"] == ""
        text = (root / "out" / "gap.md").read_text(encoding="utf-8")
        assert "## Arm 2" in text and "| set_a | orca_path | 4 |" in text
        assert (root / "out" / "fig_gap_map.png").stat().st_size > 0


@test
def test_gap_wrong_map_without_a_route_is_recorded_not_dropped():
    import json
    import gap as gapmod
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = gap_fixture(root, n=4)
        p = root / "parsed" / "set_a" / "scene_00002.json"
        rec = json.loads(p.read_text())
        x0, y0, x1, y1 = rec["goal_region"]
        ring = [[x0 - 1, y0 - 1], [x1 + 1, y0 - 1], [x1 + 1, y1 + 1], [x0 - 1, y1 + 1]]
        rec["walls"] += [[ring[i], ring[(i + 1) % 4]] for i in range(4)]    # walls round the exit
        p.write_text(json.dumps(rec))
        gapmod.main(args)
        rows = {r["scene"]: r for r in gapmod.numeric(gapmod.read_rows(root / "out" / "per_scene.csv"))}
        r = rows["scene_00002"]
        assert r["parse_ok"] == 1 and r["wrongmap_no_path_agents"] > 0 and r["wrongmap_no_path_steps"] > 0
        assert not math.isnan(r["orca_wrongmap_success"]) and r["orca_error"] == ""
        assert rows["scene_00003"]["wrongmap_no_path_agents"] == 0
        assert "no path at start" in (root / "out" / "gap.md").read_text(encoding="utf-8")


@test
def test_gap_orca_error_leaves_the_policy_row_standing():
    import gap as gapmod
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = gap_fixture(root, n=3)
        keep = gapmod.ORCAController
        def broken(*a, **k):
            raise RuntimeError("boom")
        gapmod.ORCAController = broken
        try:
            gapmod.main(args)
        finally:
            gapmod.ORCAController = keep
        rows = gapmod.numeric(gapmod.read_rows(root / "out" / "per_scene.csv"))
        assert all("boom" in r["orca_error"] for r in rows)
        assert rows[2]["parse_ok"] == 1 and not math.isnan(rows[2]["parsed_success"])
        assert "3 scene(s) with an error in an ORCA run" in (root / "out" / "gap.md").read_text(encoding="utf-8")


@test
def test_gap_refuses_rows_from_before_the_orca_arms():
    import json
    import gap as gapmod
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        args = gap_fixture(root, n=3)
        gapmod.main(args)
        out = root / "out"
        identity = json.loads((out / "identity.json").read_text())
        for key in ("arms", "orca_sha256"):                # an older run's identity, or another orca.py
            old = {k: v for k, v in identity.items() if k != key}
            (out / "identity.json").write_text(json.dumps(old))
            try:
                gapmod.main(args)
            except SystemExit as e:
                assert "another policy or parse" in str(e)
            else:
                raise AssertionError(f"identity without {key} was accepted")
        # same identity but the old columns: also refused, the file left alone
        (out / "identity.json").write_text(json.dumps(identity))
        old_columns = [c for c in gapmod.COLUMNS if "orca" not in c and "map" not in c]
        (out / "per_scene.csv").write_text(",".join(old_columns) + "\n")
        before = (out / "per_scene.csv").read_text()
        try:
            gapmod.main(args)
        except SystemExit as e:
            assert "other columns" in str(e)
        else:
            raise AssertionError("old columns were accepted")
        assert (out / "per_scene.csv").read_text() == before


@test
def test_gap_driver_dry_run_prints_the_windows_commands():
    import contextlib
    import io
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_gap
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_gap.main(["--dry-run"])
    text = buf.getvalue()
    flags = "--heat-thresh 0.4 --heat-nms-factor 0.8 --letter-thresh 0.2"
    for s in run_gap.SETS:
        assert f"--dir E:\\cnn-data\\{s} --out E:\\policy-eval\\gap\\sketch7\\parsed\\{s} {flags}" in text
        assert f"--data E:\\cnn-data\\{s} --out E:\\policy-eval\\gap\\sketch7\\perception\\{s} --no-figures {flags}" in text
    assert "gap.py --ckpt" in text and "--data-root E:\\cnn-data" in text and "--workers 6" in text
    assert "git pull --rebase --autostash" in text and "git push" in text
    assert '"Behaviour gap: e3_both_seed0 on true vs parsed scenes"' in text


@test
def test_gap_driver_end_to_end_on_cpu():
    """Real sketch scenes, the real perception model, spawn workers, then a run that skips it all."""
    import subprocess
    import gap as gapmod
    here = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        subprocess.run([sys.executable, "generate.py", "--n", "3", "--out", str(root / "data" / "test_sketch7"),
                        "--layout", "mixed5", "--agents", "1-8", "--obstacles", "0-6", "--style", "sketch7",
                        "--seed", "370"], cwd=here.parent / "sim", check=True, stdout=subprocess.DEVNULL)
        cmd = [sys.executable, "scripts/run_gap.py", "--data-root", str(root / "data"),
               "--work", str(root / "work"), "--sets", "test_sketch7", "--limit", "2", "--workers", "2",
               "--device", "cpu"]
        subprocess.run(cmd, cwd=here, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = root / "work" / "sketch7" / "smoke" / "e3_both_seed0"
        rows = gapmod.read_rows(out / "per_scene.csv")
        assert [r["scene"] for r in rows] == ["scene_00001", "scene_00002"]
        assert all(r["parse_differs"] == "0" for r in rows)
        for f in ("gap.md", "metrics.json", "fig_gap.png", "identity.json"):
            assert (out / f).exists(), f
        again = subprocess.run(cmd, cwd=here, check=True, capture_output=True, text=True).stdout
        assert "parsed, skipped" in again and "scored, skipped" in again
        assert "2 already done, 0 to run" in again


def main():
    only = sys.argv[1:]
    failed = 0
    for fn in TESTS:
        if only and not any(o in fn.__name__ for o in only):
            continue
        try:
            fn()
            print("  PASS  %s" % fn.__name__)
        except Exception:
            failed += 1
            print("  FAIL  %s" % fn.__name__)
            traceback.print_exc()
    ran = sum(1 for fn in TESTS if not only or any(o in fn.__name__ for o in only))
    print("\n%d/%d passed" % (ran - failed, ran))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
