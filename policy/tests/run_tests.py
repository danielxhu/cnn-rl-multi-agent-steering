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
