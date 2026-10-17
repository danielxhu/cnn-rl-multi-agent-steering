"""PPO for the shared multi-agent policy (DESIGN.md §3.2–§3.4).

Owns: ``RewardScaler``, ``gae``, ``replay`` (re-evaluate a batch), ``update``.

Invariants:
- Everything is masked by ``alive``: a slot that holds no acting agent
  contributes nothing to advantages, statistics or losses.
- Rewards are scaled by the running std of the discounted return; value
  targets live in scaled units, reported returns never do.
- Sequences of length ``L`` are cut along time per (row); every one starts from
  the recurrent state stored at collection time. ``mlp`` / ``stack`` ignore it.
- ``replay`` of an unchanged model reproduces the stored ``logp`` / ``value``
  (test 7); ``update`` relies on the same function.
"""
from __future__ import annotations

import _paths  # noqa: F401

import numpy as np
import torch

from nets import entropy, log_prob


class RewardScaler:
    """Running std of the per-row discounted return (CleanRL ``NormalizeReward``)."""

    def __init__(self, gamma=0.99, eps=1e-8):
        self.gamma, self.eps = gamma, eps
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4
        self.ret = None                                 # (N,) carry-over per row

    def update(self, rewards, dones, alive):
        """rewards, dones, alive: (T, N), in time order. Updates the statistics."""
        T, N = rewards.shape
        if self.ret is None or len(self.ret) != N:
            self.ret = np.zeros(N)
        vals = []
        for t in range(T):
            self.ret = self.ret * self.gamma + rewards[t]
            vals.append(self.ret[alive[t]].copy())
            self.ret = np.where(dones[t], 0.0, self.ret)
        x = np.concatenate(vals)
        if len(x) == 0:
            return
        bm, bv, bc = x.mean(), x.var(), len(x)
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    @property
    def std(self):
        return float(np.sqrt(self.var + self.eps))

    def scale(self, rewards):
        return rewards / self.std

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count,
                "ret": None if self.ret is None else self.ret.copy(), "gamma": self.gamma}

    def load_state_dict(self, d):
        self.mean, self.var, self.count, self.gamma = d["mean"], d["var"], d["count"], d["gamma"]
        self.ret = None if d["ret"] is None else np.array(d["ret"])


def gae(batch, gamma, lam, reward=None):
    """Advantages and returns, (T, N) each, as in §3.3. ``reward`` overrides the raw one."""
    r = batch["reward"] if reward is None else reward
    V, alive = batch["value"], batch["alive"]
    done = batch["terminal"] | batch["truncated"]
    B = batch["bootstrap"] * batch["truncated"]          # in the scale of ``r``
    T, N = V.shape
    A = np.zeros((T, N), np.float32)
    for t in range(T - 1, -1, -1):
        nv = batch["last_value"] if t == T - 1 else V[t + 1]
        nA = 0.0 if t == T - 1 else A[t + 1]
        notdone = ~done[t]
        delta = r[t] + gamma * nv * notdone + gamma * B[t] - V[t]
        a = delta + gamma * lam * notdone * nA
        A[t] = np.where(alive[t], a, 0.0)
    return A, A + V


def _sequences(batch, L):
    """Index helpers: chunk c covers time [c*L, (c+1)*L)."""
    T, N = batch["alive"].shape
    return T // L, N


def replay(model, batch, L):
    """Re-evaluate the batch with ``sequence``: (logp, value, entropy), each (T, N) numpy."""
    C, N = _sequences(batch, L)
    dev = next(model.parameters()).device
    obs = torch.as_tensor(batch["obs"], device=dev)
    first = torch.as_tensor(batch["first"], device=dev)
    act = torch.as_tensor(batch["action"], device=dev)
    logp, val, ent = [], [], []
    with torch.no_grad():
        for c in range(C):
            sl = slice(c * L, (c + 1) * L)
            h0 = (torch.as_tensor(batch["h_actor"][c], device=dev).unsqueeze(0),
                  torch.as_tensor(batch["h_critic"][c], device=dev).unsqueeze(0))
            dist, v = model.sequence(obs[sl], h0, first[sl])
            logp.append(log_prob(dist, act[sl]))
            ent.append(entropy(dist))
            val.append(v)
    return tuple(torch.cat(x).cpu().numpy() for x in (logp, val, ent))


def update(model, optimiser, batch, adv, ret, cfg, device, rng):
    """One PPO update (``cfg.epochs`` epochs over the sequences); returns loss statistics."""
    L = cfg.L
    C, N = _sequences(batch, L)
    t = lambda x: torch.as_tensor(x, device=device)
    obs, act, first = t(batch["obs"]), t(batch["action"]), t(batch["first"])
    old_logp, alive = t(batch["logp"]), t(batch["alive"])
    adv_t, ret_t = t(adv), t(ret)
    h_a, h_c = t(batch["h_actor"]), t(batch["h_critic"])
    old_v = t(batch["value"])

    # sequences = (chunk, row) pairs holding at least one alive step
    has = batch["alive"].reshape(C, L, N).any(1)
    seq = np.argwhere(has)                                   # (M, 2): chunk, row
    M = len(seq)
    stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")}
    if M == 0:
        return {k: 0.0 for k in stats} | {"explained_variance": 0.0}
    steps = torch.arange(L, device=device)

    for _ in range(cfg.epochs):
        perm = rng.permutation(M)
        for part in np.array_split(perm, min(cfg.n_minibatches, M)):   # no tiny tail batch
            idx = seq[part]
            ch = torch.as_tensor(idx[:, 0], device=device)
            rw = torch.as_tensor(idx[:, 1], device=device)
            tt = (ch * L)[None, :] + steps[:, None]           # (L, B) time index
            o, a_, f = obs[tt, rw[None]], act[tt, rw[None]], first[tt, rw[None]]
            m = alive[tt, rw[None]]
            if not m.any():
                continue
            h0 = (h_a[ch, rw].unsqueeze(0), h_c[ch, rw].unsqueeze(0))
            dist, v = model.sequence(o, h0, f)
            logp = log_prob(dist, a_)
            ratio = (logp - old_logp[tt, rw[None]]).exp()
            A = adv_t[tt, rw[None]]
            mean = A[m].mean()
            std = A[m].std() + 1e-8 if int(m.sum()) > 1 else 1.0     # std of one value is NaN
            A = (A - mean) / std
            pg = torch.max(-A * ratio, -A * ratio.clamp(1 - cfg.clip, 1 + cfg.clip))
            vl = 0.5 * (v - ret_t[tt, rw[None]]) ** 2
            ent = entropy(dist)
            loss = (pg[m].mean() + cfg.vf_coef * vl[m].mean() - cfg.ent_coef * ent[m].mean())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimiser.step()
            with torch.no_grad():
                lr_ = logp - old_logp[tt, rw[None]]
                stats["policy_loss"].append(pg[m].mean().item())
                stats["value_loss"].append(vl[m].mean().item())
                stats["entropy"].append(ent[m].mean().item())
                stats["approx_kl"].append(((ratio - 1) - lr_)[m].mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float()[m].mean().item())

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
    m_all = batch["alive"]
    var = ret[m_all].var()
    out["explained_variance"] = float(1 - np.var(ret[m_all] - batch["value"][m_all]) / var) if var > 1e-12 else 0.0
    return out
