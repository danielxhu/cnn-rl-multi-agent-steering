"""Prioritized Level Replay over a fixed level bank (Jiang et al. 2021; DESIGN_FRONTIER.md §3).

Owns: ``LevelSampler`` (scores, staleness, the sampling distribution) and
``level_scores`` (score per level from one batch).

The bank is ``env.bank_scene(i)`` for i < ``EnvConfig.level_bank``: the same
levels as the JAX arm's bank (``jaxsim/bank.py``). Each training episode
draws its level from ``LevelSampler.probs()``:

- seen levels (replay), with probability ``rho`` (all of it once every level
  has been seen): P = (1 - s) P_score + s P_stale, P_score rank-based,
  h(rank) = rank^(-1 / beta), P_stale proportional to updates since the
  level was last scored;
- unseen levels, uniformly, otherwise.

Score: positive value loss, the mean over the level's alive steps in the
batch of max(GAE advantage, 0). Off by default (``EnvConfig.plr = "off"``):
then nothing here runs and scenes are sampled as before.
"""
from __future__ import annotations

import numpy as np


class LevelSampler:
    def __init__(self, n_levels: int, rho=0.5, beta=0.3, staleness=0.3):
        self.n, self.rho, self.beta, self.s = int(n_levels), rho, beta, staleness
        self.score = np.zeros(self.n)
        self.seen = np.zeros(self.n, bool)
        self.last = np.zeros(self.n, np.int64)
        self.clock = 0

    def probs(self) -> np.ndarray:
        seen = np.flatnonzero(self.seen)
        p = np.zeros(self.n)
        unseen = self.n - len(seen)
        if len(seen) == 0:
            return np.full(self.n, 1.0 / self.n)
        order = np.argsort(-self.score[seen], kind="stable")
        rank = np.empty(len(seen))
        rank[order] = np.arange(1, len(seen) + 1)
        h = rank ** (-1.0 / self.beta)
        ps = h / h.sum()
        st = (self.clock - self.last[seen]).astype(float)
        pc = st / st.sum() if st.sum() > 0 else np.full(len(seen), 1.0 / len(seen))
        replay = (1 - self.s) * ps + self.s * pc
        w = 1.0 if unseen == 0 else self.rho
        p[seen] = w * replay
        if unseen:
            p[~self.seen] = (1 - w) / unseen
        return p / p.sum()

    def update(self, scores: dict):
        """``scores``: level -> score from the latest batch."""
        self.clock += 1
        for lvl, sc in scores.items():
            self.score[lvl] = sc
            self.seen[lvl] = True
            self.last[lvl] = self.clock

    def state_dict(self) -> dict:
        return {"score": self.score.copy(), "seen": self.seen.copy(), "last": self.last.copy(),
                "clock": self.clock}

    def load_state_dict(self, d: dict):
        self.score, self.seen = np.array(d["score"]), np.array(d["seen"])
        self.last, self.clock = np.array(d["last"]), int(d["clock"])


def level_scores(level: np.ndarray, adv: np.ndarray, alive: np.ndarray) -> dict:
    """(T, N) level ids (-1 = none), advantages, alive -> {level: mean max(adv, 0)}."""
    m = alive & (level >= 0)
    if not m.any():
        return {}
    lv, a = level[m], np.maximum(adv[m], 0.0)
    ids, inv = np.unique(lv, return_inverse=True)
    tot = np.bincount(inv, weights=a)
    cnt = np.bincount(inv)
    return {int(i): float(t / c) for i, t, c in zip(ids, tot, cnt)}
