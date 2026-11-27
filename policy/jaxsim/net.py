"""The ``nets.ActorCritic`` (``gru``) in JAX, with torch's parameter layout.

Parameters are a flat dict keyed exactly like the torch ``state_dict``
(``actor_body.fc1.weight`` (H, D), ``actor_body.gru.weight_ih_l0`` (3H, H)
in torch's r, z, n gate order, ...), so ``to_torch_state`` is a plain copy
and a JAX-trained policy runs unchanged in ``evaluate.PolicyController``.

GRU (torch.nn.GRU):
    r = s(W_ir x + b_ir + W_hr h + b_hr)
    z = s(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    h' = (1 - z) * n + z * h
State is zeroed where ``first`` is set, before the step (as ``nets.Body.step``).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

BODIES = ("actor_body", "critic_body")


def _orth(key, shape, gain):
    """Orthogonal (rows or columns) like ``torch.nn.init.orthogonal_``."""
    rows, cols = shape
    a = jax.random.normal(key, (max(rows, cols), min(rows, cols)))
    q, r = jnp.linalg.qr(a)
    q = q * jnp.sign(jnp.diag(r))
    if rows < cols:
        q = q.T
    return gain * q[:rows, :cols]


def init(key, in_dim: int, hidden: int, log_std_init: float = -0.5) -> dict:
    H = hidden
    ks = iter(jax.random.split(key, 8))
    p = {"log_std": jnp.full((2,), log_std_init)}
    for b in BODIES:
        p[f"{b}.fc1.weight"] = _orth(next(ks), (H, in_dim), math.sqrt(2))
        p[f"{b}.fc1.bias"] = jnp.zeros(H)
        p[f"{b}.gru.weight_ih_l0"] = _orth(next(ks), (3 * H, H), 1.0)
        p[f"{b}.gru.weight_hh_l0"] = _orth(next(ks), (3 * H, H), 1.0)
        p[f"{b}.gru.bias_ih_l0"] = jnp.zeros(3 * H)
        p[f"{b}.gru.bias_hh_l0"] = jnp.zeros(3 * H)
    p["mean.weight"] = _orth(next(ks), (2, H), 0.01)
    p["mean.bias"] = jnp.zeros(2)
    p["value.weight"] = _orth(next(ks), (1, H), 1.0)
    p["value.bias"] = jnp.zeros(1)
    return p


def _body(p, b, x, h, first):
    """x (n, D), h (n, H), first (n,) -> features (n, H), new h."""
    z_in = jnp.tanh(x @ p[f"{b}.fc1.weight"].T + p[f"{b}.fc1.bias"])
    h = jnp.where(first[:, None], 0.0, h)
    gi = z_in @ p[f"{b}.gru.weight_ih_l0"].T + p[f"{b}.gru.bias_ih_l0"]
    gh = h @ p[f"{b}.gru.weight_hh_l0"].T + p[f"{b}.gru.bias_hh_l0"]
    H = h.shape[-1]
    r = jax.nn.sigmoid(gi[:, :H] + gh[:, :H])
    z = jax.nn.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
    n = jnp.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
    h = (1.0 - z) * n + z * h
    return h, h


def step(p, obs, state, first):
    """-> (mean (n, 2), log_std (2,), value (n,), new state (ha, hc))."""
    ha, hc = state
    fa, ha = _body(p, "actor_body", obs, ha, first)
    fc, hc = _body(p, "critic_body", obs, hc, first)
    mean = fa @ p["mean.weight"].T + p["mean.bias"]
    value = (fc @ p["value.weight"].T + p["value.bias"])[:, 0]
    return mean, p["log_std"], value, (ha, hc)


def initial_state(n: int, hidden: int):
    z = jnp.zeros((n, hidden))
    return z, z


def log_prob(mean, log_std, a):
    return (-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std
            - 0.5 * math.log(2 * math.pi)).sum(-1)


def entropy(log_std):
    return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum()


def actions_to_sim(a):
    """``env.actions_to_sim`` without noise: (turn in [-1, 1], speed in [0, 1])."""
    return jnp.stack([jnp.clip(a[..., 0], -1, 1), jnp.clip((a[..., 1] + 1) / 2, 0, 1)], -1)


def to_torch_state(p) -> dict:
    """numpy arrays keyed like ``nets.ActorCritic.state_dict()``."""
    return {k: np.asarray(v, np.float32) for k, v in p.items()}
