"""Actor-critic networks for the three memory variants (DESIGN.md §2).

Owns: ``ActorCritic`` with one interface for ``mlp``, ``stack`` and ``gru``,
and numpy weight transfer for the rollout workers.

Invariants:
- Actor and critic are separate networks of the same shape; nothing is shared.
- ``sequence`` is ``step`` applied L times with the same resets: both go
  through the same ``nn.GRU`` calls (length-1 sequences), so they agree to
  float precision (test 6). This is what keeps recurrent PPO honest.
- Recurrent state has shape ``(1, n, hidden)`` for ``gru`` and ``(1, n, 0)``
  otherwise, so the learner has a single code path. State is zeroed where
  ``first`` is set, *before* the step that consumes the new observation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _orth(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Body(nn.Module):
    """Linear-tanh stem followed by a second tanh layer (mlp, stack) or a GRU."""

    def __init__(self, in_dim: int, hidden: int, recurrent: bool):
        super().__init__()
        self.hidden, self.recurrent = hidden, recurrent
        self.fc1 = _orth(nn.Linear(in_dim, hidden), math.sqrt(2))
        if recurrent:
            self.gru = nn.GRU(hidden, hidden)
            for name, p in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(p)
                else:
                    nn.init.orthogonal_(p, 1.0)
        else:
            self.fc2 = _orth(nn.Linear(hidden, hidden), math.sqrt(2))

    def step(self, x, h, first):
        """x (n, D), h (1, n, H or 0), first (n,) bool -> features (n, H), new h."""
        z = torch.tanh(self.fc1(x))
        if not self.recurrent:
            return torch.tanh(self.fc2(z)), h
        h = h * (~first).to(h.dtype).view(1, -1, 1)
        out, h = self.gru(z.unsqueeze(0), h)
        return out[0], h

    def sequence(self, x, h, first):
        """x (L, n, D), first (L, n) -> features (L, n, H)."""
        z = torch.tanh(self.fc1(x))
        if not self.recurrent:
            return torch.tanh(self.fc2(z)), h
        outs = []
        for t in range(x.shape[0]):
            h = h * (~first[t]).to(h.dtype).view(1, -1, 1)
            o, h = self.gru(z[t].unsqueeze(0), h)
            outs.append(o[0])
        return torch.stack(outs), h


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, net_cfg):
        super().__init__()
        self.cfg = net_cfg
        H = net_cfg.hidden
        rec = net_cfg.memory == "gru"
        self.H = H if rec else 0
        self.actor_body = Body(in_dim, H, rec)
        self.critic_body = Body(in_dim, H, rec)
        self.mean = _orth(nn.Linear(H, 2), 0.01)
        self.value = _orth(nn.Linear(H, 1), 1.0)
        self.log_std = nn.Parameter(torch.full((2,), float(net_cfg.log_std_init)))

    def initial_state(self, n: int, device=None):
        z = torch.zeros(1, n, self.H, device=device)
        return z, z.clone()

    def _dist(self, feat):
        mean = self.mean(feat)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def step(self, obs, state, first):
        ha, hc = state
        fa, ha = self.actor_body.step(obs, ha, first)
        fc, hc = self.critic_body.step(obs, hc, first)
        return self._dist(fa), self.value(fc).squeeze(-1), (ha, hc)

    def sequence(self, obs, state0, first):
        ha, hc = state0
        fa, _ = self.actor_body.sequence(obs, ha, first)
        fc, _ = self.critic_body.sequence(obs, hc, first)
        return self._dist(fa), self.value(fc).squeeze(-1)

    @torch.no_grad()
    def value_only(self, obs, state, first):
        """V(obs) without advancing the caller's state (bootstrap values)."""
        _, hc = state
        fc, _ = self.critic_body.step(obs, hc, first)
        return self.value(fc).squeeze(-1)

    # -- weight transfer to the CPU workers ---------------------------------
    def get_weights(self) -> dict:
        return {k: v.detach().cpu().numpy().copy() for k, v in self.state_dict().items()}

    def set_weights(self, w: dict):
        self.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})


def log_prob(dist, action):
    return dist.log_prob(action).sum(-1)


def entropy(dist):
    return dist.entropy().sum(-1)
