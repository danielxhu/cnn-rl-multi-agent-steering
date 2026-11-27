"""JAX ``net`` vs torch ``nets.ActorCritic`` on the same weights and inputs.

    <jax python>   jaxsim/net_check.py make  OUT.npz     # weights, inputs, JAX outputs
    <torch python> jaxsim/net_check.py check OUT.npz     # torch outputs, max difference

Two interpreters because the JAX and torch environments are separate.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY))
sys.path.insert(0, str(POLICY / "jaxsim"))

import numpy as np  # noqa: E402

H, D, N, L = 256, 50, 6, 30
TOL = 1e-4


def make(out):
    import jax
    import jax.numpy as jnp
    import net
    rng = np.random.default_rng(0)
    p = net.init(jax.random.PRNGKey(0), D, H)
    p = {k: v + 0.05 * rng.standard_normal(v.shape) for k, v in p.items()}   # non-zero biases
    obs = rng.uniform(-1, 1, (L, N, D)).astype(np.float32)
    first = rng.uniform(size=(L, N)) < 0.15
    first[0] = True
    st = net.initial_state(N, H)
    means, values = [], []
    for t in range(L):
        m, _, v, st = net.step(p, jnp.asarray(obs[t]), st, jnp.asarray(first[t]))
        means.append(np.asarray(m))
        values.append(np.asarray(v))
    w = net.to_torch_state(p)
    np.savez(out, obs=obs, first=first, mean=np.stack(means), value=np.stack(values),
             **{"w:" + k: v for k, v in w.items()})
    print("written", out)


def check(path):
    import _paths  # noqa: F401
    import torch
    from nets import ActorCritic
    from policyconfig import NetConfig
    z = np.load(path)
    model = ActorCritic(D, NetConfig(memory="gru", hidden=H))
    model.set_weights({k[2:]: z[k] for k in z.files if k.startswith("w:")})
    st = model.initial_state(N)
    dm = dv = 0.0
    with torch.no_grad():
        for t in range(L):
            dist, v, st = model.step(torch.from_numpy(z["obs"][t]), st, torch.from_numpy(z["first"][t]))
            dm = max(dm, float(np.abs(dist.mean.numpy() - z["mean"][t]).max()))
            dv = max(dv, float(np.abs(v.numpy() - z["value"][t]).max()))
    print(f"max |mean diff| {dm:.2e}, max |value diff| {dv:.2e}")
    sys.exit(0 if max(dm, dv) < TOL else 1)


if __name__ == "__main__":
    {"make": make, "check": check}[sys.argv[1]](sys.argv[2])
