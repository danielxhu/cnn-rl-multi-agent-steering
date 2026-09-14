# Pilot: Image → Red Path via a Perception–Policy Pipeline

Controlled pilot study for the proposal *"From Image to Path: A Perception–Policy
Pipeline for Discovering Steering Behavior Without Hand-Crafted Models"*
(Daniel Hu, July 2026).

Everything runs on a laptop CPU; no GPU or manual labels required.

## Layout

| File | Purpose |
|---|---|
| `sim.py` | 2D steering simulator, random scene generator, statement-style renderer (RGB + pixel-perfect label maps) |
| `perception.py` | Small U-Net pixel classifier + OpenCV geometry extraction (agent pose, obstacles, goal, walls) |
| `rl_env.py` | Gymnasium environment; 19-dim ego-centric observation, (turn, speed) action |
| `train_rl.py` | PPO training (Stable-Baselines3), randomized open scenes, 3 seeds |
| `evaluate.py` | Policy vs. go-straight baseline across single / multi / hallway (zero-shot) layouts |
| `end_to_end.py` | Full image → parse → simulate → red-path chain, figures, and pipeline-vs-oracle metrics |
| `results/` | Trained weights, metrics JSONs, and figures |

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install torch stable-baselines3 gymnasium opencv-python pillow
.venv/bin/python perception.py        # ~5 min  -> results/unet.pt, perception_metrics.json
.venv/bin/python train_rl.py 0 1 2    # ~15 min -> results/ppo_seed{0,1,2}.zip
.venv/bin/python evaluate.py          # ~2 min  -> results/policy_metrics.json
.venv/bin/python end_to_end.py        # ~3 min  -> figures + end_to_end_metrics.json
```

## Design notes

- **No manual labels.** The renderer emits (image, label-map) pairs; the U-Net
  trains on those alone, with light brightness/noise augmentation.
- **Hallways are held out.** The policy trains only on open scenes with 1–3
  obstacles; corridor layouts are used exclusively for zero-shot evaluation.
- **Honest end-to-end accounting.** In `end_to_end.py` the policy acts inside
  the *parsed* scene (perception errors propagate), and the trajectory is then
  validated against the *ground-truth* scene. The oracle condition (policy on
  ground-truth geometry) isolates the cost of perception errors.
- **Known pilot limitation.** At 128×128 the agent body spans a few pixels, so
  heading is estimated from the FOV-arc centroid (the arc is symmetric about
  the heading); see `perception.extract_scene`.
