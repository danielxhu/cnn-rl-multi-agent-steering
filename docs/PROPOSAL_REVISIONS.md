# Proposal revisions (to do)

Changes to `Detailed Proposal.pdf` collected from the reviews of 2026-10-17 – 10-20.
Daniel edits the proposal himself; tick items off here as they land.
Items marked *(after main line)* depend on the main-storyline decision and stay open until it is made.

## Agreed changes

- [ ] **E1 depends on E2.** E1's second half (same policy on parsed vs ground-truth scenes) needs the E2 policy. Split E1: segmentation/extraction metrics first, the parsed-vs-GT policy gap after E2.
- [ ] **Related work.** Add a related-work paragraph and the missing references:
  - Long, Fan, et al. 2018, "Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning", ICRA (arXiv:1709.10082) — sensor-level shared policy, PPO, many robots.
  - Fan, Long, et al. 2020, "Distributed multi-robot collision avoidance via deep reinforcement learning for navigation in complex scenarios", IJRR — journal follow-up.
  - Liang et al. 2020, "CrowdSteer", arXiv:2004.03089 — PPO, lidar + depth, robot in a crowd.
  - Hu, Kapadia, Pavlovic, Berseth, Faloutsos et al. 2021, IEEE TVCG — one shared, goal-conditioned policy for diverse crowd simulation.
  - Theodoropoulou et al. 2026, "From Real-World Images to Agent-Based Crowd Simulations: An End-to-End Pipeline", AAMAS demo (doi:10.65109/TLYB5574) — closest image→simulation system.
  - Le & Hu 2025, "Automated discovery of candidate simulation models for steering behavior simulation", *Simulation* 101(7):769–784 — journal version; the proposal cites only the 2020/2022 conference papers.
  - Optional: CrowdVLA (arXiv:2604.05525), ACDC "digital cousins" (arXiv:2410.07408), MPACT (CGF 2025).
  - For each, one sentence on which axis this project differs (input type, learned vs discovered, evaluation in behaviour terms).

## Text fixes

- [ ] "photos" → "images" (§6.1 "5,000 photos", §9 deliverable 2); describe inputs as hand drawings first, more input types later *(wording after main line)*.
- [ ] **E4 wording.** State the training set: leave-one-layout-out (train on 9 of the 10 layouts, test on the held-out one, rotate). Current text reads as "train on all, test on all".
- [ ] **Sensing description.** Either "16 rays, one per sector" or implement a true per-sector minimum; the text says sectors, `sim/physics.py` casts one ray per sector centre (agents beyond ~17 units can fall between rays).
- [ ] **Observation size.** 50 numbers (bearing only, no goal distance), per `policy/DESIGN.md` §1.3; proposal and `sim/physics.py` docstring say 51.
- [ ] **Policy section.** Own PPO (not Stable-Baselines3), two variants `mlp` / `gru`; update §3, §7 and the SB3 reference.
- [ ] **Layouts.** 10 families, 2–8 agents (E2 design); proposal says 7 layouts, 4 agents.
- [ ] **Compute numbers (§6).** 100 × 35 min ≈ 58 core-hours, not 350; §6.2 (32 cores < 2 h) and §6.5 (12 cores ≈ 5 h) disagree; the GPU estimate covers 256 px only, 512 px runs cost ~4×.
- [ ] **Reward note for E3.** State whether reward normalisation is on, since it partly cancels reward-scale changes.
- [ ] **Multiple start/goal groups.** Say how parsed agents are matched to their goal region.
- [ ] **Limitations.** Name the known failure modes up front: memoryless policy in cul-de-sacs, rays missing small agents at range, perception error vs agent spacing.

## After the main-line decision

- [ ] Rewrite the summary and §1 around the chosen main line.
- [ ] Add the missing "limitation 3" that motivates image input (scenarios are entered by hand; no path from a drawing to a simulation).
- [ ] Cut contributions to 3–4; the scene generator is infrastructure, not a contribution.
- [ ] Either add policy distillation to the deliverables or drop week 13.
- [ ] Re-scope E3/E5/E6 to fit the main line.

## Already resolved in code

- Agent–agent collision asymmetry and pass-through — fixed in `03e457a`.
