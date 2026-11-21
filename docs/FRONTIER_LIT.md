# Frontier literature we can build on (21 Nov 2026)

Companion to [`RELATED_WORK.md`](RELATED_WORK.md), which covers novelty. This
file covers what we can **use**: for each open problem, the prior work to
build on, what exactly to change in our code or experiments, the cost, and a
priority.

**How this was searched.** Four parallel searches over 2021–2026, one each
for coordination, generalisation and memory, perception, and E6 with
evaluation. Titles, authors, years and venues were confirmed on the official
or arXiv pages; full texts were mostly not read, so method details come from
abstracts. Entries marked *(preprint)* have no confirmed peer-reviewed venue.

## Priority list

| # | What to do | Builds on | Problem it targets | Cost | Priority |
|---|---|---|---|---|---|
| 1 | Report every comparison with IQM, 95 % stratified-bootstrap CIs and probability of improvement (`rliable`) | Agarwal et al., NeurIPS 2021 | Seed variance; which claims hold at 3–5 seeds | CPU minutes, no retraining | **Do first** |
| 2 | Zero-shot test on new layout families and agent counts; add deadlock rate, stop time, flow rate, path-length ratio | SteerBench (Singh et al. 2009); Kapadia et al. 2011 (scenario-space coverage); Chandra et al. 2023 (deadlock metrics); An et al. 2024 (eval protocol 2–24 entities) | Generalisation (5), crowd size (7), blocked exits (6) | Generator work + CPU evaluation | **High** (already planned) |
| 3 | Neighbour-set observation: K nearest visible agents (relative position and velocity), masked, pooled by attention or DeepSets; train with 2–12 agents | Kwiatkowski et al., MIG 2022; An et al. 2024 *(preprint)*; DMCA *(preprint)*; HEIGHT, T-ASE | Head-on jams (1), crowd size (7) | 1–2 days code + 3–5 × 50 M runs | **High** |
| 4 | Centralised critic (MAPPO): critic sees all positions and goals, actor unchanged | Yu et al., NeurIPS 2022 | Jams (1), seed variance (4) | ~1 day + runs; can share runs with #3 as a 2 × 2 | High |
| 5 | Training-only episodic novelty bonus on the E7 raster (count or NovelD form) | NovelD, NeurIPS 2021; E3B *(preprint)* | Residual loops (3), cul_de_sac 21 % | ~1 day + 3-run β sweep; deployed observation unchanged | Medium-high |
| 6 | Prioritised level replay, then ACCEL-style mutations (move a wall, narrow a gap, close an exit), filtering unsolvable levels | PLR *(preprint)*; Robust PLR, NeurIPS 2021; ACCEL *(preprint)*; Beukman et al., ICML 2024 | Generalisation (5), loops (3), blocked exits (6) | Sampler change, same compute per run | Medium (after #2 shows the gap) |
| 7 | Stronger, tuned classical baselines: TTC power law and social force next to ORCA; tune all with CMA-ES on the training layouts | Karamouzas et al., PRL 2014; SteerFit (Berseth et al., SCA 2014); Wolinski et al., CGF 2014 | Fair comparison; blocks the "weak baselines" objection | ~1–2 days, CPU | Medium-high (before any paper) |
| 8 | E6: treat the trained policy as the oracle; compare three imitators — Le & Hu GA models, VIPER-style trees (depth 3–12), DSP anchoring — on success and fidelity; report the interpretability–performance curve | Le & Hu, *Simulation* 2025; VIPER, NeurIPS 2018; DSP, ICML 2021 | E6 | Medium, CPU; needs the advisor's code | High (strongest novelty) |
| 9 | Perception → behaviour metric: same policy on true vs parsed scenes, success gap next to IoU | PKL (Philion et al., CVPR 2020); Ivanovic & Pavone, IV 2022 | E1 / behaviour gap | Low; the gap tooling exists | Medium-high |
| 10 | Zero-shot perception baselines on the same photos: SAM / Grounded SAM, and a VLM writing scene JSON | SAM, ICCV 2023; Grounded SAM *(tech report)*; DeFazio et al. *(preprint)*; Sketch2BIM, Autom. Constr. 2026 | Positions the 1.09 M-param synthetic-trained U-Net | 2–3 days | Medium |
| 11 | Real floor plans: CubiCasa5K, then the Raster-to-Graph set; score with room/corner F1 | CubiCasa5K, SCIA 2019; Raster-to-Graph, CGF 2024; RoomFormer, CVPR 2023 | Planned floor-plan input | 1–2 weeks | Later |
| 12 | Satellite input: HEAT on SpaceNet tiles → obstacle polygons | HEAT, CVPR 2022; SpaceNet 2018 | Planned satellite input | Days | Later / optional |

## Notes by area

### Coordination and crowd size (problems 1, 7)

- **Why the rays cannot fix head-on jams.** Our rays report only the nearest
  hit distance per sector, so an agent cannot see whether another agent is
  coming toward it. Kwiatkowski et al. (2022) found that direct neighbour
  information generally beats raycasting.
- **What to change.** Add a short neighbour list next to the rays (#3). It
  keeps the agents decentralised, needs no global position, and with a
  permutation-invariant pool the same network runs at any agent count.
- **If jams persist.** Two cheap options:
  - an energy reward term that penalises acceleration, so agents slow early
    (Kwiatkowski et al., MIG 2023);
  - a rule fallback: if an agent makes no progress for N steps, switch it to
    a local yield rule (Wang et al. 2025 *(preprint)*, the same style as the
    E7 exit rule).
- **Curriculum over agent count.** Only if #3 still degrades beyond 12
  agents: increase the agent count during training (Zhao et al., ICML 2025).

### Memory, loops, generalisation (problems 3, 4, 5, 6)

- **Supports the E7 design.**
  - Memory Gym (JMLR): transformers beat GRUs only with an extra
    reconstruction loss, and GRUs win on long tasks.
  - Ni et al. (ICML 2022): a tuned recurrent model-free agent is a strong
    baseline.
  - Together these back our choice of GRU plus an explicit map over a
    transformer, and match our finding that a longer BPTT window adds nothing.
- **Remaining loops.** The bonus in #5 uses the E7 raster as the visit count,
  so no extra network is needed. It runs in training only, so the policy
  observation is unchanged at evaluation.
- **Generalisation.** First measure the gap on new families (#2). Only if the
  gap is large, change how training levels are sampled (#6).

### Perception

- The closest work on hand-drawn plans (Sketch2BIM, 2026) needs large
  language models and a human in the loop. We use a small model and no hand
  labels. That contrast is worth stating in the paper.
- Planner-centric metrics (PKL, Ivanovic & Pavone) give a formal frame to
  the behaviour-gap experiment that already exists in `policy/DESIGN_GAP.md`.

### E6 and evaluation

- **E6 design.**
  - Le & Hu (2025) already frame model discovery as "find symbolic models
    that reproduce an oracle's behaviour". Our trained policy can be that
    oracle.
  - VIPER needs Q-values; our PPO has only V. Weight states by advantage
    instead.
  - The GRU's hidden state cannot go into a tree. Distil the `mlp` policy, or
    give the tree the E7 direction as an input.
- **Le & Hu after 2025.** No newer work and no citing papers turned up. Ask
  the advisor directly.

## Verification gaps

- Not opened, so excluded: PAIRED, GTrXL, state-space-model RL, SAM 2 and
  Grounding DINO primary pages, GREIL-Crowds.
- Title not confirmed: arXiv 2308.12779 (mAP vs driving performance).
- Partly checked: the perceptually validated crowd metric (arXiv
  2108.12346).
- Seed counts: E7 has 3 seeds, not 5. `rliable` intervals at n = 3 will be
  wide. Say so rather than hide it.
