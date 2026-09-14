# Learning Multi-Agent Steering Behavior from Images
### Detailed project proposal and computing resource request

**Daniel Hu** · Department of Computer Science · September 2026

---

## 1. Summary

This project asks whether an algorithm, rather than a modeler, can discover how
agents should move — and whether it can do so starting from a picture. A scene
image is parsed by a convolutional network into a structured description, and a
reinforcement-learning policy drives every agent in that scene from a local,
ego-centric observation. The output is the path each agent walks to its goal.

The work is a learned counterpart to the automated model discovery framework of
Le and Hu (2020; 2022), which searches a symbolic space of sensing–filtering–acting
behaviors with a genetic algorithm. Running both routes on the same scenarios
and the same metrics turns a methodological preference into a measurable
comparison.

The scene generator is built and validated (Section 4). The remaining work is
four experiment families (Section 5). **The requested allocation is 40 GPU-hours,
350 CPU core-hours and 50 GB of storage** (Section 6); a single clearly bounded
optional experiment would add 150 GPU-hours.

---

## 2. Objective and scope

The research statement sets a simple objective with a demanding generality
requirement: take a scene image — hollow-circle agents with heading lines, black
circular obstacles, a goal — and use any AI approach to produce the path to the
goal, in a way that is not tied to one layout.

Four scope decisions were settled with the advisor:

| Decision | Value |
|---|---|
| Agents per scene | **Multiple.** Agents avoid obstacles, walls *and each other* |
| Sensing geometry | **From the research statement**: 220° field of view, range 22 in a 100×100 world |
| First scenario | **Obstacle avoidance**, and explicitly *not* restricted to hallway layouts |
| Metrics | **Both families** — success/collision/timeout rates, and the collision-violation, finish-time and remaining-agent counters used in the existing framework |

Leader-following and moving obstacles are out of scope for this semester and are
listed as extensions in Section 8.

---

## 3. Approach

A two-stage pipeline. Both stages train from the simulator alone: the renderer
emits pixel-perfect labels, and the policy learns from reward. No hand-labeled
data and no hand-written steering rules enter the system.

| Stage | Component | Input → Output |
|---|---|---|
| 1 | Scene image | 512×512 PNG |
| 2 | U-Net pixel classifier | image → 6-class label map |
| 3 | Geometry extraction | label map → agent poses, obstacles, walls, regions |
| 4 | Scene instantiation | structured description → simulator state |
| 5 | Ego-centric observation | simulator state → 51 numbers per agent per step |
| 6 | Shared policy (PPO) | observation → (turn, speed) |
| 7 | Synchronous simulation | actions → trajectories |
| 8 | Path rendering | trajectories → paths drawn over the input |

The observation is what carries generality, and what makes the comparison
possible. Each agent sees 16 field-of-view sectors reported separately for three
entity categories — obstacles, walls, other agents — plus goal distance, goal
bearing and its own speed. Distances are clipped and normalised, so the vector
never encodes absolute position: a corridor simply appears as near readings in
some sectors, the same input format as any other layout.

Splitting sectors by entity category is deliberate. The existing framework's
discovered models filter entities by type (*"steer to the farthest space
entity"*, *"move along the wall"*, *"slow down if there is an agent in front"*).
Giving the learned policy the same categorical distinction means both methods
read the same information, and a difference in behavior is a difference in
discovery method rather than in what the two could perceive.

All agents share one policy, matching the homogeneous-agent setting of the prior
work. Agents step synchronously: every action is computed from the same world
state, then applied together.

---

## 4. Infrastructure already in place

The scene generator and simulator are complete and validated. This is what makes
the compute estimates in Section 6 measurements rather than guesses.

**Scene generation.** Seven layout families — open, corridor, room, doorway,
two-doorway, cul-de-sac, and a plus-shaped crossing. Each scene is written as an
RGB image, a label map whose pixel values are class indices, and a JSON ground
truth with exact agent poses, obstacle geometry, wall segments and regions.

**Validation.** Every scene is checked in configuration space before it is
written: obstacles are inflated by the agent radius via a Euclidean distance
transform, so "a disk of radius R fits here" is exact rather than approximated
on a grid. The tightest passage from start to goal is measured and recorded with
the scene, giving every sample a difficulty label. Obstacles are placed one at a
time with the check re-run after each, so a scene can never be sealed shut.

**Tests.** Thirteen unit tests cover the failure modes that are silent rather
than loud, including the case a naive grid check gets wrong (a 1.5-radius gap
that looks connected but no agent can pass) and the world-y-up / image-y-down
flip.

**Measured throughput** (single CPU core, laptop):

| Operation | Cost |
|---|---|
| Scene sample + validate + render + write | 50 ms (≈20 scenes/s) |
| Simulation step | 0.07 ms per agent-step (≈14,000 agent-steps/s) |
| Render 512×512 with labels | 8.1 ms |
| Disk footprint | 15.4 KB per scene |

Simulation cost is near-linear in agent count — 0.071 ms/agent-step at one
agent, 0.078 at sixteen — so multi-agent scaling is bounded by the number of
transitions, not by a blow-up in the environment.

**Scenario difficulty is characterised, not assumed.** A greedy controller that
turns toward the goal at full speed, over 20 scenes per layout with 4 agents and
3 obstacles:

| | open | room | two-doorway | doorway | corridor | cul-de-sac | crossing |
|---|---|---|---|---|---|---|---|
| Greedy success | 56% | 34% | 12% | 6% | 5% | 5% | 4% |

The spread is the point: the suite spans a wide difficulty range, and the null
model is far from solving any of it. Cul-de-sac and crossing are the two
families deliberately built to break a locally-sensing policy — the first
because a sealed pocket and a real exit look identical from outside, the second
because avoidance must be negotiated with other agents rather than read off
static geometry.

---

## 5. Experiment plan

### E1 — Layout generality

Train on a subset of layout families, evaluate on all seven, including families
held out entirely. Report both metric families per layout, stratified by the
recorded bottleneck width so that results separate "narrow passage" difficulty
from "wrong behavior".

*Conditions:* 3 training mixtures × 5 seeds = **15 runs**.

### E2 — Reward design as residual bias

Learning removes hand-written rules but the reward is still chosen by the
designer. Vary progress-shaping weight, collision penalty, step cost,
agent-collision penalty, and a clearance-bonus variant; measure how the
*behavior* changes — berth kept from obstacles, willingness to pass between two
obstacles, path length against safety. This is the learned-policy analogue of
fitness-function sensitivity in genetic search, and it is treated as a research
question rather than a footnote.

*Conditions:* 6 reward variants × 5 seeds = **30 runs**.

### E3 — Multi-agent scaling

Agent count is the axis the prior framework's scenarios turn on. Train and
evaluate at 2, 4, 8 and 16 agents; report collision violations and finish time
alongside success rate, and test whether a policy trained at one density
transfers to another.

*Conditions:* 4 agent counts × 5 seeds = **20 runs**.

### E4 — Comparison with discovered symbolic models

The motivating comparison. Align scenarios, metrics and evaluation protocol with
the existing framework's obstacle-avoidance models, then compare them against
learned policies on the same suite. Beyond aggregate numbers, compare the paths:
do the two discovery routes converge on similar solutions, or explore different
parts of behavior space?

A secondary analysis distils the learned policy into readable rules — a shallow
decision tree over the same sector-and-category observation — and reports both
the rules and how much performance the distillation costs. A policy that turns
out to be equivalent to *"steer toward the farthest open sector"* would be a
result worth reporting: two very different search procedures arriving at the
same behavior.

*Conditions:* 2 protocol conditions × 5 seeds = **10 runs**, plus evaluation.

### E5 — Perception robustness

Train the pixel classifier at 128, 256 and 512; sweep augmentation strength
(colour jitter, line-width jitter, background variation, noise, blur); and
measure the downstream cost of perception error by running the same policy on
parsed geometry and on ground-truth geometry. The gap between the two isolates
what perception costs, and says whether effort belongs in perception or in the
policy.

Multi-agent scenes make this stage substantially harder than single-agent ones:
recovering N poses is instance separation, not semantic segmentation, and agent
outlines merge when agents are close. The generator exposes spawn spacing as a
parameter, so detection error can be reported *as a function of nearest-neighbour
distance* — a curve that says precisely where the method breaks rather than one
aggregate number.

*Conditions:* ≈**45 training runs** across resolution, augmentation and
architecture variants.

---

## 6. Computing resources requested

### 6.1 Perception training — the GPU case

The classifier is a 4-level U-Net, 1.09M parameters, 6 output classes. One
training step at 256×256, batch 8, measured on 8 CPU threads: **1.32 s**. A run
of 60 epochs over 5,000 images is therefore **13.7 CPU-hours**.

A mid-range datacenter GPU (T4, A10 or V100 class) runs a model of this shape
roughly 20–40× faster than the CPU measured above, putting one run at **20–40
minutes**. Across the ≈45 runs of E5 that is **≈23 GPU-hours**; requesting **40
GPU-hours** covers failed jobs and re-runs.

No high-end accelerator is needed. A 1M-parameter segmentation model at 256×256
does not benefit from an A100; a T4 or A10 is the right size, and being explicit
about that should make the allocation easier to place.

### 6.2 Policy training — the CPU case

Reinforcement learning here is CPU-bound: the policy is a small MLP, and the
cost is environment stepping. Measured environment throughput is 14,000
agent-steps/s per core; with policy inference and updates, a realistic
end-to-end rate is 4,000–6,000 agent-steps/s.

At a budget of 10M agent-steps per run, one run takes **≈35 minutes on one
core**. Across E1–E4 that is 75 runs, plus roughly 25 re-runs and
hyperparameter sanity checks, so **100 runs ≈ 60 core-hours**. Sixteen-agent
configurations roughly double the per-run cost, bringing the realistic total to
**≈100 core-hours**; requesting **300 CPU core-hours** leaves working room.

What the cluster provides here is **parallelism, not scale**: these are ~100
independent 35-minute single-core jobs. On 32 cores a full sweep completes in
under two hours of wall-clock time, which is the difference between running a
reward ablation once and iterating on it.

### 6.3 Data generation and evaluation

Dataset generation runs at 20 scenes/s on one core; the datasets needed across
the semester total well under **5 core-hours**. Evaluation rollouts — 100
episodes × 7 layouts × 5 seeds per condition — cost about 10 minutes per
condition, or **≈20 core-hours** including analysis.

### 6.4 Storage

A scene occupies 15.4 KB on disk including image, label map and ground truth, so
even 100,000 scenes is under 2 GB. Model checkpoints, training logs and figures
add little. **50 GB is generous** and is requested only to avoid returning for
more.

### 6.5 Totals

| Resource | Requested | Derived from |
|---|---|---|
| GPU-hours | **40** | 45 perception runs × 20–40 min (§6.1) |
| GPU class | T4 / A10 / V100 | 1.09M-parameter U-Net at 256×256 |
| CPU core-hours | **350** | 100 policy runs × ~35 min, plus generation and evaluation (§6.2–6.3) |
| Concurrent cores | 32 preferred | many independent short jobs; not a parallel-job requirement |
| Storage | **50 GB** | 15.4 KB/scene plus checkpoints and logs (§6.4) |
| Wall-clock window | Sept 2026 – Jan 2027 | one semester, ~5 hours/week |

### 6.6 Optional extension — end-to-end pixel baseline

**+150 GPU-hours, 3 runs, clearly bounded.**

This project argues for an explicit perception stage rather than learning a
policy straight from pixels. That argument is currently an assertion. Training a
convolutional policy directly on 128×128 images would turn it into a measured
comparison, and it is the one experiment here that genuinely needs sustained GPU
time: rendering is cheap (0.13 ms/frame, and the static portion of a scene can
be cached per episode), but a convolutional policy over 10M steps is roughly two
orders of magnitude more expensive per step than the MLP.

Estimated 30–50 GPU-hours per seed × 3 seeds. This is listed separately because
the project's core contributions do not depend on it: if the allocation is
tight, it is the line to cut.

---

## 7. Software environment

Python 3.11+; NumPy, OpenCV, Pillow for the simulator; PyTorch for the
classifier; Stable-Baselines3 for PPO. The simulator itself has no dependency
beyond NumPy, OpenCV and Pillow, and runs on CPU. No licensed software, no
container registry access and no external data transfer are required — every
dataset is generated locally from the simulator.

---

## 8. Milestones

| Weeks | Work | Output |
|---|---|---|
| 1 | Scene generator, validation, tests, documentation | **complete** |
| 2–3 | Perception training pipeline; E5 resolution and augmentation sweeps | perception metrics, error-vs-separation curve |
| 4–5 | Multi-agent RL environment; first policies on open layouts | training curves, E1 baseline |
| 6–7 | E1 layout generality across all seven families | per-layout results, both metric families |
| 8–9 | E2 reward ablation and behavior analysis | behavior-vs-reward table |
| 10 | E3 multi-agent scaling | results at 2/4/8/16 agents |
| 11–12 | E4 comparison protocol and runs | aligned comparison against discovered models |
| 13 | Policy distillation into readable rules | rule set plus distillation cost |
| 14 | Consolidation and write-up | report and figures |

Extensions beyond this semester: leader-following, moving obstacles, and
perception on hand-drawn or photographed inputs.

---

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| **Instance separation in multi-agent perception** | Overlapping agent outlines merge; N poses cannot be recovered | The main technical risk. Spawn spacing is a controlled parameter, so the failure is measured as a curve rather than discovered late; circle detection with a locked radius replaces connected components |
| **Comparison protocol cannot be aligned** | E4 does not produce a like-for-like result | Confirm canonical scenarios and fitness definitions with the advisor early (Section 10); the simulator already reports both metric families, so alignment is a matter of protocol, not code |
| **Access to the existing framework's code** | Comparison must run against a re-implementation | Section 10; if re-implementation is needed, dynamics alignment is documented explicitly and the limitation stated |
| **Reward shaping consumes time** | E2 crowds out E4 | E4 has priority; E2 is capped at six variants and its runs are short and parallel |
| **Scope** | 5 hours/week is a real constraint | E4 is the semester's must-finish item; leader-following is explicitly out of scope, not a stretch goal that quietly absorbs weeks |

---

## 10. Items to confirm with the advisor

1. **Access to the existing framework's source.** Can the discovered models be
   run in their original simulator, or should the comparison use a
   re-implementation with documented dynamics alignment? This determines the
   shape of E4 and is the item most worth settling early.
2. **Canonical scenarios and fitness definitions** for the obstacle-avoidance
   comparison, so both methods are scored the same way.
3. **Deliverable form.** Is the trained pipeline plus quantitative comparison
   sufficient, or should the discovered behavior also be reported as readable
   rules? This determines how much of E4's distillation analysis is core rather
   than optional.
4. **Input scope.** Synthetic rendering only, or eventually hand-drawn and
   photographed scenes? This sets the weight of E5.

---

## 11. Deliverables

1. A validated multi-agent scene generator and simulator with documentation and
   tests — **complete**.
2. A perception module recovering scene structure from images with no manual
   annotation, characterised by detection error against agent separation.
3. A shared multi-agent steering policy learned from reward alone, evaluated
   across seven layout families under both metric families.
4. A quantified account of how much designer bias survives in the reward.
5. A scenario-aligned comparison between genetically discovered symbolic models
   and learned policies, including the learned behavior expressed as readable
   rules.

---

## References

Keller, N., and X. Hu. 2019. "Towards Data-Driven Simulation Modeling for Mobile
Agent-Based Systems." *ACM Transactions on Modeling and Computer Simulation*
29(1):1–26.

Le, H., and X. Hu. 2020. "Extended Model Space Specification for Mobile
Agent-Based Systems to Support Automated Discovery of Simulation Models." In
*Proceedings of the 2020 Winter Simulation Conference*, 2233–2244. IEEE.

Le, H., and X. Hu. 2022. "Automated Model Discovery for Steering Behavior
Simulation." In *Proceedings of the 2022 Annual Modeling and Simulation
Conference (ANNSIM '22)*, 54–65. SCS.

Raffin, A., A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann.
2021. "Stable-Baselines3: Reliable Reinforcement Learning Implementations."
*Journal of Machine Learning Research* 22(268):1–8.

Reynolds, C. W. 1999. "Steering Behaviors for Autonomous Characters." In
*Proceedings of the Game Developers Conference 1999*, 763–782.

Ronneberger, O., P. Fischer, and T. Brox. 2015. "U-Net: Convolutional Networks
for Biomedical Image Segmentation." In *MICCAI 2015*, 234–241. Springer.

Schulman, J., F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. 2017.
"Proximal Policy Optimization Algorithms." arXiv:1707.06347.

Tai, L., G. Paolo, and M. Liu. 2017. "Virtual-to-Real Deep Reinforcement
Learning: Continuous Control of Mobile Robots for Mapless Navigation." In
*IEEE/RSJ IROS 2017*, 31–36.

Tobin, J., R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel. 2017.
"Domain Randomization for Transferring Deep Neural Networks from Simulation to
the Real World." In *IEEE/RSJ IROS 2017*, 23–30.
