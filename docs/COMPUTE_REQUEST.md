# Compute request — upper bound for main line B

Date: 2026-11-03. Purpose: one request that covers the whole remaining project (multiple input types → one scene format; behaviour gap under perception error; site-specific fine-tuning; error-tolerant policy), so it does not run out later. Every figure is an **upper bound** built from measured run times on the current machine plus a 30 % margin for failed runs, re-runs and tuning.

## 1. Summary

| Resource | Upper bound | Basis |
|---|---|---|
| GPU time | **≈ 450 A100-class GPU-hours** (≈ 700 on A10 / RTX 3090; ≈ 1,050 on T4 / GTX 1070) | §2 |
| GPU memory | **≥ 24 GB per GPU** (40 GB preferred) | 512 px at batch 4 already uses 6 GB on the 8 GB card; the unified and larger models need more |
| CPU time | **≈ 17,000 core-hours** (expected ≈ 6,500) | §3 |
| CPU node | **≥ 32 cores**, 2–4 GB RAM per core | many independent CPU-bound policy runs |
| RAM on the GPU node | **64 GB** | the unified data set at 512 px (≈ 50k images ≈ 40 GB as uint8) cannot be cached in 16 GB |
| Storage | **500 GB** | §4 |
| Software | Python 3.11+, PyTorch with CUDA, NumPy, OpenCV, Pillow | no licensed software, no container registry |

Wall-clock with one A100 and one 64-core node: about 3 weeks of GPU time and about 11 days of CPU time if run continuously; the two run in parallel.

## 2. GPU (perception)

Measured on the GTX 1070 (`perception/DESIGN.md` §6.1, 5,000 images, 60 epochs, fp32): 74 s/epoch at 256 px (1.23 h per run), 281 s/epoch at 512 px (4.68 h per run); time scales linearly with the number of images.

| Work | Runs | Setting | GTX 1070-hours |
|---|---|---|---|
| Further hand-drawing versions | 8 | 256 px, 10k images | 20 |
| Floor plans (CubiCasa5K train, MSD test) | 12 | 512 px (thin lines), 10k | 112 |
| Satellite / drone (S2-NAIP + OSM) | 12 | 256 px, 20k | 59 |
| Rendered OSM maps | 8 | 256 px, 10k | 20 |
| Resolution / augmentation sweep, three new input types | 3 × 24 | as the existing E5 sweep (≈ 40 h each) | 120 |
| Unified model over all input types | 4 | 512 px, 50k | 187 |
| Unified model, larger architecture (`unet24x5`, ≈ 3× cost) | 2 | 512 px, 50k | 281 |
| Inference to build parsed scenes for the policy studies | — | — | 10 |
| **Total** | | | **809** |
| **+ 30 % margin** | | | **≈ 1,050** |

Conversion to other GPUs uses deliberately low speed-ups over the GTX 1070 (no tensor cores, fp32): A100 / RTX 4090 / L40S ×2.5, A10 / RTX 3090 ×1.5, T4 ×1.0. With mixed precision the real speed-up on an A100 is likely higher, so these numbers err on the safe side.

## 3. CPU (policy)

The policy network is small (≈ 25k parameters MLP, ≈ 0.2M GRU); simulation dominates, so policy training runs on CPU workers. One full run is 50 M agent-steps (`policy/DESIGN.md` §3). Measured on the Mac: ≈ 7,000 agent-steps/s on ≈ 3 cores for the MLP on open scenes. Per full run this report assumes **15 core-hours expected, 40 core-hours upper bound** (GRU, harder layouts). Cost per agent-step grows roughly linearly with the number of agents, which the crowd-size factor below accounts for.

| Work | Runs | Cost factor per run | Upper core-hours |
|---|---|---|---|
| Layout transfer, leave-one-out: 10 folds × 2 variants × 5 seeds | 100 | 1 | 4,000 |
| Crowd size 2 / 4 / 8 / 16 agents × 2 variants × 5 seeds | 40 | 1.9 (mean of 0.5, 1, 2, 4) | 3,000 |
| Crowd size 32 agents × 2 variants × 5 seeds (drone swarms; optional) | 10 | 8 | 3,200 |
| Retrain after the polygon-obstacle simulator change × 2 variants × 5 seeds | 10 | 1 | 400 |
| Site-specific fine-tuning: 30 sites × 3 seeds × 2 conditions (site vs all layouts) × 2 budgets (2 M, 5 M steps) | 360 | 0.07 | 1,010 |
| Error-tolerant training on parsed / noisy scenes: 3 levels × 2 variants × 5 seeds | 30 | 1 | 1,200 |
| ORCA baseline and behaviour-gap evaluations | — | — | 220 |
| **Total** | | | **≈ 13,000** |
| **+ 30 % margin** | | | **≈ 17,000** |

Expected case (15 core-hours per run): ≈ 5,000, or ≈ 6,500 with the margin.

## 4. Storage

| Item | Size |
|---|---|
| CubiCasa5K, MSD | ≈ 10–20 GB |
| S2-NAIP subset (NAIP tiles + OSM GeoJSON only — the full data set is far larger and is **not** downloaded) | ≈ 50 GB |
| Generated training sets (hand drawings, rendered maps, composited marks), 512 px | ≈ 100 GB |
| Perception checkpoints, previews, evaluation outputs | ≈ 50 GB |
| Policy runs, checkpoints, logs, evaluation outputs | ≈ 50 GB |
| Scratch for building data sets (two sets coexist while one is rebuilt) | ≈ 100 GB |
| **Total** | **≈ 350 GB → request 500 GB** |

## 5. Assumptions to confirm

- The GRU's agent-steps/s on the training machine is not recorded in the repo; `policy/scripts/peek_e2.py` on the Windows box gives it and can replace the 15 / 40 core-hour assumption.
- The sizes of CubiCasa5K and MSD are estimates; check the download pages.
- The 32-agent arm and the larger unified model are the two largest optional items (≈ 3,200 core-hours and ≈ 280 GTX 1070-hours); dropping both lowers the CPU request by about a quarter and the GPU request by about a third.
