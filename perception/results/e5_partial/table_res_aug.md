# Resolution × training augmentation

Each cell: agent F1 · centre error (units) · heading error (deg) · scene_usable rate, mean ± std over seeds. Empty cells are off the star design.

## `test` — unet24x4

| resolution | aug2 |
|---|---|
| 128 | 0.999 ± 0.000 · 0.29 ± 0.00 · 8.6 ± 0.2 · 0.99 ± 0.00 (n=2) |

## `test` — unet24x4+inst

| resolution | aug2 |
|---|---|
| 128 | 1.000 ± 0.000 · 0.12 ± 0.02 · 1.8 ± 0.5 · 1.00 ± 0.00 (n=2) |

## `test_aug3` — unet24x4

| resolution | aug2 |
|---|---|
| 128 | 0.999 ± 0.000 · 0.29 ± 0.03 · 9.6 ± 0.1 · 0.94 ± 0.03 (n=2) |

## `test_aug3` — unet24x4+inst

| resolution | aug2 |
|---|---|
| 128 | 1.000 ± 0.000 · 0.10 ± 0.00 · 1.9 ± 0.5 · 0.99 ± 0.01 (n=2) |
