# Resolution × training augmentation

Each cell: agent F1 · centre error (units) · heading error (deg) · scene_usable rate, mean ± std over seeds. Empty cells are off the star design.

## `test` — unet24x4+inst

| resolution | aug3 | sketch+aug3 |
|---|---|---|
| 256 | 1.000 ± 0.000 · 0.09 ± 0.00 · 2.0 ± 0.0 · 1.00 ± 0.00 (n=2) | 1.000 ± 0.000 · 0.11 ± 0.00 · 1.7 ± 0.3 · 1.00 ± 0.00 (n=2) |

## `test_aug3` — unet24x4+inst

| resolution | aug3 | sketch+aug3 |
|---|---|---|
| 256 | 1.000 ± 0.000 · 0.10 ± 0.00 · 2.1 ± 0.0 · 1.00 ± 0.00 (n=2) | 1.000 ± 0.000 · 0.10 ± 0.00 · 1.7 ± 0.2 · 1.00 ± 0.00 (n=2) |
