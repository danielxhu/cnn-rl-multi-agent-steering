# Resolution × training augmentation

Each cell: agent F1 · centre error (units) · heading error (deg) · scene_usable rate, mean ± std over seeds. Empty cells are off the star design.

## `test` — unet24x4+inst

| resolution | aug3 | sketch+aug3 | sketch4+aug3 | sketch5+aug3 |
|---|---|---|---|---|
| 256 |  |  | 1.000 · 0.11 · 1.0 · 1.00 (n=1) | 1.000 · 0.09 · 0.9 · 1.00 (n=1) |

## `test_aug3` — unet24x4+inst

| resolution | aug3 | sketch+aug3 | sketch4+aug3 | sketch5+aug3 |
|---|---|---|---|---|
| 256 |  |  | 1.000 · 0.11 · 0.9 · 1.00 (n=1) | 1.000 · 0.07 · 0.9 · 1.00 (n=1) |
