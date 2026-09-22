# Architecture: seg-only vs instance head

On `test`, best augmentation per cell by agent F1; mean ± std over seeds.

| resolution | variant | aug | F1 | cerr | head° | usable | n |
|---|---|---|---|---|---|---|---|
| 128 | unet24x4 | 2 | 0.999 ± 0.000 | 0.29 ± 0.00 | 8.6 ± 0.2 | 0.99 ± 0.00 | 2 |
| 128 | unet24x4 + inst | 2 | 1.000 ± 0.000 | 0.12 ± 0.02 | 1.8 ± 0.5 | 1.00 ± 0.00 | 2 |
