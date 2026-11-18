# E7: test_mixed

**Success rate** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.956 | 0.958 | 0.992 ± 0.003 | 0.995 | 0.992 ± 0.002 |
| corridor | 0.725 | 0.845 | 0.815 ± 0.013 | 0.820 | 0.890 ± 0.024 |
| room | 0.904 | 0.921 | 0.959 ± 0.008 | 0.962 | 0.955 ± 0.010 |
| doorway | 0.216 | 0.873 | 0.833 ± 0.008 | 0.818 | 0.913 ± 0.012 |
| two_doorway | 0.356 | 0.895 | 0.873 ± 0.016 | 0.880 | 0.904 ± 0.016 |
| cul_de_sac | 0.212 | 0.912 | 0.805 ± 0.018 | 0.804 | 0.889 ± 0.028 |
| crossing | 0.651 | 0.682 | 0.794 ± 0.056 | 0.793 | 0.794 ± 0.023 |
| boxrooms | 0.545 | 0.872 | 0.882 ± 0.008 | 0.868 | 0.904 ± 0.006 |
| walls_multi | 0.875 | 0.981 | 0.977 ± 0.003 | 0.986 | 0.989 ± 0.005 |
| rooms_multi | 0.555 | 0.909 | 0.944 ± 0.016 | 0.940 | 0.949 ± 0.007 |
| **overall** | 0.599 | 0.885 | 0.887 ± 0.013 | 0.887 | 0.918 ± 0.009 |

**Static (wall / obstacle) collision rate** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.017 | 0.014 | 0.004 ± 0.003 | 0.005 | 0.006 ± 0.001 |
| corridor | 0.071 | 0.020 | 0.127 ± 0.008 | 0.128 | 0.069 ± 0.005 |
| room | 0.019 | 0.005 | 0.031 ± 0.008 | 0.029 | 0.040 ± 0.010 |
| doorway | 0.553 | 0.028 | 0.068 ± 0.012 | 0.073 | 0.043 ± 0.004 |
| two_doorway | 0.391 | 0.010 | 0.075 ± 0.007 | 0.064 | 0.063 ± 0.013 |
| cul_de_sac | 0.400 | 0.036 | 0.056 ± 0.006 | 0.074 | 0.056 ± 0.010 |
| crossing | 0.087 | 0.039 | 0.094 ± 0.020 | 0.088 | 0.057 ± 0.002 |
| boxrooms | 0.340 | 0.096 | 0.069 ± 0.008 | 0.072 | 0.045 ± 0.011 |
| walls_multi | 0.112 | 0.013 | 0.007 ± 0.001 | 0.004 | 0.007 ± 0.002 |
| rooms_multi | 0.374 | 0.038 | 0.013 ± 0.001 | 0.037 | 0.025 ± 0.002 |
| **overall** | 0.236 | 0.030 | 0.054 ± 0.003 | 0.057 | 0.041 ± 0.003 |

**Agent collision rate** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.003 | 0.013 | 0.003 ± 0.003 | 0.000 | 0.001 ± 0.002 |
| corridor | 0.051 | 0.123 | 0.027 ± 0.006 | 0.015 | 0.008 ± 0.004 |
| room | 0.019 | 0.038 | 0.008 ± 0.008 | 0.003 | 0.003 ± 0.002 |
| doorway | 0.035 | 0.092 | 0.046 ± 0.018 | 0.038 | 0.012 ± 0.009 |
| two_doorway | 0.009 | 0.076 | 0.023 ± 0.015 | 0.018 | 0.011 ± 0.002 |
| cul_de_sac | 0.052 | 0.046 | 0.041 ± 0.017 | 0.012 | 0.013 ± 0.002 |
| crossing | 0.119 | 0.227 | 0.091 ± 0.057 | 0.079 | 0.101 ± 0.043 |
| boxrooms | 0.043 | 0.032 | 0.024 ± 0.010 | 0.024 | 0.028 ± 0.011 |
| walls_multi | 0.007 | 0.006 | 0.011 ± 0.004 | 0.010 | 0.002 ± 0.002 |
| rooms_multi | 0.043 | 0.048 | 0.033 ± 0.014 | 0.008 | 0.023 ± 0.004 |
| **overall** | 0.038 | 0.070 | 0.031 ± 0.015 | 0.021 | 0.020 ± 0.006 |

**Timeout rate** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.025 | 0.015 | 0.000 ± 0.000 | 0.000 | 0.001 ± 0.001 |
| corridor | 0.154 | 0.012 | 0.031 ± 0.013 | 0.037 | 0.034 ± 0.024 |
| room | 0.059 | 0.035 | 0.001 ± 0.001 | 0.006 | 0.002 ± 0.002 |
| doorway | 0.196 | 0.007 | 0.053 ± 0.005 | 0.071 | 0.032 ± 0.007 |
| two_doorway | 0.245 | 0.019 | 0.029 ± 0.008 | 0.037 | 0.021 ± 0.009 |
| cul_de_sac | 0.336 | 0.006 | 0.099 ± 0.002 | 0.109 | 0.043 ± 0.040 |
| crossing | 0.143 | 0.051 | 0.020 ± 0.005 | 0.041 | 0.049 ± 0.047 |
| boxrooms | 0.072 | 0.000 | 0.025 ± 0.011 | 0.035 | 0.023 ± 0.014 |
| walls_multi | 0.006 | 0.000 | 0.005 ± 0.000 | 0.000 | 0.002 ± 0.003 |
| rooms_multi | 0.027 | 0.005 | 0.011 ± 0.001 | 0.014 | 0.002 ± 0.003 |
| **overall** | 0.126 | 0.015 | 0.027 ± 0.002 | 0.035 | 0.021 ± 0.015 |

**Loop rate (cumulative heading change spans >= 360 deg)** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.000 | 0.000 | 0.000 ± 0.001 | 0.005 | 0.003 ± 0.003 |
| corridor | 0.000 | 0.000 | 0.086 ± 0.005 | 0.076 | 0.042 ± 0.021 |
| room | 0.000 | 0.003 | 0.010 ± 0.002 | 0.018 | 0.009 ± 0.003 |
| doorway | 0.016 | 0.000 | 0.222 ± 0.043 | 0.233 | 0.128 ± 0.047 |
| two_doorway | 0.014 | 0.003 | 0.132 ± 0.034 | 0.141 | 0.078 ± 0.012 |
| cul_de_sac | 0.032 | 0.000 | 0.320 ± 0.034 | 0.303 | 0.224 ± 0.044 |
| crossing | 0.015 | 0.014 | 0.173 ± 0.013 | 0.198 | 0.105 ± 0.008 |
| boxrooms | 0.017 | 0.006 | 0.111 ± 0.018 | 0.114 | 0.084 ± 0.006 |
| walls_multi | 0.000 | 0.000 | 0.022 ± 0.008 | 0.025 | 0.003 ± 0.002 |
| rooms_multi | 0.003 | 0.005 | 0.080 ± 0.031 | 0.078 | 0.077 ± 0.006 |
| **overall** | 0.010 | 0.003 | 0.116 ± 0.016 | 0.119 | 0.075 ± 0.013 |

**Success rate per bottleneck width** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| bottleneck | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| [5.2,7) | 0.385 | 0.701 | 0.594 ± 0.018 | 0.552 | 0.745 ± 0.010 |
| [7,10) | 0.369 | 0.823 | 0.850 ± 0.023 | 0.855 | 0.881 ± 0.016 |
| [10,15) | 0.439 | 0.889 | 0.907 ± 0.017 | 0.916 | 0.931 ± 0.017 |
| [15,inf) | 0.873 | 0.971 | 0.984 ± 0.005 | 0.988 | 0.982 ± 0.002 |

**Multi-exit subset: success** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| subset | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| multi-exit scenes | 0.760 | 0.957 | 0.964 ± 0.012 | 0.967 | 0.970 ± 0.005 |

**Multi-exit subset: loop rate** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| subset | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| multi-exit scenes | 0.006 | 0.001 | 0.033 ± 0.016 | 0.032 | 0.024 ± 0.001 |

**Multi-exit subset: exit switches per agent** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| subset | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| multi-exit scenes | 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 |

**Success of looping agents** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

|  | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| **overall** | 0.116 | 0.750 | 0.740 ± 0.034 | 0.670 | 0.845 ± 0.018 |

**Success of the other agents** (`test_mixed`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

|  | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| **overall** | 0.602 | 0.884 | 0.902 ± 0.016 | 0.913 | 0.922 ± 0.010 |
