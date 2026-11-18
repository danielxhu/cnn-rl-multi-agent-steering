# E7: test4

**Success rate** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.965 | 0.985 | 0.994 ± 0.003 | 0.998 | 0.990 ± 0.002 |
| corridor | 0.770 | 0.835 | 0.802 ± 0.011 | 0.850 | 0.882 ± 0.013 |
| room | 0.907 | 0.965 | 0.979 ± 0.012 | 0.990 | 0.982 ± 0.008 |
| doorway | 0.207 | 0.892 | 0.878 ± 0.025 | 0.880 | 0.940 ± 0.007 |
| two_doorway | 0.388 | 0.887 | 0.902 ± 0.020 | 0.880 | 0.925 ± 0.005 |
| cul_de_sac | 0.193 | 0.905 | 0.808 ± 0.029 | 0.802 | 0.863 ± 0.026 |
| crossing | 0.743 | 0.765 | 0.799 ± 0.032 | 0.792 | 0.835 ± 0.009 |
| boxrooms | 0.535 | 0.890 | 0.842 ± 0.026 | 0.856 | 0.893 ± 0.008 |
| walls_multi | 0.892 | 0.998 | 0.987 ± 0.004 | 0.995 | 0.997 ± 0.001 |
| rooms_multi | 0.570 | 0.920 | 0.937 ± 0.016 | 0.940 | 0.954 ± 0.005 |
| **overall** | 0.617 | 0.904 | 0.893 ± 0.013 | 0.898 | 0.926 ± 0.004 |

**Static (wall / obstacle) collision rate** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.015 | 0.010 | 0.006 ± 0.003 | 0.000 | 0.004 ± 0.006 |
| corridor | 0.037 | 0.020 | 0.136 ± 0.017 | 0.110 | 0.076 ± 0.017 |
| room | 0.025 | 0.003 | 0.015 ± 0.008 | 0.007 | 0.012 ± 0.002 |
| doorway | 0.570 | 0.030 | 0.061 ± 0.006 | 0.048 | 0.039 ± 0.006 |
| two_doorway | 0.385 | 0.007 | 0.050 ± 0.005 | 0.065 | 0.048 ± 0.012 |
| cul_de_sac | 0.422 | 0.040 | 0.069 ± 0.015 | 0.083 | 0.065 ± 0.015 |
| crossing | 0.065 | 0.062 | 0.083 ± 0.011 | 0.060 | 0.053 ± 0.002 |
| boxrooms | 0.307 | 0.075 | 0.106 ± 0.021 | 0.104 | 0.082 ± 0.008 |
| walls_multi | 0.090 | 0.003 | 0.008 ± 0.003 | 0.003 | 0.003 ± 0.001 |
| rooms_multi | 0.362 | 0.043 | 0.017 ± 0.006 | 0.015 | 0.036 ± 0.001 |
| **overall** | 0.228 | 0.029 | 0.055 ± 0.004 | 0.049 | 0.042 ± 0.006 |

**Agent collision rate** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.005 | 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 |
| corridor | 0.035 | 0.113 | 0.025 ± 0.008 | 0.010 | 0.017 ± 0.002 |
| room | 0.015 | 0.010 | 0.003 ± 0.005 | 0.000 | 0.002 ± 0.002 |
| doorway | 0.030 | 0.077 | 0.028 ± 0.026 | 0.025 | 0.010 ± 0.004 |
| two_doorway | 0.010 | 0.075 | 0.025 ± 0.032 | 0.015 | 0.003 ± 0.002 |
| cul_de_sac | 0.050 | 0.040 | 0.027 ± 0.012 | 0.015 | 0.028 ± 0.022 |
| crossing | 0.055 | 0.130 | 0.082 ± 0.024 | 0.100 | 0.070 ± 0.031 |
| boxrooms | 0.052 | 0.022 | 0.023 ± 0.012 | 0.010 | 0.007 ± 0.005 |
| walls_multi | 0.005 | 0.000 | 0.002 ± 0.002 | 0.000 | 0.000 ± 0.000 |
| rooms_multi | 0.045 | 0.033 | 0.015 ± 0.012 | 0.000 | 0.003 ± 0.005 |
| **overall** | 0.030 | 0.050 | 0.023 ± 0.012 | 0.018 | 0.014 ± 0.006 |

**Timeout rate** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.015 | 0.005 | 0.000 ± 0.000 | 0.003 | 0.006 ± 0.004 |
| corridor | 0.158 | 0.033 | 0.037 ± 0.012 | 0.030 | 0.026 ± 0.023 |
| room | 0.052 | 0.022 | 0.003 ± 0.004 | 0.003 | 0.005 ± 0.007 |
| doorway | 0.193 | 0.000 | 0.033 ± 0.005 | 0.048 | 0.011 ± 0.008 |
| two_doorway | 0.217 | 0.030 | 0.023 ± 0.009 | 0.040 | 0.023 ± 0.011 |
| cul_de_sac | 0.335 | 0.015 | 0.096 ± 0.004 | 0.100 | 0.043 ± 0.041 |
| crossing | 0.138 | 0.043 | 0.036 ± 0.008 | 0.048 | 0.042 ± 0.030 |
| boxrooms | 0.105 | 0.013 | 0.029 ± 0.012 | 0.030 | 0.018 ± 0.012 |
| walls_multi | 0.013 | 0.000 | 0.003 ± 0.000 | 0.003 | 0.000 ± 0.000 |
| rooms_multi | 0.022 | 0.005 | 0.032 ± 0.008 | 0.045 | 0.007 ± 0.001 |
| **overall** | 0.125 | 0.017 | 0.029 ± 0.001 | 0.035 | 0.018 ± 0.013 |

**Loop rate (cumulative heading change spans >= 360 deg)** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| layout | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| open | 0.000 | 0.000 | 0.003 ± 0.002 | 0.000 | 0.005 ± 0.000 |
| corridor | 0.000 | 0.000 | 0.075 ± 0.011 | 0.110 | 0.052 ± 0.020 |
| room | 0.003 | 0.000 | 0.010 ± 0.009 | 0.010 | 0.012 ± 0.003 |
| doorway | 0.007 | 0.000 | 0.202 ± 0.051 | 0.233 | 0.140 ± 0.010 |
| two_doorway | 0.000 | 0.003 | 0.111 ± 0.022 | 0.122 | 0.099 ± 0.025 |
| cul_de_sac | 0.037 | 0.000 | 0.327 ± 0.023 | 0.320 | 0.210 ± 0.037 |
| crossing | 0.007 | 0.013 | 0.161 ± 0.017 | 0.185 | 0.086 ± 0.008 |
| boxrooms | 0.013 | 0.003 | 0.120 ± 0.011 | 0.115 | 0.081 ± 0.007 |
| walls_multi | 0.000 | 0.000 | 0.015 ± 0.002 | 0.013 | 0.009 ± 0.004 |
| rooms_multi | 0.005 | 0.003 | 0.085 ± 0.023 | 0.055 | 0.051 ± 0.031 |
| **overall** | 0.007 | 0.002 | 0.111 ± 0.011 | 0.116 | 0.074 ± 0.013 |

**Success rate per bottleneck width** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| bottleneck | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| [5.2,7) | 0.377 | 0.750 | 0.613 ± 0.036 | 0.598 | 0.749 ± 0.004 |
| [7,10) | 0.444 | 0.850 | 0.845 ± 0.020 | 0.865 | 0.892 ± 0.014 |
| [10,15) | 0.457 | 0.914 | 0.925 ± 0.012 | 0.930 | 0.943 ± 0.007 |
| [15,inf) | 0.875 | 0.976 | 0.985 ± 0.005 | 0.988 | 0.988 ± 0.000 |

**Multi-exit subset: success** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| subset | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| multi-exit scenes | 0.704 | 0.951 | 0.923 ± 0.015 | 0.939 | 0.948 ± 0.006 |

**Multi-exit subset: loop rate** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| subset | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| multi-exit scenes | 0.006 | 0.002 | 0.062 ± 0.007 | 0.062 | 0.039 ± 0.010 |

**Multi-exit subset: exit switches per agent** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

| subset | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| multi-exit scenes | 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 |

**Success of looping agents** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

|  | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| **overall** | 0.146 | 0.857 | 0.701 ± 0.034 | 0.684 | 0.861 ± 0.016 |

**Success of the other agents** (`test4`, mean ± std over seeds; runs: orca_straight 1, orca_path 1, e3_both 3, e7_longgru 1, e7_map 3)

|  | orca_straight | orca_path | e3_both | e7_longgru | e7_map |
|---|---|---|---|---|---|
| **overall** | 0.618 | 0.904 | 0.910 ± 0.014 | 0.923 | 0.928 ± 0.006 |
