# Behaviour gap: e3_both_seed0 on true vs parsed scenes

Perception `sketch7`. One deterministic episode per scene. S = success, C = collision, T = timeout, as rates over a scene's agents, averaged over scenes. gap = parsed - true, on the scenes with a parse (DESIGN_GAP.md).

## Arm 1: gap per set, each controller in the true and in the parsed world

| set | controller | scenes | true S | parsed S | gap S (95% CI) | mean abs gap S | true C | parsed C | gap C | true T | parsed T | gap T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_sketch7 | policy | 998 | 0.828 | 0.843 | +0.016 ± 0.015 | 0.107 | 0.137 | 0.125 | -0.012 | 0.035 | 0.032 | -0.003 |
| test_sketch7 | orca_path | 998 | 0.824 | 0.837 | +0.012 ± 0.015 | 0.114 | 0.170 | 0.154 | -0.016 | 0.006 | 0.010 | +0.004 |
| test_sketch7_free | policy | 498 | 0.900 | 0.895 | -0.005 ± 0.014 | 0.065 | 0.070 | 0.068 | -0.002 | 0.030 | 0.037 | +0.007 |
| test_sketch7_free | orca_path | 498 | 0.911 | 0.901 | -0.010 ± 0.016 | 0.078 | 0.083 | 0.091 | +0.009 | 0.007 | 0.008 | +0.001 |
| test_sketch7_tilt | policy | 498 | 0.842 | 0.833 | -0.009 ± 0.020 | 0.109 | 0.126 | 0.139 | +0.013 | 0.031 | 0.028 | -0.004 |
| test_sketch7_tilt | orca_path | 498 | 0.840 | 0.825 | -0.014 ± 0.021 | 0.115 | 0.151 | 0.166 | +0.015 | 0.009 | 0.008 | -0.001 |
| test_sketch7_lines | policy | 492 | 0.800 | 0.798 | -0.002 ± 0.022 | 0.123 | 0.169 | 0.159 | -0.010 | 0.031 | 0.043 | +0.012 |
| test_sketch7_lines | orca_path | 492 | 0.818 | 0.798 | -0.020 ± 0.023 | 0.138 | 0.169 | 0.193 | +0.024 | 0.013 | 0.009 | -0.004 |
| overall | policy | 2486 | 0.840 | 0.843 | +0.003 ± 0.009 | 0.102 | 0.128 | 0.123 | -0.005 | 0.033 | 0.034 | +0.002 |
| overall | orca_path | 2486 | 0.843 | 0.840 | -0.004 ± 0.009 | 0.112 | 0.148 | 0.151 | +0.003 | 0.008 | 0.009 | +0.001 |

`overall` pools the sets unweighted; `test_sketch7` is the headline, the others are probes of hard cases.

## Arm 2: orca_path with the parsed map, in the true world

Physics, agents and goals are the true scene's, also for the policy, which reads the true world through its rays and uses no map. Map cost = ORCA wrong map - ORCA right map. This tests map errors only, not goal or exit errors. `no path at start` = scenes where at least one agent had no finite path on the parsed map (it then follows the straight bearing).

| group | scenes | policy S (no map) | ORCA right map S | ORCA wrong map S | map cost S (95% CI) | mean abs S | ORCA right C | wrong C | of which static | ORCA right T | wrong T | no path at start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_sketch7 | 998 | 0.828 | 0.824 | 0.727 | -0.097 ± 0.017 | 0.142 | 0.170 | 0.265 | 0.161 | 0.006 | 0.008 | 2 |
| test_sketch7_free | 498 | 0.900 | 0.911 | 0.803 | -0.108 ± 0.021 | 0.136 | 0.083 | 0.190 | 0.158 | 0.007 | 0.007 | 4 |
| test_sketch7_tilt | 498 | 0.842 | 0.840 | 0.738 | -0.101 ± 0.023 | 0.133 | 0.151 | 0.257 | 0.149 | 0.009 | 0.005 | 4 |
| test_sketch7_lines | 492 | 0.800 | 0.818 | 0.692 | -0.125 ± 0.025 | 0.171 | 0.169 | 0.303 | 0.212 | 0.013 | 0.005 | 1 |
| overall | 2486 | 0.840 | 0.843 | 0.737 | -0.106 ± 0.010 | 0.145 | 0.148 | 0.256 | 0.168 | 0.008 | 0.006 | 11 |
| usable: yes | 2200 | 0.845 | 0.850 | 0.747 | -0.102 ± 0.011 | 0.142 | 0.141 | 0.246 | 0.164 | 0.009 | 0.007 | 5 |
| usable: no | 286 | 0.795 | 0.795 | 0.662 | -0.133 ± 0.030 | 0.162 | 0.203 | 0.333 | 0.198 | 0.001 | 0.005 | 6 |
| path at start | 2475 | 0.841 | 0.844 | 0.738 | -0.106 ± 0.011 | 0.144 | 0.148 | 0.256 | 0.168 | 0.008 | 0.006 | 0 |
| no path at start | 11 | 0.623 | 0.708 | 0.544 | -0.164 ± 0.135 | 0.164 | 0.161 | 0.389 | 0.187 | 0.132 | 0.067 | 11 |

## Failed parses

End-to-end S counts a failed parse as 0 success.

| set | scenes | failed | failed rate | true S, all scenes | true S, failed scenes | end-to-end S |
|---|---|---|---|---|---|---|
| test_sketch7 | 1000 | 2 | 0.002 | 0.828 | 1.000 | 0.842 |
| test_sketch7_free | 499 | 1 | 0.002 | 0.900 | 0.625 | 0.894 |
| test_sketch7_tilt | 499 | 1 | 0.002 | 0.843 | 1.000 | 0.831 |
| test_sketch7_lines | 492 | 0 | 0.000 | 0.800 | - | 0.798 |
| overall | 2490 | 4 | 0.002 | 0.840 | 0.906 | 0.841 |

- 4 x `ValueError: perception marked this drawing as not parsed`

## Gap by perception error

`usable` = perception's `scene_usable`; `agents` = parsed minus true agent count.

| group | controller | scenes | true S | parsed S | gap S (95% CI) | mean abs gap S | true C | parsed C | gap C | true T | parsed T | gap T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| usable: yes | policy | 2200 | 0.845 | 0.854 | +0.008 ± 0.009 | 0.095 | 0.121 | 0.111 | -0.009 | 0.034 | 0.035 | +0.001 |
| usable: yes | orca_path | 2200 | 0.850 | 0.850 | +0.001 ± 0.009 | 0.103 | 0.141 | 0.140 | -0.001 | 0.009 | 0.010 | +0.000 |
| usable: no | policy | 286 | 0.795 | 0.758 | -0.037 ± 0.031 | 0.159 | 0.181 | 0.211 | +0.030 | 0.024 | 0.031 | +0.007 |
| usable: no | orca_path | 286 | 0.795 | 0.759 | -0.036 ± 0.034 | 0.179 | 0.203 | 0.237 | +0.034 | 0.001 | 0.004 | +0.003 |
| agents <= -2 | policy | 19 | 0.711 | 0.717 | +0.006 ± 0.121 | 0.174 | 0.289 | 0.283 | -0.006 | 0.000 | 0.000 | +0.000 |
| agents <= -2 | orca_path | 19 | 0.701 | 0.744 | +0.043 ± 0.104 | 0.154 | 0.293 | 0.256 | -0.036 | 0.007 | 0.000 | -0.007 |
| agents -1 | policy | 95 | 0.795 | 0.827 | +0.032 ± 0.049 | 0.130 | 0.177 | 0.150 | -0.027 | 0.028 | 0.023 | -0.004 |
| agents -1 | orca_path | 95 | 0.754 | 0.817 | +0.063 ± 0.052 | 0.153 | 0.246 | 0.177 | -0.069 | 0.000 | 0.006 | +0.006 |
| agents 0 | policy | 2294 | 0.843 | 0.850 | +0.007 ± 0.009 | 0.096 | 0.123 | 0.115 | -0.008 | 0.034 | 0.035 | +0.001 |
| agents 0 | orca_path | 2294 | 0.849 | 0.848 | -0.002 ± 0.009 | 0.106 | 0.142 | 0.143 | +0.001 | 0.009 | 0.009 | +0.000 |
| agents +1 | policy | 75 | 0.816 | 0.675 | -0.141 ± 0.066 | 0.214 | 0.175 | 0.283 | +0.108 | 0.009 | 0.042 | +0.033 |
| agents +1 | orca_path | 75 | 0.811 | 0.658 | -0.152 ± 0.058 | 0.219 | 0.189 | 0.337 | +0.148 | 0.000 | 0.004 | +0.004 |
| agents >= +2 | policy | 3 | 0.778 | 0.542 | -0.235 ± 0.394 | 0.347 | 0.222 | 0.458 | +0.235 | 0.000 | 0.000 | +0.000 |
| agents >= +2 | orca_path | 3 | 0.778 | 0.590 | -0.188 ± 0.359 | 0.299 | 0.222 | 0.410 | +0.188 | 0.000 | 0.000 | +0.000 |

## Gap per layout

`*` = not among the policy's training layouts. `crossing`: true scenes give each agent its own goal, parsed scenes any exit (42 scenes with per-agent goals).

| layout | controller | scenes | true S | parsed S | gap S (95% CI) | mean abs gap S | true C | parsed C | gap C | true T | parsed T | gap T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boxrooms | policy | 278 | 0.849 | 0.858 | +0.010 ± 0.028 | 0.113 | 0.128 | 0.118 | -0.010 | 0.023 | 0.023 | +0.000 |
| boxrooms | orca_path | 278 | 0.852 | 0.860 | +0.008 ± 0.032 | 0.134 | 0.141 | 0.136 | -0.006 | 0.007 | 0.004 | -0.002 |
| corridor | policy | 136 | 0.710 | 0.747 | +0.038 ± 0.035 | 0.097 | 0.239 | 0.192 | -0.047 | 0.052 | 0.060 | +0.009 |
| corridor | orca_path | 136 | 0.813 | 0.844 | +0.032 ± 0.038 | 0.103 | 0.182 | 0.144 | -0.038 | 0.005 | 0.011 | +0.006 |
| crossing | policy | 134 | 0.845 | 0.892 | +0.047 ± 0.039 | 0.116 | 0.124 | 0.101 | -0.023 | 0.032 | 0.008 | -0.024 |
| crossing | orca_path | 134 | 0.827 | 0.829 | +0.002 ± 0.047 | 0.147 | 0.140 | 0.148 | +0.008 | 0.032 | 0.023 | -0.009 |
| cul_de_sac | policy | 140 | 0.743 | 0.742 | -0.001 ± 0.035 | 0.104 | 0.145 | 0.155 | +0.011 | 0.112 | 0.102 | -0.010 |
| cul_de_sac | orca_path | 140 | 0.817 | 0.805 | -0.013 ± 0.039 | 0.124 | 0.176 | 0.186 | +0.010 | 0.006 | 0.009 | +0.003 |
| doorway | policy | 140 | 0.823 | 0.799 | -0.025 ± 0.045 | 0.134 | 0.134 | 0.143 | +0.008 | 0.042 | 0.058 | +0.016 |
| doorway | orca_path | 140 | 0.803 | 0.780 | -0.023 ± 0.036 | 0.111 | 0.189 | 0.209 | +0.020 | 0.008 | 0.011 | +0.003 |
| open | policy | 140 | 0.913 | 0.925 | +0.012 ± 0.024 | 0.050 | 0.087 | 0.069 | -0.018 | 0.000 | 0.006 | +0.006 |
| open | orca_path | 140 | 0.899 | 0.892 | -0.007 ± 0.031 | 0.077 | 0.100 | 0.095 | -0.005 | 0.002 | 0.013 | +0.011 |
| room | policy | 140 | 0.814 | 0.861 | +0.047 ± 0.044 | 0.119 | 0.164 | 0.111 | -0.053 | 0.022 | 0.028 | +0.006 |
| room | orca_path | 140 | 0.860 | 0.872 | +0.012 ± 0.028 | 0.062 | 0.131 | 0.113 | -0.018 | 0.009 | 0.015 | +0.006 |
| rooms * | policy | 277 | 0.784 | 0.755 | -0.030 ± 0.031 | 0.132 | 0.154 | 0.172 | +0.018 | 0.062 | 0.073 | +0.011 |
| rooms * | orca_path | 277 | 0.756 | 0.742 | -0.015 ± 0.032 | 0.161 | 0.239 | 0.251 | +0.012 | 0.005 | 0.007 | +0.002 |
| rooms_multi | policy | 271 | 0.853 | 0.853 | +0.000 ± 0.026 | 0.098 | 0.113 | 0.107 | -0.007 | 0.034 | 0.041 | +0.007 |
| rooms_multi | orca_path | 271 | 0.877 | 0.856 | -0.021 ± 0.028 | 0.112 | 0.121 | 0.132 | +0.011 | 0.002 | 0.012 | +0.010 |
| two_doorway | policy | 140 | 0.857 | 0.859 | +0.002 ± 0.038 | 0.102 | 0.133 | 0.125 | -0.008 | 0.010 | 0.017 | +0.006 |
| two_doorway | orca_path | 140 | 0.825 | 0.808 | -0.016 ± 0.040 | 0.110 | 0.160 | 0.178 | +0.018 | 0.015 | 0.014 | -0.002 |
| walls * | policy | 419 | 0.871 | 0.868 | -0.003 ± 0.020 | 0.093 | 0.108 | 0.114 | +0.006 | 0.021 | 0.018 | -0.003 |
| walls * | orca_path | 419 | 0.849 | 0.854 | +0.005 ± 0.021 | 0.108 | 0.143 | 0.144 | +0.001 | 0.009 | 0.002 | -0.006 |
| walls_multi | policy | 271 | 0.913 | 0.905 | -0.008 ± 0.021 | 0.076 | 0.079 | 0.089 | +0.010 | 0.008 | 0.005 | -0.003 |
| walls_multi | orca_path | 271 | 0.910 | 0.904 | -0.007 ± 0.021 | 0.071 | 0.080 | 0.090 | +0.010 | 0.009 | 0.006 | -0.004 |

### Arm 2 per layout

| layout | scenes | policy S (no map) | ORCA right map S | ORCA wrong map S | map cost S (95% CI) | mean abs S | ORCA right C | wrong C | of which static | ORCA right T | wrong T | no path at start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boxrooms | 278 | 0.849 | 0.852 | 0.614 | -0.238 ± 0.040 | 0.249 | 0.141 | 0.381 | 0.348 | 0.007 | 0.005 | 1 |
| corridor | 136 | 0.710 | 0.813 | 0.754 | -0.059 ± 0.037 | 0.095 | 0.182 | 0.238 | 0.101 | 0.005 | 0.008 | 1 |
| crossing | 134 | 0.845 | 0.827 | 0.683 | -0.145 ± 0.044 | 0.159 | 0.140 | 0.291 | 0.209 | 0.032 | 0.026 | 3 |
| cul_de_sac | 140 | 0.743 | 0.817 | 0.706 | -0.112 ± 0.047 | 0.176 | 0.176 | 0.287 | 0.198 | 0.006 | 0.007 | 0 |
| doorway | 140 | 0.823 | 0.803 | 0.685 | -0.118 ± 0.052 | 0.189 | 0.189 | 0.304 | 0.165 | 0.008 | 0.011 | 0 |
| open | 140 | 0.913 | 0.899 | 0.856 | -0.042 ± 0.031 | 0.080 | 0.100 | 0.140 | 0.063 | 0.002 | 0.004 | 1 |
| room | 140 | 0.814 | 0.860 | 0.811 | -0.049 ± 0.026 | 0.065 | 0.131 | 0.178 | 0.083 | 0.009 | 0.011 | 1 |
| rooms * | 277 | 0.784 | 0.756 | 0.632 | -0.124 ± 0.036 | 0.189 | 0.239 | 0.365 | 0.220 | 0.005 | 0.003 | 0 |
| rooms_multi | 271 | 0.853 | 0.877 | 0.737 | -0.140 ± 0.033 | 0.177 | 0.121 | 0.257 | 0.195 | 0.002 | 0.006 | 1 |
| two_doorway | 140 | 0.857 | 0.825 | 0.692 | -0.133 ± 0.054 | 0.202 | 0.160 | 0.300 | 0.174 | 0.015 | 0.008 | 1 |
| walls * | 419 | 0.871 | 0.849 | 0.790 | -0.058 ± 0.021 | 0.099 | 0.143 | 0.207 | 0.123 | 0.009 | 0.003 | 1 |
| walls_multi | 271 | 0.913 | 0.910 | 0.877 | -0.034 ± 0.017 | 0.053 | 0.080 | 0.121 | 0.068 | 0.009 | 0.002 | 1 |

Check: `predict.py` and `evaluate.py` disagree on 0 parse(s) (expected 0).
Check: 0 scene(s) with an error in an ORCA run (expected 0).
