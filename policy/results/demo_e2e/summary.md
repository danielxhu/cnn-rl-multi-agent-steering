# End-to-end demo: drawings -> e3_both_seed0 (gru, 47,348,908 steps)

| drawing | agents | starts | exits | inner_walls | obstacles | success | collision | static_coll | agent_coll | timeout | steps | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30f1b2b84fe24c50b895aaaa2f5fa22f.png~tplv-a9rns2rl98-image | 6 | 1 | 4 | 0 | 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 91 | 1 starts / 4 exits |
| 823f12a470ef4aae853d3a56a036931f.png~tplv-a9rns2rl98-image | 6 | 3 | 1 | 4 | 1 | 0.17 | 0.83 | 0.83 | 0.00 | 0.00 | 231 | 3 starts / 1 exits |
| a7e14ce9a34245409e1fcf136b2ed2f0.png~tplv-a9rns2rl98-image | 7 | 2 | 2 | 6 | 3 | 0.43 | 0.57 | 0.14 | 0.43 | 0.00 | 110 | 2 starts / 2 exits; agent 1 starts touching an agent; agent 2 starts touching an agent; agent 3 starts touching an agent |
| f2e8b7520a844db4aa5c80d5765f467c.png~tplv-a9rns2rl98-image | 3 | 1 | 2 | 3 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 42 | 1 starts / 2 exits |
| IMG_046D71A32042-1 | 3 | 1 | 2 | 2 | 0 | 0.67 | 0.33 | 0.33 | 0.00 | 0.00 | 45 | 1 starts / 2 exits |
| IMG_1B879CED3B57-1 | 5 | 1 | 3 | 1 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 82 | 1 starts / 3 exits |
| IMG_5842BA104D41-1 | 3 | 1 | 3 | 3 | 2 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 66 | 1 starts / 3 exits |
| IMG_A3D4108B5547-1 | 3 | 1 | 1 | 0 | 3 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 73 |  |
| IMG_C12B4A92FE16-1 | 4 | 1 | 3 | 1 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 46 | 1 starts / 3 exits |

- success, pooled over 40 agents: 0.750
- collision, pooled over 40 agents: 0.250
- timeout, pooled over 40 agents: 0.000
