# Drawings from the advisor meeting (19 Oct 2026)

Five drawings parsed live with the v6 model (`scripts/demo.py`), one of them
(`IMG_A3D4108B5547-1`) drawn by the advisor. The original photos were not
kept; `input/` holds the photo panel cut out of each demo sheet (about 600 px
tall, corner markers included), which still rectifies and parses as the
original did. `v6_<name>.png` = photo | overlay | clean re-rendering.

| Drawing | v6 result |
|---|---|
| 046D | all correct |
| 1B87 | a 1-unit wall stub where an agent's tail crosses the S box side |
| 5842 | slanted room wall broken (a false door); two wall stubs on the S box |
| C12B | the long free-standing wall lost (its middle read as a narrow G box); the small G box read as an obstacle |
| A3D4 (advisor) | 2 of 3 agents (tail drawn through the circle); two sides of the tilted S box read as walls |

Advisor's remarks: agents need not be drawn inside a start box, and lines
(walls, boxes) may be slanted. Diagnostic only, as the other real drawings.
