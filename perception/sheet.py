"""Printed drawing-sheet geometry, shared by the sheet generator and the rectifier.

A hand-drawn scene lives on an A4 page.  The world square (100 x 100 units) is
printed as a frame; four filled squares outside it locate the page in a photo,
and a row of small boxes below it encodes the scene id.  Everything here is in
millimetres on the page, origin at the page's bottom-left corner, y up (the
same orientation as the world, so there is no flip between world and page).

Invariants:
- `world_to_page` / `page_to_world` are the only conversion between the two.
- The top-left marker is larger than the other three; that is how the
  rectifier knows which way up the photo is.
- Every marker and id box lies outside the world square, so rectifying to the
  square crops them away.
"""
from __future__ import annotations

PAGE_W, PAGE_H = 210.0, 297.0          # A4 portrait, mm
SQ = 160.0                             # world square side on paper, mm (1.6 mm per world unit)
SQ_X0, SQ_Y0 = 25.0, 100.0             # bottom-left corner of the world square
WORLD = 100.0                          # world side, units (sim Config.world)
FRAME_BAND = 1.5 * SQ / WORLD          # printed boundary wall inside the square: half of the
                                       # sim wall thickness (3.0 units), 2.4 mm

MARKER_OFFSET = 12.0                   # marker centre, diagonally outside each square corner
MARKER_BIG, MARKER_SMALL = 12.0, 8.0   # side lengths; the big one is top-left

ID_BITS = 7                            # scene ids 0..127
ID_BOX = 5.0                           # id box side
ID_GAP = 2.0
ID_Y = 92.0                            # id box centres, between the square and the bottom markers


def world_to_page(x, y):
    s = SQ / WORLD
    return SQ_X0 + x * s, SQ_Y0 + y * s


def page_to_world(px, py):
    s = SQ / WORLD
    return (px - SQ_X0) / s, (py - SQ_Y0) / s


def marker_centres() -> dict:
    """{'tl', 'tr', 'br', 'bl'} -> (x, y) page mm."""
    o = MARKER_OFFSET
    x0, x1, y0, y1 = SQ_X0, SQ_X0 + SQ, SQ_Y0, SQ_Y0 + SQ
    return {"tl": (x0 - o, y1 + o), "tr": (x1 + o, y1 + o),
            "br": (x1 + o, y0 - o), "bl": (x0 - o, y0 - o)}


def marker_size(name: str) -> float:
    return MARKER_BIG if name == "tl" else MARKER_SMALL


def id_box_centres() -> list:
    """Centres of the ID_BITS + 1 boxes (data bits, then one even-parity bit), page mm."""
    n = ID_BITS + 1
    width = n * ID_BOX + (n - 1) * ID_GAP
    x = PAGE_W / 2 - width / 2 + ID_BOX / 2
    return [(x + i * (ID_BOX + ID_GAP), ID_Y) for i in range(n)]


def encode_id(sid: int) -> list:
    """Bits for the boxes: ID_BITS data bits (MSB first) + an even-parity bit."""
    if not 0 <= sid < 2 ** ID_BITS:
        raise ValueError(f"scene id {sid} does not fit in {ID_BITS} bits")
    bits = [(sid >> (ID_BITS - 1 - i)) & 1 for i in range(ID_BITS)]
    return bits + [sum(bits) % 2]


def decode_id(bits: list):
    """Inverse of `encode_id`; None when the parity check fails."""
    data, parity = bits[:ID_BITS], bits[ID_BITS]
    if sum(data) % 2 != parity:
        return None
    v = 0
    for b in data:
        v = (v << 1) | int(b)
    return v


def frame_points(n_per_side: int = 15) -> list:
    """Points on the centre line of the printed frame band, page mm."""
    c = FRAME_BAND / 2
    x0, x1, y0, y1 = SQ_X0 + c, SQ_X0 + SQ - c, SQ_Y0 + c, SQ_Y0 + SQ - c
    t = [(i + 0.5) / n_per_side for i in range(n_per_side)]
    return ([(x0 + f * (x1 - x0), y0) for f in t] + [(x0 + f * (x1 - x0), y1) for f in t]
            + [(x0, y0 + f * (y1 - y0)) for f in t] + [(x1, y0 + f * (y1 - y0)) for f in t])
