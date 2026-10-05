"""Photo of a drawing sheet -> the world square as a 512 x 512 image, plus the scene id.

    world_img, info = rectify(photo_rgb)          # info: sheet_id, markers_px, homography

Steps: flatten uneven lighting (divide by a heavily blurred copy), binarise
with Otsu (inside the paper only), keep solid four-cornered blobs, try every
set of four whose sizes match the printed markers (one 2.25x the others),
keep the set whose homography puts ink on the printed frame, and warp the
world square out.
The id boxes are then read through the same homography.

numpy and cv2 only. Geometry lives in `sheet.py`.

Invariants:
- Output pixel (row 0, col 0) is world (0, WORLD): the usual image convention,
  so the result feeds `Predictor` exactly like a rendered scene.
- Raises `RectifyError` rather than guessing when four markers are not found.
"""
from __future__ import annotations

import numpy as np
import cv2

import _paths  # noqa: F401
import sheet


class RectifyError(RuntimeError):
    pass


def load_photo(path) -> np.ndarray:
    """A photo file as RGB uint8, turned upright by the EXIF orientation tag phones write."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), np.uint8)


def _odd(n: float) -> int:
    n = int(n)
    return n + 1 if n % 2 == 0 else n


def page_mask(gray: np.ndarray):
    """The sheet of paper, or None when it cannot be told from its surroundings.

    Paper is the largest bright region; marks on it are holes, so the mask is
    the region's convex hull. Without this, the dark desk next to the page edge
    binarises as a thick band of ink that swallows the corner markers.
    """
    small = cv2.GaussianBlur(gray, (0, 0), max(gray.shape) / 400)
    t, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # below Otsu: paper in a shadow is darker than lit paper but still lighter than a dark desk
    _, bright = cv2.threshold(small, 0.75 * t, 255, cv2.THRESH_BINARY)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    frac = stats[k, cv2.CC_STAT_AREA] / gray.size
    if not 0.15 < frac < 0.97:                       # no clear paper/desk contrast: do not mask
        return None
    cs, _ = cv2.findContours((cc == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [cv2.convexHull(max(cs, key=cv2.contourArea))], 255)
    edge = _odd(max(gray.shape) / 250)
    return cv2.erode(mask, np.ones((edge, edge), np.uint8))


def binarise(photo: np.ndarray, use_page_mask: bool = True) -> np.ndarray:
    """Ink = 255, inside the page only. Illumination is divided out first so
    shadows do not become ink."""
    gray = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY) if photo.ndim == 3 else photo
    k = _odd(max(gray.shape) / 12)
    bg = cv2.GaussianBlur(cv2.dilate(gray, np.ones((15, 15), np.uint8)), (k, k), 0)
    flat = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    mask = page_mask(gray) if use_page_mask else None
    if mask is not None:
        flat = np.where(mask > 0, flat, 255).astype(np.uint8)
    _, bw = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def _marker_candidates(bw: np.ndarray) -> list:
    """[(area, (cx, cy))] for every solid four-cornered blob.

    RETR_LIST, not RETR_EXTERNAL, so marks nested in other contours are seen.
    The shape tests survive perspective: a square seen at an angle is still a
    convex quadrilateral whose area its four-corner approximation reproduces,
    while a filled circle forced to four corners loses about a third (2/pi).
    """
    min_area = (max(bw.shape) / 150) ** 2
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        area = cv2.contourArea(c)
        if area < min_area or area > 0.01 * bw.size:
            continue
        approx = cv2.approxPolyDP(c, 0.05 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        if area / max(cv2.contourArea(approx), 1e-9) > 1.15:
            continue                                          # round, not square
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if w * h <= 0 or not 0.4 < w / h < 2.5:
            continue
        mask = np.zeros(bw.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        if bw[mask > 0].mean() / 255 < 0.85:                 # hollow: frame, id box, inner edges
            continue
        m = cv2.moments(c)
        out.append((area, (m["m10"] / m["m00"], m["m01"] / m["m00"])))
    return out


def _order(four) -> dict:
    """Largest is top-left; the rest clockwise in the image (y down)."""
    tl = np.array(four[0][1])
    rest = [np.array(p) for _, p in four[1:]]
    centre = (tl + sum(rest)) / 4
    ang0 = np.arctan2(*(tl - centre)[::-1])
    rest.sort(key=lambda p: (np.arctan2(*(p - centre)[::-1]) - ang0) % (2 * np.pi))
    tr, br, bl = rest
    return {"tl": tuple(tl), "tr": tuple(tr), "br": tuple(br), "bl": tuple(bl)}


def _sample(bw: np.ndarray, H_photo_to_page: np.ndarray, pts_mm) -> np.ndarray:
    """bw values (0/255) at page-mm points, looked up in the photo."""
    px = cv2.perspectiveTransform(np.float32(pts_mm)[None], np.linalg.inv(H_photo_to_page))[0]
    xs, ys = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
    inside = (xs >= 0) & (xs < bw.shape[1]) & (ys >= 0) & (ys < bw.shape[0])
    v = np.zeros(len(px))
    v[inside] = bw[ys[inside], xs[inside]]
    return v


def frame_score(bw: np.ndarray, markers_px: dict) -> float:
    """Fraction of the printed frame found where these markers say it should be."""
    try:
        H = page_homography(markers_px)
    except cv2.error:
        return 0.0
    return float((_sample(bw, H, sheet.frame_points()) > 127).mean())


def find_markers(bw: np.ndarray) -> dict:
    """{'tl','tr','br','bl'} -> (x, y) photo px of the four corner markers.

    Candidate sets must match the printed size pattern (one marker about
    (MARKER_BIG / MARKER_SMALL)^2 = 2.25 times the area of the other three,
    which agree in size, loosely, since perspective shrinks far ones). Each
    set is then *verified*: its homography must put ink on the printed frame.
    The best-verified set wins; below 0.7 the photo is rejected.
    """
    from itertools import combinations
    cands = sorted(_marker_candidates(bw), key=lambda t: -t[0])[:12]
    want = (sheet.MARKER_BIG / sheet.MARKER_SMALL) ** 2
    best, best_score = None, 0.0
    for four in combinations(cands, 4):
        big, rest = four[0], four[1:]
        small = np.array([a for a, _ in rest])
        if small.max() > 2.5 * small.min() or not 0.4 * want < big[0] / small.mean() < 2.5 * want:
            continue
        markers = _order(four)
        quad = np.float32([markers[n] for n in ("tl", "tr", "br", "bl")])
        if (not cv2.isContourConvex(quad.reshape(-1, 1, 2))
                or cv2.contourArea(quad) < 0.1 * bw.size):   # the markers span most of the page
            continue
        score = frame_score(bw, markers)
        if score > best_score:
            best, best_score = markers, score
    if best is None or best_score < 0.7:
        raise RectifyError(f"no set of four corner markers verified against the frame "
                           f"({len(cands)} candidates, best frame score {best_score:.2f}); "
                           "are all four corner squares in the photo?")
    return best


def page_homography(markers_px: dict) -> np.ndarray:
    """3x3 mapping photo px -> page mm."""
    names = ("tl", "tr", "br", "bl")
    src = np.float32([markers_px[n] for n in names])
    dst = np.float32([sheet.marker_centres()[n] for n in names])
    return cv2.getPerspectiveTransform(src, dst)


def world_square_matrix(out: int) -> np.ndarray:
    """3x3 mapping page mm -> output px of the world square (row 0 = top)."""
    s = out / sheet.SQ
    return np.array([[s, 0, -sheet.SQ_X0 * s],
                     [0, -s, (sheet.SQ_Y0 + sheet.SQ) * s],
                     [0, 0, 1]], np.float64)


def read_id(bw: np.ndarray, H_photo_to_page: np.ndarray):
    """Decode the id boxes; None if the parity check fails."""
    bits = []
    for cx, cy in sheet.id_box_centres():
        r = sheet.ID_BOX * 0.3
        pts = [(cx + dx, cy + dy) for dx in np.linspace(-r, r, 5) for dy in np.linspace(-r, r, 5)]
        bits.append(int(_sample(bw, H_photo_to_page, pts).mean() > 127))
    return sheet.decode_id(bits)


def rectify(photo: np.ndarray, out: int = 512):
    """(world image (out, out, 3) uint8, info dict)."""
    photo = np.asarray(photo)
    bw = binarise(photo)
    try:
        markers = find_markers(bw)
    except RectifyError:
        # The page mask keeps the largest bright region. On an image cropped close
        # to the sheet (a scan, a tablet drawing) that is the inside of the frame,
        # and the corner markers outside it get masked away: retry on the whole
        # image. The frame check in find_markers still rejects a wrong set.
        bw = binarise(photo, use_page_mask=False)
        markers = find_markers(bw)
    H = page_homography(markers)
    M = world_square_matrix(out) @ H
    img = cv2.warpPerspective(photo, M, (out, out), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REPLICATE)
    return img, {"sheet_id": read_id(bw, H), "markers_px": markers, "homography": M}
