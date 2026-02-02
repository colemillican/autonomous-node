# src/zones.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional

Point = Tuple[float, float]          # (x, y) normalized in [0..1]
PointPx = Tuple[int, int]            # (x, y) in pixels


@dataclass(frozen=True)
class Zone:
    name: str
    polygon_norm: List[Point]        # polygon vertices in normalized coords (0..1)


def normalize_point(cx_px: float, cy_px: float, w: int, h: int) -> Point:
    """Convert pixel point -> normalized point."""
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid frame size w={w}, h={h}")
    return (cx_px / float(w), cy_px / float(h))


def polygon_norm_to_px(poly_norm: Iterable[Point], w: int, h: int) -> List[PointPx]:
    """Convert normalized polygon -> pixel polygon for drawing overlays."""
    pts: List[PointPx] = []
    for x, y in poly_norm:
        pts.append((int(round(x * w)), int(round(y * h))))
    return pts


def point_in_polygon(p: Point, poly: List[Point]) -> bool:
    """
    Ray-casting point-in-polygon test (deterministic, fast).
    p: normalized (x,y)
    poly: list of normalized vertices
    Returns True if point is inside polygon (including edges is treated as inside-ish).
    """
    x, y = p
    inside = False

    n = len(poly)
    if n < 3:
        return False

    # iterate edges (xi, yi) -> (xj, yj)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]

        # Check if edge crosses horizontal ray at y
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-12) + xi
        )
        if intersects:
            inside = not inside

        j = i

    return inside


def build_default_zones() -> List[Zone]:
    """
    Day 6: start with ONE simple zone.
    You will tweak these polygon points after you see them drawn on the video.
    """
    return [
        Zone(
            name="zone_a",
            polygon_norm=[
                (0.10, 0.20),
                (0.45, 0.20),
                (0.45, 0.80),
                (0.10, 0.80),
            ],
        )
    ]


def compute_zone_hits(
    objects_px: Dict[int, Tuple[int, int]],
    w: int,
    h: int,
    zones: List[Zone],
) -> Dict[int, Set[str]]:
    """
    For each tracked ID, determine which zones its centroid is inside.
    Returns: {id: {"zone_a", ...}, ...}
    """
    hits: Dict[int, Set[str]] = {}
    for obj_id, (cx, cy) in objects_px.items():
        p_norm = normalize_point(cx, cy, w, h)
        in_zones: Set[str] = set()
        for z in zones:
            if point_in_polygon(p_norm, z.polygon_norm):
                in_zones.add(z.name)
        hits[obj_id] = in_zones
    return hits


# ---- quick self-test (no camera required) ----
def _self_test() -> None:
    zones = build_default_zones()
    w, h = 640, 480

    # Pick a point that should be inside zone_a based on default polygon
    inside_px = (int(0.20 * w), int(0.50 * h))
    outside_px = (int(0.80 * w), int(0.50 * h))

    objects = {1: inside_px, 2: outside_px}
    hits = compute_zone_hits(objects, w, h, zones)

    print("Zones:", [z.name for z in zones])
    print("Hits:", hits)

    assert "zone_a" in hits[1], "ID 1 should be inside zone_a"
    assert "zone_a" not in hits[2], "ID 2 should be outside zone_a"
    print("zones.py self-test PASSED")


if __name__ == "__main__":
    _self_test()
