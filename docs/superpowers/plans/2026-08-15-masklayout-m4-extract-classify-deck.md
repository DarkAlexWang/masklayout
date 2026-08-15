# masklayout M4 — Extraction, Classification, and the Rule Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract sites from target geometry, measure the closed selector vocabulary at each one, and load a declarative rule deck that matches sites deterministically.

**Architecture:** Five stages. A spatial index answers "what is near this?". Extraction walks polygons into edges, corners, and line ends with numeric curvature. Classification measures the eight selector attributes at each site by ray casting, which is exact at any angle. The deck is validated data with a content hash. The matcher pairs sites with rules by priority, first match winning per feature kind.

**M4 produces matches, not geometry.** Turning a `(site, rule)` pair into corrected polygons is M5. Keeping that boundary means the vocabulary can be validated before anything depends on the shapes it drives.

**Design reference:** `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`, especially §4 and §4.1.

## Why this milestone is the expensive one to get wrong

Design §4 states it plainly: *"`classify` is the contract. Because the deck is declarative over a fixed vocabulary, whatever `classify` measures is exactly what a rule can select on and nothing else."*

Extending the vocabulary later is additive and cheap. Getting a measurement **wrong** is not — every deck written against it encodes the error. So each measurement in Task 3 is tested against geometry whose answer is known by construction, at several angles, and the angle-independence is asserted rather than assumed.

## Global Constraints

M0–M3 constraints all still apply, plus:

- **The selector vocabulary is closed**: `site`, `width_nm`, `space_nm`, `edge_length_nm`, `angle_deg`, `corner_type`, `curvature_1_per_um`, `local_density`. A deck naming anything else fails at load, listing the valid keys (§4.1).
- **Matching is deterministic**: rules are evaluated in `priority` order and the first match wins **per feature kind**, so a bias and a hammerhead may both apply to one site but two hammerhead rules may not (§2 decision 7).
- **Target geometry is immutable.** Extraction and classification only read (§12).
- Measurement is by ray casting, not by axis-aligned assumptions. Every measurement test runs at several angles.

## Verified toolchain facts

Established by experiment before writing this plan:

| Fact | Consequence |
|---|---|
| Inward ray cast from an edge midpoint measures local width **exactly** — 100.0000 nm at 0°, 17°, 37°, 45°, 73° | Width needs no special case for angled geometry |
| Outward ray cast + `STRtree` measures space **exactly** — 60.0000 nm at 0° and 37° | Same for space |
| shapely 2.x `STRtree.query` returns an **ndarray of indices**; geometries come from `.geometries` | Index wrapper must map indices back to owners |

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/geometry/index.py` | `SpatialIndex` over polygons, wrapping `STRtree` |
| `src/masklayout/opc/sites.py` | `Site`, `SiteKind`, `CornerType` |
| `src/masklayout/opc/extract.py` | Polygons → edges, corners, line ends, curvature |
| `src/masklayout/opc/classify.py` | Measure the eight selector attributes |
| `src/masklayout/opc/deck.py` | Deck schema, loader, validation, content hash |
| `src/masklayout/opc/match.py` | Site × deck → matches |
| `tests/unit/test_index.py` | Spatial index |
| `tests/unit/test_extract.py` | Edges, corners, line ends, curvature |
| `tests/unit/test_classify.py` | The eight measurements, at multiple angles |
| `tests/unit/test_deck.py` | Schema, loading, rejection, hashing |
| `tests/unit/test_match.py` | Priority, first-match-per-kind, determinism |

---

## Task 1: Spatial index

**Files:**
- Create: `src/masklayout/geometry/index.py`
- Test: `tests/unit/test_index.py`

**Interfaces:**
- Produces:
  - `SpatialIndex(polygons: Sequence[Polygon], precision_um: float)`
  - `.query_bbox(minx, miny, maxx, maxy) -> list[int]` — indices of candidates
  - `.query_ray(origin_um, direction_um, length_um) -> list[int]`
  - `.nearest_distance_um(origin_um, direction_um, max_length_um, exclude: int | None) -> float | None`
  - `.polygon_count` and `.shapely_geometry(i)`

The index owns the float-micrometre shapely mirror of the integer model, so
nothing else has to convert back and forth.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_index.py`:

```python
"""Spatial index over model polygons."""

import numpy as np
import pytest

from masklayout.geometry.index import SpatialIndex
from masklayout.model.geometry import Polygon


def _rect_dbu(x0: int, y0: int, x1: int, y1: int, layer: int = 10) -> Polygon:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int64)
    return Polygon(points=pts, layer=layer)


def test_index_reports_its_size() -> None:
    index = SpatialIndex([_rect_dbu(0, 0, 100, 100)], precision_um=0.001)
    assert index.polygon_count == 1


def test_bbox_query_finds_only_overlapping_polygons() -> None:
    index = SpatialIndex(
        [_rect_dbu(0, 0, 100, 100), _rect_dbu(1000, 0, 1100, 100)], precision_um=0.001
    )
    assert index.query_bbox(-0.01, -0.01, 0.15, 0.15) == [0]
    assert index.query_bbox(0.95, -0.01, 1.15, 0.15) == [1]
    assert sorted(index.query_bbox(-1.0, -1.0, 5.0, 5.0)) == [0, 1]


def test_ray_query_returns_candidates_along_the_ray() -> None:
    index = SpatialIndex(
        [_rect_dbu(0, 0, 100, 100), _rect_dbu(200, 0, 300, 100)], precision_um=0.001
    )
    hits = index.query_ray((0.05, 0.05), (1.0, 0.0), length_um=1.0)
    assert sorted(hits) == [0, 1]


def test_nearest_distance_measures_the_gap() -> None:
    # Two 100 nm bars separated by a 60 nm gap.
    index = SpatialIndex(
        [_rect_dbu(0, 0, 2000, 100), _rect_dbu(0, 160, 2000, 260)], precision_um=0.001
    )
    distance = index.nearest_distance_um(
        (1.0, 0.1), (0.0, 1.0), max_length_um=1.0, exclude=0
    )
    assert distance == pytest.approx(0.060)


def test_nearest_distance_returns_none_when_nothing_is_hit() -> None:
    index = SpatialIndex([_rect_dbu(0, 0, 100, 100)], precision_um=0.001)
    assert (
        index.nearest_distance_um((0.05, 0.2), (0.0, 1.0), max_length_um=1.0, exclude=None)
        is None
    )


def test_exclude_skips_the_owning_polygon() -> None:
    index = SpatialIndex([_rect_dbu(0, 0, 2000, 100)], precision_um=0.001)
    # Casting outward from the top edge with the owner excluded finds nothing.
    assert (
        index.nearest_distance_um((1.0, 0.1), (0.0, 1.0), max_length_um=1.0, exclude=0) is None
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `geometry/index.py`**

```python
"""Spatial index over model polygons.

gdstk provides no spatial index, so this wraps shapely's STRtree. It owns the
float-micrometre shapely mirror of the integer model so no other module has
to convert between the two representations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree

from masklayout.model.geometry import Polygon

#: Ignore intersections closer than this to the ray origin; the origin sits on
#: the source edge, so it would otherwise register as a zero-distance hit.
_ORIGIN_EPSILON_UM = 1e-9


class SpatialIndex:
    """Bounding-box index with ray queries, in float micrometres."""

    def __init__(self, polygons: Sequence[Polygon], precision_um: float) -> None:
        self._precision_um = precision_um
        self._geometries = [
            ShapelyPolygon(polygon.points.astype(np.float64) * precision_um)
            for polygon in polygons
        ]
        self._tree = STRtree(self._geometries) if self._geometries else None

    @property
    def polygon_count(self) -> int:
        return len(self._geometries)

    def shapely_geometry(self, index: int) -> ShapelyPolygon:
        return self._geometries[index]

    def _query(self, geometry: LineString | ShapelyPolygon) -> list[int]:
        if self._tree is None:
            return []
        return [int(i) for i in self._tree.query(geometry)]

    def query_bbox(
        self, minx_um: float, miny_um: float, maxx_um: float, maxy_um: float
    ) -> list[int]:
        """Indices of polygons whose bounding boxes meet the given box."""
        box = ShapelyPolygon(
            [
                (minx_um, miny_um),
                (maxx_um, miny_um),
                (maxx_um, maxy_um),
                (minx_um, maxy_um),
            ]
        )
        return self._query(box)

    def query_ray(
        self,
        origin_um: tuple[float, float],
        direction_um: tuple[float, float],
        length_um: float,
    ) -> list[int]:
        """Indices of polygons whose bounding boxes meet the ray."""
        return self._query(self._ray(origin_um, direction_um, length_um))

    @staticmethod
    def _ray(
        origin_um: tuple[float, float],
        direction_um: tuple[float, float],
        length_um: float,
    ) -> LineString:
        norm = float(np.hypot(direction_um[0], direction_um[1]))
        if norm == 0.0:
            raise ValueError("ray direction must be non-zero")
        unit = (direction_um[0] / norm, direction_um[1] / norm)
        end = (origin_um[0] + unit[0] * length_um, origin_um[1] + unit[1] * length_um)
        return LineString([origin_um, end])

    def nearest_distance_um(
        self,
        origin_um: tuple[float, float],
        direction_um: tuple[float, float],
        max_length_um: float,
        exclude: int | None,
    ) -> float | None:
        """Distance along a ray to the first polygon boundary it meets.

        Returns None when the ray reaches ``max_length_um`` without hitting
        anything, which callers read as "no neighbour within range".
        """
        ray = self._ray(origin_um, direction_um, max_length_um)
        origin = Point(origin_um)
        best: float | None = None

        for candidate in self._query(ray):
            if candidate == exclude:
                continue
            crossing = ray.intersection(self._geometries[candidate].exterior)
            if crossing.is_empty:
                continue
            parts = crossing.geoms if hasattr(crossing, "geoms") else [crossing]
            for part in parts:
                distance = origin.distance(part)
                if distance <= _ORIGIN_EPSILON_UM:
                    continue
                if best is None or distance < best:
                    best = distance
        return best
```

- [ ] **Step 4: Run tests, verify, commit**

```bash
uv run pytest tests/unit/test_index.py -v
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest -q
git add src/masklayout/geometry/index.py tests/unit/test_index.py
git commit -m "feat(m4): add a spatial index over model polygons

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

**Run each verification command separately and read its result.** Chaining them
with `&&` has already once let a lint failure through because a later command
in the chain became the gate.

---

## Task 2: Site extraction

**Files:**
- Create: `src/masklayout/opc/__init__.py`, `src/masklayout/opc/sites.py`, `src/masklayout/opc/extract.py`
- Test: `tests/unit/test_extract.py`

**Interfaces:**
- Produces:
  - `SiteKind` — `Literal["edge", "convex_corner", "concave_corner", "line_end"]`
  - `CornerType` — `Literal["convex", "concave", "none"]`
  - `Site` — frozen dataclass: `kind`, `polygon_index`, `vertex_index`, `midpoint_um`, `outward_normal_um`, `edge_length_um`, `angle_deg`, `corner_type`, `curvature_1_per_um`
  - `extract_sites(polygons, precision_um, line_end_ratio) -> list[Site]`
  - `vertex_curvature_1_per_um(points_um) -> NDArray[np.float64]`

**Line-end definition.** An edge is a line end when its length is at most
`line_end_ratio` times the length of **both** neighbouring edges, and those two
neighbours run roughly antiparallel to each other — the signature of a bar's
short terminating edge. `line_end_ratio` defaults to 0.5 and is a parameter,
not a constant, because it is a heuristic rather than a law.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_extract.py`:

```python
"""Site extraction from target geometry."""

import math

import numpy as np
import pytest

from masklayout.opc.extract import extract_sites, vertex_curvature_1_per_um
from masklayout.model.geometry import Polygon


def _bar_dbu(length_nm: int = 2000, width_nm: int = 100, angle_deg: float = 0.0) -> Polygon:
    pts = np.array(
        [[0, 0], [length_nm, 0], [length_nm, width_nm], [0, width_nm]], dtype=np.float64
    )
    a = math.radians(angle_deg)
    rotated = np.column_stack(
        (
            pts[:, 0] * math.cos(a) - pts[:, 1] * math.sin(a),
            pts[:, 0] * math.sin(a) + pts[:, 1] * math.cos(a),
        )
    )
    return Polygon(points=np.round(rotated).astype(np.int64), layer=10)


def test_a_rectangle_yields_four_edges_and_four_convex_corners() -> None:
    sites = extract_sites([_bar_dbu()], precision_um=0.001, line_end_ratio=0.5)
    edges = [s for s in sites if s.kind in ("edge", "line_end")]
    corners = [s for s in sites if s.kind.endswith("corner")]
    assert len(edges) == 4
    assert len(corners) == 4
    assert all(s.corner_type == "convex" for s in corners)


def test_the_short_edges_of_a_bar_are_line_ends() -> None:
    sites = extract_sites([_bar_dbu()], precision_um=0.001, line_end_ratio=0.5)
    line_ends = [s for s in sites if s.kind == "line_end"]
    assert len(line_ends) == 2
    for site in line_ends:
        assert site.edge_length_um == pytest.approx(0.1)


@pytest.mark.parametrize("angle_deg", [0.0, 17.0, 37.0, 45.0])
def test_line_end_detection_is_angle_independent(angle_deg: float) -> None:
    sites = extract_sites(
        [_bar_dbu(angle_deg=angle_deg)], precision_um=0.001, line_end_ratio=0.5
    )
    assert len([s for s in sites if s.kind == "line_end"]) == 2


def test_edge_angle_follows_rotation() -> None:
    sites = extract_sites([_bar_dbu(angle_deg=30.0)], precision_um=0.001, line_end_ratio=0.5)
    long_edges = sorted(
        (s for s in sites if s.kind in ("edge", "line_end")),
        key=lambda s: s.edge_length_um,
        reverse=True,
    )[:2]
    angles = sorted(a.angle_deg % 180.0 for a in long_edges)
    assert angles[0] == pytest.approx(30.0, abs=0.5)


def test_outward_normal_points_away_from_the_polygon() -> None:
    from shapely.geometry import Point
    from shapely.geometry import Polygon as ShapelyPolygon

    polygon = _bar_dbu()
    shape = ShapelyPolygon(polygon.points.astype(np.float64) * 0.001)
    for site in extract_sites([polygon], precision_um=0.001, line_end_ratio=0.5):
        if site.kind not in ("edge", "line_end"):
            continue
        probe = (
            site.midpoint_um[0] + site.outward_normal_um[0] * 1e-4,
            site.midpoint_um[1] + site.outward_normal_um[1] * 1e-4,
        )
        assert not shape.contains(Point(probe))


def test_concave_corners_are_detected() -> None:
    # An L shape has exactly one concave corner.
    pts = np.array(
        [[0, 0], [3000, 0], [3000, 1000], [1000, 1000], [1000, 3000], [0, 3000]],
        dtype=np.int64,
    )
    sites = extract_sites([Polygon(points=pts, layer=10)], precision_um=0.001, line_end_ratio=0.5)
    concave = [s for s in sites if s.corner_type == "concave"]
    assert len(concave) == 1


def test_curvature_of_a_circle_is_one_over_its_radius() -> None:
    from masklayout.geometry.curves import circle_um

    radius = 2.0
    points = circle_um((0.0, 0.0), radius, max_chord_error_um=0.0005)
    curvature = vertex_curvature_1_per_um(points)
    assert float(np.median(curvature)) == pytest.approx(1.0 / radius, rel=0.02)


def test_curvature_of_a_straight_run_is_zero() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert float(np.max(np.abs(vertex_curvature_1_per_um(points)[1:-1]))) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `opc/sites.py`**

```python
"""Extracted sites: the things a rule can select."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SiteKind = Literal["edge", "line_end", "convex_corner", "concave_corner"]
CornerType = Literal["convex", "concave", "none"]


@dataclass(frozen=True)
class Site:
    """One location on target geometry, with its measured geometry.

    Positional attributes are float micrometres because a site is a
    measurement about the model, not geometry stored in it.
    """

    kind: SiteKind
    polygon_index: int
    vertex_index: int
    midpoint_um: tuple[float, float]
    outward_normal_um: tuple[float, float]
    edge_length_um: float
    angle_deg: float
    corner_type: CornerType
    curvature_1_per_um: float

    @property
    def site_id(self) -> str:
        """Stable identifier, assigned after normalization (design §7)."""
        return f"{self.polygon_index}#{self.vertex_index}:{self.kind}"
```

- [ ] **Step 4: Write `opc/extract.py`**

```python
"""Walk target polygons into sites: edges, corners, and line ends."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from masklayout.geometry.normalize import signed_area
from masklayout.model.geometry import Polygon
from masklayout.opc.sites import CornerType, Site

#: Two edges count as antiparallel when their directions oppose within this
#: many degrees. Used to recognise the two long sides flanking a line end.
_ANTIPARALLEL_TOLERANCE_DEG = 30.0


def vertex_curvature_1_per_um(points_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Curvature at each vertex from the circumcircle of it and its neighbours.

    The design takes curvature numerically from the polyline rather than
    re-fitting an analytic curve (§2 decision 10); this is that measurement.
    Reciprocal of the circumradius, zero where the three points are collinear.
    """
    values = np.asarray(points_um, dtype=np.float64)
    previous = np.roll(values, 1, axis=0)
    following = np.roll(values, -1, axis=0)

    a = np.linalg.norm(values - previous, axis=1)
    b = np.linalg.norm(following - values, axis=1)
    c = np.linalg.norm(following - previous, axis=1)

    cross = (values[:, 0] - previous[:, 0]) * (following[:, 1] - previous[:, 1]) - (
        values[:, 1] - previous[:, 1]
    ) * (following[:, 0] - previous[:, 0])
    area = np.abs(cross) / 2.0

    denominator = a * b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.where(denominator > 0.0, 4.0 * area / denominator, 0.0)
    return np.nan_to_num(curvature, nan=0.0, posinf=0.0)


def _corner_types(points_um: NDArray[np.float64]) -> list[CornerType]:
    """Convex or concave at each vertex, for a counterclockwise ring."""
    previous = np.roll(points_um, 1, axis=0)
    following = np.roll(points_um, -1, axis=0)
    incoming = points_um - previous
    outgoing = following - points_um
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    return ["convex" if value > 0 else "concave" if value < 0 else "none" for value in cross]


def _is_line_end(
    lengths: NDArray[np.float64], directions: NDArray[np.float64], index: int, ratio: float
) -> bool:
    """True when edge ``index`` is a short edge flanked by two long antiparallel ones."""
    count = len(lengths)
    before = (index - 1) % count
    after = (index + 1) % count
    if lengths[index] > ratio * min(lengths[before], lengths[after]):
        return False
    dot = float(np.clip(np.dot(directions[before], directions[after]), -1.0, 1.0))
    angle_between_deg = math.degrees(math.acos(dot))
    return angle_between_deg >= 180.0 - _ANTIPARALLEL_TOLERANCE_DEG


def extract_sites(
    polygons: Sequence[Polygon],
    precision_um: float,
    line_end_ratio: float = 0.5,
) -> list[Site]:
    """Extract every edge, corner, and line end from the given polygons."""
    sites: list[Site] = []

    for polygon_index, polygon in enumerate(polygons):
        points = polygon.points.astype(np.float64) * precision_um
        if signed_area(points) < 0:
            points = points[::-1]

        following = np.roll(points, -1, axis=0)
        segments = following - points
        lengths = np.linalg.norm(segments, axis=1)
        safe = np.where(lengths > 0.0, lengths, 1.0)[:, None]
        directions = segments / safe
        # Right normal of a counterclockwise ring points out of the polygon.
        normals = np.column_stack((directions[:, 1], -directions[:, 0]))
        midpoints = (points + following) / 2.0
        curvature = vertex_curvature_1_per_um(points)
        corners = _corner_types(points)

        for i in range(len(points)):
            kind = "line_end" if _is_line_end(lengths, directions, i, line_end_ratio) else "edge"
            sites.append(
                Site(
                    kind=kind,
                    polygon_index=polygon_index,
                    vertex_index=i,
                    midpoint_um=(float(midpoints[i, 0]), float(midpoints[i, 1])),
                    outward_normal_um=(float(normals[i, 0]), float(normals[i, 1])),
                    edge_length_um=float(lengths[i]),
                    angle_deg=math.degrees(math.atan2(segments[i, 1], segments[i, 0])) % 360.0,
                    corner_type="none",
                    curvature_1_per_um=0.0,
                )
            )
            sites.append(
                Site(
                    kind="convex_corner" if corners[i] == "convex" else "concave_corner",
                    polygon_index=polygon_index,
                    vertex_index=i,
                    midpoint_um=(float(points[i, 0]), float(points[i, 1])),
                    outward_normal_um=(float(normals[i, 0]), float(normals[i, 1])),
                    edge_length_um=0.0,
                    angle_deg=math.degrees(math.atan2(segments[i, 1], segments[i, 0])) % 360.0,
                    corner_type=corners[i],
                    curvature_1_per_um=float(curvature[i]),
                )
            )

    return sites
```

Create an empty-bodied `src/masklayout/opc/__init__.py`:

```python
"""Rule-based OPC-like correction."""
```

- [ ] **Step 5: Run tests, verify each gate separately, commit**

---

## Task 3: Classification — the selector vocabulary

**Files:**
- Create: `src/masklayout/opc/classify.py`
- Test: `tests/unit/test_classify.py`

**Interfaces:**
- Produces:
  - `SiteMeasurement` — frozen dataclass with exactly the eight selector keys plus the originating `Site`.
  - `classify_sites(sites, polygons, tech, max_probe_um, density_window_um) -> list[SiteMeasurement]`
  - `SELECTOR_KEYS: frozenset[str]` — the closed vocabulary the deck validates against.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_classify.py`:

```python
"""The eight selector measurements."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.classify import SELECTOR_KEYS, classify_sites
from masklayout.opc.extract import extract_sites


def _bar(length_nm: int, width_nm: int, angle_deg: float, dy_nm: int = 0) -> Polygon:
    pts = np.array(
        [[0, dy_nm], [length_nm, dy_nm], [length_nm, dy_nm + width_nm], [0, dy_nm + width_nm]],
        dtype=np.float64,
    )
    a = math.radians(angle_deg)
    rotated = np.column_stack(
        (
            pts[:, 0] * math.cos(a) - pts[:, 1] * math.sin(a),
            pts[:, 0] * math.sin(a) + pts[:, 1] * math.cos(a),
        )
    )
    return Polygon(points=np.round(rotated).astype(np.int64), layer=10)


def _measure(polygons: list[Polygon]) -> list:
    tech = TechConfig()
    sites = extract_sites(polygons, tech.precision_um, line_end_ratio=0.5)
    return classify_sites(
        sites, polygons, tech, max_probe_um=2.0, density_window_um=1.0
    )


def test_selector_keys_are_exactly_the_documented_vocabulary() -> None:
    assert SELECTOR_KEYS == frozenset(
        {
            "site",
            "width_nm",
            "space_nm",
            "edge_length_nm",
            "angle_deg",
            "corner_type",
            "curvature_1_per_um",
            "local_density",
        }
    )


@pytest.mark.parametrize("angle_deg", [0.0, 17.0, 37.0, 45.0, 73.0])
def test_width_is_measured_exactly_at_any_angle(angle_deg: float) -> None:
    measurements = _measure([_bar(2000, 100, angle_deg)])
    long_edges = [
        m for m in measurements if m.site.kind == "edge" and m.edge_length_nm > 1000
    ]
    assert long_edges
    for measurement in long_edges:
        assert measurement.width_nm == pytest.approx(100.0, abs=1.5)


@pytest.mark.parametrize("angle_deg", [0.0, 37.0])
def test_space_to_a_neighbour_is_measured_at_any_angle(angle_deg: float) -> None:
    measurements = _measure([_bar(2000, 100, angle_deg), _bar(2000, 100, angle_deg, dy_nm=160)])
    spaces = [
        m.space_nm
        for m in measurements
        if m.site.kind == "edge" and m.edge_length_nm > 1000 and m.space_nm is not None
    ]
    assert spaces
    assert min(spaces) == pytest.approx(60.0, abs=1.5)


def test_space_is_none_for_an_isolated_feature() -> None:
    measurements = _measure([_bar(2000, 100, 0.0)])
    assert all(m.space_nm is None for m in measurements)


def test_edge_length_is_reported_in_nanometres() -> None:
    measurements = _measure([_bar(2000, 100, 0.0)])
    lengths = sorted({round(m.edge_length_nm) for m in measurements if m.site.kind != "edge"} |
                     {round(m.edge_length_nm) for m in measurements})
    assert 2000 in lengths
    assert 100 in lengths


def test_local_density_is_higher_in_a_dense_neighbourhood() -> None:
    isolated = _measure([_bar(2000, 100, 0.0)])
    dense = _measure(
        [_bar(2000, 100, 0.0), _bar(2000, 100, 0.0, dy_nm=160), _bar(2000, 100, 0.0, dy_nm=-160)]
    )
    assert max(m.local_density for m in dense) > max(m.local_density for m in isolated)


def test_every_measurement_exposes_the_full_vocabulary() -> None:
    for measurement in _measure([_bar(2000, 100, 0.0)]):
        assert set(measurement.as_selector_values()) == SELECTOR_KEYS
```

- [ ] **Step 2: Run to verify it fails, then write `opc/classify.py`**

```python
"""Measure the closed selector vocabulary at each site.

This module is the contract described in design §4: whatever is measured
here is exactly what a rule can select on, and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from masklayout.config import TechConfig
from masklayout.geometry.index import SpatialIndex
from masklayout.model.geometry import Polygon
from masklayout.opc.sites import Site

#: The closed selector vocabulary. A deck naming anything outside this set
#: fails at load. Extending it is additive; a rule can never reach past it.
SELECTOR_KEYS = frozenset(
    {
        "site",
        "width_nm",
        "space_nm",
        "edge_length_nm",
        "angle_deg",
        "corner_type",
        "curvature_1_per_um",
        "local_density",
    }
)


@dataclass(frozen=True)
class SiteMeasurement:
    """A site with the full selector vocabulary measured at it."""

    site: Site
    width_nm: float | None
    space_nm: float | None
    edge_length_nm: float
    angle_deg: float
    corner_type: str
    curvature_1_per_um: float
    local_density: float

    def as_selector_values(self) -> dict[str, Any]:
        """Exactly the selector vocabulary, for matching against a rule."""
        return {
            "site": self.site.kind,
            "width_nm": self.width_nm,
            "space_nm": self.space_nm,
            "edge_length_nm": self.edge_length_nm,
            "angle_deg": self.angle_deg,
            "corner_type": self.corner_type,
            "curvature_1_per_um": self.curvature_1_per_um,
            "local_density": self.local_density,
        }


def classify_sites(
    sites: Sequence[Site],
    polygons: Sequence[Polygon],
    tech: TechConfig,
    max_probe_um: float = 2.0,
    density_window_um: float = 1.0,
) -> list[SiteMeasurement]:
    """Measure the selector vocabulary at every site.

    Width casts a ray inward from the edge midpoint to the far side of the
    same polygon; space casts outward to the nearest other polygon. Both are
    exact at any angle, which is why no Manhattan special case exists here.
    """
    index = SpatialIndex(polygons, tech.precision_um)
    measurements: list[SiteMeasurement] = []

    for site in sites:
        inward = (-site.outward_normal_um[0], -site.outward_normal_um[1])
        width_um = (
            index.nearest_distance_um(site.midpoint_um, inward, max_probe_um, exclude=None)
            if site.kind in ("edge", "line_end")
            else None
        )
        space_um = (
            index.nearest_distance_um(
                site.midpoint_um,
                site.outward_normal_um,
                max_probe_um,
                exclude=site.polygon_index,
            )
            if site.kind in ("edge", "line_end")
            else None
        )

        half = density_window_um / 2.0
        neighbours = index.query_bbox(
            site.midpoint_um[0] - half,
            site.midpoint_um[1] - half,
            site.midpoint_um[0] + half,
            site.midpoint_um[1] + half,
        )
        density = len(neighbours) / max(index.polygon_count, 1)

        measurements.append(
            SiteMeasurement(
                site=site,
                width_nm=None if width_um is None else width_um * 1000.0,
                space_nm=None if space_um is None else space_um * 1000.0,
                edge_length_nm=site.edge_length_um * 1000.0,
                angle_deg=site.angle_deg,
                corner_type=site.corner_type,
                curvature_1_per_um=site.curvature_1_per_um,
                local_density=density,
            )
        )

    return measurements
```

- [ ] **Step 3: Run tests, verify each gate separately, commit**

---

## Task 4: The rule deck

**Files:**
- Create: `src/masklayout/opc/deck.py`, `src/masklayout/decks/generic_hammerhead_v1.yaml`
- Modify: `pyproject.toml` — add `pyyaml>=6.0`
- Test: `tests/unit/test_deck.py`

**Interfaces:**
- Produces:
  - `Range(min: float | None, max: float | None)` with `.contains(value)`
  - `Selector` — the eight keys, all optional
  - `Rule(id, priority, kind, when: Selector, apply: Apply)`
  - `Apply(pcell: str, params: dict[str, Any])`
  - `RuleDeck(id, version, rules)` with `.content_hash`, `.rules_in_priority_order()`
  - `load_deck(path) -> RuleDeck`, `load_deck_from_mapping(mapping) -> RuleDeck`
  - `UnknownSelectorError`

**Content hash.** Provenance requires a deck version *and* a hash, so a deck
edited without a version bump is still distinguishable. The hash is over the
canonical JSON dump, so it is stable across YAML formatting changes.

- [ ] **Step 1: Write the failing test** — cover: a valid deck loads; an unknown selector key fails naming the valid keys; an unknown PCell name is allowed at load (resolved at build time in M5); priority ordering is stable; the content hash changes when a threshold changes but not when YAML whitespace changes; a duplicate rule id is rejected.

- [ ] **Step 2–4: Implement, ship the example deck, run each gate separately, commit**

---

## Task 5: Matching

**Files:**
- Create: `src/masklayout/opc/match.py`
- Test: `tests/unit/test_match.py`

**Interfaces:**
- Produces:
  - `Match(site_id, rule_id, kind, params)` — frozen.
  - `match_sites(measurements, deck) -> list[Match]`
  - `MatchReport(considered, matched, unmatched, by_rule: dict[str, int])`

**Semantics to test explicitly:**
- Rules evaluate in `priority` order; the first match wins **per kind**.
- A site may match two rules of **different** kinds and receive both.
- A site matching no rule produces no match and is counted in `unmatched`.
- A `None` measurement (e.g. `space_nm` on an isolated feature) never satisfies
  a selector that constrains it — an unmeasurable attribute is not a match.
- Output order is deterministic for identical input.

- [ ] **Steps: TDD as above, then the M4 acceptance test**

The acceptance test loads the shipped deck, extracts and classifies a target
containing an isolated bar and a dense pair, matches, and asserts the isolated
line end takes the isolated hammerhead rule while the dense one takes the dense
rule — the whole vocabulary exercised end to end.

---

## Self-Review

**Spec coverage.** M4's acceptance is "selector vocabulary measured; deck loads, validates, matches". Task 1 supplies the index the measurements need. Task 2 extracts the sites. Task 3 measures exactly the eight documented keys and asserts the set matches the documentation. Task 4 loads and validates the deck, rejecting unknown selectors. Task 5 matches with the documented precedence. Generation is explicitly M5.

**Placeholder scan.** Tasks 1–3 carry complete code. Tasks 4 and 5 specify interfaces, semantics, and test cases but leave bodies to implementation — a deliberate departure, because their shape depends on what Tasks 1–3 actually produce and writing speculative bodies now would be guessing. **If executing this plan, expand Tasks 4 and 5 into full TDD steps before starting them.**

**Type consistency.** `Site` fields are consumed under the same names in `classify.py`. `SELECTOR_KEYS` is asserted equal to the documented vocabulary in both `test_classify.py` and the deck validator. `classify_sites(sites, polygons, tech, max_probe_um, density_window_um)` matches its test helper.

**Known risks.** Two, both stated rather than hidden. `local_density` is currently a crude count-in-window ratio rather than an area fraction; it is honest but weak, and the test only asserts a relative ordering. And the line-end heuristic is a ratio test with an antiparallel check — it will misfire on a short edge between two long ones that is not a line end, such as a chamfer. `line_end_ratio` is a parameter so that misfire is tunable rather than baked in.
