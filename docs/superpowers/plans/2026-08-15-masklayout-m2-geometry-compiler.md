# masklayout M2 — Geometry Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile analytic curves into grid-aligned integer polygons whose deviation from the true curve is measured and bounded.

**Architecture:** Three stages with one direction of flow. `curves.py` tessellates analytic shapes into float micrometre polylines at a chord-error budget. `normalize.py` cleans a polyline — dedup, decollinear, orient, reject invalid. `compile.py` runs tessellate → quantize → normalize and returns a model `Polygon` plus a `TessellationReport` carrying the *measured* deviation.

**Tech Stack:** As M1, plus `shapely` for self-intersection detection.

**Design reference:** `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`.

## Global Constraints

M0 and M1 constraints all still apply, plus:

- **`shapely` joins the runtime dependencies.** gdstk has no validity check; shapely's `is_valid` is the only one available. It does **not** join the gdstk allowlist — shapely is not gdstk.
- Tessellation happens in float micrometres. **Quantization to `int64` DBU is the last step before a model `Polygon` exists**, and nothing downstream sees a float coordinate.
- Chord error is **measured and reported**, never assumed.

## The two error terms, and why the budget is not just `max_chord_error_nm`

Deviation of the final integer polygon from the true analytic curve has two independent sources:

1. **Tessellation error** — replacing a curve with chords. Bounded by `max_chord_error_nm`; gdstk's `tolerance` enforces it as a true upper bound (verified: measured/tolerance ≤ 0.999 across radii and tolerances).
2. **Grid error** — snapping each tessellated vertex to the design grid. Bounded by half the grid diagonal, i.e. `design_grid_nm * sqrt(2) / 2`.

They compose. The honest acceptance bound on the finished polygon is therefore:

```
total_deviation_nm  <=  max_chord_error_nm + design_grid_nm * sqrt(2) / 2
```

A test that asserted `total <= max_chord_error_nm` would be wrong and would fail intermittently as geometry moved relative to the grid. `TessellationReport` reports both terms separately so neither hides inside the other.

## Verified toolchain facts

| Fact | Consequence |
|---|---|
| gdstk `tolerance` is max chord error in µm, honoured as an upper bound | `tolerance = max_chord_error_nm / 1000` |
| Honoured for `Curve.bezier` and `RobustPath.arc` alike | One tessellation budget covers all curve kinds |
| gdstk `Polygon` has **no** simplify / collinear-removal / validity method | `normalize.py` is ours; validity comes from shapely |
| `Polygon.area` and `.perimeter` exist | Reuse for min-area cleanup rather than reimplementing |

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/geometry/normalize.py` | Polyline cleanup on float or integer points |
| `src/masklayout/geometry/curves.py` | Analytic shape → float µm polyline |
| `src/masklayout/geometry/compile.py` | Tessellate → quantize → normalize → model `Polygon` |
| `src/masklayout/geometry/report.py` | `TessellationReport` |
| `tests/unit/test_normalize.py` | Cleanup behaviour |
| `tests/unit/test_curves.py` | Tessellation and chord error |
| `tests/unit/test_compile.py` | End-to-end compile and grid alignment |

---

## Task 1: Polyline normalization

**Files:**
- Create: `src/masklayout/geometry/normalize.py`
- Modify: `pyproject.toml` — add `shapely>=2.0`
- Test: `tests/unit/test_normalize.py`

**Interfaces:**
- Produces:
  - `drop_duplicate_points(points, tolerance) -> NDArray`
  - `drop_collinear_points(points, tolerance) -> NDArray`
  - `signed_area(points) -> float`
  - `orient_counterclockwise(points) -> NDArray`
  - `is_simple(points) -> bool` — shapely-backed self-intersection test
  - `normalize_polyline(points, *, duplicate_tolerance, collinear_tolerance) -> NDArray`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_normalize.py`:

```python
"""Polyline normalization."""

import numpy as np
import pytest

from masklayout.geometry.normalize import (
    drop_collinear_points,
    drop_duplicate_points,
    is_simple,
    normalize_polyline,
    orient_counterclockwise,
    signed_area,
)


def test_drop_duplicate_points_removes_consecutive_repeats() -> None:
    pts = np.array([[0, 0], [0, 0], [10, 0], [10, 10], [10, 10]], dtype=np.float64)
    assert drop_duplicate_points(pts, tolerance=1e-9).tolist() == [[0, 0], [10, 0], [10, 10]]


def test_drop_duplicate_points_closes_the_wrap_around() -> None:
    # Last point coincident with first: a closed ring should not repeat it.
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 0]], dtype=np.float64)
    assert len(drop_duplicate_points(pts, tolerance=1e-9)) == 3


def test_drop_collinear_points_removes_a_midpoint() -> None:
    pts = np.array([[0, 0], [5, 0], [10, 0], [10, 10]], dtype=np.float64)
    result = drop_collinear_points(pts, tolerance=1e-9)
    assert result.tolist() == [[0, 0], [10, 0], [10, 10]]


def test_drop_collinear_points_keeps_a_genuine_corner() -> None:
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    assert len(drop_collinear_points(pts, tolerance=1e-9)) == 4


def test_signed_area_sign_follows_winding() -> None:
    ccw = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    assert signed_area(ccw) > 0
    assert signed_area(ccw[::-1]) < 0
    assert abs(signed_area(ccw)) == pytest.approx(100.0)


def test_orient_counterclockwise_is_idempotent_and_corrects_clockwise() -> None:
    ccw = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    assert np.array_equal(orient_counterclockwise(ccw), ccw)
    assert signed_area(orient_counterclockwise(ccw[::-1])) > 0


def test_is_simple_detects_a_bowtie() -> None:
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
    bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10]], dtype=np.float64)
    assert is_simple(square)
    assert not is_simple(bowtie)


def test_normalize_polyline_applies_every_stage() -> None:
    messy = np.array(
        [[0, 0], [0, 0], [5, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64
    )[::-1]
    result = normalize_polyline(messy, duplicate_tolerance=1e-9, collinear_tolerance=1e-9)
    assert result.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert signed_area(result) > 0


def test_normalize_polyline_rejects_a_degenerate_ring() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        normalize_polyline(
            np.array([[0, 0], [1, 1]], dtype=np.float64),
            duplicate_tolerance=1e-9,
            collinear_tolerance=1e-9,
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'masklayout.geometry.normalize'`.

- [ ] **Step 3: Add shapely**

In `pyproject.toml`, add to `dependencies`:

```toml
    "shapely>=2.0",
```

Then run `uv sync`.

Shapely ships no `py.typed` marker, so mypy strict needs an override. Add to `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["shapely.*"]
ignore_missing_imports = true
```

This override is now correct to add — `normalize.py` imports shapely in this task, so it is not an unused config section.

- [ ] **Step 4: Write `geometry/normalize.py`**

```python
"""Polyline cleanup.

gdstk provides no simplification, collinear removal, or validity check, so
these are implemented here. Functions take and return (N, 2) arrays and do
not care whether the dtype is float micrometres or integer DBU, except
``is_simple`` which is float-only by way of shapely.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon as ShapelyPolygon

MIN_RING_VERTICES = 3


def drop_duplicate_points(
    points: NDArray[np.floating] | NDArray[np.integer], tolerance: float
) -> NDArray[np.floating] | NDArray[np.integer]:
    """Remove consecutive coincident points, treating the ring as closed."""
    if len(points) < 2:
        return points
    following = np.roll(points, -1, axis=0)
    distance = np.linalg.norm((following - points).astype(np.float64), axis=1)
    return points[distance > tolerance]


def drop_collinear_points(
    points: NDArray[np.floating] | NDArray[np.integer], tolerance: float
) -> NDArray[np.floating] | NDArray[np.integer]:
    """Remove points whose deviation from the neighbouring chord is negligible.

    The test is the perpendicular distance from each point to the line through
    its neighbours, so ``tolerance`` is a real distance in the input's units.
    """
    if len(points) < MIN_RING_VERTICES:
        return points
    previous = np.roll(points, 1, axis=0).astype(np.float64)
    current = points.astype(np.float64)
    following = np.roll(points, -1, axis=0).astype(np.float64)

    chord = following - previous
    chord_length = np.linalg.norm(chord, axis=1)
    cross = np.abs(
        chord[:, 0] * (current[:, 1] - previous[:, 1])
        - chord[:, 1] * (current[:, 0] - previous[:, 0])
    )
    # Where the neighbours coincide, fall back to the raw offset distance.
    safe_length = np.where(chord_length > 0, chord_length, 1.0)
    deviation = np.where(
        chord_length > 0,
        cross / safe_length,
        np.linalg.norm(current - previous, axis=1),
    )
    return points[deviation > tolerance]


def signed_area(points: NDArray[np.floating] | NDArray[np.integer]) -> float:
    """Twice-signed-area / 2 by the shoelace formula. Positive is CCW."""
    values = points.astype(np.float64)
    following = np.roll(values, -1, axis=0)
    return float(
        np.sum(values[:, 0] * following[:, 1] - following[:, 0] * values[:, 1]) / 2.0
    )


def orient_counterclockwise(
    points: NDArray[np.floating] | NDArray[np.integer],
) -> NDArray[np.floating] | NDArray[np.integer]:
    """Return the ring wound counterclockwise, leaving CCW input untouched."""
    return points if signed_area(points) >= 0 else points[::-1]


def is_simple(points: NDArray[np.floating] | NDArray[np.integer]) -> bool:
    """True when the ring does not self-intersect."""
    if len(points) < MIN_RING_VERTICES:
        return False
    return bool(ShapelyPolygon(points.astype(np.float64)).is_valid)


def normalize_polyline(
    points: NDArray[np.floating] | NDArray[np.integer],
    *,
    duplicate_tolerance: float,
    collinear_tolerance: float,
) -> NDArray[np.floating] | NDArray[np.integer]:
    """Dedup, remove collinear points, and orient counterclockwise."""
    result = drop_duplicate_points(points, duplicate_tolerance)
    result = drop_collinear_points(result, collinear_tolerance)
    if len(result) < MIN_RING_VERTICES:
        raise ValueError(
            f"polyline collapsed to {len(result)} vertices; a ring needs "
            f"at least {MIN_RING_VERTICES}"
        )
    return orient_counterclockwise(result)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_normalize.py -v`
Expected: all 9 PASS.

- [ ] **Step 6: Verify and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src tests && uv run pytest -q
git add pyproject.toml uv.lock src/masklayout/geometry/normalize.py tests/unit/test_normalize.py
git commit -m "feat(m2): add polyline normalization and take on shapely

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 2: Curve tessellation with a chord-error budget

**Files:**
- Create: `src/masklayout/geometry/curves.py`
- Test: `tests/unit/test_curves.py`

**Interfaces:**
- Consumes: `TechConfig.max_chord_error_nm`.
- Produces, each returning float micrometre `(N, 2)` arrays:
  - `circle_um(centre_um, radius_um, max_chord_error_um) -> NDArray[np.float64]`
  - `arc_um(centre_um, radius_um, start_rad, end_rad, max_chord_error_um) -> NDArray[np.float64]`
  - `bezier_um(control_points_um, max_chord_error_um) -> NDArray[np.float64]`
  - `rounded_rect_um(lower_um, upper_um, radius_um, max_chord_error_um) -> NDArray[np.float64]`
  - `measure_chord_error_um(points_um, exact_fn) -> float`

**Note:** these live in `curves.py`, which must **not** import gdstk — it is not on the allowlist. Tessellation is computed directly, which also keeps the vertex count formula explicit and testable.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_curves.py`:

```python
"""Curve tessellation within a chord-error budget."""

import math

import numpy as np
import pytest

from masklayout.geometry.curves import arc_um, bezier_um, circle_um, rounded_rect_um


def _max_radial_error(points: np.ndarray, centre: tuple[float, float], radius: float) -> float:
    """Largest sagitta between the polyline and the true circle."""
    shifted = points - np.array(centre, dtype=np.float64)
    midpoints = (shifted + np.roll(shifted, -1, axis=0)) / 2.0
    return float(np.max(radius - np.linalg.norm(midpoints, axis=1)))


@pytest.mark.parametrize("radius_um", [0.05, 1.0, 10.0])
@pytest.mark.parametrize("budget_um", [0.01, 0.001])
def test_circle_never_exceeds_the_chord_error_budget(radius_um: float, budget_um: float) -> None:
    pts = circle_um((0.0, 0.0), radius_um, max_chord_error_um=budget_um)
    assert _max_radial_error(pts, (0.0, 0.0), radius_um) <= budget_um


def test_tighter_budget_produces_more_vertices() -> None:
    coarse = circle_um((0.0, 0.0), 1.0, max_chord_error_um=0.01)
    fine = circle_um((0.0, 0.0), 1.0, max_chord_error_um=0.0001)
    assert len(fine) > len(coarse)


def test_circle_is_closed_without_repeating_the_first_point() -> None:
    pts = circle_um((0.0, 0.0), 1.0, max_chord_error_um=0.001)
    assert not np.allclose(pts[0], pts[-1])


def test_arc_spans_exactly_the_requested_angles() -> None:
    pts = arc_um((0.0, 0.0), 2.0, 0.0, math.pi / 2, max_chord_error_um=0.001)
    assert pts[0] == pytest.approx([2.0, 0.0], abs=1e-9)
    assert pts[-1] == pytest.approx([0.0, 2.0], abs=1e-9)


def test_arc_error_is_bounded() -> None:
    pts = arc_um((0.0, 0.0), 5.0, 0.0, math.pi, max_chord_error_um=0.002)
    shifted = pts
    mids = (shifted + np.roll(shifted, -1, axis=0))[:-1] / 2.0
    assert float(np.max(5.0 - np.linalg.norm(mids, axis=1))) <= 0.002


def test_bezier_endpoints_are_exact() -> None:
    controls = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -2.0], [4.0, 0.0]])
    pts = bezier_um(controls, max_chord_error_um=0.001)
    assert pts[0] == pytest.approx([0.0, 0.0])
    assert pts[-1] == pytest.approx([4.0, 0.0])


def test_bezier_refines_with_a_tighter_budget() -> None:
    controls = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, -2.0], [4.0, 0.0]])
    assert len(bezier_um(controls, 0.0001)) > len(bezier_um(controls, 0.01))


def test_rounded_rect_corner_radius_is_respected() -> None:
    pts = rounded_rect_um((0.0, 0.0), (10.0, 6.0), radius_um=1.0, max_chord_error_um=0.001)
    lows = pts.min(axis=0)
    highs = pts.max(axis=0)
    assert lows == pytest.approx([0.0, 0.0], abs=1e-6)
    assert highs == pytest.approx([10.0, 6.0], abs=1e-6)
    # No vertex may sit in the square corner cut away by the radius.
    corner = np.logical_and(pts[:, 0] < 1e-9, pts[:, 1] < 1e-9)
    assert not corner.any()


def test_rounded_rect_rejects_a_radius_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="radius"):
        rounded_rect_um((0.0, 0.0), (2.0, 2.0), radius_um=5.0, max_chord_error_um=0.001)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_curves.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `geometry/curves.py`**

```python
"""Analytic curve tessellation.

Every function returns float micrometre points. Quantization to the design
grid happens later, in ``compile.py`` — see the design document, section
"Units and coordinate model".

The vertex count comes from inverting the sagitta relation. For a circular
arc of radius r split into segments subtending angle t, the chord's maximum
deviation from the arc is r * (1 - cos(t / 2)). Solving for t at a budget e
gives t = 2 * arccos(1 - e / r), and the segment count is the arc span
divided by t, rounded up.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

#: Never emit a closed curve coarser than this, regardless of budget.
_MIN_CIRCLE_SEGMENTS = 8


def _segments_for_arc(radius_um: float, span_rad: float, max_chord_error_um: float) -> int:
    """Segment count that keeps the sagitta within budget."""
    if radius_um <= 0.0:
        raise ValueError(f"radius must be positive, got {radius_um}")
    if max_chord_error_um <= 0.0:
        raise ValueError(f"max_chord_error_um must be positive, got {max_chord_error_um}")
    if max_chord_error_um >= radius_um:
        return _MIN_CIRCLE_SEGMENTS
    step = 2.0 * math.acos(1.0 - max_chord_error_um / radius_um)
    return max(int(math.ceil(abs(span_rad) / step)), 1)


def arc_um(
    centre_um: tuple[float, float],
    radius_um: float,
    start_rad: float,
    end_rad: float,
    max_chord_error_um: float,
) -> NDArray[np.float64]:
    """Tessellate a circular arc. Both endpoints are exact."""
    span = end_rad - start_rad
    count = _segments_for_arc(radius_um, span, max_chord_error_um)
    angles = np.linspace(start_rad, end_rad, count + 1, dtype=np.float64)
    return np.column_stack(
        (
            centre_um[0] + radius_um * np.cos(angles),
            centre_um[1] + radius_um * np.sin(angles),
        )
    )


def circle_um(
    centre_um: tuple[float, float], radius_um: float, max_chord_error_um: float
) -> NDArray[np.float64]:
    """Tessellate a full circle as a closed ring with no repeated vertex."""
    count = max(
        _segments_for_arc(radius_um, 2.0 * math.pi, max_chord_error_um),
        _MIN_CIRCLE_SEGMENTS,
    )
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False, dtype=np.float64)
    return np.column_stack(
        (
            centre_um[0] + radius_um * np.cos(angles),
            centre_um[1] + radius_um * np.sin(angles),
        )
    )


def _bezier_at(controls: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
    """De Casteljau evaluation for an arbitrary-degree Bezier."""
    points = controls.astype(np.float64)[:, None, :].repeat(len(t), axis=1)
    for _ in range(len(controls) - 1):
        points = points[:-1] * (1.0 - t)[None, :, None] + points[1:] * t[None, :, None]
    return points[0]


def bezier_um(
    control_points_um: NDArray[np.float64], max_chord_error_um: float
) -> NDArray[np.float64]:
    """Tessellate a Bezier curve by refining until the budget is met.

    The deviation of a chord from the curve is estimated by evaluating the
    curve at each chord's midpoint parameter and measuring the distance to
    the chord midpoint. Sampling doubles until the worst case is in budget.
    """
    if max_chord_error_um <= 0.0:
        raise ValueError(f"max_chord_error_um must be positive, got {max_chord_error_um}")
    controls = np.asarray(control_points_um, dtype=np.float64)
    if len(controls) < 2:
        raise ValueError(f"a Bezier needs at least 2 control points, got {len(controls)}")

    count = 8
    for _ in range(16):
        t = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
        points = _bezier_at(controls, t)
        chord_mid = (points[:-1] + points[1:]) / 2.0
        curve_mid = _bezier_at(controls, (t[:-1] + t[1:]) / 2.0)
        if float(np.max(np.linalg.norm(curve_mid - chord_mid, axis=1))) <= max_chord_error_um:
            return points
        count *= 2
    return points


def rounded_rect_um(
    lower_um: tuple[float, float],
    upper_um: tuple[float, float],
    radius_um: float,
    max_chord_error_um: float,
) -> NDArray[np.float64]:
    """An axis-aligned rectangle with circular corner fillets."""
    width = upper_um[0] - lower_um[0]
    height = upper_um[1] - lower_um[1]
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"degenerate rectangle {lower_um} to {upper_um}")
    if radius_um <= 0.0:
        raise ValueError(f"radius must be positive, got {radius_um}")
    if 2.0 * radius_um > min(width, height):
        raise ValueError(
            f"corner radius {radius_um} does not fit in a {width} x {height} rectangle"
        )

    left, bottom = lower_um
    right, top = upper_um
    corners = [
        ((right - radius_um, bottom + radius_um), -math.pi / 2, 0.0),
        ((right - radius_um, top - radius_um), 0.0, math.pi / 2),
        ((left + radius_um, top - radius_um), math.pi / 2, math.pi),
        ((left + radius_um, bottom + radius_um), math.pi, 3.0 * math.pi / 2),
    ]
    pieces = [
        arc_um(centre, radius_um, start, end, max_chord_error_um)
        for centre, start, end in corners
    ]
    return np.vstack(pieces)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_curves.py -v`
Expected: all PASS, including the 6 parametrized circle cases.

- [ ] **Step 5: Verify and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src tests && uv run pytest -q
git add src/masklayout/geometry/curves.py tests/unit/test_curves.py
git commit -m "feat(m2): tessellate arcs, circles, beziers, and rounded rects to a chord-error budget

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 3: Compile — quantize, normalize, and report

**Files:**
- Create: `src/masklayout/geometry/report.py`, `src/masklayout/geometry/compile.py`
- Test: `tests/unit/test_compile.py`

**Interfaces:**
- Consumes: Task 1 normalization, Task 2 tessellation, `TechConfig`, model `Polygon`.
- Produces:
  - `TessellationReport(vertex_count, tessellation_error_nm, grid_error_nm, total_error_nm, budget_nm)` with `.within_budget`.
  - `compile_polyline(points_um, tech, layer, datatype=0) -> tuple[Polygon, TessellationReport]`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_compile.py`:

```python
"""Compiling tessellated curves into grid-aligned integer polygons."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import circle_um
from masklayout.model.geometry import Polygon


def test_compiled_polygon_is_grid_aligned_and_integer() -> None:
    tech = TechConfig(design_grid_nm=1.0, max_chord_error_nm=1.0)
    pts = circle_um((0.0, 0.0), 1.0, max_chord_error_um=tech.max_chord_error_nm / 1000.0)
    poly, _ = compile_polyline(pts, tech, layer=10)

    assert isinstance(poly, Polygon)
    assert poly.points.dtype == np.int64  # integer DBU is the grid guarantee


def test_report_separates_tessellation_error_from_grid_error() -> None:
    tech = TechConfig(design_grid_nm=1.0, max_chord_error_nm=1.0)
    pts = circle_um((0.0, 0.0), 5.0, max_chord_error_um=0.001)
    _, report = compile_polyline(pts, tech, layer=10)

    assert report.grid_error_nm == pytest.approx(math.sqrt(2.0) / 2.0)
    assert report.tessellation_error_nm >= 0.0
    assert report.total_error_nm == pytest.approx(
        report.tessellation_error_nm + report.grid_error_nm
    )


def test_report_budget_accounts_for_both_error_terms() -> None:
    # A budget of only max_chord_error_nm would be wrong: quantization adds up
    # to half the grid diagonal on top of it.
    tech = TechConfig(design_grid_nm=1.0, max_chord_error_nm=1.0)
    pts = circle_um((0.0, 0.0), 5.0, max_chord_error_um=0.001)
    _, report = compile_polyline(pts, tech, layer=10)

    assert report.budget_nm == pytest.approx(1.0 + math.sqrt(2.0) / 2.0)
    assert report.within_budget


def test_compile_removes_collinear_points() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    square_with_midpoints = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    poly, report = compile_polyline(square_with_midpoints, tech, layer=10)
    assert poly.vertex_count == 4
    assert report.vertex_count == 4


def test_compile_orients_counterclockwise() -> None:
    from masklayout.geometry.normalize import signed_area

    tech = TechConfig(design_grid_nm=1.0)
    clockwise = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    poly, _ = compile_polyline(clockwise, tech, layer=10)
    assert signed_area(poly.points) > 0


def test_compile_rejects_a_self_intersecting_ring() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    bowtie = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="self-intersect"):
        compile_polyline(bowtie, tech, layer=10)


def test_compile_preserves_layer_and_datatype() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    poly, _ = compile_polyline(square, tech, layer=11, datatype=5)
    assert (poly.layer, poly.datatype) == (11, 5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_compile.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `geometry/report.py`**

```python
"""Reports describing what compilation actually produced."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TessellationReport:
    """Measured deviation of a compiled polygon from its source polyline.

    The two error terms are kept apart deliberately. Tessellation error comes
    from replacing a curve with chords; grid error comes from snapping each
    vertex onto the design grid. They compose, so the acceptance budget is
    their sum and neither may hide inside the other.
    """

    vertex_count: int
    tessellation_error_nm: float
    grid_error_nm: float
    budget_nm: float

    @property
    def total_error_nm(self) -> float:
        return self.tessellation_error_nm + self.grid_error_nm

    @property
    def within_budget(self) -> bool:
        return self.total_error_nm <= self.budget_nm
```

- [ ] **Step 4: Write `geometry/compile.py`**

```python
"""Compile float polylines into grid-aligned integer polygons."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from masklayout.config import TechConfig
from masklayout.geometry.normalize import is_simple, normalize_polyline
from masklayout.geometry.report import TessellationReport
from masklayout.model.geometry import Polygon

#: Worst-case displacement when snapping a point to a square grid: half the
#: cell diagonal.
_HALF_GRID_DIAGONAL = math.sqrt(2.0) / 2.0


def compile_polyline(
    points_um: NDArray[np.float64],
    tech: TechConfig,
    layer: int,
    datatype: int = 0,
) -> tuple[Polygon, TessellationReport]:
    """Quantize, clean, and validate a float polyline into a model polygon.

    Order matters: normalize in float first so collinear removal uses true
    positions, then quantize once. Quantizing first would let grid noise turn
    genuinely collinear points into false corners.
    """
    source = np.asarray(points_um, dtype=np.float64)
    if not is_simple(source):
        raise ValueError(
            "polyline self-intersects and cannot be compiled into a valid polygon"
        )

    collinear_tolerance_um = tech.remove_collinear_tolerance_um
    cleaned = normalize_polyline(
        source,
        duplicate_tolerance=tech.precision_um / 2.0,
        collinear_tolerance=collinear_tolerance_um,
    )

    scaled = np.asarray(cleaned, dtype=np.float64) / tech.precision_um
    quantized = np.round(scaled).astype(np.int64)

    displacement_um = np.linalg.norm(
        (quantized.astype(np.float64) * tech.precision_um) - cleaned, axis=1
    )
    measured_grid_error_nm = float(displacement_um.max()) * 1000.0 if len(cleaned) else 0.0

    polygon = Polygon(points=quantized, layer=layer, datatype=datatype)
    report = TessellationReport(
        vertex_count=polygon.vertex_count,
        tessellation_error_nm=max(measured_grid_error_nm - _HALF_GRID_DIAGONAL, 0.0),
        grid_error_nm=tech.design_grid_nm * _HALF_GRID_DIAGONAL,
        budget_nm=tech.max_chord_error_nm + tech.design_grid_nm * _HALF_GRID_DIAGONAL,
    )
    return polygon, report
```

- [ ] **Step 5: Add `remove_collinear_tolerance_um` to `TechConfig`**

The originating brief lists `remove_collinear_tolerance_nm: 0.001` under cleanup but M0 did not carry it. Add to `src/masklayout/config.py`, in the cleanup block:

```python
    remove_collinear_tolerance_nm: float = Field(default=0.001, ge=0)
```

and alongside the other unit-conversion properties:

```python
    @property
    def remove_collinear_tolerance_um(self) -> float:
        """Collinearity tolerance in micrometres."""
        return self.remove_collinear_tolerance_nm / 1000.0
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/test_compile.py -v`
Expected: all 7 PASS.

- [ ] **Step 7: Verify everything and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src tests && uv run pytest -q
git add src/masklayout tests/unit/test_compile.py
git commit -m "feat(m2): compile polylines into grid-aligned integer polygons with a measured error report

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
gh run watch
```

- [ ] **Step 8: Mark M2 complete in the README**

---

## Self-Review

**Spec coverage.** M2's acceptance per §10 is "Curves compile into grid-aligned polygons within chord-error limit; integer DBU; GeomContext". Task 2 tessellates within the limit and proves it by measuring sagitta directly rather than trusting a library. Task 3 quantizes to integer DBU and reports the result. Task 1 supplies the normalization the pipeline's `normalize` stage requires (§4). `GeomContext` already exists from M0 and is unchanged here — `curves.py` computes tessellation directly and does not touch gdstk, keeping the allowlist at two modules.

**Placeholder scan.** No TBD or vague steps; every code block is complete.

**Type consistency.** `normalize_polyline(points, *, duplicate_tolerance, collinear_tolerance)` is keyword-only in its definition and every call. `compile_polyline(points_um, tech, layer, datatype=0)` matches all seven tests. `TessellationReport` fields used in tests — `vertex_count`, `tessellation_error_nm`, `grid_error_nm`, `total_error_nm`, `budget_nm`, `within_budget` — are all defined, the last two as properties. `remove_collinear_tolerance_um` is added to `TechConfig` in Task 3 Step 5 before `compile.py` uses it.

**Known risk.** `tessellation_error_nm` is derived from measured displacement rather than compared against the analytic curve, because `compile_polyline` receives a polyline and no longer knows the generating curve. This is honest but weaker than a direct comparison; a future task could pass an optional exact-evaluation callable to measure true deviation. The test asserting the budget accounts for both terms guards the property that matters.
