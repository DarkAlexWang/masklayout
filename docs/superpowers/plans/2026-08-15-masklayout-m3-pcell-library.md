# masklayout M3 — PCell Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author layouts from scratch — curvilinear wires, line ends, contacts, and hierarchical arrays — through named, validated, serializable parameterized cells.

**Architecture:** A PCell is a pydantic params model plus a `build` that returns model `Polygon`s. Params are validated and serializable because M4's rule deck references PCells **by name with a params dict**, so the registry built here is the same mechanism the deck will use. Wires are constructed by offsetting a centerline along its normals, which handles varying width naturally; everything routes through `compile_polyline` so grid alignment and validity are enforced in one place.

**Tech Stack:** As M2. No new dependencies.

**Design reference:** `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`.

## Scope

M3's acceptance (§10) is "from-scratch layout generates curvilinear wires, line ends, contacts, and arrays". This plan delivers exactly that, plus `rounded_rect` since `curves.py` already tessellates it:

| PCell | Why it is in M3 |
|---|---|
| `bezier_wire` | curvilinear wire |
| `tapered_wire` | curvilinear wire with varying width |
| `line_end` | line ends |
| `contact` | contacts |
| `contact_array` | arrays |
| `rounded_rect` | already tessellated by `curves.py` |

**Deliberately deferred:** `serpentine`, `ring`, `racetrack`, `arbitrary_contour`, and standalone `arc_segment`. Each is a parameter variation on machinery this milestone builds, not new capability. Shipping them would grow the surface without making the milestone more true. They land when something needs them.

## Global Constraints

M0–M2 constraints all still apply, plus:

- **PCell params are pydantic and frozen.** M4's deck names a PCell and supplies a params dict; that path must validate and round-trip.
- **Every PCell returns model `Polygon`s via `compile_polyline`.** No PCell quantizes by hand, so grid alignment and self-intersection rejection have exactly one implementation.
- **Arrays produce hierarchy, not flattened copies.** `contact_array` emits a `Cell` plus a `Reference` carrying a `RectangularRepetition` — the design forbids implicit flattening (§12).
- PCells must not import gdstk. The allowlist stays at two modules.

## The wire construction, and why not shapely buffer

A wire is a centerline plus a width. `shapely.buffer` strokes a constant-width line, but **cannot express a taper**, and its join and cap approximations are governed by `quad_segs` rather than by our chord-error budget.

Instead, offset the centerline along its own normals:

```
for each centerline point i:
    tangent  = normalize(p[i+1] - p[i-1])        # central difference
    normal   = (-tangent.y, tangent.x)
    left[i]  = p[i] + normal * width(i) / 2
    right[i] = p[i] - normal * width(i) / 2
ring = concatenate(left, reverse(right))
```

Width may vary per point, so taper is free. Endpoints use one-sided differences. On a curve whose radius falls below half the local width the offset self-intersects; `compile_polyline` already rejects that with a clear message, so the failure is caught rather than silently producing a bowtie.

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/pcells/base.py` | `PCellParams`, `PCell` protocol, the name registry |
| `src/masklayout/pcells/wires.py` | `bezier_wire`, `tapered_wire`, the offset helper |
| `src/masklayout/pcells/shapes.py` | `rounded_rect`, `line_end` |
| `src/masklayout/pcells/contacts.py` | `contact`, `contact_array` |
| `src/masklayout/pcells/__init__.py` | Public surface; registry population |
| `tests/unit/test_pcell_base.py` | Registry and params validation |
| `tests/unit/test_pcell_wires.py` | Wire construction and taper |
| `tests/unit/test_pcell_shapes.py` | Rounded rect and line end |
| `tests/unit/test_pcell_contacts.py` | Contacts and hierarchical arrays |

---

## Task 1: PCell foundation — params, protocol, registry

**Files:**
- Create: `src/masklayout/pcells/__init__.py`, `src/masklayout/pcells/base.py`
- Test: `tests/unit/test_pcell_base.py`

**Interfaces:**
- Produces:
  - `PCellParams` — frozen pydantic base for all PCell parameter models.
  - `PCellBuilder` — `Protocol` with `params_model: type[PCellParams]` and `build(params, tech, layer, datatype) -> list[Polygon]`.
  - `register(name, params_model)` — decorator registering a build function.
  - `build_pcell(name, params, tech, layer, datatype=0) -> list[Polygon]` — the by-name entry point M4's deck will call.
  - `registered_names() -> list[str]`
  - `UnknownPCellError`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pcell_base.py`:

```python
"""PCell registry and parameter validation."""

import numpy as np
import pytest
from pydantic import ValidationError

from masklayout.config import TechConfig
from masklayout.pcells.base import (
    UnknownPCellError,
    build_pcell,
    registered_names,
)


def test_registry_exposes_the_m3_pcells() -> None:
    names = registered_names()
    for expected in ("bezier_wire", "tapered_wire", "line_end", "contact", "rounded_rect"):
        assert expected in names


def test_registered_names_is_sorted_for_determinism() -> None:
    assert registered_names() == sorted(registered_names())


def test_build_by_name_accepts_a_plain_params_dict() -> None:
    # This is exactly the path M4's rule deck will take.
    polygons = build_pcell(
        "rounded_rect",
        {"lower_um": (0.0, 0.0), "upper_um": (2.0, 1.0), "radius_um": 0.2},
        TechConfig(),
        layer=10,
    )
    assert polygons
    assert polygons[0].points.dtype == np.int64
    assert polygons[0].layer == 10


def test_unknown_pcell_lists_the_known_ones() -> None:
    with pytest.raises(UnknownPCellError) as excinfo:
        build_pcell("no_such_pcell", {}, TechConfig(), layer=10)
    assert "rounded_rect" in str(excinfo.value)


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_pcell(
            "rounded_rect",
            {
                "lower_um": (0.0, 0.0),
                "upper_um": (2.0, 1.0),
                "radius_um": 0.2,
                "bogus": 1,
            },
            TechConfig(),
            layer=10,
        )


def test_params_are_frozen() -> None:
    from masklayout.pcells.shapes import RoundedRectParams

    params = RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(2.0, 1.0), radius_um=0.2)
    with pytest.raises(ValidationError):
        params.radius_um = 0.5


def test_params_round_trip_through_json() -> None:
    from masklayout.pcells.shapes import RoundedRectParams

    original = RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(2.0, 1.0), radius_um=0.2)
    assert RoundedRectParams.model_validate_json(original.model_dump_json()) == original
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_pcell_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'masklayout.pcells'`.

- [ ] **Step 3: Write `pcells/base.py`**

```python
"""PCell parameter models, the build protocol, and the name registry.

The registry exists because M4's rule deck references a PCell by name and
supplies parameters as data. Building that path here means the deck reuses
one validated mechanism rather than introducing a second one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon


class UnknownPCellError(KeyError):
    """A PCell was requested by a name that is not registered."""


class PCellParams(BaseModel):
    """Base for every PCell's parameters.

    Frozen so a built cell cannot be silently re-parameterized, and strict so
    a typo in a recipe fails loudly instead of being ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


BuildFn = Callable[[Any, TechConfig, int, int], list[Polygon]]


class PCellBuilder(Protocol):
    """What a registered PCell provides."""

    params_model: type[PCellParams]

    def __call__(
        self, params: Any, tech: TechConfig, layer: int, datatype: int
    ) -> list[Polygon]: ...


_REGISTRY: dict[str, tuple[type[PCellParams], BuildFn]] = {}


def register(name: str, params_model: type[PCellParams]) -> Callable[[BuildFn], BuildFn]:
    """Register a build function under a name, with its params model."""

    def decorate(build: BuildFn) -> BuildFn:
        if name in _REGISTRY:
            raise ValueError(f"PCell {name!r} is already registered")
        _REGISTRY[name] = (params_model, build)
        return build

    return decorate


def registered_names() -> list[str]:
    """Every registered PCell name, sorted for determinism."""
    return sorted(_REGISTRY)


def build_pcell(
    name: str,
    params: PCellParams | dict[str, Any],
    tech: TechConfig,
    layer: int,
    datatype: int = 0,
) -> list[Polygon]:
    """Build a PCell by name from either a params model or a plain dict."""
    try:
        params_model, build = _REGISTRY[name]
    except KeyError:
        raise UnknownPCellError(
            f"unknown PCell {name!r}; registered: {registered_names()}"
        ) from None
    validated = params if isinstance(params, params_model) else params_model.model_validate(params)
    return build(validated, tech, layer, datatype)
```

- [ ] **Step 4: Write `pcells/__init__.py`**

Importing the modules is what populates the registry, so the public surface and
the registry are established together.

```python
"""Parameterized cells for authoring layouts from scratch."""

from masklayout.pcells import contacts, shapes, wires  # noqa: F401  (registers PCells)
from masklayout.pcells.base import (
    PCellParams,
    UnknownPCellError,
    build_pcell,
    register,
    registered_names,
)

__all__ = [
    "PCellParams",
    "UnknownPCellError",
    "build_pcell",
    "register",
    "registered_names",
]
```

This import runs before `shapes`, `wires`, and `contacts` exist, so complete
Tasks 2–4 before expecting Task 1's tests to pass. To keep the TDD cycle
honest, temporarily reduce the import to `from masklayout.pcells import shapes`
once Task 2 lands, and add the others as they arrive.

- [ ] **Step 5: Defer running the tests**

Task 1's tests exercise `rounded_rect`, which Task 2 creates. Run:

`uv run pytest tests/unit/test_pcell_base.py -v`

Expected right now: FAIL with `ModuleNotFoundError: No module named 'masklayout.pcells.shapes'`. That is the correct state — proceed to Task 2 and return here.

---

## Task 2: Shape PCells — rounded_rect and line_end

**Files:**
- Create: `src/masklayout/pcells/shapes.py`
- Test: `tests/unit/test_pcell_shapes.py`

**Interfaces:**
- Produces:
  - `RoundedRectParams(lower_um, upper_um, radius_um)` and the `"rounded_rect"` registration.
  - `LineEndParams(centre_um, width_um, extension_um, angle_rad, corner_radius_um)` and `"line_end"`.

A line end is the terminating cap of a drawn line: a rectangle of `width_um`
by `extension_um` extending from `centre_um` along `angle_rad`, with
optionally rounded outer corners. It is built in edge-local coordinates and
rotated into place, which is what makes the same PCell work at any angle
(§ "Feature-local coordinates" in the originating brief).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pcell_shapes.py`:

```python
"""Shape PCells."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.pcells.shapes import LineEndParams, RoundedRectParams, build_line_end
from masklayout.pcells.shapes import build_rounded_rect


def test_rounded_rect_spans_the_requested_extent() -> None:
    tech = TechConfig()
    polys = build_rounded_rect(
        RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(4.0, 2.0), radius_um=0.5),
        tech,
        10,
        0,
    )
    assert len(polys) == 1
    low_x, low_y, high_x, high_y = polys[0].bounds_dbu
    assert (low_x, low_y) == (0, 0)
    assert (high_x, high_y) == (4000, 2000)


def test_rounded_rect_is_counterclockwise_and_grid_aligned() -> None:
    polys = build_rounded_rect(
        RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(4.0, 2.0), radius_um=0.5),
        TechConfig(),
        10,
        0,
    )
    assert polys[0].points.dtype == np.int64
    assert signed_area(polys[0].points) > 0


def test_line_end_at_zero_angle_has_the_expected_extent() -> None:
    tech = TechConfig()
    polys = build_line_end(
        LineEndParams(
            centre_um=(0.0, 0.0), width_um=0.1, extension_um=0.04, angle_rad=0.0
        ),
        tech,
        11,
        0,
    )
    low_x, low_y, high_x, high_y = polys[0].bounds_dbu
    assert (low_x, high_x) == (0, 40)  # extends forward only
    assert (low_y, high_y) == (-50, 50)  # centred on width


def test_line_end_rotates_rigidly() -> None:
    tech = TechConfig()
    flat = build_line_end(
        LineEndParams(centre_um=(0.0, 0.0), width_um=0.1, extension_um=0.04, angle_rad=0.0),
        tech,
        11,
        0,
    )[0]
    turned = build_line_end(
        LineEndParams(
            centre_um=(0.0, 0.0), width_um=0.1, extension_um=0.04, angle_rad=math.pi / 2
        ),
        tech,
        11,
        0,
    )[0]
    # A rigid rotation preserves vertex count and bounding-box dimensions, swapped.
    assert turned.vertex_count == flat.vertex_count
    fx0, fy0, fx1, fy1 = flat.bounds_dbu
    tx0, ty0, tx1, ty1 = turned.bounds_dbu
    assert (tx1 - tx0, ty1 - ty0) == (fy1 - fy0, fx1 - fx0)


def test_line_end_rejects_a_corner_radius_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="radius"):
        build_line_end(
            LineEndParams(
                centre_um=(0.0, 0.0),
                width_um=0.1,
                extension_um=0.04,
                angle_rad=0.0,
                corner_radius_um=0.09,
            ),
            TechConfig(),
            11,
            0,
        )


def test_rounded_rect_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError):
        RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(0.0, 2.0), radius_um=0.1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_pcell_shapes.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `pcells/shapes.py`**

```python
"""Shape PCells: rounded rectangles and line ends."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import arc_um, rounded_rect_um
from masklayout.model.geometry import Polygon
from masklayout.pcells.base import PCellParams, register


def rotate_um(
    points_um: NDArray[np.float64], angle_rad: float, origin_um: tuple[float, float]
) -> NDArray[np.float64]:
    """Rotate points about an origin. Used to place edge-local geometry."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    shifted = points_um - np.array(origin_um, dtype=np.float64)
    rotated = np.column_stack(
        (
            shifted[:, 0] * cos_a - shifted[:, 1] * sin_a,
            shifted[:, 0] * sin_a + shifted[:, 1] * cos_a,
        )
    )
    return rotated + np.array(origin_um, dtype=np.float64)


class RoundedRectParams(PCellParams):
    """An axis-aligned rectangle with circular corner fillets."""

    lower_um: tuple[float, float]
    upper_um: tuple[float, float]
    radius_um: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_extent(self) -> RoundedRectParams:
        width = self.upper_um[0] - self.lower_um[0]
        height = self.upper_um[1] - self.lower_um[1]
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"degenerate rectangle {self.lower_um} to {self.upper_um}: "
                f"width={width}, height={height}"
            )
        return self


@register("rounded_rect", RoundedRectParams)
def build_rounded_rect(
    params: RoundedRectParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    points = rounded_rect_um(
        params.lower_um,
        params.upper_um,
        params.radius_um,
        tech.max_chord_error_nm / 1000.0,
    )
    polygon, _ = compile_polyline(points, tech, layer, datatype)
    return [polygon]


class LineEndParams(PCellParams):
    """The terminating cap of a drawn line, in edge-local coordinates.

    The cap extends forward from ``centre_um`` along ``angle_rad`` by
    ``extension_um``, spanning ``width_um`` across. Building it edge-local and
    rotating into place is what lets one PCell serve any line angle.
    """

    centre_um: tuple[float, float]
    width_um: float = Field(gt=0)
    extension_um: float = Field(gt=0)
    angle_rad: float = 0.0
    corner_radius_um: float = Field(default=0.0, ge=0)


@register("line_end", LineEndParams)
def build_line_end(
    params: LineEndParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    budget_um = tech.max_chord_error_nm / 1000.0
    half_width = params.width_um / 2.0
    radius = params.corner_radius_um

    if radius > 0.0:
        if 2.0 * radius > min(params.width_um, params.extension_um):
            raise ValueError(
                f"corner radius {radius} does not fit a line end "
                f"{params.width_um} wide by {params.extension_um} long"
            )
        # Square inner edge, rounded outer corners.
        outer = params.extension_um
        pieces = [
            np.array([[0.0, -half_width]]),
            np.array([[outer - radius, -half_width]]),
            arc_um((outer - radius, -half_width + radius), radius, -math.pi / 2, 0.0, budget_um),
            arc_um((outer - radius, half_width - radius), radius, 0.0, math.pi / 2, budget_um),
            np.array([[0.0, half_width]]),
        ]
        local = np.vstack(pieces)
    else:
        local = np.array(
            [
                [0.0, -half_width],
                [params.extension_um, -half_width],
                [params.extension_um, half_width],
                [0.0, half_width],
            ],
            dtype=np.float64,
        )

    placed = rotate_um(local, params.angle_rad, (0.0, 0.0))
    placed = placed + np.array(params.centre_um, dtype=np.float64)
    polygon, _ = compile_polyline(placed, tech, layer, datatype)
    return [polygon]
```

- [ ] **Step 4: Reduce `pcells/__init__.py` to what exists**

While Tasks 3 and 4 are outstanding, the import line must name only `shapes`:

```python
from masklayout.pcells import shapes  # noqa: F401  (registers PCells)
```

- [ ] **Step 5: Run both test files**

Run: `uv run pytest tests/unit/test_pcell_shapes.py tests/unit/test_pcell_base.py -v`

Expected: `test_pcell_shapes.py` all PASS. In `test_pcell_base.py`,
`test_registry_exposes_the_m3_pcells` still FAILS because `bezier_wire`,
`tapered_wire`, and `contact` are not registered yet; every other test PASSES.
Return to it after Task 4.

- [ ] **Step 6: Verify and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src tests
uv run pytest -q -k "not test_registry_exposes_the_m3_pcells"
git add src/masklayout/pcells tests/unit/test_pcell_shapes.py tests/unit/test_pcell_base.py
git commit -m "feat(m3): add the PCell registry with rounded_rect and line_end

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 3: Wire PCells

**Files:**
- Create: `src/masklayout/pcells/wires.py`
- Modify: `src/masklayout/pcells/__init__.py` — add `wires`
- Test: `tests/unit/test_pcell_wires.py`

**Interfaces:**
- Produces:
  - `offset_centerline_um(centerline_um, widths_um) -> NDArray[np.float64]` — the ring formed by offsetting a polyline along its normals by a per-point width.
  - `BezierWireParams(control_points_um, width_um)` and `"bezier_wire"`.
  - `TaperedWireParams(control_points_um, start_width_um, end_width_um)` and `"tapered_wire"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pcell_wires.py`:

```python
"""Wire PCells."""

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.pcells.wires import (
    BezierWireParams,
    TaperedWireParams,
    build_bezier_wire,
    build_tapered_wire,
    offset_centerline_um,
)


def test_offset_of_a_straight_line_is_a_rectangle() -> None:
    centerline = np.array([[0.0, 0.0], [10.0, 0.0]])
    widths = np.array([2.0, 2.0])
    ring = offset_centerline_um(centerline, widths)
    assert ring.shape == (4, 2)
    assert ring[:, 1].min() == pytest.approx(-1.0)
    assert ring[:, 1].max() == pytest.approx(1.0)


def test_offset_width_varies_along_a_taper() -> None:
    centerline = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    widths = np.array([2.0, 1.0, 0.5])
    ring = offset_centerline_um(centerline, widths)
    # Ring is left side then reversed right side; sample matching x positions.
    at_start = ring[np.isclose(ring[:, 0], 0.0)][:, 1]
    at_end = ring[np.isclose(ring[:, 0], 10.0)][:, 1]
    assert at_start.max() - at_start.min() == pytest.approx(2.0)
    assert at_end.max() - at_end.min() == pytest.approx(0.5)


def test_offset_rejects_mismatched_width_count() -> None:
    with pytest.raises(ValueError, match="one width per"):
        offset_centerline_um(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([1.0]))


def test_bezier_wire_is_a_valid_grid_aligned_polygon() -> None:
    tech = TechConfig()
    polys = build_bezier_wire(
        BezierWireParams(
            control_points_um=((0.0, 0.0), (2.0, 3.0), (6.0, -3.0), (8.0, 0.0)),
            width_um=0.4,
        ),
        tech,
        10,
        0,
    )
    assert len(polys) == 1
    assert polys[0].points.dtype == np.int64
    assert signed_area(polys[0].points) > 0
    assert polys[0].vertex_count > 8


def test_bezier_wire_width_is_respected_at_the_start() -> None:
    tech = TechConfig()
    polys = build_bezier_wire(
        BezierWireParams(
            control_points_um=((0.0, 0.0), (3.0, 0.0), (7.0, 0.0), (10.0, 0.0)),
            width_um=0.5,
        ),
        tech,
        10,
        0,
    )
    low_x, low_y, high_x, high_y = polys[0].bounds_dbu
    assert high_y - low_y == 500  # 0.5 um at a 1 nm grid


def test_tapered_wire_narrows_from_start_to_end() -> None:
    tech = TechConfig()
    polys = build_tapered_wire(
        TaperedWireParams(
            control_points_um=((0.0, 0.0), (3.0, 0.0), (7.0, 0.0), (10.0, 0.0)),
            start_width_um=1.0,
            end_width_um=0.2,
        ),
        tech,
        10,
        0,
    )
    pts = polys[0].points
    near_start = pts[pts[:, 0] < 100]
    near_end = pts[pts[:, 0] > 9900]
    start_span = near_start[:, 1].max() - near_start[:, 1].min()
    end_span = near_end[:, 1].max() - near_end[:, 1].min()
    assert start_span == pytest.approx(1000, abs=2)
    assert end_span == pytest.approx(200, abs=2)
    assert start_span > end_span


def test_wire_rejects_a_width_that_self_intersects_on_a_tight_curve() -> None:
    # A wire far wider than its curve radius folds through itself.
    tech = TechConfig()
    with pytest.raises(ValueError, match="self-intersect"):
        build_bezier_wire(
            BezierWireParams(
                control_points_um=((0.0, 0.0), (0.5, 4.0), (-0.5, 4.0), (0.0, 0.0)),
                width_um=6.0,
            ),
            tech,
            10,
            0,
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_pcell_wires.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `pcells/wires.py`**

```python
"""Wire PCells built by offsetting a centerline along its normals.

shapely's buffer would stroke a constant-width line, but cannot express a
taper and approximates joins on its own terms rather than against our
chord-error budget. Offsetting along per-point normals handles varying width
directly and keeps the budget in one place.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import bezier_um
from masklayout.model.geometry import Polygon
from masklayout.pcells.base import PCellParams, register

MIN_CENTERLINE_POINTS = 2


def _unit_tangents(centerline_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tangent at each point: central differences inside, one-sided at the ends."""
    tangents = np.empty_like(centerline_um)
    tangents[1:-1] = centerline_um[2:] - centerline_um[:-2]
    tangents[0] = centerline_um[1] - centerline_um[0]
    tangents[-1] = centerline_um[-1] - centerline_um[-2]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / np.where(lengths > 0.0, lengths, 1.0)


def offset_centerline_um(
    centerline_um: NDArray[np.float64], widths_um: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Build the closed ring of a wire of varying width about a centerline."""
    centre = np.asarray(centerline_um, dtype=np.float64)
    widths = np.asarray(widths_um, dtype=np.float64)
    if len(centre) < MIN_CENTERLINE_POINTS:
        raise ValueError(f"a centerline needs at least {MIN_CENTERLINE_POINTS} points")
    if len(widths) != len(centre):
        raise ValueError(
            f"need one width per centerline point: {len(widths)} widths "
            f"for {len(centre)} points"
        )
    if np.any(widths <= 0.0):
        raise ValueError("every width must be positive")

    tangents = _unit_tangents(centre)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    half = (widths / 2.0)[:, None]
    left = centre + normals * half
    right = centre - normals * half
    return np.vstack((left, right[::-1]))


class BezierWireParams(PCellParams):
    """A constant-width wire following a Bezier centerline."""

    control_points_um: tuple[tuple[float, float], ...]
    width_um: float = Field(gt=0)


@register("bezier_wire", BezierWireParams)
def build_bezier_wire(
    params: BezierWireParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    centerline = bezier_um(
        np.array(params.control_points_um, dtype=np.float64),
        tech.max_chord_error_nm / 1000.0,
    )
    widths = np.full(len(centerline), params.width_um, dtype=np.float64)
    polygon, _ = compile_polyline(
        offset_centerline_um(centerline, widths), tech, layer, datatype
    )
    return [polygon]


class TaperedWireParams(PCellParams):
    """A wire whose width varies linearly along a Bezier centerline."""

    control_points_um: tuple[tuple[float, float], ...]
    start_width_um: float = Field(gt=0)
    end_width_um: float = Field(gt=0)


@register("tapered_wire", TaperedWireParams)
def build_tapered_wire(
    params: TaperedWireParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    centerline = bezier_um(
        np.array(params.control_points_um, dtype=np.float64),
        tech.max_chord_error_nm / 1000.0,
    )
    widths = np.linspace(
        params.start_width_um, params.end_width_um, len(centerline), dtype=np.float64
    )
    polygon, _ = compile_polyline(
        offset_centerline_um(centerline, widths), tech, layer, datatype
    )
    return [polygon]
```

- [ ] **Step 4: Add `wires` to `pcells/__init__.py`**

```python
from masklayout.pcells import shapes, wires  # noqa: F401  (registers PCells)
```

- [ ] **Step 5: Run tests, verify, commit**

```bash
uv run pytest tests/unit/test_pcell_wires.py -v
uv run ruff format . && uv run ruff check . && uv run mypy src tests
git add src/masklayout/pcells tests/unit/test_pcell_wires.py
git commit -m "feat(m3): add bezier and tapered wire PCells

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 4: Contacts and hierarchical arrays

**Files:**
- Create: `src/masklayout/pcells/contacts.py`
- Modify: `src/masklayout/pcells/__init__.py` — add `contacts`
- Test: `tests/unit/test_pcell_contacts.py`

**Interfaces:**
- Produces:
  - `ContactParams(centre_um, size_um, corner_radius_um)` and `"contact"`.
  - `place_contact_array(layout, cell_name, params, columns, rows, pitch_um, layer, origin_um, datatype=0) -> Reference` — creates the contact cell if absent, appends a `Reference` with a `RectangularRepetition` to the named parent cell, and returns it.

**Why a function rather than a registered PCell:** `contact_array` produces
hierarchy — a cell plus a placement — not a list of polygons, so it cannot
satisfy the registry's build signature. Flattening it into polygons to fit
would violate the rule against implicit flattening (§12). The registry holds
shape-producing PCells; array placement is a layout operation.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_pcell_contacts.py`:

```python
"""Contact PCell and hierarchical array placement."""

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.cell import Cell, RectangularRepetition
from masklayout.model.layout import Layout
from masklayout.pcells.contacts import ContactParams, build_contact, place_contact_array


def test_contact_is_centred_on_its_origin() -> None:
    polys = build_contact(
        ContactParams(centre_um=(1.0, 1.0), size_um=(0.2, 0.2)), TechConfig(), 12, 0
    )
    low_x, low_y, high_x, high_y = polys[0].bounds_dbu
    assert (low_x, low_y, high_x, high_y) == (900, 900, 1100, 1100)


def test_contact_with_a_corner_radius_has_more_vertices() -> None:
    tech = TechConfig()
    square = build_contact(ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)), tech, 12, 0)
    rounded = build_contact(
        ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2), corner_radius_um=0.05),
        tech,
        12,
        0,
    )
    assert rounded[0].vertex_count > square[0].vertex_count


def test_contact_rejects_a_radius_larger_than_half_its_size() -> None:
    with pytest.raises(ValueError, match="radius"):
        build_contact(
            ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2), corner_radius_um=0.2),
            TechConfig(),
            12,
            0,
        )


def test_array_placement_preserves_hierarchy() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP"))

    reference = place_contact_array(
        layout,
        parent_cell="TOP",
        contact_cell_name="CONTACT",
        params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
        columns=4,
        rows=3,
        pitch_um=(0.5, 0.5),
        layer=12,
    )

    # A cell plus a placement, not 12 flattened copies.
    assert "CONTACT" in layout.cells
    assert len(layout.cells["CONTACT"].polygons) == 1
    assert layout.cells["TOP"].polygons == []
    assert layout.cells["TOP"].references == [reference]
    assert layout.dependencies("TOP") == {"CONTACT"}


def test_array_repetition_describes_the_grid() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP"))
    reference = place_contact_array(
        layout,
        parent_cell="TOP",
        contact_cell_name="CONTACT",
        params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
        columns=4,
        rows=3,
        pitch_um=(0.5, 0.25),
        layer=12,
    )
    rep = reference.repetition
    assert isinstance(rep, RectangularRepetition)
    assert (rep.columns, rep.rows) == (4, 3)
    assert rep.spacing_dbu == (500, 250)
    assert rep.offsets_dbu().shape == (12, 2)


def test_array_reuses_an_existing_contact_cell() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP"))
    params = ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2))
    for _ in range(2):
        place_contact_array(
            layout,
            parent_cell="TOP",
            contact_cell_name="CONTACT",
            params=params,
            columns=2,
            rows=2,
            pitch_um=(0.5, 0.5),
            layer=12,
        )
    assert len(layout.cells) == 2  # TOP and CONTACT, not TOP and two contacts
    assert len(layout.cells["TOP"].references) == 2


def test_array_rejects_an_unknown_parent_cell() -> None:
    layout = Layout(name="LIB")
    with pytest.raises(KeyError, match="NOPE"):
        place_contact_array(
            layout,
            parent_cell="NOPE",
            contact_cell_name="CONTACT",
            params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
            columns=2,
            rows=2,
            pitch_um=(0.5, 0.5),
            layer=12,
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_pcell_contacts.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `pcells/contacts.py`**

```python
"""Contact PCell and hierarchical array placement."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import rounded_rect_um
from masklayout.model.cell import Cell, RectangularRepetition, Reference
from masklayout.model.geometry import Polygon
from masklayout.model.layout import Layout, UnknownCellError
from masklayout.pcells.base import PCellParams, register


class ContactParams(PCellParams):
    """A rectangular contact or via, optionally with rounded corners."""

    centre_um: tuple[float, float]
    size_um: tuple[float, float]
    corner_radius_um: float = Field(default=0.0, ge=0)


@register("contact", ContactParams)
def build_contact(
    params: ContactParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    half_x = params.size_um[0] / 2.0
    half_y = params.size_um[1] / 2.0
    lower = (params.centre_um[0] - half_x, params.centre_um[1] - half_y)
    upper = (params.centre_um[0] + half_x, params.centre_um[1] + half_y)

    if params.corner_radius_um > 0.0:
        if 2.0 * params.corner_radius_um > min(params.size_um):
            raise ValueError(
                f"corner radius {params.corner_radius_um} does not fit a contact "
                f"of size {params.size_um}"
            )
        points = rounded_rect_um(
            lower, upper, params.corner_radius_um, tech.max_chord_error_nm / 1000.0
        )
    else:
        points = np.array(
            [
                [lower[0], lower[1]],
                [upper[0], lower[1]],
                [upper[0], upper[1]],
                [lower[0], upper[1]],
            ],
            dtype=np.float64,
        )

    polygon, _ = compile_polyline(points, tech, layer, datatype)
    return [polygon]


def place_contact_array(
    layout: Layout,
    parent_cell: str,
    contact_cell_name: str,
    params: ContactParams,
    columns: int,
    rows: int,
    pitch_um: tuple[float, float],
    layer: int,
    origin_um: tuple[float, float] = (0.0, 0.0),
    datatype: int = 0,
) -> Reference:
    """Place a contact array as hierarchy: one cell, one repeated reference.

    The design forbids implicit flattening, so this emits a Reference carrying
    a RectangularRepetition rather than N copies of the contact geometry.
    """
    if parent_cell not in layout.cells:
        raise UnknownCellError(
            f"unknown parent cell {parent_cell!r}; known cells: {sorted(layout.cells)}"
        )
    if contact_cell_name not in layout.cells:
        cell = layout.add(Cell(name=contact_cell_name))
        cell.polygons.extend(build_contact(params, layout.tech, layer, datatype))

    precision_um = layout.tech.precision_um
    reference = Reference(
        cell_name=contact_cell_name,
        origin_dbu=(
            int(round(origin_um[0] / precision_um)),
            int(round(origin_um[1] / precision_um)),
        ),
        repetition=RectangularRepetition(
            columns=columns,
            rows=rows,
            spacing_dbu=(
                int(round(pitch_um[0] / precision_um)),
                int(round(pitch_um[1] / precision_um)),
            ),
        ),
    )
    layout.cells[parent_cell].references.append(reference)
    return reference
```

- [ ] **Step 4: Restore the full `pcells/__init__.py`**

```python
from masklayout.pcells import contacts, shapes, wires  # noqa: F401  (registers PCells)
```

- [ ] **Step 5: Run the whole suite, including Task 1's deferred test**

```bash
uv run pytest tests/unit/test_pcell_contacts.py -v
uv run pytest tests/unit/test_pcell_base.py -v   # now fully passing
uv run ruff format . && uv run ruff check . && uv run mypy src tests && uv run pytest -q
```

- [ ] **Step 6: Commit, push, verify CI**

```bash
git add src/masklayout/pcells tests/unit/test_pcell_contacts.py
git commit -m "feat(m3): add contact PCell and hierarchical array placement

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
gh run watch
```

- [ ] **Step 7: End-to-end M3 acceptance check**

M3's acceptance is that a from-scratch layout produces curvilinear wires, line
ends, contacts, and arrays. Write `tests/integration/test_author_from_scratch.py`:

```python
"""M3 acceptance: author a layout from scratch and round-trip it."""

from pathlib import Path

from masklayout.config import TechConfig
from masklayout.io.streams import read_gds, write_gds
from masklayout.model.cell import Cell
from masklayout.model.layout import Layout
from masklayout.pcells.contacts import ContactParams, place_contact_array
from masklayout.pcells.shapes import LineEndParams, build_line_end
from masklayout.pcells.wires import BezierWireParams, build_bezier_wire


def test_authored_layout_round_trips_through_gds(tmp_path: Path) -> None:
    tech = TechConfig()
    layout = Layout(name="AUTHORED", tech=tech)
    top = layout.add(Cell(name="TOP"))

    top.polygons.extend(
        build_bezier_wire(
            BezierWireParams(
                control_points_um=((0.0, 0.0), (2.0, 3.0), (6.0, -3.0), (8.0, 0.0)),
                width_um=0.4,
            ),
            tech,
            10,
            0,
        )
    )
    top.polygons.extend(
        build_line_end(
            LineEndParams(centre_um=(8.0, 0.0), width_um=0.4, extension_um=0.05),
            tech,
            10,
            0,
        )
    )
    place_contact_array(
        layout,
        parent_cell="TOP",
        contact_cell_name="CONTACT",
        params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
        columns=3,
        rows=3,
        pitch_um=(0.5, 0.5),
        layer=12,
        origin_um=(10.0, 0.0),
    )

    path = tmp_path / "authored.gds"
    write_gds(layout, path)
    restored, report = read_gds(path)

    assert sorted(restored.cells) == ["CONTACT", "TOP"]
    assert restored.top_cells() == ["TOP"]
    assert restored.dependencies("TOP") == {"CONTACT"}
    assert report.polygon_count == 3  # wire, line end, one contact — not nine
```

Then commit and mark M3 complete in the README.

---

## Self-Review

**Spec coverage.** M3's acceptance is "from-scratch layout generates curvilinear wires, line ends, contacts, and arrays". Task 3 covers curvilinear wires including taper, Task 2 covers line ends built edge-local so they work at any angle, Task 4 covers contacts and arrays as real hierarchy. Task 1's registry is what M4's deck will call to build a PCell by name from a params dict. Task 4 Step 7 exercises all four together and round-trips them.

**Placeholder scan.** No TBD or vague steps. The staged `pcells/__init__.py` is called out explicitly at each task rather than left to guesswork, since importing a module that does not yet exist is the one thing that would break the TDD cycle here.

**Type consistency.** Every registered build has signature `(params, tech: TechConfig, layer: int, datatype: int) -> list[Polygon]`, matching `BuildFn` and the calls in every test. `ContactParams`, `LineEndParams`, `RoundedRectParams`, `BezierWireParams`, and `TaperedWireParams` all derive from `PCellParams`. `place_contact_array` takes keyword arguments in the same order in its definition and all five of its tests.

**Known risk.** `test_line_end_rotates_rigidly` compares bounding boxes after a 90° rotation. That is exact for 90° but would not hold for an arbitrary angle once quantization is applied, so the test deliberately uses a right angle. Arbitrary-angle placement is exercised by M4's decorator work, where it matters.
