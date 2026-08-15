# masklayout M1 — GDS/OASIS I/O and Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read GDSII and OASIS into a typed, gdstk-free model with integer coordinates and preserved hierarchy, and write it back out with round-trip fidelity.

**Architecture:** A typed model (`Polygon`, `Label`, `Reference`, `Cell`, `Layout`) holds `int64` coordinates in design database units and knows nothing about gdstk. A single bridge module converts between that model and `gdstk.Library`. Readers return a `ReadReport` alongside the layout, so nothing — converted paths, dropped entities, cell counts — changes without being counted.

**Tech Stack:** As M0, plus `numpy` for coordinate arrays.

**Design reference:** `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`. Section numbers below refer to it.

## Global Constraints

Everything in the M0 plan's Global Constraints still applies, plus:

- **gdstk allowlist is now two modules**: `geometry/context.py` and `io/_gdstk_bridge.py` (§3). No others. The AST guard enforces this.
- **Model coordinates are `int64` in DBU.** Floats appear only in `_gdstk_bridge.py` during conversion. Rotation and magnification are floats — they are not coordinates.
- **Grid mismatch is a hard error.** If a file's database precision differs from `tech.precision_m`, raise naming both values. Never adopt the file's grid, never resample (decision taken at M1 planning).
- **Paths are converted to polygons on read and counted in the `ReadReport`.** Labels are preserved. Nothing is dropped silently (decision taken at M1 planning).
- The model must never expose a gdstk type in a public signature (§12).

## Verified toolchain facts

Established by direct experiment against gdstk 1.0.1 before writing this plan. These drive several test designs:

| Fact | Consequence |
|---|---|
| `Reference.cell` is a `Cell` object, in memory and after read | Read `.cell.name` for the target name |
| `Repetition` has **no** `.type` attribute | Discriminate on `columns`/`rows`/`spacing` vs `offsets` being non-`None` |
| Rectangular arrays expose `columns`, `rows`, `spacing`; `get_offsets()` returns all placements | Model rectangular repetitions structurally; fall back to explicit offsets |
| **gdstk writes `FlexPath` as BOUNDARY records** — a path written then read comes back as a polygon, and `.paths` is empty | A path-conversion test **must** use an in-memory `gdstk.Library`, never a file round-trip |
| `Polygon.points` is `float64` micrometres | Conversion to `int64` DBU is required on every read |
| `Library.top_level()` returns top cells; `Cell.dependencies(True)` walks the tree | Hierarchy inspection can be verified against gdstk's own answer |

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/model/geometry.py` | `Polygon`, `Label` — integer-coordinate geometry |
| `src/masklayout/model/cell.py` | `Reference`, `RectangularRepetition`, `ExplicitRepetition`, `Cell` |
| `src/masklayout/model/layout.py` | `Layout` and hierarchy inspection |
| `src/masklayout/io/__init__.py` | Public I/O surface |
| `src/masklayout/io/report.py` | `ReadReport` |
| `src/masklayout/io/_gdstk_bridge.py` | Model ↔ `gdstk.Library`; the only new gdstk importer |
| `src/masklayout/io/streams.py` | `read_gds`, `write_gds`, `read_oas`, `write_oas` |
| `tests/unit/test_model_geometry.py` | Model types |
| `tests/unit/test_layout_hierarchy.py` | Hierarchy inspection |
| `tests/unit/test_bridge_units.py` | Coordinate conversion and grid mismatch |
| `tests/unit/test_streams.py` | Read, write, round-trip |

---

## Task 1: Typed model with integer coordinates

**Files:**
- Create: `src/masklayout/model/geometry.py`, `src/masklayout/model/cell.py`, `src/masklayout/model/layout.py`
- Test: `tests/unit/test_model_geometry.py`, `tests/unit/test_layout_hierarchy.py`

**Interfaces:**
- Consumes: `TechConfig` (M0 Task 3), `LayerMap` (M0 Task 2).
- Produces:
  - `Polygon(points: NDArray[np.int64], layer: int, datatype: int)` with `.vertex_count`, `.bounds_dbu`.
  - `Label(text: str, origin_dbu: tuple[int, int], layer: int, datatype: int)`.
  - `RectangularRepetition(columns: int, rows: int, spacing_dbu: tuple[int, int])` with `.offsets_dbu()`.
  - `ExplicitRepetition(offsets_dbu: NDArray[np.int64])` with `.offsets_dbu()`.
  - `Reference(cell_name: str, origin_dbu, rotation_rad, magnification, x_reflection, repetition)`.
  - `Cell(name: str, polygons, labels, references)`.
  - `Layout(name, cells: dict[str, Cell], tech, layers)` with `.top_cells()`, `.dependencies(name)`, `.depth()`, `.polygon_count()`.

**Note on equality:** these dataclasses hold numpy arrays, so `eq=True` would raise "truth value of an array is ambiguous" on comparison. Types holding arrays use `eq=False`; tests compare with `np.array_equal`.

- [ ] **Step 1: Write the failing model test**

Create `tests/unit/test_model_geometry.py`:

```python
"""Typed geometry model."""

import numpy as np
import pytest

from masklayout.model.cell import (
    Cell,
    ExplicitRepetition,
    RectangularRepetition,
    Reference,
)
from masklayout.model.geometry import Label, Polygon


def test_polygon_holds_integer_coordinates() -> None:
    poly = Polygon(points=np.array([[0, 0], [100, 0], [100, 50]], dtype=np.int64), layer=10)
    assert poly.points.dtype == np.int64
    assert poly.vertex_count == 3
    assert poly.datatype == 0


def test_polygon_rejects_non_integer_coordinates() -> None:
    with pytest.raises(TypeError, match="int64"):
        Polygon(points=np.array([[0.0, 0.5]], dtype=np.float64), layer=10)


def test_polygon_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        Polygon(points=np.array([0, 1, 2], dtype=np.int64), layer=10)


def test_polygon_bounds() -> None:
    poly = Polygon(points=np.array([[0, 0], [100, 0], [100, 50]], dtype=np.int64), layer=10)
    assert poly.bounds_dbu == (0, 0, 100, 50)


def test_label_holds_integer_origin() -> None:
    label = Label(text="A1", origin_dbu=(10, 20), layer=12)
    assert label.origin_dbu == (10, 20)
    assert label.text == "A1"


def test_rectangular_repetition_expands_to_offsets() -> None:
    rep = RectangularRepetition(columns=3, rows=2, spacing_dbu=(1000, 500))
    offsets = rep.offsets_dbu()
    assert offsets.dtype == np.int64
    assert offsets.shape == (6, 2)
    assert (0, 0) in {tuple(row) for row in offsets.tolist()}
    assert (2000, 500) in {tuple(row) for row in offsets.tolist()}


def test_rectangular_repetition_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError, match="columns"):
        RectangularRepetition(columns=0, rows=2, spacing_dbu=(10, 10))


def test_explicit_repetition_returns_its_offsets() -> None:
    given = np.array([[0, 0], [7, 9]], dtype=np.int64)
    assert np.array_equal(ExplicitRepetition(offsets_dbu=given).offsets_dbu(), given)


def test_reference_defaults_are_identity() -> None:
    ref = Reference(cell_name="LEAF", origin_dbu=(0, 0))
    assert ref.rotation_rad == 0.0
    assert ref.magnification == 1.0
    assert ref.x_reflection is False
    assert ref.repetition is None


def test_cell_starts_empty() -> None:
    cell = Cell(name="TOP")
    assert cell.polygons == []
    assert cell.labels == []
    assert cell.references == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_model_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'masklayout.model.geometry'`.

- [ ] **Step 3: Write `model/geometry.py`**

```python
"""Integer-coordinate geometry types.

All coordinates are int64 in design database units (DBU). No float
coordinate exists at this layer; see the design document, section
"Units and coordinate model".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _check_integer_points(points: NDArray[np.int64]) -> None:
    if points.dtype != np.int64:
        raise TypeError(
            f"coordinates must be int64 design database units, got {points.dtype}"
        )
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got shape {points.shape}")


@dataclass(frozen=True, eq=False)
class Polygon:
    """A closed polygon in design database units."""

    points: NDArray[np.int64]
    layer: int
    datatype: int = 0

    def __post_init__(self) -> None:
        _check_integer_points(self.points)

    @property
    def vertex_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def bounds_dbu(self) -> tuple[int, int, int, int]:
        """(min_x, min_y, max_x, max_y)."""
        low = self.points.min(axis=0)
        high = self.points.max(axis=0)
        return (int(low[0]), int(low[1]), int(high[0]), int(high[1]))


@dataclass(frozen=True)
class Label:
    """A text annotation. Carried through I/O but never used as geometry."""

    text: str
    origin_dbu: tuple[int, int]
    layer: int
    datatype: int = 0
```

- [ ] **Step 4: Write `model/cell.py`**

```python
"""Cells, references, and array repetitions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from masklayout.model.geometry import Label, Polygon


@dataclass(frozen=True)
class RectangularRepetition:
    """A regular grid of placements, as GDSII AREF expresses it."""

    columns: int
    rows: int
    spacing_dbu: tuple[int, int]

    def __post_init__(self) -> None:
        if self.columns < 1:
            raise ValueError(f"columns must be >= 1, got {self.columns}")
        if self.rows < 1:
            raise ValueError(f"rows must be >= 1, got {self.rows}")

    def offsets_dbu(self) -> NDArray[np.int64]:
        """Every placement offset, row-major, including (0, 0)."""
        dx, dy = self.spacing_dbu
        cols = np.arange(self.columns, dtype=np.int64) * dx
        rows = np.arange(self.rows, dtype=np.int64) * dy
        grid_x, grid_y = np.meshgrid(cols, rows, indexing="ij")
        return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.int64)


@dataclass(frozen=True, eq=False)
class ExplicitRepetition:
    """An arbitrary list of placement offsets."""

    offsets_dbu_array: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.offsets_dbu_array.dtype != np.int64:
            raise TypeError(
                f"offsets must be int64 DBU, got {self.offsets_dbu_array.dtype}"
            )

    def offsets_dbu(self) -> NDArray[np.int64]:
        return self.offsets_dbu_array


Repetition = RectangularRepetition | ExplicitRepetition


@dataclass(frozen=True)
class Reference:
    """A placement of another cell.

    ``rotation_rad`` and ``magnification`` are floats because they are not
    coordinates; the integer-coordinate rule applies to positions only.
    """

    cell_name: str
    origin_dbu: tuple[int, int]
    rotation_rad: float = 0.0
    magnification: float = 1.0
    x_reflection: bool = False
    repetition: Repetition | None = None


@dataclass
class Cell:
    """A named container of geometry and placements."""

    name: str
    polygons: list[Polygon] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
```

Note: `ExplicitRepetition` takes `offsets_dbu_array` as its field so the
`offsets_dbu()` method name stays free on both repetition kinds. Tests
construct it with the keyword `offsets_dbu=` — so the field must be named
`offsets_dbu_array` **and** the test updated to match. Use
`ExplicitRepetition(offsets_dbu_array=given)` in the test from Step 1.

- [ ] **Step 5: Write `model/layout.py`**

```python
"""The top-level layout container and hierarchy inspection."""

from __future__ import annotations

from dataclasses import dataclass, field

from masklayout.config import TechConfig
from masklayout.model.cell import Cell
from masklayout.model.layers import LayerMap


class UnknownCellError(KeyError):
    """A reference names a cell that is not in the layout."""


@dataclass
class Layout:
    """A library of cells sharing one technology configuration."""

    name: str
    cells: dict[str, Cell] = field(default_factory=dict)
    tech: TechConfig = field(default_factory=TechConfig)
    layers: LayerMap = field(default_factory=LayerMap.default)

    def add(self, cell: Cell) -> Cell:
        if cell.name in self.cells:
            raise ValueError(f"cell {cell.name!r} already exists in layout {self.name!r}")
        self.cells[cell.name] = cell
        return cell

    def referenced_names(self) -> set[str]:
        """Every cell name that appears as a reference target."""
        return {ref.cell_name for cell in self.cells.values() for ref in cell.references}

    def top_cells(self) -> list[str]:
        """Cells that nothing else references, sorted for determinism."""
        referenced = self.referenced_names()
        return sorted(name for name in self.cells if name not in referenced)

    def dependencies(self, cell_name: str) -> set[str]:
        """Every cell reachable from ``cell_name``, excluding itself."""
        if cell_name not in self.cells:
            raise UnknownCellError(
                f"unknown cell {cell_name!r}; known cells: {sorted(self.cells)}"
            )
        seen: set[str] = set()
        stack = [cell_name]
        while stack:
            current = stack.pop()
            for ref in self.cells[current].references:
                if ref.cell_name in seen:
                    continue
                if ref.cell_name not in self.cells:
                    raise UnknownCellError(
                        f"cell {current!r} references unknown cell {ref.cell_name!r}"
                    )
                seen.add(ref.cell_name)
                stack.append(ref.cell_name)
        seen.discard(cell_name)
        return seen

    def depth(self) -> int:
        """Longest reference chain. A flat layout has depth 0."""
        memo: dict[str, int] = {}

        def walk(name: str, path: frozenset[str]) -> int:
            if name in path:
                raise ValueError(f"reference cycle detected at cell {name!r}")
            if name in memo:
                return memo[name]
            refs = self.cells[name].references
            best = 0 if not refs else 1 + max(
                walk(ref.cell_name, path | {name}) for ref in refs
            )
            memo[name] = best
            return best

        return max((walk(name, frozenset()) for name in self.cells), default=0)

    def polygon_count(self) -> int:
        return sum(len(cell.polygons) for cell in self.cells.values())
```

- [ ] **Step 6: Write the hierarchy test**

Create `tests/unit/test_layout_hierarchy.py`:

```python
"""Hierarchy inspection."""

import numpy as np
import pytest

from masklayout.model.cell import Cell, Reference
from masklayout.model.geometry import Polygon
from masklayout.model.layout import Layout, UnknownCellError


def _square(size: int = 100) -> Polygon:
    pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.int64)
    return Polygon(points=pts, layer=10)


def _nested() -> Layout:
    layout = Layout(name="LIB")
    layout.add(Cell(name="LEAF", polygons=[_square()]))
    layout.add(Cell(name="MID", references=[Reference("LEAF", (0, 0))]))
    layout.add(Cell(name="TOP", references=[Reference("MID", (0, 0))]))
    return layout


def test_top_cells_excludes_referenced_cells() -> None:
    assert _nested().top_cells() == ["TOP"]


def test_dependencies_are_transitive() -> None:
    assert _nested().dependencies("TOP") == {"MID", "LEAF"}
    assert _nested().dependencies("LEAF") == set()


def test_depth_counts_the_longest_chain() -> None:
    assert _nested().depth() == 2
    flat = Layout(name="LIB")
    flat.add(Cell(name="ONLY", polygons=[_square()]))
    assert flat.depth() == 0


def test_polygon_count_sums_all_cells() -> None:
    assert _nested().polygon_count() == 1


def test_duplicate_cell_name_is_rejected() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="A"))
    with pytest.raises(ValueError, match="already exists"):
        layout.add(Cell(name="A"))


def test_dependencies_on_unknown_cell_lists_known_cells() -> None:
    with pytest.raises(UnknownCellError, match="LEAF"):
        _nested().dependencies("NOPE")


def test_dangling_reference_is_reported() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP", references=[Reference("MISSING", (0, 0))]))
    with pytest.raises(UnknownCellError, match="MISSING"):
        layout.dependencies("TOP")


def test_reference_cycle_is_detected() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="A", references=[Reference("B", (0, 0))]))
    layout.add(Cell(name="B", references=[Reference("A", (0, 0))]))
    with pytest.raises(ValueError, match="cycle"):
        layout.depth()
```

- [ ] **Step 7: Run both test files**

Run: `uv run pytest tests/unit/test_model_geometry.py tests/unit/test_layout_hierarchy.py -v`
Expected: all PASS.

- [ ] **Step 8: Verify lint, format, types, whole suite**

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```
Expected: clean.

- [ ] **Step 9: Commit and push**

```bash
git add src/masklayout/model tests/unit/test_model_geometry.py tests/unit/test_layout_hierarchy.py
git commit -m "feat(m1): add typed model with integer coordinates and hierarchy inspection

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 2: Coordinate conversion and grid validation

**Files:**
- Create: `src/masklayout/io/__init__.py`, `src/masklayout/io/errors.py`, `src/masklayout/io/_gdstk_bridge.py` (conversion helpers only in this task)
- Modify: `tests/test_architecture.py` — add `io/_gdstk_bridge.py` to `ALLOWED`
- Test: `tests/unit/test_bridge_units.py`

**Interfaces:**
- Consumes: `TechConfig.precision_um`, `TechConfig.precision_m`.
- Produces:
  - `GridMismatchError`, `OffGridCoordinateError` in `io/errors.py`.
  - `um_to_dbu(points_um: NDArray[np.float64], precision_um: float) -> NDArray[np.int64]`
  - `dbu_to_um(points_dbu: NDArray[np.int64], precision_um: float) -> NDArray[np.float64]`
  - `check_library_grid(unit: float, precision_m: float, tech: TechConfig) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_bridge_units.py`:

```python
"""Coordinate conversion between float micrometres and integer DBU."""

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.io._gdstk_bridge import check_library_grid, dbu_to_um, um_to_dbu
from masklayout.io.errors import GridMismatchError, OffGridCoordinateError


def test_um_to_dbu_is_exact_on_grid() -> None:
    points = np.array([[0.0, 0.001], [1.5, -0.25]], dtype=np.float64)
    result = um_to_dbu(points, precision_um=0.001)
    assert result.dtype == np.int64
    assert result.tolist() == [[0, 1], [1500, -250]]


def test_um_to_dbu_tolerates_float_representation_error() -> None:
    # 0.3 / 0.001 evaluates to 299.99999999999994 in IEEE 754.
    result = um_to_dbu(np.array([[0.3, 0.7]], dtype=np.float64), precision_um=0.001)
    assert result.tolist() == [[300, 700]]


def test_um_to_dbu_rejects_genuinely_off_grid_coordinates() -> None:
    with pytest.raises(OffGridCoordinateError, match=r"0\.0005"):
        um_to_dbu(np.array([[0.0005, 0.0]], dtype=np.float64), precision_um=0.001)


def test_dbu_to_um_round_trips() -> None:
    original = np.array([[0, 1], [1500, -250]], dtype=np.int64)
    back = um_to_dbu(dbu_to_um(original, precision_um=0.001), precision_um=0.001)
    assert np.array_equal(back, original)


def test_check_library_grid_accepts_matching_precision() -> None:
    check_library_grid(unit=1e-6, precision_m=1e-9, tech=TechConfig(design_grid_nm=1.0))


def test_check_library_grid_rejects_mismatch_naming_both() -> None:
    with pytest.raises(GridMismatchError) as excinfo:
        check_library_grid(unit=1e-6, precision_m=2.5e-10, tech=TechConfig(design_grid_nm=1.0))
    message = str(excinfo.value)
    assert "2.5e-10" in message
    assert "1e-09" in message


def test_check_library_grid_rejects_unexpected_unit() -> None:
    with pytest.raises(GridMismatchError, match="unit"):
        check_library_grid(unit=1e-3, precision_m=1e-9, tech=TechConfig(design_grid_nm=1.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_bridge_units.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'masklayout.io'`.

- [ ] **Step 3: Write `io/errors.py`**

```python
"""I/O error types."""

from __future__ import annotations


class GridMismatchError(ValueError):
    """A file's database grid does not match the configured design grid."""


class OffGridCoordinateError(ValueError):
    """A coordinate does not lie on the design grid."""


class UnsupportedEntityError(ValueError):
    """A stream contains an entity kind this version cannot represent."""
```

- [ ] **Step 4: Write the conversion helpers in `io/_gdstk_bridge.py`**

Start the file with only these helpers; Task 3 adds structural conversion.

```python
"""Conversion between the typed model and gdstk.

This is one of exactly two modules permitted to import gdstk; see the design
document, section "The gdstk boundary". Nothing here may leak a gdstk type
into a public signature.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from masklayout.config import TechConfig
from masklayout.io.errors import GridMismatchError, OffGridCoordinateError

#: Maximum acceptable deviation, in grid units, before a coordinate counts as
#: off-grid. Generous enough for float64 division noise, tight enough to catch
#: a genuine half-grid coordinate.
_ON_GRID_TOLERANCE = 1e-6

#: GDSII user unit expected by this toolkit: 1 micrometre.
_EXPECTED_UNIT_M = 1e-6


def um_to_dbu(points_um: NDArray[np.float64], precision_um: float) -> NDArray[np.int64]:
    """Convert float micrometres to integer design database units."""
    scaled = np.asarray(points_um, dtype=np.float64) / precision_um
    rounded = np.round(scaled)
    residue = np.abs(scaled - rounded)
    if residue.size and residue.max() > _ON_GRID_TOLERANCE:
        worst = np.unravel_index(int(np.argmax(residue)), residue.shape)
        value = float(np.asarray(points_um)[worst])
        raise OffGridCoordinateError(
            f"coordinate {value!r} um is not a multiple of the design grid "
            f"{precision_um} um (off by {float(residue.max())} grid units)"
        )
    return rounded.astype(np.int64)


def dbu_to_um(points_dbu: NDArray[np.int64], precision_um: float) -> NDArray[np.float64]:
    """Convert integer design database units to float micrometres."""
    return np.asarray(points_dbu, dtype=np.float64) * precision_um


def check_library_grid(unit: float, precision_m: float, tech: TechConfig) -> None:
    """Reject a stream whose grid differs from the configured design grid.

    The design forbids silently adopting a file's grid or resampling onto
    ours: target geometry is immutable, so a mismatch is an error.
    """
    if not np.isclose(unit, _EXPECTED_UNIT_M, rtol=0.0, atol=1e-18):
        raise GridMismatchError(
            f"unsupported user unit {unit!r} m; masklayout expects {_EXPECTED_UNIT_M!r} m"
        )
    if not np.isclose(precision_m, tech.precision_m, rtol=1e-12, atol=0.0):
        raise GridMismatchError(
            f"file database precision {precision_m!r} m does not match the configured "
            f"design grid {tech.precision_m!r} m "
            f"(design_grid_nm={tech.design_grid_nm}); "
            "set design_grid_nm to match the file, or convert the file"
        )
```

Create an empty-bodied `src/masklayout/io/__init__.py`:

```python
"""Stream I/O for GDSII and OASIS."""
```

- [ ] **Step 5: Add the bridge to the architecture allowlist**

In `tests/test_architecture.py`, change:

```python
ALLOWED = {"geometry/context.py"}
```

to:

```python
# The gdstk boundary is a closed, explicit allowlist — see the design document,
# section "The gdstk boundary". Adding an entry is a design change.
ALLOWED = {"geometry/context.py", "io/_gdstk_bridge.py"}
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/test_bridge_units.py tests/test_architecture.py -v`
Expected: all PASS.

- [ ] **Step 7: Verify lint, format, types, whole suite**

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

- [ ] **Step 8: Commit and push**

```bash
git add src/masklayout/io tests/unit/test_bridge_units.py tests/test_architecture.py
git commit -m "feat(m1): add DBU conversion and hard grid-mismatch validation

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 3: Reader — streams to model, with a report

**Files:**
- Create: `src/masklayout/io/report.py`
- Modify: `src/masklayout/io/_gdstk_bridge.py` — add `library_to_layout`
- Test: `tests/unit/test_streams.py` (read cases)

**Interfaces:**
- Consumes: Task 1 model types, Task 2 conversion helpers.
- Produces:
  - `ReadReport(source, cell_count, polygon_count, label_count, reference_count, paths_converted, file_precision_m, top_cells)`.
  - `library_to_layout(library, tech, layers, source) -> tuple[Layout, ReadReport]` — takes a `gdstk.Library`, returns model types only.

**Critical test-design note:** gdstk writes `FlexPath` as BOUNDARY records, so a path written to GDS and read back is already a polygon and `.paths` is empty. The `paths_converted` test **must** build a `gdstk.Library` in memory and pass it straight to `library_to_layout`. A file round-trip cannot exercise that code path.

- [ ] **Step 1: Write the failing read tests**

Create `tests/unit/test_streams.py`:

```python
"""Stream reading, writing, and round-trip fidelity."""

import math
from pathlib import Path

import gdstk
import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.io._gdstk_bridge import library_to_layout
from masklayout.io.errors import GridMismatchError
from masklayout.model.cell import RectangularRepetition
from masklayout.model.layers import LayerMap


def _hierarchical_library() -> gdstk.Library:
    lib = gdstk.Library("LIB", unit=1e-6, precision=1e-9)
    leaf = lib.new_cell("LEAF")
    leaf.add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5), layer=10, datatype=0))
    leaf.add(gdstk.Label("tag", (0.5, 0.25), layer=12))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(leaf, (5.0, 5.0), rotation=math.pi / 4, magnification=2.0))
    top.add(gdstk.Reference(leaf, (0.0, 0.0), columns=3, rows=2, spacing=(10.0, 10.0)))
    return lib


def test_read_preserves_hierarchy_and_reports_counts() -> None:
    layout, report = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    assert sorted(layout.cells) == ["LEAF", "TOP"]
    assert layout.top_cells() == ["TOP"]
    assert layout.dependencies("TOP") == {"LEAF"}
    assert report.cell_count == 2
    assert report.polygon_count == 1
    assert report.label_count == 1
    assert report.reference_count == 2
    assert report.paths_converted == 0


def test_read_converts_polygon_coordinates_to_dbu() -> None:
    layout, _ = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    poly = layout.cells["LEAF"].polygons[0]
    assert poly.points.dtype == np.int64
    assert poly.bounds_dbu == (0, 0, 1000, 500)
    assert poly.layer == 10


def test_read_preserves_reference_transform() -> None:
    layout, _ = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    ref = layout.cells["TOP"].references[0]
    assert ref.cell_name == "LEAF"
    assert ref.origin_dbu == (5000, 5000)
    assert ref.rotation_rad == pytest.approx(math.pi / 4)
    assert ref.magnification == pytest.approx(2.0)


def test_read_preserves_rectangular_repetition() -> None:
    layout, _ = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    rep = layout.cells["TOP"].references[1].repetition
    assert isinstance(rep, RectangularRepetition)
    assert (rep.columns, rep.rows) == (3, 2)
    assert rep.spacing_dbu == (10000, 10000)
    assert rep.offsets_dbu().shape == (6, 2)


def test_read_converts_paths_to_polygons_and_counts_them() -> None:
    # gdstk writes FlexPath as BOUNDARY, so this must NOT go through a file.
    lib = gdstk.Library("LIB", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("WITH_PATH")
    cell.add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5), layer=10))
    cell.add(gdstk.FlexPath([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)], 0.2, layer=11))
    assert len(cell.paths) == 1, "in-memory cell must hold a real path"

    layout, report = library_to_layout(lib, TechConfig(), LayerMap.default(), source="memory")

    assert report.paths_converted == 1
    assert len(layout.cells["WITH_PATH"].polygons) == 2
    assert {p.layer for p in layout.cells["WITH_PATH"].polygons} == {10, 11}


def test_read_rejects_a_library_whose_grid_differs() -> None:
    lib = gdstk.Library("LIB", unit=1e-6, precision=2.5e-10)
    lib.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5), layer=10))
    with pytest.raises(GridMismatchError, match="design grid"):
        library_to_layout(lib, TechConfig(design_grid_nm=1.0), LayerMap.default(), "memory")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_streams.py -v`
Expected: FAIL — `ImportError: cannot import name 'library_to_layout'`.

- [ ] **Step 3: Write `io/report.py`**

```python
"""Structured reports describing what a read actually did."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadReport:
    """What was found, converted, and preserved when reading a stream.

    ``paths_converted`` exists so that path-to-polygon conversion is always
    visible: geometry may change representation on import, but never silently.
    """

    source: str
    cell_count: int
    polygon_count: int
    label_count: int
    reference_count: int
    paths_converted: int
    file_precision_m: float
    top_cells: tuple[str, ...]

    def summary(self) -> str:
        parts = [
            f"{self.cell_count} cells",
            f"{self.polygon_count} polygons",
            f"{self.reference_count} references",
            f"{self.label_count} labels",
        ]
        if self.paths_converted:
            parts.append(f"{self.paths_converted} paths converted to polygons")
        return f"{self.source}: " + ", ".join(parts)
```

- [ ] **Step 4: Add `library_to_layout` to `io/_gdstk_bridge.py`**

Append these imports to the existing import block:

```python
import gdstk

from masklayout.io.report import ReadReport
from masklayout.model.cell import (
    Cell,
    ExplicitRepetition,
    RectangularRepetition,
    Reference,
    Repetition,
)
from masklayout.model.geometry import Label, Polygon
from masklayout.model.layers import LayerMap
from masklayout.model.layout import Layout
```

Then append:

```python
def _repetition_to_model(
    repetition: gdstk.Repetition | None, precision_um: float
) -> Repetition | None:
    """Convert a gdstk repetition.

    gdstk's Repetition has no ``type`` attribute; the kind is inferred from
    which fields are populated.
    """
    if repetition is None:
        return None
    if repetition.columns is not None and repetition.rows is not None:
        spacing = repetition.spacing
        if spacing is not None:
            dx, dy = um_to_dbu(np.array([spacing], dtype=np.float64), precision_um)[0]
            return RectangularRepetition(
                columns=int(repetition.columns),
                rows=int(repetition.rows),
                spacing_dbu=(int(dx), int(dy)),
            )
    offsets = np.asarray(repetition.get_offsets(), dtype=np.float64)
    return ExplicitRepetition(offsets_dbu_array=um_to_dbu(offsets, precision_um))


def library_to_layout(
    library: gdstk.Library,
    tech: TechConfig,
    layers: LayerMap,
    source: str,
) -> tuple[Layout, ReadReport]:
    """Convert a gdstk library into the typed model.

    Paths are converted to polygons and counted. Labels are preserved.
    A grid mismatch is an error, never a silent regrid.
    """
    check_library_grid(library.unit, library.precision, tech)
    precision_um = tech.precision_um

    layout = Layout(name=library.name, tech=tech, layers=layers)
    paths_converted = 0
    label_count = 0
    reference_count = 0

    for gcell in library.cells:
        if not isinstance(gcell, gdstk.Cell):
            raise UnsupportedEntityError(
                f"{source}: cell {gcell!r} is a raw cell and cannot be modelled"
            )
        cell = Cell(name=gcell.name)

        for gpoly in gcell.polygons:
            cell.polygons.append(
                Polygon(
                    points=um_to_dbu(np.asarray(gpoly.points), precision_um),
                    layer=int(gpoly.layer),
                    datatype=int(gpoly.datatype),
                )
            )

        for gpath in gcell.paths:
            paths_converted += 1
            for gpoly in gpath.to_polygons():
                cell.polygons.append(
                    Polygon(
                        points=um_to_dbu(np.asarray(gpoly.points), precision_um),
                        layer=int(gpoly.layer),
                        datatype=int(gpoly.datatype),
                    )
                )

        for glabel in gcell.labels:
            label_count += 1
            origin = um_to_dbu(np.array([glabel.origin], dtype=np.float64), precision_um)[0]
            cell.labels.append(
                Label(
                    text=glabel.text,
                    origin_dbu=(int(origin[0]), int(origin[1])),
                    layer=int(glabel.layer),
                    datatype=int(glabel.texttype),
                )
            )

        for gref in gcell.references:
            reference_count += 1
            origin = um_to_dbu(np.array([gref.origin], dtype=np.float64), precision_um)[0]
            cell.references.append(
                Reference(
                    cell_name=gref.cell.name,
                    origin_dbu=(int(origin[0]), int(origin[1])),
                    rotation_rad=float(gref.rotation),
                    magnification=float(gref.magnification),
                    x_reflection=bool(gref.x_reflection),
                    repetition=_repetition_to_model(gref.repetition, precision_um),
                )
            )

        layout.add(cell)

    report = ReadReport(
        source=source,
        cell_count=len(layout.cells),
        polygon_count=layout.polygon_count(),
        label_count=label_count,
        reference_count=reference_count,
        paths_converted=paths_converted,
        file_precision_m=float(library.precision),
        top_cells=tuple(layout.top_cells()),
    )
    return layout, report
```

Add `UnsupportedEntityError` to the `masklayout.io.errors` import in this file.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_streams.py -v`
Expected: all 6 read tests PASS.

- [ ] **Step 6: Verify and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src tests && uv run pytest -q
git add src/masklayout/io tests/unit/test_streams.py
git commit -m "feat(m1): read GDS/OASIS into the typed model with a read report

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 4: Writer and round-trip

**Files:**
- Modify: `src/masklayout/io/_gdstk_bridge.py` — add `layout_to_library`
- Modify: `src/masklayout/geometry/context.py` — add `read_gds`, `read_oas`
- Create: `src/masklayout/io/streams.py`
- Modify: `src/masklayout/io/__init__.py` — export the public surface
- Test: `tests/unit/test_streams.py` (write and round-trip cases)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `layout_to_library(layout) -> gdstk.Library`
  - `read_gds(path, tech=None, layers=None) -> tuple[Layout, ReadReport]`
  - `read_oas(path, tech=None, layers=None) -> tuple[Layout, ReadReport]`
  - `write_gds(layout, path, timestamp=None) -> None`
  - `write_oas(layout, path) -> None`

- [ ] **Step 1: Append the failing round-trip tests to `tests/unit/test_streams.py`**

```python
def _model_layout():
    from masklayout.io._gdstk_bridge import library_to_layout

    layout, _ = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    return layout


def test_gds_round_trip_preserves_hierarchy_and_geometry(tmp_path: Path) -> None:
    from masklayout.io.streams import read_gds, write_gds

    original = _model_layout()
    path = tmp_path / "rt.gds"
    write_gds(original, path)
    restored, report = read_gds(path)

    assert sorted(restored.cells) == sorted(original.cells)
    assert restored.top_cells() == original.top_cells()
    assert restored.dependencies("TOP") == {"LEAF"}
    assert report.source == str(path)

    before = original.cells["LEAF"].polygons[0]
    after = restored.cells["LEAF"].polygons[0]
    assert np.array_equal(np.sort(before.points, axis=0), np.sort(after.points, axis=0))
    assert (after.layer, after.datatype) == (before.layer, before.datatype)


def test_gds_round_trip_preserves_reference_transform(tmp_path: Path) -> None:
    from masklayout.io.streams import read_gds, write_gds

    path = tmp_path / "refs.gds"
    write_gds(_model_layout(), path)
    restored, _ = read_gds(path)

    ref = restored.cells["TOP"].references[0]
    assert ref.cell_name == "LEAF"
    assert ref.origin_dbu == (5000, 5000)
    assert ref.rotation_rad == pytest.approx(math.pi / 4, abs=1e-9)
    assert ref.magnification == pytest.approx(2.0)


def test_gds_round_trip_preserves_rectangular_repetition(tmp_path: Path) -> None:
    from masklayout.io.streams import read_gds, write_gds

    path = tmp_path / "array.gds"
    write_gds(_model_layout(), path)
    restored, _ = read_gds(path)

    rep = restored.cells["TOP"].references[1].repetition
    assert isinstance(rep, RectangularRepetition)
    assert (rep.columns, rep.rows) == (3, 2)
    assert rep.spacing_dbu == (10000, 10000)


def test_oasis_round_trip_preserves_hierarchy(tmp_path: Path) -> None:
    from masklayout.io.streams import read_oas, write_oas

    path = tmp_path / "rt.oas"
    write_oas(_model_layout(), path)
    restored, report = read_oas(path)

    assert sorted(restored.cells) == ["LEAF", "TOP"]
    assert restored.top_cells() == ["TOP"]
    assert report.polygon_count == 1


def test_write_gds_is_byte_reproducible(tmp_path: Path) -> None:
    import hashlib

    from masklayout.io.streams import write_gds

    digests = []
    for name in ("a.gds", "b.gds"):
        path = tmp_path / name
        write_gds(_model_layout(), path)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert digests[0] == digests[1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_streams.py -k round_trip -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'masklayout.io.streams'`.

- [ ] **Step 3: Add reading to `GeomContext`**

In `src/masklayout/geometry/context.py`, add two methods to `GeomContext`:

```python
    def read_gds(self, path: Path | str) -> gdstk.Library:
        """Read a GDSII stream into a gdstk library."""
        return gdstk.read_gds(path)

    def read_oas(self, path: Path | str) -> gdstk.Library:
        """Read an OASIS stream into a gdstk library."""
        return gdstk.read_oas(path)
```

- [ ] **Step 4: Add `layout_to_library` to `io/_gdstk_bridge.py`**

```python
def _model_repetition_to_gdstk(
    repetition: Repetition | None, precision_um: float
) -> gdstk.Repetition | None:
    if repetition is None:
        return None
    if isinstance(repetition, RectangularRepetition):
        dx, dy = dbu_to_um(np.array(repetition.spacing_dbu, dtype=np.int64), precision_um)
        return gdstk.Repetition(
            columns=repetition.columns,
            rows=repetition.rows,
            spacing=(float(dx), float(dy)),
        )
    offsets = dbu_to_um(repetition.offsets_dbu(), precision_um)
    return gdstk.Repetition(offsets=[tuple(row) for row in offsets.tolist()])


def layout_to_library(layout: Layout) -> gdstk.Library:
    """Build a gdstk library from the typed model.

    Cells are emitted in sorted name order and their contents in stored
    order, so output is deterministic for identical input.
    """
    precision_um = layout.tech.precision_um
    library = gdstk.Library(
        layout.name, unit=_EXPECTED_UNIT_M, precision=layout.tech.precision_m
    )

    built: dict[str, gdstk.Cell] = {}
    for name in sorted(layout.cells):
        built[name] = library.new_cell(name)

    for name in sorted(layout.cells):
        cell = layout.cells[name]
        gcell = built[name]

        for poly in cell.polygons:
            gcell.add(
                gdstk.Polygon(
                    dbu_to_um(poly.points, precision_um),
                    layer=poly.layer,
                    datatype=poly.datatype,
                )
            )

        for label in cell.labels:
            origin = dbu_to_um(np.array(label.origin_dbu, dtype=np.int64), precision_um)
            gcell.add(
                gdstk.Label(
                    label.text,
                    (float(origin[0]), float(origin[1])),
                    layer=label.layer,
                    texttype=label.datatype,
                )
            )

        for ref in cell.references:
            if ref.cell_name not in built:
                raise UnknownCellError(
                    f"cell {name!r} references unknown cell {ref.cell_name!r}"
                )
            origin = dbu_to_um(np.array(ref.origin_dbu, dtype=np.int64), precision_um)
            greference = gdstk.Reference(
                built[ref.cell_name],
                (float(origin[0]), float(origin[1])),
                rotation=ref.rotation_rad,
                magnification=ref.magnification,
                x_reflection=ref.x_reflection,
            )
            greference.repetition = _model_repetition_to_gdstk(ref.repetition, precision_um)
            gcell.add(greference)

    return library
```

Add `from masklayout.model.layout import Layout, UnknownCellError` to the imports.

- [ ] **Step 5: Write `io/streams.py`**

```python
"""Public stream I/O: GDSII and OASIS."""

from __future__ import annotations

import datetime
from pathlib import Path

from masklayout.config import TechConfig
from masklayout.geometry.context import GeomContext
from masklayout.io._gdstk_bridge import layout_to_library, library_to_layout
from masklayout.io.report import ReadReport
from masklayout.model.layers import LayerMap
from masklayout.model.layout import Layout


def _context(layout_or_tech: Layout | TechConfig, timestamp: datetime.datetime | None):
    tech = layout_or_tech.tech if isinstance(layout_or_tech, Layout) else layout_or_tech
    return GeomContext(tech, timestamp=timestamp)


def read_gds(
    path: Path | str,
    tech: TechConfig | None = None,
    layers: LayerMap | None = None,
) -> tuple[Layout, ReadReport]:
    """Read a GDSII file into the typed model."""
    tech = tech or TechConfig()
    library = GeomContext(tech).read_gds(path)
    return library_to_layout(library, tech, layers or LayerMap.default(), str(path))


def read_oas(
    path: Path | str,
    tech: TechConfig | None = None,
    layers: LayerMap | None = None,
) -> tuple[Layout, ReadReport]:
    """Read an OASIS file into the typed model."""
    tech = tech or TechConfig()
    library = GeomContext(tech).read_oas(path)
    return library_to_layout(library, tech, layers or LayerMap.default(), str(path))


def write_gds(
    layout: Layout,
    path: Path | str,
    timestamp: datetime.datetime | None = None,
) -> None:
    """Write the layout as GDSII, with a pinned timestamp for reproducibility."""
    _context(layout, timestamp).write_gds(layout_to_library(layout), path)


def write_oas(layout: Layout, path: Path | str) -> None:
    """Write the layout as OASIS."""
    _context(layout, None).write_oas(layout_to_library(layout), path)
```

- [ ] **Step 6: Export the public surface in `io/__init__.py`**

```python
"""Stream I/O for GDSII and OASIS."""

from masklayout.io.errors import (
    GridMismatchError,
    OffGridCoordinateError,
    UnsupportedEntityError,
)
from masklayout.io.report import ReadReport
from masklayout.io.streams import read_gds, read_oas, write_gds, write_oas

__all__ = [
    "GridMismatchError",
    "OffGridCoordinateError",
    "ReadReport",
    "UnsupportedEntityError",
    "read_gds",
    "read_oas",
    "write_gds",
    "write_oas",
]
```

- [ ] **Step 7: Run the full stream test file**

Run: `uv run pytest tests/unit/test_streams.py -v`
Expected: all PASS.

- [ ] **Step 8: Verify everything**

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

- [ ] **Step 9: Commit, push, verify CI**

```bash
git add src/masklayout tests/unit/test_streams.py
git commit -m "feat(m1): write GDS/OASIS from the model with round-trip fidelity

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
gh run watch
```

- [ ] **Step 10: Mark M1 complete in the README**

Change the M1 row to `**complete**`, commit, push.

---

## Self-Review

**Spec coverage.** M1's acceptance per §10 is "Import, preserve references, and round-trip export valid layouts", plus pinned timestamps. Task 1 builds the model that holds references and hierarchy. Task 2 enforces the grid contract that makes integer coordinates exact. Task 3 imports, preserving references and repetitions, reporting path conversion. Task 4 exports and proves round-trip fidelity plus byte reproducibility. The two M1 planning decisions — hard error on grid mismatch, convert-and-count paths — are each covered by a dedicated test.

**Placeholder scan.** No TBD or "handle edge cases". Every code step is complete and runnable; every test step names the command and expected result.

**Type consistency.** `ExplicitRepetition`'s field is `offsets_dbu_array` and its method is `offsets_dbu()`; Task 1 Step 4 flags that the Step 1 test must use the keyword `offsets_dbu_array=`. `library_to_layout(library, tech, layers, source)` has the same four-parameter signature in Task 3's definition and in every Task 3 and Task 4 test. `Polygon(points=, layer=, datatype=)` is consistent throughout. `GeomContext.read_gds`/`read_oas` added in Task 4 Step 3 are used only in `streams.py`, keeping the gdstk allowlist at two modules.

**Known deferrals.** OASIS repetition kinds beyond rectangular collapse to explicit offsets — lossless in placement, larger on disk. Region selection and flattening (`materialize`) belong to M4/M5, not here.
