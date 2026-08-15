# masklayout M0 — Scaffold and Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `masklayout` project skeleton with reproducible tooling, a validated `TechConfig`, a layer map, and the `GeomContext` boundary that owns every `gdstk` call — with CI green.

**Architecture:** A `src/` layout package managed by `uv`. Configuration and layer models are pydantic (validated, versioned, serializable). `GeomContext` is the single module permitted to import `gdstk`; it injects the configured precision and fracture limit into every call so gdstk's defaults can never silently replace them. An AST-based test enforces that boundary.

**Tech Stack:** Python 3.12+, `uv`, `gdstk`, `numpy`, `pydantic` v2, `pytest`, `ruff`, `mypy`, GitHub Actions.

**Design reference:** `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`. Section numbers below refer to it.

## Global Constraints

Every task's requirements implicitly include this section.

- Python `>=3.12`.
- **No module under `src/` may import `gdstk` except `src/masklayout/geometry/context.py`.** Test files may import it freely (§3, §8).
- **Never rely on a gdstk default for `precision` or `max_points`.** gdstk defaults to `precision=1e-3` and `max_points=199`, which would silently override the configured grid and fracture limit (§3, §12).
- The public API is independent of `gdstk` types (§12).
- Explicit units in every parameter name: `*_nm`, `*_um` (§12).
- Config errors fail at load, naming the offending values (§6.3).
- Deterministic output: GDS writes use a pinned timestamp, default epoch 0 (§5).
- `mrc_deburr_nm` derives from `design_grid_nm / 2` unless explicitly overridden (§3).
- Line length 100. `ruff format` clean. `mypy` strict on `src`.
- M0 dependencies are `gdstk`, `numpy`, `pydantic` only. `shapely` arrives at M2 with the spatial index; `pyyaml` at M4 with rule decks. Do not add them now — mypy `strict` warns on override sections for modules that are never imported.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, tool configuration |
| `.python-version` | Pins 3.12 for `uv` |
| `.github/workflows/ci.yml` | Lint, format check, type check, test |
| `src/masklayout/__init__.py` | Package version and public surface |
| `src/masklayout/py.typed` | Marks the package as typed |
| `src/masklayout/config.py` | `TechConfig` and its validation |
| `src/masklayout/model/layers.py` | `Layer`, `LayerMap`, default layer table |
| `src/masklayout/geometry/context.py` | `GeomContext` — the only gdstk importer |
| `tests/unit/test_package.py` | Package smoke test |
| `tests/unit/test_layers.py` | Layer map behaviour |
| `tests/unit/test_config.py` | Config validation |
| `tests/unit/test_context.py` | GeomContext behaviour |
| `tests/test_architecture.py` | gdstk import boundary guard |

---

## Task 1: Project scaffold, tooling, and CI

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.github/workflows/ci.yml`
- Create: `src/masklayout/__init__.py`, `src/masklayout/py.typed`
- Test: `tests/unit/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `masklayout.__version__` (`str`). An installed, importable `masklayout` package and working `uv run pytest` / `ruff` / `mypy` commands that every later task depends on.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_package.py`:

```python
"""Smoke test: the package is importable and declares a version."""

import masklayout


def test_package_exposes_version() -> None:
    assert isinstance(masklayout.__version__, str)
    assert masklayout.__version__.count(".") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_package.py -v`
Expected: FAIL — `uv` cannot sync (no `pyproject.toml`), or `ModuleNotFoundError: No module named 'masklayout'`.

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "masklayout"
version = "0.1.0"
description = "Recipe-driven computational mask-layout toolkit for semiconductor lithography masks"
readme = "README.md"
requires-python = ">=3.12"
license = "Apache-2.0"
authors = [{ name = "Alex Wang", email = "wangzhihuan0815@gmail.com" }]
dependencies = [
    "gdstk>=1.0.1",
    "numpy>=2.0",
    "pydantic>=2.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/masklayout"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "mypy>=1.11",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src", "tests"]

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

- [ ] **Step 4: Create `.python-version`**

```text
3.12
```

- [ ] **Step 5: Create the package**

`src/masklayout/__init__.py`:

```python
"""masklayout — a computational mask-layout toolkit for lithography masks."""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

Create an empty `src/masklayout/py.typed`:

```bash
touch src/masklayout/py.typed
```

- [ ] **Step 6: Sync and run the test**

Run:
```bash
uv sync
uv run pytest tests/unit/test_package.py -v
```
Expected: PASS.

- [ ] **Step 7: Verify lint, format, and types are clean**

Run:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy src tests
```
Expected: all clean. `ruff format` may rewrite files — that is fine, commit the result.

- [ ] **Step 8: Create the CI workflow**

`.github/workflows/ci.yml`. Note both action tags are pinned to versions verified to exist — `astral-sh/setup-uv` has **no floating `v10` major tag**, so the exact release must be used:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.0
        with:
          enable-cache: true

      - name: Sync dependencies
        run: uv sync

      - name: Lint
        run: uv run ruff check .

      - name: Format check
        run: uv run ruff format --check .

      - name: Type check
        run: uv run mypy src tests

      - name: Test
        run: uv run pytest -v
```

- [ ] **Step 9: Commit and push**

```bash
git add pyproject.toml uv.lock .python-version .github src tests
git commit -m "feat(m0): project scaffold, tooling, and CI

src/ layout managed by uv, with ruff, mypy strict, and pytest wired up.
CI action tags pinned to verified releases: setup-uv has no floating
v10 major tag, so v10.0.0 is used explicitly.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

- [ ] **Step 10: Verify CI passes**

Run: `gh run watch` (or `gh run list --limit 1`)
Expected: the `check` job succeeds. If it fails, fix before starting Task 2.

---

## Task 2: Layer and LayerMap

**Files:**
- Create: `src/masklayout/model/__init__.py`, `src/masklayout/model/layers.py`
- Test: `tests/unit/test_layers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Layer(number: int, datatype: int, name: str)` — frozen pydantic model.
  - `DEFAULT_LAYERS: dict[str, tuple[int, int]]`
  - `LayerMap(layers: dict[str, Layer])` — frozen; `LayerMap.default() -> LayerMap`; `layer_map["TARGET"] -> Layer`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layers.py`:

```python
"""Layer map behaviour."""

import pytest
from pydantic import ValidationError

from masklayout.model.layers import Layer, LayerMap


def test_default_map_contains_the_engineering_layers() -> None:
    lm = LayerMap.default()
    assert lm["TARGET"] == Layer(number=10, datatype=0, name="TARGET")
    assert lm["POST_OPC"] == Layer(number=11, datatype=0, name="POST_OPC")
    assert lm["SRAF"] == Layer(number=12, datatype=0, name="SRAF")
    assert lm["DEBUG_SOURCE"] == Layer(number=200, datatype=0, name="DEBUG_SOURCE")
    assert lm["DEBUG_MARKERS"] == Layer(number=201, datatype=0, name="DEBUG_MARKERS")
    assert lm["OVERLAY_ADD"] == Layer(number=202, datatype=0, name="OVERLAY_ADD")
    assert lm["OVERLAY_REMOVE"] == Layer(number=203, datatype=0, name="OVERLAY_REMOVE")


def test_default_map_contains_field_layer_for_tone_inversion() -> None:
    # FIELD is required by design decision 5: tone inversion needs an explicit extent.
    assert LayerMap.default()["FIELD"] == Layer(number=20, datatype=0, name="FIELD")


def test_layer_is_frozen() -> None:
    layer = Layer(number=10, datatype=0, name="TARGET")
    with pytest.raises(ValidationError):
        layer.number = 11


def test_layer_number_out_of_gds_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Layer(number=70000, datatype=0, name="TOO_BIG")
    with pytest.raises(ValidationError):
        Layer(number=-1, datatype=0, name="NEGATIVE")


def test_duplicate_layer_datatype_pair_is_rejected_naming_both() -> None:
    with pytest.raises(ValidationError) as excinfo:
        LayerMap(
            layers={
                "TARGET": Layer(number=10, datatype=0, name="TARGET"),
                "SHADOW": Layer(number=10, datatype=0, name="SHADOW"),
            }
        )
    message = str(excinfo.value)
    assert "TARGET" in message
    assert "SHADOW" in message
    assert "10/0" in message


def test_unknown_layer_lookup_lists_known_layers() -> None:
    with pytest.raises(KeyError) as excinfo:
        LayerMap.default()["NOPE"]
    assert "TARGET" in str(excinfo.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_layers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'masklayout.model'`.

- [ ] **Step 3: Write the implementation**

Create empty `src/masklayout/model/__init__.py`:

```python
"""Typed layout models."""
```

Create `src/masklayout/model/layers.py`:

```python
"""Logical layer definitions and the layer map.

Layer numbers are configurable. The defaults below are the engineering
convention from the V1 design, section "Layer policy".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

GDS_MAX = 65535

DEFAULT_LAYERS: dict[str, tuple[int, int]] = {
    "TARGET": (10, 0),
    "POST_OPC": (11, 0),
    "SRAF": (12, 0),
    "FIELD": (20, 0),
    "DEBUG_SOURCE": (200, 0),
    "DEBUG_MARKERS": (201, 0),
    "OVERLAY_ADD": (202, 0),
    "OVERLAY_REMOVE": (203, 0),
}


class Layer(BaseModel):
    """A GDSII layer/datatype pair with a logical name."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=0, le=GDS_MAX)
    datatype: int = Field(ge=0, le=GDS_MAX)
    name: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"{self.name}({self.number}/{self.datatype})"


class LayerMap(BaseModel):
    """Maps logical layer names onto GDSII layer/datatype pairs."""

    model_config = ConfigDict(frozen=True)

    layers: dict[str, Layer]

    @model_validator(mode="after")
    def _reject_duplicate_pairs(self) -> LayerMap:
        seen: dict[tuple[int, int], str] = {}
        for name, layer in self.layers.items():
            key = (layer.number, layer.datatype)
            if key in seen:
                raise ValueError(
                    f"layer/datatype {key[0]}/{key[1]} is assigned to both "
                    f"{seen[key]!r} and {name!r}; logical layers must be distinct"
                )
            seen[key] = name
        return self

    @classmethod
    def default(cls) -> LayerMap:
        """The engineering-convention layer map."""
        return cls(
            layers={
                name: Layer(number=number, datatype=datatype, name=name)
                for name, (number, datatype) in DEFAULT_LAYERS.items()
            }
        )

    def __getitem__(self, name: str) -> Layer:
        try:
            return self.layers[name]
        except KeyError:
            raise KeyError(
                f"unknown logical layer {name!r}; known layers: {sorted(self.layers)}"
            ) from None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_layers.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Verify lint, format, and types**

Run:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy src tests
```
Expected: clean.

- [ ] **Step 6: Commit and push**

```bash
git add src/masklayout/model tests/unit/test_layers.py
git commit -m "feat(m0): add Layer and LayerMap with the default layer table

Includes FIELD (20/0), required by the design's tone-inversion decision.
Duplicate layer/datatype pairs are rejected, naming both logical layers.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 3: TechConfig with grid validation

**Files:**
- Create: `src/masklayout/config.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `TechConfig` — frozen pydantic model with fields `name`, `design_grid_nm`, `mask_grid_nm`, `magnification`, `tone`, `max_chord_error_nm`, `max_segment_length_nm`, `min_segment_length_nm`, `max_vertices_per_polygon`, `min_polygon_area_nm2`, `min_edge_length_nm`, `fracture_vertex_limit`, `mrc_deburr_nm`; and properties `effective_mrc_deburr_nm -> float`, `precision_um -> float`, `precision_m -> float`. Task 4 consumes `precision_um`, `precision_m`, and `fracture_vertex_limit`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_config.py`:

```python
"""TechConfig validation."""

import pytest
from pydantic import ValidationError

from masklayout.config import TechConfig


def test_defaults_are_valid() -> None:
    tech = TechConfig()
    assert tech.design_grid_nm == 1.0
    assert tech.mask_grid_nm == 0.5
    assert tech.magnification == 4
    assert tech.tone == "clear"
    assert tech.fracture_vertex_limit == 4000


def test_config_is_frozen() -> None:
    tech = TechConfig()
    with pytest.raises(ValidationError):
        tech.design_grid_nm = 2.0


def test_precision_properties_convert_grid_correctly() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    assert tech.precision_um == pytest.approx(0.001)
    assert tech.precision_m == pytest.approx(1e-9)


def test_mrc_deburr_derives_from_half_the_design_grid() -> None:
    assert TechConfig(design_grid_nm=1.0).effective_mrc_deburr_nm == pytest.approx(0.5)
    # A finer grid needs a mask grid it divides exactly: 4 * 0.4 / 0.4 == 4.
    # Passing design_grid_nm=0.4 alone is INVALID against the default
    # mask_grid_nm=0.5, because 4 * 0.4 / 0.5 == 3.2.
    finer = TechConfig(design_grid_nm=0.4, mask_grid_nm=0.4)
    assert finer.effective_mrc_deburr_nm == pytest.approx(0.2)


def test_mrc_deburr_override_is_respected() -> None:
    tech = TechConfig(design_grid_nm=1.0, mrc_deburr_nm=0.25)
    assert tech.effective_mrc_deburr_nm == pytest.approx(0.25)


def test_mask_grid_multiple_violation_names_all_three_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TechConfig(design_grid_nm=1.0, mask_grid_nm=0.3, magnification=4)
    message = str(excinfo.value)
    assert "magnification" in message
    assert "design_grid_nm" in message
    assert "mask_grid_nm" in message


def test_mask_grid_multiple_tolerates_float_representation_error() -> None:
    # 4 * 0.1 / 0.05 evaluates to 8.000000000000002 in IEEE 754.
    # This is a legal configuration and must not be rejected.
    tech = TechConfig(design_grid_nm=0.1, mask_grid_nm=0.05, magnification=4)
    assert tech.design_grid_nm == 0.1


def test_segment_length_ordering_is_enforced() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TechConfig(min_segment_length_nm=20.0, max_segment_length_nm=10.0)
    assert "min_segment_length_nm" in str(excinfo.value)


def test_non_positive_grid_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechConfig(design_grid_nm=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'masklayout.config'`.

- [ ] **Step 3: Write the implementation**

Create `src/masklayout/config.py`:

```python
"""Technology configuration.

Every value here is a software default. None of them are foundry,
mask-writer, or process-node rules. All are configurable and are recorded
in generated manifests.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Tone = Literal["clear", "dark"]

_MULTIPLE_TOLERANCE = 1e-9


class TechConfig(BaseModel):
    """Geometry, grid, and export configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = "generic_mask_geometry_v1"

    # Grid and mask scale.
    design_grid_nm: float = Field(default=1.0, gt=0)
    mask_grid_nm: float = Field(default=0.5, gt=0)
    magnification: int = Field(default=4, ge=1)
    tone: Tone = "clear"

    # Curve tessellation.
    max_chord_error_nm: float = Field(default=1.0, gt=0)
    max_segment_length_nm: float = Field(default=10.0, gt=0)
    min_segment_length_nm: float = Field(default=1.0, gt=0)
    max_vertices_per_polygon: int = Field(default=4000, ge=4)

    # Cleanup thresholds.
    min_polygon_area_nm2: float = Field(default=4.0, ge=0)
    min_edge_length_nm: float = Field(default=1.0, ge=0)

    # Export.
    fracture_vertex_limit: int = Field(default=4000, ge=4)

    # Verification. None means "derive from design_grid_nm".
    mrc_deburr_nm: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_mask_grid_multiple(self) -> TechConfig:
        """Scaling to mask scale must be an exact integer multiply.

        If this holds, export needs no re-snap and introduces no second
        grid-error term.
        """
        scaled = self.magnification * self.design_grid_nm
        ratio = scaled / self.mask_grid_nm
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=_MULTIPLE_TOLERANCE):
            raise ValueError(
                "magnification * design_grid_nm must be an exact multiple of mask_grid_nm: "
                f"magnification={self.magnification}, "
                f"design_grid_nm={self.design_grid_nm}, "
                f"mask_grid_nm={self.mask_grid_nm} "
                f"(gives {scaled} / {self.mask_grid_nm} = {ratio})"
            )
        return self

    @model_validator(mode="after")
    def _validate_segment_lengths(self) -> TechConfig:
        if self.min_segment_length_nm > self.max_segment_length_nm:
            raise ValueError(
                "min_segment_length_nm must not exceed max_segment_length_nm: "
                f"min_segment_length_nm={self.min_segment_length_nm}, "
                f"max_segment_length_nm={self.max_segment_length_nm}"
            )
        return self

    @property
    def effective_mrc_deburr_nm(self) -> float:
        """Deburr radius used by MRC.

        Compensates the quantization error accumulated by the erode/dilate
        round trip, which is a function of the grid — so it derives from the
        grid unless explicitly overridden.
        """
        if self.mrc_deburr_nm is None:
            return self.design_grid_nm / 2.0
        return self.mrc_deburr_nm

    @property
    def precision_um(self) -> float:
        """Design grid in micrometres, for gdstk boolean/offset/fracture."""
        return self.design_grid_nm / 1000.0

    @property
    def precision_m(self) -> float:
        """Design grid in metres, for the gdstk Library precision."""
        return self.design_grid_nm * 1e-9
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Verify lint, format, and types**

Run:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy src tests
```
Expected: clean.

- [ ] **Step 6: Commit and push**

```bash
git add src/masklayout/config.py tests/unit/test_config.py
git commit -m "feat(m0): add TechConfig with exact-multiple grid validation

magnification * design_grid_nm must land exactly on mask_grid_nm, so
export scaling is a pure integer multiply with no re-snap and no second
grid-error term. Validation tolerates IEEE 754 representation error and
names all three values on failure.

mrc_deburr_nm derives from design_grid_nm / 2 unless overridden.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Task 4: GeomContext and the gdstk import boundary

**Files:**
- Create: `src/masklayout/geometry/__init__.py`, `src/masklayout/geometry/context.py`
- Test: `tests/unit/test_context.py`, `tests/test_architecture.py`

**Interfaces:**
- Consumes: `TechConfig` from Task 3 — specifically `precision_um`, `precision_m`, `fracture_vertex_limit`.
- Produces: `GeomContext(tech: TechConfig, timestamp: datetime | None = None)` with `precision_um` property and methods `boolean`, `offset`, `fracture`, `new_library`, `write_gds`, `write_oas`. Every later milestone routes its geometry operations through this object.

**Why this task exists:** gdstk's defaults (`precision=1e-3`, `max_points=199`) silently override configuration. Step 1's third test is the regression test for exactly that failure.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_context.py`:

```python
"""GeomContext behaviour.

Test files may import gdstk directly; the import boundary applies to src/ only.
"""

import datetime
import hashlib
import math
from pathlib import Path

import gdstk
import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.context import GeomContext


def test_boolean_output_lands_exactly_on_the_design_grid() -> None:
    ctx = GeomContext(TechConfig(design_grid_nm=1.0))
    first = gdstk.rectangle((0.0, 0.0), (1.0, 0.2))
    first.rotate(math.radians(37))
    second = gdstk.rectangle((0.5, 0.0), (1.5, 0.2))
    second.rotate(math.radians(37))

    result = ctx.boolean(first, second, "or")

    assert result
    for polygon in result:
        scaled = np.asarray(polygon.points) / ctx.precision_um
        residue = np.abs(scaled - np.round(scaled)).max()
        assert residue < 1e-9, f"vertex off the design grid by {residue} grid units"


def test_write_gds_honours_the_configured_fracture_limit(tmp_path: Path) -> None:
    # Regression test: gdstk's write_gds defaults to max_points=199, which would
    # silently fracture this polygon regardless of fracture_vertex_limit.
    tech = TechConfig(fracture_vertex_limit=4000)
    ctx = GeomContext(tech)
    library = ctx.new_library("TOP")
    cell = library.new_cell("TOP")

    circle = gdstk.ellipse((0.0, 0.0), 10.0, tolerance=1e-4)
    assert len(circle.points) > 199, "test is meaningless unless it exceeds gdstk's default"
    assert len(circle.points) <= tech.fracture_vertex_limit
    cell.add(circle)

    out = tmp_path / "circle.gds"
    ctx.write_gds(library, out)

    read_back = gdstk.read_gds(out)
    polygons = read_back.cells[0].polygons
    assert len(polygons) == 1, "polygon was fractured; gdstk's max_points=199 default leaked"


def test_write_gds_is_byte_reproducible(tmp_path: Path) -> None:
    digests = []
    for name in ("first.gds", "second.gds"):
        ctx = GeomContext(TechConfig())
        library = ctx.new_library("TOP")
        cell = library.new_cell("TOP")
        cell.add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
        path = tmp_path / name
        ctx.write_gds(library, path)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert digests[0] == digests[1]


def test_write_gds_uses_the_pinned_timestamp(tmp_path: Path) -> None:
    pinned = datetime.datetime(2001, 2, 3, 4, 5, 6)
    ctx = GeomContext(TechConfig(), timestamp=pinned)
    library = ctx.new_library("TOP")
    library.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
    path = tmp_path / "stamped.gds"
    ctx.write_gds(library, path)

    other = GeomContext(TechConfig(), timestamp=datetime.datetime(1999, 1, 1))
    other_library = other.new_library("TOP")
    other_library.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
    other_path = tmp_path / "other.gds"
    other.write_gds(other_library, other_path)

    assert path.read_bytes() != other_path.read_bytes()


def test_write_oas_round_trips(tmp_path: Path) -> None:
    ctx = GeomContext(TechConfig())
    library = ctx.new_library("TOP")
    library.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
    path = tmp_path / "out.oas"
    ctx.write_oas(library, path)

    read_back = gdstk.read_oas(path)
    assert [cell.name for cell in read_back.cells] == ["TOP"]
    assert len(read_back.cells[0].polygons) == 1


def test_fracture_uses_the_configured_limit() -> None:
    ctx = GeomContext(TechConfig(fracture_vertex_limit=100))
    circle = gdstk.ellipse((0.0, 0.0), 10.0, tolerance=1e-4)
    assert len(circle.points) > 100

    pieces = ctx.fracture(circle)

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece.points) <= 100


def test_new_library_uses_the_configured_precision() -> None:
    ctx = GeomContext(TechConfig(design_grid_nm=1.0))
    library = ctx.new_library("TOP")
    assert library.precision == 1e-9
    assert library.unit == 1e-6
```

Create `tests/test_architecture.py`:

```python
"""Architecture guard: gdstk may only be imported by GeomContext."""

import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "masklayout"
ALLOWED = {"geometry/context.py"}


def _imports_gdstk(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "gdstk" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "gdstk":
                return True
    return False


def test_only_geomcontext_imports_gdstk() -> None:
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if path.relative_to(SRC).as_posix() not in ALLOWED
        and _imports_gdstk(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offenders, (
        f"gdstk imported outside GeomContext: {offenders}. "
        "Route the call through masklayout.geometry.context instead."
    )


def test_the_guard_can_actually_detect_an_import() -> None:
    # Guards that never fire are worthless; prove this one fires.
    assert _imports_gdstk(ast.parse("import gdstk"))
    assert _imports_gdstk(ast.parse("from gdstk import boolean"))
    assert not _imports_gdstk(ast.parse("import numpy"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_context.py tests/test_architecture.py -v`
Expected: `test_context.py` tests FAIL with `ModuleNotFoundError: No module named 'masklayout.geometry'`. The `test_architecture.py` tests may already pass — that is expected, since no `src/` module imports gdstk yet.

- [ ] **Step 3: Write the implementation**

Create `src/masklayout/geometry/__init__.py`:

```python
"""Geometry primitives and the gdstk boundary."""
```

Create `src/masklayout/geometry/context.py`:

```python
"""The single boundary between masklayout and gdstk.

This is the only module in the package permitted to import gdstk, enforced
by tests/test_architecture.py.

It exists because gdstk's defaults silently override configuration:
``boolean`` and ``offset`` default to ``precision=1e-3``, and ``write_gds``
defaults to ``max_points=199``. Calling gdstk directly anywhere else would
quietly substitute those for the configured grid and fracture limit.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

import gdstk

from masklayout.config import TechConfig

BooleanOperation = Literal["or", "and", "xor", "not"]
#: Join styles accepted by ``gdstk.offset``. Narrower than FlexPath's join
#: styles: "natural" and "smooth" are path joins and are rejected here.
JoinStyle = Literal["miter", "bevel", "round"]

#: Default pinned timestamp. GDSII embeds a header timestamp that would
#: otherwise make output non-reproducible between runs.
PINNED_TIMESTAMP = datetime.datetime(1970, 1, 1)

#: GDSII user unit: 1 micrometre.
USER_UNIT_M = 1e-6


class GeomContext:
    """Carries the configured precision and fracture limit into every gdstk call."""

    def __init__(self, tech: TechConfig, timestamp: datetime.datetime | None = None) -> None:
        self._tech = tech
        self._timestamp = PINNED_TIMESTAMP if timestamp is None else timestamp

    @property
    def tech(self) -> TechConfig:
        return self._tech

    @property
    def precision_um(self) -> float:
        """The design grid, in micrometres."""
        return self._tech.precision_um

    def boolean(
        self,
        operand1: gdstk.Polygon | list[gdstk.Polygon],
        operand2: gdstk.Polygon | list[gdstk.Polygon],
        operation: BooleanOperation,
    ) -> list[gdstk.Polygon]:
        """Boolean operation at the configured grid.

        Because precision equals the design grid, the result is grid-aligned
        by construction and needs no separate snapping pass.
        """
        return gdstk.boolean(operand1, operand2, operation, precision=self.precision_um)

    def offset(
        self,
        polygons: gdstk.Polygon | list[gdstk.Polygon],
        distance_um: float,
        join: JoinStyle = "miter",
        tolerance: int = 2,
        use_union: bool = True,
    ) -> list[gdstk.Polygon]:
        """Dilate (positive distance) or erode (negative) at the configured grid."""
        return gdstk.offset(
            polygons,
            distance_um,
            join=join,
            tolerance=tolerance,
            precision=self.precision_um,
            use_union=use_union,
        )

    def fracture(self, polygon: gdstk.Polygon) -> list[gdstk.Polygon]:
        """Split a polygon to the configured vertex limit."""
        return polygon.fracture(
            max_points=self._tech.fracture_vertex_limit,
            precision=self.precision_um,
        )

    def new_library(self, name: str) -> gdstk.Library:
        """A library whose database precision matches the design grid."""
        return gdstk.Library(name, unit=USER_UNIT_M, precision=self._tech.precision_m)

    def write_gds(self, library: gdstk.Library, path: Path | str) -> None:
        """Write GDSII with the configured vertex limit and a pinned timestamp."""
        library.write_gds(
            path,
            max_points=self._tech.fracture_vertex_limit,
            timestamp=self._timestamp,
        )

    def write_oas(self, library: gdstk.Library, path: Path | str) -> None:
        """Write OASIS.

        OASIS carries no timestamp, so output is reproducible without one.
        """
        library.write_oas(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_context.py tests/test_architecture.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Prove the architecture guard actually fires**

Temporarily add `import gdstk` to the top of `src/masklayout/config.py`, then run:

```bash
uv run pytest tests/test_architecture.py::test_only_geomcontext_imports_gdstk -v
```

Expected: FAIL, naming `config.py`. **Remove the import again** and re-run to confirm it passes. A guard that has never been seen to fail is not a guard.

- [ ] **Step 6: Run the whole suite**

Run:
```bash
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest -v
```
Expected: all clean, all tests pass.

- [ ] **Step 7: Commit and push**

```bash
git add src/masklayout/geometry tests/unit/test_context.py tests/test_architecture.py
git commit -m "feat(m0): add GeomContext as the sole gdstk boundary

gdstk's defaults silently override configuration: boolean and offset
default to precision=1e-3, write_gds to max_points=199. GeomContext
injects the configured grid and fracture limit into every call, and an
AST-based test enforces that no other module under src/ imports gdstk.

Includes regression tests for the two defaults, for byte-reproducible
GDS output via a pinned timestamp, and for grid-aligned boolean results.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

- [ ] **Step 8: Verify CI passes and M0 is complete**

Run: `gh run watch`
Expected: green. M0 acceptance condition — `uv run pytest`, `ruff`, and `mypy` all clean in a reproducible environment — is now met.

- [ ] **Step 9: Update the README milestone table**

Change the M0 row status from `not started` to `complete`, then:

```bash
git add README.md
git commit -m "docs: mark M0 complete

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Self-Review

**Spec coverage.** M0's scope per design §10 and the planning-scope note is: scaffold, tooling, `TechConfig` with §3 validation, the layer map, `GeomContext`, and passing CI. Task 1 covers scaffold, tooling, and CI. Task 2 covers the layer map including `FIELD`. Task 3 covers `TechConfig` with the exact-multiple validation and the derived deburr. Task 4 covers `GeomContext` and the import boundary. No OPC algorithms appear, as required. Nothing in M0's scope is unassigned.

**Deliberately deferred, with the milestone that picks them up:** `shapely` and `geometry/index.py` (M2), `cli.py` (M4), `opc/deck.py` (M4), `io/mask.py` (M8). Deferring `shapely` and `pyyaml` is load-bearing, not cosmetic — mypy `strict` enables `warn_unused_configs`, so an override section for a module nothing imports produces noise in CI.

**Placeholder scan.** No TBD, TODO, "implement later", "add appropriate error handling", or "similar to Task N". Every code step carries complete, runnable code. Every test step names the exact command and expected outcome.

**Type consistency.** `TechConfig.precision_um`, `precision_m`, and `fracture_vertex_limit` are defined in Task 3 and consumed under those exact names in Task 4. `LayerMap.default()` and `Layer(number=, datatype=, name=)` are consistent between Task 2's implementation and its tests. `GeomContext(tech, timestamp=None)` matches its usage in every Task 4 test. `masklayout.__version__` is defined in Task 1 and asserted in Task 1's test only.
