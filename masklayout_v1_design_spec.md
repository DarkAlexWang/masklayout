# masklayout — V1 Design Specification

## Purpose

`masklayout` is a pure-Python, recipe-driven computational mask-layout toolkit for semiconductor lithography masks. It supports both:

1. Authoring complete layouts from scratch with parameterized cells (PCells).
2. Reading a target-layout GDS and generating deterministic, OPC-like geometric corrections.

V1 is a geometry compiler and verifier. It does **not** claim wafer-print prediction or wafer edge-placement-error (EPE) validation; those require an explicit optical, resist, and etch process model and are outside V1.

## Product goals

- Author non-Manhattan, curvilinear-intent layouts.
- Import target GDS layouts while preserving hierarchy by default.
- Generate piecewise-linear post-OPC mask polygons.
- Support rule/template-based hammerheads, serifs, jogs, line-end extensions, local bias, and SRAFs.
- Provide strict geometric controls for curve tessellation, polygon complexity, grid quantization, and cleanup.
- Write both GDSII and OASIS outputs.
- Produce machine-readable manifests, debug layers, overlays, and preview renderings.
- Be reproducible, deterministic, testable, and friendly to command-line/CI use.

## Non-goals for V1

- Full model-based OPC (MBOPC).
- Inverse lithography technology (ILT) optimization.
- Aerial-image, resist, or etch simulation.
- Foundry-qualified DRC or mask-rule-deck replacement.
- Exact wafer EPE calculation.
- A graphical layout editor.

## Technology choices

| Concern | Decision |
|---|---|
| Language | Python 3.12+ |
| Environment/package manager | `uv` |
| Test framework | `pytest` |
| Geometry/GDS backend | `gdstk` |
| Linting | `ruff` |
| Static typing | `mypy` |
| Input | GDSII target layout, Python layout API, YAML/JSON recipes |
| Output | GDSII, OASIS, JSON manifest, SVG/PNG preview |
| Primary data style | Typed Python models; do not expose raw `gdstk` objects as the core public API |

`gdstk` is used as the low-level geometry and stream-format backend. GDSII output is polygonal: curves and curved-intent features must be tessellated into valid polygons during compilation.

## V1 decisions

- Input modes: support both full from-scratch authoring and target-GDS decoration.
- OPC mode: deterministic rule/template-based correction derived from target geometry.
- Initial pattern scope: line ends, hammerheads, serifs, jogs, contacts/vias, curvilinear contours, and SRAFs.
- Output geometry: piecewise-linear polygons with strict chord-error and segment-length limits.
- Output streams: GDSII and OASIS.
- Hierarchy: preserve by default; only flatten or materialize explicitly selected cells/regions.
- Simulation: excluded from V1.
- Interface: first-class Python API plus recipe files and CLI.

## Terminology and bounds

### Geometric quantities controlled in V1

- **Chord error**: maximum deviation between an analytic curve and its tessellated polygonal contour.
- **Mask-space displacement**: geometric difference between target and post-OPC layout contours.
- **Grid error**: error introduced by snapping coordinates to the configured database grid.
- **Polygon complexity**: vertex count and fracture count after compilation.

### Quantity intentionally not controlled in V1

- **Wafer EPE**: requires an imaging/resist/etch model, illumination conditions, mask model, and a wafer-contour definition. Do not describe geometry-only results as wafer EPE results.

## Default technology configuration

These are software defaults only. They are not foundry, mask-writer, or process-node rules. Every value must be configurable and recorded in generated manifests.

```yaml
technology:
  name: generic_mask_geometry_v1
  layout_unit: um
  database_precision: 0.001_um       # 1 nm internal output grid
  coordinate_snap_nm: 1.0

  curve:
    max_chord_error_nm: 1.0
    max_segment_length_nm: 10.0
    min_segment_length_nm: 1.0
    max_vertices_per_polygon: 4000

  cleanup:
    min_polygon_area_nm2: 4.0
    min_edge_length_nm: 1.0
    remove_collinear_tolerance_nm: 0.001

  export:
    fracture_vertex_limit: 4000
    preserve_hierarchy: true
    output_formats: [gds, oasis]
```

## Layer policy

Layer numbers and datatypes must be configurable through a layer map. The following is the default engineering convention.

| Logical layer | Default GDS layer/datatype | Purpose |
|---|---:|---|
| `TARGET` | 10 / 0 | Desired target pattern |
| `POST_OPC` | 11 / 0 | Final generated post-OPC geometry |
| `SRAF` | 12 / 0 | Assist features retained separately for debug |
| `DEBUG_SOURCE` | 200 / 0 | Selected source edges, corners, and candidate sites |
| `DEBUG_MARKERS` | 201 / 0 | Geometry violations, conflicts, and rejected candidates |
| `OVERLAY_ADD` | 202 / 0 | `POST_OPC - TARGET` |
| `OVERLAY_REMOVE` | 203 / 0 | `TARGET - POST_OPC` |

A production export may map `POST_OPC` and `SRAF` to a required output layer. Keep a separate engineering/debug export with the logical layers intact.

## Architecture

```text
Intent geometry
  └── Parametric primitives and PCells
        └── Compiled geometry
              └── Quantized, cleaned polygons
                    └── GDSII/OASIS serialization
```

```text
Target GDS / PCell recipe
  → hierarchy selection
  → polygon normalization and grid quantization
  → edge/corner/line-end extraction
  → local-context measurement and classification
  → candidate OPC-feature generation
  → collision and keep-out resolution
  → Boolean merge / healing / simplification
  → geometric verification
  → GDS/OASIS export + manifest + preview
```

## Core data model

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Layer:
    number: int
    datatype: int
    name: str


@dataclass(frozen=True)
class TechConfig:
    grid_nm: float
    max_chord_error_nm: float
    max_segment_length_nm: float
    max_vertices_per_polygon: int
    min_area_nm2: float
    min_edge_length_nm: float


@dataclass
class Feature:
    id: str
    kind: str
    polygons: list["Polygon"]
    source_feature_id: str | None = None
    parameters: dict[str, float | str | bool] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)


@dataclass
class Layout:
    cells: dict[str, "Cell"]
    layers: "LayerMap"
    tech: TechConfig
```

Generated features must preserve provenance. At minimum, record:

- Feature ID and type.
- Source cell and source polygon identifier.
- Source edge or vertex identifier when applicable.
- Rule identifier and rule-deck version.
- Parameter values.
- Tool version, technology-config version, and generation timestamp.

## Public API targets

```python
layout = masklayout.Layout.new("TOP", tech=tech, layers=layers)

# Author from scratch.
cell = layout.cell("TOP")
cell.add(masklayout.pcells.bezier_wire(...))
cell.add(masklayout.pcells.contact_array(...))

# Decorate an existing target layout.
layout = masklayout.read_gds("target.gds", tech=tech, layers=layers)
result = masklayout.compile_post_opc(
    layout,
    target_layer="TARGET",
    rule_deck=rule_deck,
)

masklayout.write_gds(result, "post_opc.gds")
masklayout.write_oasis(result, "post_opc.oas")
masklayout.write_manifest(result, "post_opc.manifest.json")
masklayout.render_svg(result, "post_opc.svg")
```

## PCell library

### Geometry PCells

Implement these first:

- `rounded_rect`
- `arc_segment`
- `bezier_wire`
- `tapered_wire`
- `serpentine`
- `ring`
- `racetrack`
- `arbitrary_contour`
- `line_end`
- `contact`
- `contact_array`

### OPC PCells

Implement in this order:

1. `edge_bias`
2. `line_end_extension`
3. `hammerhead`
4. `serif`
5. `jog`
6. `curvilinear_edge_perturbation`
7. `sraf_bar`
8. `sraf_array`
9. Contact/via local correction templates

All PCells must support physical-unit parameters, grid snapping, rotation, mirroring, layer selection, analytic preview where relevant, and compiled polygon output.

## OPC extraction and classification

V1 extracts and classifies the following target-layout context:

- Isolated, semi-isolated, and dense lines.
- Horizontal, vertical, 45-degree, and arbitrary-angle line ends.
- Convex and concave corners.
- Acute and obtuse corners.
- Narrow necks and spaces.
- Contacts/vias and contact arrays.
- Periodic/repetitive local patterns.
- Curvilinear segments by local tangent and curvature.
- Candidate SRAF zones constrained by spacing and keep-out rules.

### Feature-local coordinates

Generate hammerheads, serifs, and line-end extensions in an edge-local coordinate system, then transform them into global layout coordinates. This makes the same rule work for Manhattan and arbitrary-angle geometry.

```python
add_hammerhead(
    edge=line_end,
    extension_nm=28,
    head_width_nm=42,
    neck_width_nm=20,
    corner_radius_nm=2,
    side="both",
)
```

## Geometry compiler requirements

The compiler must:

1. Tessellate analytic geometry subject to maximum chord error and segment-length limits.
2. Quantize compiled vertices onto the configured grid.
3. Normalize polygons: remove duplicate points, remove near-collinear segments, establish orientation, and reject invalid contours.
4. Perform Boolean union/difference/intersection as required.
5. Resolve collisions and enforce keep-out geometry.
6. Remove slivers, small-area remnants, and edges below the configured thresholds.
7. Simplify without exceeding configured geometric tolerances.
8. Fracture polygons before export when vertex limits require it.
9. Emit deterministic output ordering for reproducible diffs and regression tests.

## Verification

V1 verification is geometric and structural, not process-aware.

Checks include:

- All vertices aligned to the configured grid.
- No polygon exceeds the configured export vertex limit after fracture.
- No self-intersections or invalid polygons.
- Minimum polygon area and minimum edge length.
- Minimum width/space checks where an implementation is available.
- Acute-angle and notch markers.
- SRAF/target and SRAF/SRAF collision/keep-out checks.
- Polygon count, vertex count, area, perimeter, and fracture statistics.
- Target/post-OPC difference layers and metrics.
- Curve tessellation report including maximum estimated chord error.

## Repository layout

```text
masklayout/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── src/masklayout/
│   ├── api.py
│   ├── config.py
│   ├── model/
│   │   ├── layout.py
│   │   ├── feature.py
│   │   └── layers.py
│   ├── io/
│   │   ├── gds.py
│   │   ├── oasis.py
│   │   └── manifest.py
│   ├── geometry/
│   │   ├── quantize.py
│   │   ├── curves.py
│   │   ├── polygon_ops.py
│   │   ├── fracture.py
│   │   └── validate.py
│   ├── pcells/
│   │   ├── basic.py
│   │   ├── curvilinear.py
│   │   ├── line_end.py
│   │   └── contacts.py
│   ├── opc/
│   │   ├── extract.py
│   │   ├── classify.py
│   │   ├── rules.py
│   │   ├── hammerhead.py
│   │   ├── serif.py
│   │   ├── sraf.py
│   │   └── compile.py
│   ├── verify/
│   │   ├── rules.py
│   │   ├── report.py
│   │   └── overlay.py
│   └── render/
│       ├── svg.py
│       └── raster.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
└── examples/
    ├── author_from_scratch.py
    ├── decorate_target_gds.py
    └── recipe_hammerhead.yaml
```

## Milestones

| Milestone | Deliverable | Acceptance condition |
|---|---|---|
| M0 | Project skeleton and CI | `uv run pytest`, linting, type checking, reproducible environment |
| M1 | GDS/OASIS I/O and hierarchy inspection | Import, preserve references, and round-trip export valid layouts |
| M2 | Geometry compiler | Curves compile into grid-aligned polygons within chord-error limit |
| M3 | PCell library | From-scratch layout generates curvilinear wires, line ends, contacts, and arrays |
| M4 | Target decorator | Imported GDS generates bias, line-end extensions, hammerheads, serifs, and jogs |
| M5 | SRAF engine | Rule-constrained SRAFs with collision resolution and keep-out enforcement |
| M6 | Verification/reporting | Marker layers, overlays, JSON manifest, SVG/PNG preview, and geometric report |
| M7 | Regression corpus | Golden tests cover dense, isolated, curvilinear, contact, and pathological patterns |

## First vertical slice

Implement this narrow end-to-end slice before adding broad feature coverage:

1. Create a target layout from scratch containing a non-Manhattan tapered Bézier wire, an isolated line, a dense-line pair, and a contact array.
2. Export the target in GDSII and OASIS.
3. Import the target GDS.
4. Extract line ends.
5. Add parameterized hammerheads to selected line ends.
6. Boolean-merge, heal, simplify, and fracture the output.
7. Emit `POST_OPC`, overlay layers, a JSON parameter manifest, and an SVG preview.
8. Assert grid alignment, valid polygons, vertex-limit compliance, and chord-error compliance.

## Initial commands

```bash
uv init --python 3.12
uv add gdstk pydantic pyyaml numpy shapely
uv add --dev pytest pytest-cov ruff mypy

uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

`shapely` is optional in the long run. Prefer `gdstk` for stream-format handling and primary polygon operations; only retain extra geometry dependencies when their behavior is explicitly tested and needed.

## Development rules

- Keep the public API independent of `gdstk` types.
- Use explicit units in parameter names, especially `*_nm` and `*_um`.
- Never silently alter the configured grid, tolerance, layer map, or fracture limit.
- Never flatten hierarchy implicitly.
- Treat every generated OPC feature as traceable data, not anonymous geometry.
- Generate deterministic results for identical input, rule deck, technology config, and tool version.
- Keep target geometry immutable; write generated geometry to logical output layers.
- Require regression fixtures for every geometry bug fix.
- Distinguish geometric validation from lithographic/process validation in documentation and reports.

## Future extensions after V1

- Aerial-image and resist-model adapters.
- Kernel-based compact-model integration.
- Edge-placement-error sampling against simulated printed contours.
- Iterative model-based OPC optimization.
- Curvilinear/ILT contour optimization with constraints.
- GPU rasterization and batched patch processing.
- Integration with inspection labeling, defect databases, and wafer/mask image registration.

## Handoff prompt for Claude Code

Use this document as the initial project specification. First ask Claude Code to:

> Read this specification. Create the `masklayout` repository scaffold using `uv`, Python 3.12+, `pytest`, `ruff`, and `mypy`. Implement Milestone M0 only. Do not begin OPC algorithms yet. Add a concise `CLAUDE.md`, a typed `TechConfig`, a layer-map model, standard command configuration in `pyproject.toml`, and passing CI-quality tests. Report created files and all commands run.

Then proceed milestone by milestone, requiring tests and a review of public API decisions before moving to the next milestone.
