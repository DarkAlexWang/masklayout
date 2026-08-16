# masklayout

A recipe-driven computational mask-layout toolkit for semiconductor lithography masks,
in pure Python.

`masklayout` authors non-Manhattan, curvilinear layouts from scratch, decorates imported
target GDS with deterministic rule-based OPC-like corrections — hammerheads, serifs,
jogs, line-end extensions, local bias, SRAFs — and exports both engineering (1×) and
mask (×4, tone-applied) streams.

> **Status: M4 complete. Early development.**
> Everything through site extraction, the selector vocabulary, and the declarative
> rule deck is in place and tested: a target layout can be extracted, classified, and
> matched against rules. **Correction geometry is not generated yet** — M4 produces
> matches, not shapes; that is M5. The design is specified in
> [`docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`](docs/superpowers/specs/2026-08-14-masklayout-v1-design.md);
> everything described below as a capability is planned, not delivered, except where
> the milestone table marks it complete.

## What it is, and what it is not

V1 is a **geometry compiler and verifier**. It controls chord error, mask-space
displacement, grid error, and polygon complexity.

It does **not** predict wafer printing. There is no aerial-image, resist, or etch model,
and no edge-placement-error computation — those require an optical and process model
that is explicitly outside V1. Geometry-only results are never reported as EPE results.
This distinction is enforced throughout the documentation and the generated reports.

Also out of scope for V1: model-based OPC, inverse lithography (ILT), foundry-qualified
DRC, and any graphical editor.

## Planned capabilities

- Author curvilinear and arbitrary-angle layouts via parameterized cells.
- Import target GDS, preserving hierarchy; flatten only explicitly selected regions.
- Generate rule-based corrections from a declarative, versioned rule deck.
- Enforce strict curve tessellation, grid quantization, and polygon-complexity limits.
- Verify geometry: grid alignment, self-intersection, min area/edge, mask rule checks.
- Export GDSII and OASIS, plus JSON manifests, debug and overlay layers, and SVG previews.
- Produce byte-reproducible output for identical input, deck, config, and tool version.

## Design highlights

**Integer coordinates.** All geometry is `int64` in design database units. No float
coordinate reaches the model layer or a writer, which is what makes reproducibility a
guarantee rather than an aspiration.

**Determinism is verified, not assumed.** GDSII embeds a header timestamp that defeats
byte-reproducibility unless pinned; the writers pin it. OASIS was measured to be
reproducible with no special handling.

**Provenance on every generated feature.** Feature ID, source cell, source edge, rule ID
and deck hash, parameter values, tool and config version. Identifiers are assigned after
normalization so they survive grid snapping and collinear-point removal.

**Mask rule checks carry a stated sensitivity floor.** Morphological min-width and
min-space checking on quantized non-Manhattan geometry accumulates roughly half a grid
step of error along every edge — enough that a clean rotated bar can report more
violation area than a genuinely defective one. A half-grid deburr fixes it, at the cost
of a real detection limit near the rule value. That limit is recorded in the manifest
rather than left implicit.

The design document records which toolchain assumptions were verified by experiment,
including two that were disproved and corrected.

## Architecture

```
read GDS/OASIS → select region → materialize → normalize → index
  → extract → classify → match rule deck → generate → resolve collisions
  → merge → verify → fracture → export
```

`classify` is the contract: the rule deck is declarative over a closed selector
vocabulary, so whatever `classify` measures is exactly what a rule can select on.

## Milestones

| M | Deliverable | Status |
|---|---|---|
| M0 | Project skeleton and CI | **complete** |
| M1 | GDS/OASIS I/O and hierarchy | **complete** |
| M2 | Geometry compiler | **complete** |
| M3 | PCell library | **complete** |
| M4 | Extraction, classification, rule deck | **complete** |
| M5 | Target decorator | not started |
| M6 | SRAF engine | not started |
| M7 | Verification and reporting | not started |
| M8 | Mask export (×4, tone inversion) | not started |
| M9 | Regression corpus | not started |

## Technology

Python 3.12+, [`uv`](https://docs.astral.sh/uv/) for environment management,
[`gdstk`](https://heitzmann.github.io/gdstk/) as the geometry and stream-format backend,
`shapely` for spatial indexing, `pydantic` for configuration and rule-deck validation,
with `pytest`, `ruff`, and `mypy`.

The public API is kept independent of `gdstk` types.

## Getting started

**Requirements:** Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). Nothing else —
`uv` fetches the interpreter and every dependency.

```bash
git clone git@github.com:DarkAlexWang/masklayout.git
cd masklayout
uv sync
```

`uv sync` creates `.venv/`, installs `gdstk`, `numpy`, `pydantic`, `pyyaml`, and
`shapely`, and installs `masklayout` itself in editable mode. Every command below is
prefixed with `uv run`, so you never need to activate the virtualenv.

There is **no command-line interface yet** — `masklayout` is used as a Python library.
A CLI arrives with the recipe runner in a later milestone.

## Running the examples

Two programs exercise everything that works today. Both write only into
`examples/out/`, which is git-ignored.

### Authoring a layout from scratch

```bash
uv run python examples/author_from_scratch.py
```

Builds a curvilinear Bézier wire, a tapered wire, a line-end cap, and a 4×3 contact
array; writes GDSII and OASIS; reads the GDS back and reports the hierarchy:

```
technology      : generic_mask_geometry_v1
design grid     : 1.0 nm
mask grid x mag : 0.5 nm x 4
chord error     : 1.0 nm
MRC deburr      : 0.5 nm (derived from the grid)

bezier wire     : 256 vertices
tapered wire    : 64 vertices, 800 nm tall at its widest
line end        : 4 vertices
contact array   : 4 x 3 = 12 placements, stored as 1 cell + 1 reference

wrote authored.gds  (2962 bytes)
wrote authored.oas  (1112 bytes)

read back:
  .../authored.gds: 2 cells, 4 polygons, 1 references, 0 labels
  top cells    : ['TOP']
  hierarchy    : TOP depends on ['CONTACT']
  depth        : 1
```

The polygon count is the thing to notice: **4 polygons, not 15**. The twelve contact
placements round-tripped as one cell plus one repeated reference, because hierarchy is
never flattened implicitly.

The output files open in KLayout or any GDS viewer.

### Decorating a target layout

```bash
uv run python examples/decorate_target.py
```

Takes two collinear bars separated by a 60 nm gap, extracts sites, measures the
selector vocabulary, and matches the shipped rule deck:

```
deck            : generic_hammerhead_v1 v1.0.0
content hash    : 20cf50d39e06f354...
rules           : ['hh_isolated_line_end', 'hh_dense_line_end', 'bias_narrow_edge']

extracted       : 16 sites from 2 polygons
  convex_corner    8
  edge             4
  line_end         4

measured line ends (the closed selector vocabulary):
    x (um)    width    space  density
     2.000      100       60    0.094
     0.000      100      inf    0.050
     4.060      100      inf    0.050
     2.060      100       60    0.094

matched         : 8 sites, 8 unmatched
by rule         : {'bias_narrow_edge': 4, 'hh_dense_line_end': 2, 'hh_isolated_line_end': 2}

hammerhead decisions:
  site 0#1:line_end      -> hh_dense_line_end      {'extension_um': 0.014}
  site 0#3:line_end      -> hh_isolated_line_end   {'extension_um': 0.028, ...}
  site 1#1:line_end      -> hh_isolated_line_end   {'extension_um': 0.028, ...}
  site 1#3:line_end      -> hh_dense_line_end      {'extension_um': 0.014}
```

This is the point of the whole vocabulary: the two inner line ends measure `space = 60`
and take the **dense** rule, while the two outer ends measure `space = inf` and take the
**isolated** rule. Identical geometry, different correction, decided by context.

It stops at matching. **Turning these decisions into correction geometry is not
implemented** — that is M5.

## Verifying the build

Four independent checks. Run them separately and read each result; chaining them with
`&&` lets an earlier failure hide behind a later success.

```bash
uv run pytest -q                      # 197 tests
uv run ruff check .                   # lint
uv run ruff format --check .          # formatting
uv run mypy src tests examples        # strict type checking, 55 files
```

Expected output:

```
197 passed in 0.33s
All checks passed!
55 files already formatted
Success: no issues found in 55 source files
```

CI runs exactly these four on every push and pull request.

### Checks worth running individually

Some tests encode findings that were expensive to discover. To watch them specifically:

```bash
# Measurement is exact at any angle — no Manhattan special case anywhere.
uv run pytest tests/unit/test_classify.py -k "at_any_angle" -v

# A compiled curve stays within budget of the TRUE analytic curve. The budget is
# max_chord_error_nm + design_grid_nm*sqrt(2)/2; asserting the first term alone
# fails at every radius.
uv run pytest tests/unit/test_compile.py -k "true_curve" -v

# gdstk's defaults (precision=1e-3, max_points=199) never silently override config.
uv run pytest tests/unit/test_context.py -v

# gdstk is imported by exactly two allowlisted modules, and the guard demonstrably fires.
uv run pytest tests/test_architecture.py -v

# End-to-end: authoring, and context-driven rule selection.
uv run pytest tests/integration/ -v
```

### Reproducibility

GDSII embeds a header timestamp, so writers pin it. The same layout written twice is
byte-identical:

```bash
uv run python -c "
import hashlib, pathlib
from masklayout.config import TechConfig
from masklayout.io.streams import write_gds
from masklayout.model.cell import Cell
from masklayout.model.layout import Layout
from masklayout.pcells.wires import BezierWireParams, build_bezier_wire
tech = TechConfig()
def build(path):
    lay = Layout(name='R', tech=tech); top = lay.add(Cell(name='TOP'))
    top.polygons.extend(build_bezier_wire(BezierWireParams(
        control_points_um=((0,0),(2,3),(6,-3),(8,0)), width_um=0.4), tech, 10, 0))
    write_gds(lay, path); return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
a, b = build('/tmp/a.gds'), build('/tmp/b.gds')
print(a[:16], b[:16], 'IDENTICAL' if a == b else 'DIFFER')
"
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src tests examples
```

Design documents live in [`docs/superpowers/specs/`](docs/superpowers/specs/) and
per-milestone implementation plans in [`docs/superpowers/plans/`](docs/superpowers/plans/).
The plans record what was verified by experiment and what turned out to be wrong.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Apache-2.0 was chosen over MIT for its express patent grant: OPC, SRAF placement, and
mask synthesis are patent-dense areas, and downstream users benefit from explicit patent
permission rather than an implied one.
