# masklayout

A recipe-driven computational mask-layout toolkit for semiconductor lithography masks,
in pure Python.

`masklayout` authors non-Manhattan, curvilinear layouts from scratch, decorates imported
target GDS with deterministic rule-based OPC-like corrections — hammerheads, serifs,
line-end extensions, local bias, SRAFs — verifies the result, and exports both
engineering (1×) and mask (×4, tone-applied) streams.

> **Status: all V1 milestones delivered except `jog`.**
> The pipeline runs end to end: read → extract → classify → match a declarative rule
> deck → generate corrections → place SRAFs → merge → verify → export at 1× and ×4.
> 364 tests. See [Open items](#open-items-and-known-limitations) for what is *not*
> done — that list is deliberately specific.

## What it is, and what it is not

V1 is a **geometry compiler and verifier**. It controls chord error, mask-space
displacement, grid error, and polygon complexity.

It does **not** predict wafer printing. There is no aerial-image, resist, or etch model,
and no edge-placement-error computation — those require an optical and process model
that is explicitly outside V1. Geometry-only results are never reported as EPE results.

Also out of scope for V1: model-based OPC, inverse lithography (ILT),
foundry-qualified DRC, and any graphical editor.

## Milestones

| M | Deliverable | Status |
|---|---|---|
| M0 | Project skeleton and CI | **complete** |
| M1 | GDS/OASIS I/O and hierarchy | **complete** |
| M2 | Geometry compiler | **complete** |
| M3 | PCell library | **complete** |
| M4 | Extraction, classification, rule deck | **complete** |
| M5 | Target decorator | **complete** (`jog` deferred) |
| M6 | SRAF engine | **complete** |
| M7 | Verification and reporting | **complete** (SVG; no PNG) |
| M8 | Mask export (×4, tone inversion) | **complete** |
| M9 | Regression corpus | **complete** |

The design is specified in
[`docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`](docs/superpowers/specs/2026-08-14-masklayout-v1-design.md).

---

## Getting started

**Requirements:** Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). Nothing else —
`uv` fetches the interpreter and every dependency.

```bash
git clone git@github.com:DarkAlexWang/masklayout.git
cd masklayout
uv sync
```

Every command below is prefixed with `uv run`, so you never need to activate the
virtualenv.

**There is no command-line interface.** `masklayout` is a Python library. A CLI is not
implemented — see [Open items](#open-items-and-known-limitations).

## Usage

### Authoring a layout from scratch

```python
from masklayout.config import TechConfig
from masklayout.io.streams import write_gds, write_oas
from masklayout.model.cell import Cell
from masklayout.model.layout import Layout
from masklayout.pcells.wires import BezierWireParams, build_bezier_wire
from masklayout.pcells.contacts import ContactParams, place_contact_array

tech = TechConfig()                     # 1 nm grid, 0.5 nm mask grid, x4, clear tone
layout = Layout(name="TOP", tech=tech)
top = layout.add(Cell(name="TOP"))

top.polygons.extend(build_bezier_wire(
    BezierWireParams(
        control_points_um=((0, 0), (2, 3), (6, -3), (8, 0)),
        width_um=0.4,
    ),
    tech, layer=10, datatype=0,
))

place_contact_array(                     # emits a cell + a repeated reference,
    layout, parent_cell="TOP",           # not 12 flattened copies
    contact_cell_name="CONTACT",
    params=ContactParams(centre_um=(0, 0), size_um=(0.2, 0.2)),
    columns=4, rows=3, pitch_um=(0.5, 0.5), layer=12,
)

write_gds(layout, "out.gds")
write_oas(layout, "out.oas")
```

### Decorating a target with OPC corrections

```python
from masklayout.io.streams import read_gds
from masklayout.opc.deck import load_deck
from masklayout.opc.decorate import decorate

layout, report = read_gds("target.gds", tech=tech)
print(report.summary())

# NOTE: decorate() takes a flat polygon list, not a Layout. Hierarchy is
# preserved through I/O but the OPC path does not consume it — see Open items.
target = layout.cells["TOP"].polygons

deck = load_deck("src/masklayout/decks/generic_hammerhead_v1.yaml")
result = decorate(target, deck, tech)

print(result.report.summary())
result.post_opc        # corrected geometry, layer 11
result.overlay_add     # POST_OPC - TARGET, layer 202
result.srafs           # assist features, layer 12 — never merged into POST_OPC
result.rejected        # SRAFs that failed keep-out, with reasons
for feature in result.features:
    print(feature.provenance())   # rule id, deck hash, parameters
```

### Verifying and reporting

```python
from masklayout.verify.structural import run_structural_checks
from masklayout.verify.mrc import check_min_width, check_min_space
from masklayout.io.manifest import build_manifest, write_manifest
from masklayout.render.svg import render_svg

violations  = run_structural_checks(result.post_opc, tech)
violations += check_min_width(result.post_opc, 20.0, tech)   # deburr on by default
violations += check_min_space(result.post_opc, 60.0, tech)

geometry = {"TARGET": target, "POST_OPC": result.post_opc,
            "OVERLAY_ADD": result.overlay_add}

write_manifest("out.manifest.json", build_manifest(
    tech, tool_version="0.1.0", features=result.features,
    violations=violations, layer_geometry=geometry,
    deck_id=deck.id, deck_version=deck.version,
    deck_hash=deck.content_hash, mrc_ran=True,
))
render_svg("out.svg", geometry, tech)
```

### Exporting mask data

```python
from masklayout.io.mask import export_mask, write_mask_gds

field = [...]   # a FIELD-layer polygon; required for tone inversion, never inferred

mask = export_mask(
    result.post_opc, result.srafs, field,
    TechConfig(tone="dark"),         # FIELD - (POST_OPC | SRAF), then x4
    min_width_nm=20.0,               # MRC re-runs after inversion
)
print(mask.statistics)               # tone, magnification, on_mask_grid
assert mask.clean
write_mask_gds("mask.gds", mask, tech)
```

### Writing a rule deck

Selectors are drawn from a **closed vocabulary**; anything else fails at load.

```yaml
deck:
  id: my_deck
  version: 1.0.0

rules:
  - id: hh_isolated_line_end
    priority: 10                  # lower fires first; first match wins per kind
    kind: hammerhead
    when:                         # omitted keys are unconstrained
      site: line_end
      width_nm: {min: 60, max: 140}
      space_nm: {min: 120}        # "isolated" — an unbounded space satisfies a minimum
    apply:
      pcell: line_end
      params:
        extension_um: 0.028
        head_width_ratio: 1.4     # >1.0 flares into a hammerhead; 1.0 is a plain extension
```

Vocabulary: `site`, `width_nm`, `space_nm`, `edge_length_nm`, `angle_deg`,
`corner_type`, `curvature_1_per_um`, `local_density`.

Correction kinds with generators: `hammerhead`, `line_end_extension`, `serif`,
`edge_bias`, `sraf_bar`.

## Running the examples

Two programs exercise everything. Both write only into `examples/out/`, which is
git-ignored.

```bash
uv run python examples/author_from_scratch.py
uv run python examples/decorate_target.py
```

The authoring example reports **4 polygons, not 15** after a GDS round-trip — twelve
contact placements survived as one cell plus one reference, because hierarchy is never
flattened implicitly.

The decoration example shows the point of the whole selector vocabulary:

```
    x (um)    width    space  density
     2.000      100       60    0.094      -> hh_dense_line_end
     0.000      100      inf    0.050      -> hh_isolated_line_end

decorated       : 16 sites, 8 matched, 8 features, (edge_bias x4, hammerhead x4)
target area     : 0.400000 um^2
post-OPC area   : 0.424392 um^2  (+6.10%)

verification    : 1 violation(s)
  [error] min_space: 1 gap(s) narrower than 60.0 nm

wrote post_opc.gds, post_opc.manifest.json, post_opc.svg
```

Identical geometry, different correction, decided by context. That `min_space` failure
is real, not noise: the hammerheads extend 14 nm each into the 60 nm gap, leaving about
32 nm. **The correction created a violation and MRC caught it** — which is exactly why
MRC runs after correction rather than before.

Open `post_opc.gds` in KLayout, or `post_opc.svg` in any browser.

---

## Verifying the build

Four independent gates. **Run them separately and read each result** — chaining them
with `&&` lets an earlier failure hide behind a later success.

```bash
uv run pytest -q                      # 364 passed
uv run ruff check .                   # All checks passed!
uv run ruff format --check .          # 89 files already formatted
uv run mypy src tests examples        # Success: no issues found in 87 source files
```

CI runs exactly these four on every push and pull request.

### Verifying a single milestone

Each command is the complete test set for that milestone. The counts sum to exactly
**364** — the full suite — so no test is unattributed.

| M | What it proves | Tests |
|---|---|---:|
| M0 | Config validation, layer map, gdstk boundary guard | 25 |
| M1 | Typed model, hierarchy, GDS/OASIS round-trip | 36 |
| M2 | Normalization, tessellation, grid-aligned compile | 34 |
| M3 | PCell registry, wires, contacts, authoring | 28 |
| M4 | Selector vocabulary, deck loading, matching | 74 |
| M5 | Correction generation and merging | 52 |
| M6 | SRAF placement and keep-out | 31 |
| M7 | Structural checks, MRC, manifest, SVG | 22 |
| M8 | ×4 scaling, tone inversion, post-inversion MRC | 20 |
| M9 | Golden corpus over twelve pattern classes | 42 |

```bash
# M0 — scaffold, config, layer map, gdstk boundary
uv run pytest tests/unit/test_package.py tests/unit/test_layers.py \
              tests/unit/test_config.py tests/unit/test_context.py \
              tests/test_architecture.py -v

# M1 — typed model, hierarchy, GDS/OASIS round-trip
uv run pytest tests/unit/test_model_geometry.py tests/unit/test_layout_hierarchy.py \
              tests/unit/test_bridge_units.py tests/unit/test_streams.py -v

# M2 — normalization, curve tessellation, compile
uv run pytest tests/unit/test_normalize.py tests/unit/test_curves.py \
              tests/unit/test_compile.py -v

# M3 — PCell library and from-scratch authoring
uv run pytest tests/unit/test_pcell_base.py tests/unit/test_pcell_shapes.py \
              tests/unit/test_pcell_wires.py tests/unit/test_pcell_contacts.py \
              tests/integration/test_author_from_scratch.py -v

# M4 — extraction, classification, rule deck, matching
uv run pytest tests/unit/test_index.py tests/unit/test_extract.py \
              tests/unit/test_classify.py tests/unit/test_deck.py \
              tests/unit/test_match.py tests/integration/test_decorate_target.py -v

# M5 — feature provenance, placement, generators, decorate
uv run pytest tests/unit/test_feature.py tests/unit/test_placement.py \
              tests/unit/test_generate.py tests/unit/test_decorate.py \
              tests/integration/test_post_opc.py -v

# M6 — SRAF placement and keep-out resolution
uv run pytest tests/unit/test_sraf.py tests/unit/test_resolve.py \
              tests/integration/test_sraf_placement.py -v

# M7 — structural checks, MRC, manifest, SVG
uv run pytest tests/unit/test_verify.py \
              tests/integration/test_verification_reporting.py -v

# M8 — mask export
uv run pytest tests/unit/test_mask_export.py -v

# M9 — golden regression corpus
uv run pytest tests/regression/ -v
```

### Checks that encode expensive findings

Some tests exist because something non-obvious turned out to be true:

```bash
# Width and space are measured exactly at any angle — no Manhattan special case.
uv run pytest tests/unit/test_classify.py -k "at_any_angle" -v

# A compiled curve stays within budget of the TRUE analytic curve. The budget is
# max_chord_error_nm + design_grid_nm*sqrt(2)/2; asserting the first term alone
# fails at EVERY radius, because quantization adds a second, independent error.
uv run pytest tests/unit/test_compile.py -k "true_curve" -v

# MRC's half-grid deburr, both halves — including that DISABLING it makes a clean
# rotated bar falsely fail. Without the deburr a clean bar reports more violation
# area than a genuinely defective one.
uv run pytest tests/unit/test_verify.py -k "deburr" -v

# Tone inversion swaps width and space: geometry passing min-width in clear tone
# fails it in dark tone.
uv run pytest tests/unit/test_mask_export.py -k "width_problem" -v

# gdstk's defaults (precision=1e-3, max_points=199) never silently override config.
uv run pytest tests/unit/test_context.py -v

# gdstk is imported by exactly two allowlisted modules, and the guard demonstrably fires.
uv run pytest tests/test_architecture.py -v
```

### Reproducibility

GDSII embeds a header timestamp, so writers pin it. OASIS carries none. The same layout
written twice is byte-identical:

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

Manifests are equally reproducible: no timestamp is recorded and keys are sorted, so two
runs of the same input produce byte-identical JSON.

### Regenerating goldens

Goldens record counts and statistics, never coordinates, so a failure says *what*
changed (`features.hammerhead: 4 -> 3`). Regeneration is explicit and never automatic —
a corpus that silently rewrites its own expectations tests nothing:

```bash
MASKLAYOUT_REGENERATE_GOLDENS=1 uv run pytest tests/regression/
```

---

## Open items and known limitations

Nothing here was discovered late; each was recorded when it was found.

### Not implemented

| Item | Detail |
|---|---|
| **Region selection / `materialize`** | The design's pipeline selects a decoration region and flattens *only* those placements. **Not wired up.** Hierarchy survives I/O, but `decorate()` takes a flat `Sequence[Polygon]`, so a caller must flatten manually and hierarchy is lost through correction. This is the largest gap. |
| **`jog` and edge fragmentation** | The only correction requiring an edge to be dissected into fragments. Deferred from M5. `decorate(skip_unknown_kinds=True)` lets a deck reference it and reports it; the default keeps it loud. |
| **Command-line interface** | Library only. The design lists a CLI and recipe runner; neither exists. |
| **PNG preview** | SVG only. Rasterising would add a dependency for no capability V1 needs. |
| **5 of 11 geometry PCells** | `serpentine`, `ring`, `racetrack`, `arbitrary_contour`, standalone `arc_segment`. Each is a parameter variation on machinery that exists. |
| **`sraf_array`, contact/via correction templates** | Listed in the originating brief's OPC PCell order; not reached. |

### Validation gaps

| Item | Detail |
|---|---|
| **Deck `apply` blocks are unvalidated** | Loading validates selectors against the closed vocabulary, but never checks the `apply` block against a generator's contract. A deck can name a `kind` with no generator, or supply parameters a generator rejects, and only fail at generation time. This let a broken rule sit in the shipped deck for two milestones. |
| **Field containment is a boolean test** | `validate_field` uses a boolean difference, not a topological containment predicate. Geometry touching the field from outside at a single vertex would pass. |
| **SRAF keep-out is a distance test** | An assist feature *inside* a ring-shaped target would measure a comfortable distance to the boundary while sitting where no assist feature belongs. The V1 pattern set has no such geometry. |

### Accuracy limits, by design

| Item | Detail |
|---|---|
| **MRC sensitivity floor** | Morphological MRC on quantized geometry can miss a violation within roughly one design-grid step of the rule value. The deburr that makes the checks usable on non-Manhattan geometry is what imposes the floor. Recorded in every manifest, not only here. |
| **Chord error is a two-term budget** | Deviation of a finished polygon from its analytic curve is `max_chord_error_nm + design_grid_nm·√2/2`. Tessellation and quantization are independent and additive — measured 1.18–1.50 nm at a 1.0 nm chord-error setting. |
| **`edge_bias` on a curved edge** | Produces a rectangle along the chord, not a true offset of the curve. Exact for a straight edge, below the grid for tessellated chords, wrong for a long gently curved edge. Not covered by tests. |
| **Line-end detection is a heuristic** | A short edge flanked by two long antiparallel ones. A chamfer can satisfy it. `line_end_ratio` is a parameter so the misfire is tunable. |
| **Goldens record counts, not coordinates** | A change altering coordinates while preserving every count and rounded area passes unnoticed. A deliberate trade: coordinate goldens fail on every legitimate improvement and train people to regenerate blindly. Area is recorded to the nearest nm², so a shift big enough to matter still shows. |
| **OASIS repetitions** | Kinds beyond rectangular collapse to explicit offsets on write. Lossless in placement, larger on disk. |

### Design amendments made during implementation

Two decisions in the design document were changed while building, both recorded in the
spec with their reasoning:

- **The gdstk allowlist grew from one module to two.** Stream conversion must construct
  `gdstk.Polygon`/`Cell`/`Reference`; folding that into `GeomContext` would give one
  module two unrelated responsibilities. The enforceable property is that the list is
  closed and tested, not that it has exactly one entry.
- **`local_density` was redefined.** The plan proposed a count-of-neighbours ratio,
  which is degenerate — one bar gives 1/1, three give 3/3, both 1.0. It is now pattern
  density: covered area over window area.

---

## Architecture

```
read GDS/OASIS → extract → classify → match rule deck → generate
  → resolve collisions → merge → verify → export (1x engineering, x4 mask)
```

`classify` is the contract: the rule deck is declarative over a **closed** selector
vocabulary, so whatever `classify` measures is exactly what a rule can select on. A deck
naming anything else fails at load.

**Integer coordinates.** All geometry is `int64` in design database units. No float
coordinate reaches the model layer or a writer, which is what makes reproducibility a
guarantee rather than an aspiration.

**Provenance on every generated feature.** Feature id, source site, rule id, deck
id/version/content-hash, and the parameters used. Identifiers are assigned *after*
normalization so they survive grid snapping and collinear-point removal.

## Technology

Python 3.12+, [`uv`](https://docs.astral.sh/uv/),
[`gdstk`](https://heitzmann.github.io/gdstk/) for geometry and stream formats, `shapely`
for spatial indexing and validity, `pydantic` for configuration and rule-deck
validation, with `pytest`, `ruff`, and `mypy` (strict).

The public API is independent of `gdstk` types; gdstk is confined to two allowlisted
modules, enforced by an AST-based test.

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
