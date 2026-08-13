# masklayout V1 — Design

Status: approved for planning
Supersedes: `masklayout_v1_design_spec.md` (retained as the originating brief)

This document records the design decisions taken during brainstorming, the reasoning
behind them, and the toolchain facts verified by direct experiment. Where it differs
from the originating brief, this document governs.

---

## 1. Scope

`masklayout` is a recipe-driven computational mask-layout toolkit. It authors
non-Manhattan, curvilinear layouts from scratch, decorates imported target GDS with
deterministic rule-based OPC-like corrections, and exports both engineering (1×) and
mask (×4, tone-applied) streams.

V1 is a **geometry compiler and verifier**. It makes no claim about wafer printing.
Chord error, mask-space displacement, grid error, and polygon complexity are
controlled. Wafer EPE is not computed and geometry-only results are never described
as EPE results.

### Not in V1

Model-based OPC, ILT, aerial/resist/etch simulation, foundry-qualified DRC, and any
graphical editor.

### Target scale

10k–1M polygons per run, single process, runtime budget in minutes.
Verified feasible: 1M polygons union in 6.7 s, offset in 13.0 s (§9).

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Integer `int64` coordinates in design database units | Makes the determinism guarantee real rather than aspirational; no float state to diverge |
| 2 | Flatten only explicitly selected cells/regions | Context-aware OPC needs cross-cell neighbourhood; selective flattening is correct within the region and leaves the rest referenced |
| 3 | Full mask-data export: ×4 scale + tone inversion | Output is directly usable as reticle data |
| 4 | `magnification × design_grid` must be an exact multiple of `mask_grid` | Export scaling becomes a pure integer multiply — no re-snap, no second grid-error term |
| 5 | Field extent from an explicit `FIELD` layer polygon | Nothing is silently inferred; the boundary is versioned with the layout |
| 6 | Declarative YAML/JSON rule deck, pydantic-validated | Decks are data: hashable for provenance, diffable in review, shippable independently of code |
| 7 | Priority order, first match wins per feature kind | Deterministic and debuggable; different kinds still compose on one site |
| 8 | Fracture = vertex-limit splitting only | Matches `fracture_vertex_limit`; true trapezoid MDP fracture is out of scope |
| 9 | MRC by morphological open/close **with half-grid deburr** | Works on arbitrary-angle geometry; the deburr is mandatory (§6.2) |
| 10 | Numeric curvature from the polyline, no analytic re-fitting | Sufficient to select rules and place edge-local perturbations |
| 11 | Fragment on demand | Keeps vertex counts low; only jog and curvilinear perturbation dissect |

---

## 3. Units and coordinate model

Internal coordinates are `int64` in **design database units**. 1 DBU =
`database_precision`, default 0.001 µm = 1 nm. No float coordinate reaches the model
layer or a writer. Floats exist only inside curve tessellation and edge-local feature
construction, and are quantized on the way back into the model.

### TechConfig additions

Beyond the originating brief:

```python
design_grid_nm: float           # 1.0
mask_grid_nm:   float           # 0.5, at mask scale
magnification:  int             # 4
tone:           Literal["clear", "dark"]
field_layer:    Layer           # FIELD, default 20/0
mrc_deburr_nm:  float           # defaults to design_grid_nm / 2; see §6.2
```

`mrc_deburr_nm` **derives from `design_grid_nm`** rather than being an independent
constant: it compensates quantization error, which is a function of the grid. It
remains explicitly overridable, and the effective value is recorded in the manifest.

### Layer policy

As the originating brief, plus `FIELD`, which decision 5 makes mandatory for any
tone-inverted run:

| Logical layer | Default | Purpose |
|---|---:|---|
| `TARGET` | 10 / 0 | Desired target pattern |
| `POST_OPC` | 11 / 0 | Final generated post-OPC geometry |
| `SRAF` | 12 / 0 | Assist features, retained separately for debug |
| `FIELD` | 20 / 0 | Field/frame extent; required when tone inversion is enabled |
| `DEBUG_SOURCE` | 200 / 0 | Selected source edges, corners, candidate sites |
| `DEBUG_MARKERS` | 201 / 0 | Violations, conflicts, rejected candidates |
| `OVERLAY_ADD` | 202 / 0 | `POST_OPC − TARGET` |
| `OVERLAY_REMOVE` | 203 / 0 | `TARGET − POST_OPC` |

All numbers are configurable through the layer map. A production export may remap
`POST_OPC` and `SRAF`; the engineering export keeps logical layers intact.

### Config-load validation

`magnification × design_grid_nm` must be an integer multiple of `mask_grid_nm`.
Violations fail at load with all three values named. Because this holds, export
scaling is an integer multiply: no re-snap, no second quantization, no
post-verification drift.

### Grid alignment comes from the boolean engine

**Verified:** `gdstk.boolean(..., precision=p)` quantizes its output exactly to `p`
(measured off-grid residue at `precision=1e-3`: `0.000000`). Setting `precision` to
the design grid means boolean output is grid-aligned by construction.

There is therefore **no separate snap step after boolean**, and no seam problem at the
junction between a generated feature and its parent body. An earlier draft of this
design specified a `seam_overlap_nm` parameter and a
construct→overlap→union→snap→heal ordering to avoid T-junction slivers on rotated
geometry. Experiment showed no such slivers occur: a hammerhead on a 37° line end
merged to a clean 8-vertex polygon with zero sub-1 nm edges under snap-then-union,
union-then-snap, and overlap-then-union alike. That parameter and that ordering are
deleted.

### GeomContext: the one place gdstk is called

gdstk's defaults silently override configuration:

- `boolean`/`offset` default to `precision=1e-3`, quantizing to 1 nm regardless of a
  finer configured grid.
- `write_gds` defaults to `max_points=199`, re-fracturing at 199 regardless of
  `fracture_vertex_limit`.

Both violate the rule that the grid and fracture limit are never silently altered.
Mitigation: a single `GeomContext` carries `precision` and `max_points`, and every
gdstk call routes through it. A test asserts that no module imports `gdstk` directly
except `geometry/context.py`.

---

## 4. Compile pipeline

```
read GDS/OASIS                 hierarchy preserved
  → select decoration region   named cells or bbox
  → materialize                selected placements flattened into a working cell;
                               everything else stays referenced
  → normalize                  snap, dedup, decollinear, orient, reject invalid
  → index                      shapely STRtree over polygon bboxes
  → extract                    edges, corners, line ends, contacts,
                               per-vertex numeric curvature
  → classify                   width, space, density, angle, line-end type
                               ← this IS the rule deck's selector vocabulary
  → match                      priority order, first match wins per feature kind
  → generate                   edge-local construction, fragment on demand
  → resolve                    collision + keep-out; rejects → DEBUG_MARKERS
  → merge                      boolean union at precision = design grid
  → verify                     §6
  → fracture                   to fracture_vertex_limit
  → export                     two paths, §5
```

`classify` is the contract. Because the deck is declarative over a fixed vocabulary,
whatever `classify` measures is exactly what a rule can select on and nothing else.
Extending it later is additive; a rule can never reach past it.

### 4.1 Rule deck schema

A deck is data. It is loaded, validated by pydantic, and hashed; the hash and version
go into every generated feature's provenance.

```yaml
deck:
  id: generic_hammerhead_v1
  version: 1.0.0

rules:
  - id: hh_isolated_line_end
    priority: 10
    kind: hammerhead                 # one rule per kind wins per site
    when:                            # selectors drawn ONLY from classify's vocabulary
      site: line_end
      width_nm:   {min: 18, max: 26}
      space_nm:   {min: 120}         # isolated
      angle_deg:  any
    apply:
      pcell: hammerhead
      params:
        extension_nm:    28
        head_width_nm:   42
        neck_width_nm:   20
        corner_radius_nm: 2
        side: both

  - id: hh_dense_line_end
    priority: 20
    kind: hammerhead
    when:
      site: line_end
      width_nm: {min: 18, max: 26}
      space_nm: {max: 60}            # dense
    apply:
      pcell: hammerhead
      params: {extension_nm: 14, head_width_nm: 30, neck_width_nm: 20, side: both}
```

Selector keys are a closed set, fixed by `classify`: `site`, `width_nm`, `space_nm`,
`edge_length_nm`, `angle_deg`, `corner_type`, `curvature_1_per_um`, `local_density`.
A deck referencing any other key fails at load, naming the unknown key and listing the
valid ones. Rules are evaluated in `priority` order; the first match per `kind` wins,
so an edge may take a bias and its line end a hammerhead, but two hammerhead rules
cannot both fire on one site.

---

## 5. Export

Two deliberately separate paths.

**Engineering export** — 1×, as-drawn tone, logical layers intact, plus `DEBUG_*` and
`OVERLAY_*`, manifest, SVG.

**Mask export** — ×4, tone applied, production layer map, no debug layers.

Separating them keeps tone inversion out of verification and keeps the debug and
overlay story in the coordinate system where rule parameters are legible.

### Tone inversion

Written geometry is `FIELD − (POST_OPC ∪ SRAF)` when tone requires inversion.

Inversion **swaps width and space**, so a deck that passes min-space pre-inversion can
fail min-width after it. Post-inversion checks are therefore field containment plus a
full MRC re-run on the inverted geometry.

### Reproducibility

**Verified:** GDSII carries a header timestamp that defeats byte-reproducibility
unless pinned. `write_gds(timestamp=...)` accepts a fixed value; with it, two runs are
byte-identical, and without it they differ. Writers take a pinned timestamp from
config, default epoch 0.

**Verified:** OASIS has no timestamp parameter and writes none. Output is
byte-identical across runs separated in time with no special handling.

Golden tests byte-compare as a fast check and fall back to canonical polygon-set
comparison to produce a readable diff on mismatch.

---

## 6. Verification

### 6.1 Tiers

1. **Structural**, always — grid alignment, self-intersection, degenerate polygons,
   min area, min edge length, vertex limit after fracture.
2. **Rule**, always — MRC min width/space (§6.2), acute-angle and notch markers,
   SRAF keep-out and collision.
3. **Post-inversion**, mask export only — field containment, MRC re-run.

Every violation is written twice: a marker polygon on `DEBUG_MARKERS` for a human, and
a structured entry in the manifest for CI.

### 6.2 MRC requires a half-grid deburr

Min width is computed by morphological opening (erode by `min_width/2`, dilate back),
min space by closing, differencing the result against the original to localize
violations.

**Naive open/close is unusable on non-Manhattan geometry.** The erode/dilate round trip
accumulates roughly half a grid step of quantization error along every edge, and that
noise scales with perimeter. Measured on a 31°-rotated bar:

| deburr | clean bar | bar with 8 nm neck |
|---|---|---|
| 0.0 nm | 3 regions, 723 nm² | 3 regions, 777 nm² |
| **0.5 nm** | **0 regions, 0 nm²** | **2 regions, 330 nm²** |
| 1.0 nm | 0 regions, 0 nm² | 0 regions, 0 nm² |

A clean bar reported *more* violation area than a genuinely defective one. `join="miter"`
does not help; the residue is along edges, not only at corners. Thresholding on total
area cannot work.

**Required:** erode each violation region by `mrc_deburr_nm` (half the design grid)
before reporting. Thin quantization slivers vanish; compact real violations survive.

**Stated limitation:** the usable deburr window is narrow and grid-tied. Min-width
violations within roughly one grid step of the rule may be missed. This sensitivity
floor is recorded in the manifest, not left implicit.

### 6.3 Error philosophy

- Config and deck errors fail at load with the offending values named. Never degrade
  silently.
- Rule violations produce markers and the pipeline continues, yielding a complete
  report rather than first-failure.
- Structurally invalid input polygons are rejected and recorded, never silently
  auto-repaired.
- `--strict` turns any violation into a nonzero exit.

---

## 7. Modules and data model

Additions to the originating brief's repository layout, each traceable to a decision:

```
src/masklayout/
  cli.py                  # CLI was a stated interface but had no module
  geometry/context.py     # GeomContext — the only module that imports gdstk
  geometry/index.py       # shapely STRtree
  opc/deck.py             # rule schema, loader, validation, hashing
  opc/fragment.py         # on-demand dissection
  io/mask.py              # ×4 scale + tone inversion export path
  decks/                  # example decks as shipped data, not code
```

### Dependency posture

`shapely` is **load-bearing, not optional**. gdstk has no spatial index of any kind
(verified), and `STRtree` is the index. The originating brief's note that shapely is
optional in the long run does not hold for this design.

### pydantic vs. dataclasses

**pydantic** for anything crossing a file boundary — `TechConfig`, `LayerMap`, rule
decks, recipes, manifest — because those need validation, versioning, and round-tripping.

**Frozen dataclasses + numpy arrays** for hot-path geometry — `Polygon`, `Edge`,
`Site`, `Feature` — because at a million polygons per-object validation is unaffordable.

### Provenance IDs

The brief requires a source edge/vertex identifier, but snapping and collinear removal
destroy naive indices. **IDs are assigned after normalization, never before.**
Normalization is deterministic, so IDs are stable across runs of the same input.

```
polygon_id = "{cell}:{layer}:{n}"     # n = index in canonical sort by bbox (min_y, min_x)
edge_id    = "{polygon_id}#{k}"       # k from canonical start vertex, CCW
feature_id = "{rule_id}@{edge_id}"    # + ordinal when one rule emits several
```

The manifest records tool version, deck version and hash, tech-config hash, and the
pinned timestamp.

---

## 8. Testing

- `unit/` — geometry primitives: quantization, tessellation chord error, boolean,
  offset, fracture.
- `integration/` — pipeline stages on synthetic patterns.
- `regression/` — golden corpus.

Fixtures are **generated by code, not checked-in GDS**, so they are reproducible; only
canonical goldens are committed. Every geometry bug fix adds a fixture.

Corpus must cover: dense, semi-isolated, and isolated lines; line ends at 0°, 45°, and
arbitrary angle; convex and concave corners; acute corners; narrow necks; contact
arrays; a curvilinear Bézier wire; and pathological cases — self-touching, zero-area,
coincident edges, and a polygon past the vertex limit.

Two tests exist specifically to protect decisions in this document:

- No module imports `gdstk` except `geometry/context.py` (§3).
- MRC reports zero violations on a clean rotated bar and at least one on a bar with a
  known sub-rule neck (§6.2).

---

## 9. Verified toolchain facts

Established by direct experiment against gdstk 1.0.1, shapely 2.1.2, numpy 2.5.2 on
Python 3.12.1. These are load-bearing; re-verify on dependency upgrade.

| Fact | Result |
|---|---|
| `boolean(precision=p)` quantizes output to `p` | off-grid residue `0.000000` at `p=1e-3`; `0.494` grid units at `p=1e-6` |
| GDS reproducibility needs a pinned timestamp | identical sha with it, differing sha without |
| OASIS is reproducible with no special handling | no timestamp param; identical sha across runs separated in time |
| OASIS round-trips | `read_oas` recovers cells and polygons |
| `fracture(max_points, precision)` is vertex-limit splitting | confirmed |
| gdstk has no spatial index | confirmed — nothing tree/index/rtree in the module |
| Naive morphological MRC false-positives on clean angled geometry | 723 nm² clean vs 777 nm² defective |
| Half-grid deburr discriminates correctly | 0 regions clean, 2 regions (330 nm²) defective |
| Scale premise holds | 100k: union 0.23 s, offset 0.44 s. 1M: union 6.7 s, offset 13.0 s |

---

## 10. Milestones

| M | Deliverable | Acceptance |
|---|---|---|
| M0 | Skeleton + CI | `pytest`, `ruff`, `mypy` clean; reproducible env |
| M1 | I/O + hierarchy | GDS+OASIS round-trip, refs preserved, pinned timestamps |
| M2 | Geometry compiler | curves → grid-aligned polygons within chord error; integer DBU; `GeomContext` |
| M3 | PCell library | curvilinear wires, line ends, contacts, arrays from scratch |
| M4 | Extract + classify + deck | selector vocabulary measured; deck loads, validates, matches |
| M5 | Target decorator | bias, line-end extension, hammerhead, serif, jog + on-demand fragmentation |
| M6 | SRAF engine | rule-constrained SRAFs, collision and keep-out resolution |
| M7 | Verification + reporting | markers, overlays, manifest, SVG/PNG, MRC with deburr |
| M8 | Mask export | ×4 + tone inversion vs `FIELD`, post-inversion MRC, production layer map |
| M9 | Regression corpus | golden coverage across the full pattern list |

M4 and M5 are split from the brief's single M4 because the deck became a first-class
artifact. M8 is new, added by the full-mask-export decision.

**Planning scope.** This design covers all of V1, but one implementation plan does not.
Each milestone gets its own plan, written against this document and reviewed before the
next begins. The first plan covers **M0 only**: scaffold, tooling, `TechConfig` with the
§3 validation, the layer map, `GeomContext`, and passing CI. No OPC algorithms.

---

## 11. First vertical slice

1. Author a target containing a non-Manhattan tapered Bézier wire, an isolated line, a
   dense-line pair, and a contact array.
2. Export the target as GDSII and OASIS.
3. Import the target GDS.
4. Extract line ends.
5. Add parameterized hammerheads to selected line ends.
6. Boolean-merge at `precision = design grid`, simplify, fracture.
7. Emit `POST_OPC`, overlay layers, a JSON manifest, and an SVG preview.
8. **Also write the ×4 mask GDS with `tone: clear`** — scale only, no inversion.
9. Assert grid alignment, valid polygons, vertex-limit compliance, and chord-error
   compliance.

Step 8 costs almost nothing and keeps the two-path export structure exercised from day
one, so M8 adds inversion to a live path rather than bolting a whole export mode onto
code that never had one.

---

## 12. Development rules

Carried from the originating brief, with additions marked.

- Keep the public API independent of `gdstk` types.
- **All gdstk calls route through `GeomContext`.** (new)
- **Never rely on a gdstk default for precision or `max_points`.** (new)
- Use explicit units in parameter names, especially `*_nm` and `*_um`.
- Never silently alter the configured grid, tolerance, layer map, or fracture limit.
- Never flatten hierarchy implicitly.
- Treat every generated OPC feature as traceable data, not anonymous geometry.
- Generate deterministic results for identical input, deck, config, and tool version.
- Keep target geometry immutable; write generated geometry to logical output layers.
- Require regression fixtures for every geometry bug fix.
- Distinguish geometric validation from lithographic validation in all documentation
  and reports.
