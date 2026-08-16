# masklayout M5 — Target Decorator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn M4's matches into correction geometry, merge it with the target, and emit `POST_OPC` with overlay layers and full provenance.

**Architecture:** One idea carries the whole milestone — **the site supplies placement, the rule supplies shape**. A generator reads position and outward normal from the site, takes shape parameters from the matched rule, and calls the existing PCell registry. Nothing new is needed to draw the shapes; M3 already built them.

**Design reference:** `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`.

## Verified before planning

| Fact | Consequence |
|---|---|
| A site's outward normal, fed to `build_line_end` as `angle_rad`, places a hammerhead correctly at 0° and 37° | Placement needs no per-angle special case |
| Target ∪ generated heads yields **one** polygon at both angles | No seam handling; M2's `boolean(precision=grid)` finding holds for real corrections |
| `width_nm` at a line end is the line's width (M4 fix) | Head width is a ratio of a meaningful number |

## Scope

M5's acceptance (§10) lists "bias, line-end extension, hammerhead, serif, jog + on-demand fragmentation". This plan delivers:

| Correction | Status |
|---|---|
| `line_end_extension` | in |
| `hammerhead` | in |
| `serif` | in |
| `edge_bias` | in |
| `jog` | **deferred** |

**`jog` is deferred, with reasons.** It is the only correction that requires dissecting an edge into fragments, and the design already isolates fragmentation as on-demand machinery serving just `jog` and curvilinear perturbation (§2 decision 11). Four working correction types with real merging, overlay layers, and provenance is a coherent milestone. A rushed jog bolted on beside them is not. It lands with `curvilinear_edge_perturbation`, where fragmentation earns its own milestone.

## Global Constraints

M0–M4 constraints all still apply, plus:

- **Placement comes from the site; shape comes from the rule.** A deck may not set `centre_um` or `angle_rad` — those are geometric facts, not authoring choices. A rule that tries fails loudly.
- **Target geometry is immutable** (§12). Corrections are new polygons written to `POST_OPC`; the `TARGET` layer is never modified in place.
- **Every generated feature carries provenance**: its own id, the source site id, the rule id, the deck id/version/hash, and the parameters used (§7).
- Corrections merge by boolean union at `precision = design grid`, which is grid-aligned by construction — no separate snap, no seam overlap.

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/opc/feature.py` | `Feature` — generated geometry plus provenance |
| `src/masklayout/opc/placement.py` | Site → placement parameters; rule-override rejection |
| `src/masklayout/opc/generate.py` | Generators per correction kind; the kind registry |
| `src/masklayout/opc/decorate.py` | Match → features → merged `POST_OPC` + overlays |
| `tests/unit/test_feature.py` | Provenance |
| `tests/unit/test_placement.py` | Placement derivation and override rejection |
| `tests/unit/test_generate.py` | Each correction kind |
| `tests/unit/test_decorate.py` | Merging, overlays, immutability |

---

## Task 1: Feature and placement

**Files:**
- Create: `src/masklayout/opc/feature.py`, `src/masklayout/opc/placement.py`
- Test: `tests/unit/test_feature.py`, `tests/unit/test_placement.py`

**Interfaces:**
- `Feature(id, kind, polygons, source_site_id, rule_id, deck_id, deck_version, deck_hash, parameters)` — frozen, with `.vertex_count` and `.provenance()` returning a JSON-ready dict.
- `PlacementOverrideError`
- `placement_for(site) -> dict[str, Any]` — `{"centre_um": ..., "angle_rad": ...}` where the angle is the outward normal's direction.
- `merge_params(placement, rule_params) -> dict[str, Any]` — raises `PlacementOverrideError` if the rule sets a placement key.

Key tests: the angle equals `atan2` of the outward normal; placement is derived at 0° and 37°; a rule supplying `centre_um` is rejected naming the key; provenance round-trips through JSON.

---

## Task 2: Line-end corrections

**Files:**
- Create: `src/masklayout/opc/generate.py`
- Test: `tests/unit/test_generate.py`

**Interfaces:**
- `generate_feature(measurement, match, tech) -> Feature | None`
- `register_generator(kind)` — decorator keyed by the rule's `kind`
- `UnknownCorrectionKindError`

**`hammerhead` and `line_end_extension` share one generator.** Both place a cap at a line end; they differ only in `head_width_ratio` — 1.0 is a plain extension, above 1.0 is a hammerhead. The generator computes `width_um = width_nm/1000 * head_width_ratio` from the measured line width, so the head scales with the line rather than being a fixed absolute.

Rule parameters: `extension_um` (required), `head_width_ratio` (default 1.0), `corner_radius_um` (default 0.0).

Key tests: a hammerhead is wider than its line and an extension is not; both extend outward, never inward; both work at 0° and 37°; a missing `extension_um` fails naming it; the feature carries the rule and deck ids.

---

## Task 3: Serif and edge bias

**Files:**
- Modify: `src/masklayout/opc/generate.py`
- Test: `tests/unit/test_generate.py`

**`serif`** places a small square at a convex corner, centred on the vertex and offset outward along the corner bisector. Rule parameters: `size_um`, optional `corner_radius_um`. It applies to `convex_corner` sites; a rule targeting a concave corner is legal but generates an inward serif, which the test pins.

**`edge_bias`** offsets an edge outward (positive) or inward (negative) by `bias_um`, producing a thin rectangle along the edge that is unioned for positive bias and subtracted for negative. The `Feature` gains `polarity: Literal["add", "subtract"]`, since M5 is the first milestone where a correction can remove material.

Key tests: a serif sits outside the target at a convex corner; positive bias yields `add` polarity and negative yields `subtract`; a zero bias produces no feature at all rather than a degenerate polygon.

---

## Task 4: Decorate, merge, and overlay

**Files:**
- Create: `src/masklayout/opc/decorate.py`
- Test: `tests/unit/test_decorate.py`, `tests/integration/test_post_opc.py`

**Interfaces:**
- `DecorateResult(post_opc, features, overlay_add, overlay_remove, report)`
- `decorate(polygons, deck, tech, layers, ...) -> DecorateResult`
- `DecorateReport(sites, matched, features_generated, features_rejected, by_kind)`

The pipeline: extract → classify → match → generate → merge. Merging is
`(target ∪ additive) − subtractive` at `precision = design grid`. Overlays are
`POST_OPC − TARGET` on `OVERLAY_ADD` and `TARGET − POST_OPC` on `OVERLAY_REMOVE`,
exactly as the layer policy defines them.

Key tests: the input polygons are unchanged after decorating (immutability); `POST_OPC`
has area greater than `TARGET` when only additive corrections fire; `OVERLAY_ADD` is
non-empty and `OVERLAY_REMOVE` empty in that case, and the reverse for negative bias;
every output polygon is grid-aligned; decorating twice gives identical output.

The integration test runs the isolated-and-dense two-bar target through `decorate` and
writes a real GDS with `TARGET`, `POST_OPC`, and both overlays on their proper layers,
then reads it back and checks the layers are present and distinct.

---

## Self-Review

**Spec coverage.** M5's acceptance is the target decorator producing bias, line-end extension, hammerhead, serif, and jog. Four of the five are delivered with merging, overlays, and provenance; `jog` is deferred with its reasoning stated above rather than quietly dropped. Task 1 supplies the provenance the design's §7 demands. Task 4 produces the `POST_OPC` and overlay layers the layer policy defines.

**Placeholder scan.** Tasks are specified by interface, semantics, and required test cases rather than full code bodies — the same style as M4's Tasks 4–5, and for the same reason: each task's shape depends on what the previous one produces. Expand each into TDD steps before starting it.

**Type consistency.** `Feature` fields are consumed under the same names in `generate.py` and `decorate.py`. `generate_feature(measurement, match, tech)` matches its registry signature and every call. `placement_for(site)` returns the two keys `merge_params` rejects rules from setting.

**Known risk.** `edge_bias` on a curved edge produces a rectangle along the chord, not a true offset of the curve. For the short fragments a curve tessellates into, the difference is below the grid; for a long straight edge it is exact. It is wrong for a long, gently curved edge, and the test suite does not currently cover that case — noted here rather than discovered later.
