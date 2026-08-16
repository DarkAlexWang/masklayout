# masklayout M7 — Verification and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Check the geometry we produce, mark what fails, and write a machine-readable record of everything the run did.

**Architecture:** Three tiers of check (structural, rule, post-inversion), each producing `Violation` records that go two places at once — a marker polygon on `DEBUG_MARKERS` for a human, and a structured entry in the manifest for CI. An SVG renderer makes the result visible without a GDS viewer.

**Design reference:** §6 of `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`.

## The finding this milestone exists to implement correctly

Design §6.2 is emphatic, and it was established by experiment before any code existed:

> Naive open/close is unusable on non-Manhattan geometry. The erode/dilate round trip accumulates roughly half a grid step of quantization error along every edge, and that noise scales with perimeter.

Measured on a 31°-rotated bar at the time: a **clean** bar reported 723 nm² of violations and a genuinely defective one 777 nm². A clean bar looked *worse* than a broken one. `join="miter"` did not help; the residue is along edges, not only at corners.

The fix is a **half-grid deburr**: erode each violation region by `design_grid_nm / 2` before reporting. At 0.5 nm the clean bar yields 0 regions and the defective one 2. This milestone must reproduce that, and a test must pin both halves — a check that only verifies "the defect is found" would pass on the unusable version.

## Global Constraints

M0–M6 constraints all still apply, plus:

- **Every violation is reported twice**: a marker polygon and a manifest record.
- **MRC carries its sensitivity floor in the manifest**, not in a comment. Violations within roughly one grid step of the rule may be missed, and a report that does not say so overstates its own coverage.
- **The manifest is the durable record**: tool version, tech config, deck id/version/hash, every feature's provenance, every violation, and the geometric statistics.

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/verify/violation.py` | `Violation`, `Severity` |
| `src/masklayout/verify/structural.py` | Grid, validity, area, edge length, vertex limit |
| `src/masklayout/verify/mrc.py` | Min width and space, with the deburr |
| `src/masklayout/verify/report.py` | `VerificationReport`, statistics |
| `src/masklayout/io/manifest.py` | JSON manifest |
| `src/masklayout/render/svg.py` | SVG preview |

---

## Task 1: Violations and structural checks

`Violation(check, severity, message, polygons, detail)`. Checks, each returning a list:

- `check_grid_alignment` — every vertex an exact multiple of the design grid.
- `check_simple` — no self-intersection (shapely).
- `check_min_area` — polygon area at least `min_polygon_area_nm2`.
- `check_min_edge_length` — every edge at least `min_edge_length_nm`.
- `check_vertex_limit` — vertex count at most `fracture_vertex_limit`.
- `run_structural_checks(polygons, tech) -> list[Violation]`

Key tests: each check fires on a deliberately broken polygon and stays silent on a good one; a bowtie is caught; a 4001-vertex polygon trips the limit at 4000; violations carry the offending geometry so it can be marked.

---

## Task 2: MRC with the half-grid deburr

- `check_min_width(polygons, min_width_nm, tech) -> list[Violation]`
- `check_min_space(polygons, min_space_nm, tech) -> list[Violation]`
- `MRC_SENSITIVITY_NOTE` — the floor, as text destined for the manifest.

Both use morphological open/close via `GeomContext.offset`, then **erode each violation region by `tech.effective_mrc_deburr_nm`** before reporting.

**Required tests, both halves:**
- A clean 31°-rotated bar produces **zero** width violations.
- A bar with a neck below the rule produces **at least one**.
- Without the deburr the clean bar produces violations — asserted directly, so the deburr cannot be removed without a test failing.
- Two bars closer than the min space are flagged; comfortably spaced ones are not.
- The reported violation region overlaps the actual defect, not some arbitrary corner.

---

## Task 3: Manifest

`write_manifest(path, tech, result, verification, tool_version, deck)` producing JSON with: tool version and timestamp (pinned), the full tech config, deck id/version/hash, every feature's provenance, every violation, geometric statistics (polygon and vertex counts, total area, perimeter), and the MRC sensitivity note.

Key tests: it round-trips through `json.load`; it names the deck hash; it contains one record per feature and per violation; two runs of the same input produce identical JSON; the sensitivity note is present whenever MRC ran.

---

## Task 4: SVG preview and acceptance

`render_svg(path, layers_to_polygons, tech, ...)` writing a standalone SVG with one group per layer, a distinct colour per logical layer, and a legend. No external dependencies — SVG is text.

The acceptance test decorates the two-bar target, verifies it, writes GDS + manifest + SVG, and asserts: the manifest names every feature and the deck hash; violations that exist appear both as markers and as manifest records; the SVG is non-empty and contains a path for each populated layer.

---

## Self-Review

**Spec coverage.** M7's acceptance is "marker layers, overlays, JSON manifest, SVG/PNG preview, and geometric report". Tasks 1–2 produce the violations that become markers; overlays already exist from M5; Task 3 is the manifest and geometric report; Task 4 is the preview. PNG is **not** delivered — SVG is the vector form the design lists first, and rasterising it would add a dependency for no capability this milestone needs. Stated rather than silently skipped.

**Known risk.** `check_min_edge_length` will flag the short chords of a finely tessellated curve, which are not defects. The tessellator's `min_segment_length_nm` is supposed to prevent them, so a violation there indicates a real inconsistency between the two settings — but on a deck that sets them inconsistently it will be noisy, and the noise will look like a geometry bug rather than a configuration one.
