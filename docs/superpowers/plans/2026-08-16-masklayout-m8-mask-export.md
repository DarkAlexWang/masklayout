# masklayout M8 — Mask Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Emit reticle data — ×4 scaled, tone applied against a declared field, on a production layer map — and re-verify it, because inversion changes what the checks mean.

**Architecture:** A separate export path from the engineering one. Scaling is a pure integer multiply, guaranteed by the config validation written back at M0. Tone inversion is `FIELD − (POST_OPC ∪ SRAF)`. Because inversion **swaps width and space**, MRC runs again on the inverted result rather than being trusted from before.

**Design reference:** §5 of `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`.

## What M0 already guarantees

`TechConfig` validates at load that `magnification × design_grid_nm` is an exact multiple of `mask_grid_nm`. That was decided at the very first design question, and it is what makes this milestone simple: **scaling is `points * magnification`, an exact integer operation.** No re-snap, no second quantization, no post-verification drift. A test asserts the scaled result lands on the mask grid.

## Global Constraints

M0–M7 constraints all still apply, plus:

- **Tone inversion requires an explicit `FIELD` polygon.** No inferred extent, no computed bounding box. A tone-inverted run without one fails, naming the layer.
- **All geometry must lie inside the field.** Anything outside is an error, not a silent clip.
- **MRC re-runs after inversion**, and the report says which pass a violation came from. A min-space violation pre-inversion becomes a min-width violation after; reporting them interchangeably would mislead.
- **The mask export carries no debug or overlay layers.** It is reticle data, not an engineering view.

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/io/mask.py` | Scaling, tone inversion, mask-layer mapping, export |
| `tests/unit/test_mask_export.py` | Scaling exactness, inversion, field validation |
| `tests/integration/test_mask_stream.py` | End-to-end mask GDS/OASIS |

---

## Task 1: Scaling and the field

**Interfaces:**
- `MaskExportError`, `FieldMissingError`, `GeometryOutsideFieldError`
- `scale_to_mask(polygons, tech) -> list[Polygon]` — multiply by `magnification`; exact.
- `validate_field(field, geometry, tech) -> None` — field present, single polygon, everything inside.
- `on_mask_grid(polygons, tech) -> bool`

Key tests: a 100 nm feature becomes 400 nm at ×4; the scaled result is on the mask grid for the default config; scaling is exact for every vertex, asserted by comparing against integer multiplication directly; a missing field raises naming `FIELD`; geometry poking outside the field raises naming the offending polygon; an empty field polygon list is an error, not an empty mask.

---

## Task 2: Tone inversion

**Interfaces:**
- `invert_tone(field, geometry, tech) -> list[Polygon]` — returns `field − geometry`.
- `apply_tone(field, post_opc, srafs, tech) -> list[Polygon]` — dispatches on `tech.tone`.

For `tone="clear"` the written geometry is `POST_OPC ∪ SRAF` unchanged. For `tone="dark"` it is `FIELD − (POST_OPC ∪ SRAF)`.

Key tests: inverting a bar in a field yields a field-with-a-hole (area = field area − bar area); inverting twice returns the original; SRAFs are included in the subtraction, not left as holes-in-holes; a clear-tone run leaves geometry untouched; the inverted result is grid-aligned.

---

## Task 3: Post-inversion verification and export

**Interfaces:**
- `MaskExportResult(geometry, violations, statistics)`
- `write_mask_gds(path, geometry, tech, layer, timestamp=None)`
- `export_mask(post_opc, srafs, field, tech, mask_layer, min_width_nm, min_space_nm) -> MaskExportResult`

The pipeline: validate field → union `POST_OPC ∪ SRAF` → apply tone → scale ×4 → **re-run MRC at mask scale** → map to the production layer → write.

MRC thresholds at mask scale are the 1× values times the magnification, since the geometry grew. Each violation records `pass="post_inversion"` so it is never confused with a pre-inversion finding.

Key tests: a dark-tone export inverts and still passes structural checks; the post-inversion MRC pass is labelled; a design that passes min-space at 1× can fail min-width after inversion, and the test constructs exactly that case; the mask GDS is byte-reproducible; the mask stream contains only the production layer, with no debug or overlay layers present.

---

## Self-Review

**Spec coverage.** M8's acceptance is "×4 + tone inversion vs `FIELD`, post-inversion MRC, production layer map". Task 1 covers scaling and the field contract, Task 2 the inversion, Task 3 the re-verification and the stream. The design's insistence that inversion swaps width and space is not merely honoured but demonstrated by a test that builds a case which passes one check before and fails the other after.

**Known risk.** Field containment is tested by bounding box and boolean difference, not by a topological containment predicate. A geometry that exactly abuts the field edge is accepted, which is correct, but one that touches from outside at a single vertex would also pass the difference test while being arguably outside. The V1 pattern set has no such case; a real frame would want a stricter predicate.
