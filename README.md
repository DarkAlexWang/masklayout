# masklayout

A recipe-driven computational mask-layout toolkit for semiconductor lithography masks,
in pure Python.

`masklayout` authors non-Manhattan, curvilinear layouts from scratch, decorates imported
target GDS with deterministic rule-based OPC-like corrections — hammerheads, serifs,
jogs, line-end extensions, local bias, SRAFs — and exports both engineering (1×) and
mask (×4, tone-applied) streams.

> **Status: M3 complete. Early development.**
> Configuration, layer map, the `GeomContext` geometry boundary, GDSII/OASIS import
> and export with preserved hierarchy, the geometry compiler, and a PCell library for
> authoring curvilinear wires, line ends, contacts, and hierarchical arrays from
> scratch are in place and tested. **No OPC algorithms exist yet.** The design is
> specified in
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
| M4 | Extraction, classification, rule deck | not started |
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

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Apache-2.0 was chosen over MIT for its express patent grant: OPC, SRAF placement, and
mask synthesis are patent-dense areas, and downstream users benefit from explicit patent
permission rather than an implied one.
