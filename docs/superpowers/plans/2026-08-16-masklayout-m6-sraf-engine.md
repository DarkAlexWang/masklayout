# masklayout M6 — SRAF Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Place rule-constrained sub-resolution assist features, resolve their collisions, and enforce keep-out — reporting every rejection rather than silently dropping it.

**Architecture:** SRAFs reuse M5's generator machinery with one difference that runs through the whole milestone: **an assist feature is not part of the main pattern.** It goes on the `SRAF` layer and is never merged into `POST_OPC`. The design settles this — tone inversion is `FIELD − (POST_OPC ∪ SRAF)`, treating the two as distinct geometry.

**Design reference:** `docs/superpowers/specs/2026-08-14-masklayout-v1-design.md`, §10 M6 and the layer policy.

## Global Constraints

M0–M5 constraints all still apply, plus:

- **`Feature.polarity` gains `"assist"`.** Assist features are collected separately, written to `SRAF`, and excluded from the `POST_OPC` boolean.
- **Every rejected SRAF is reported with a reason.** Silently dropping a candidate would make a sparse result indistinguishable from a working one. Rejections go to `DEBUG_MARKERS` and into the report.
- **Resolution is deterministic.** Candidates are considered in a stable order so the same input yields the same surviving set.

## File Structure

| File | Responsibility |
|---|---|
| `src/masklayout/opc/sraf.py` | `sraf_bar` generator |
| `src/masklayout/opc/resolve.py` | Collision and keep-out resolution |
| `tests/unit/test_sraf.py` | Placement and geometry |
| `tests/unit/test_resolve.py` | Rejection rules and determinism |

---

## Task 1: The SRAF generator

**Interfaces:**
- `sraf_bar` registered as a correction kind, producing `polarity="assist"`.
- Rule parameters: `distance_um` (edge to near side of the bar, required), `width_um` (required), `length_ratio` (default 1.0, of the source edge length).

Placement reuses `placement_for`: the bar sits along the outward normal at `distance_um + width_um/2` from the edge, running parallel to it. Because the offset is measured from the edge rather than the bar's centre, a deck author writes the number a rule deck actually specifies.

Key tests: the bar sits outside the target at the requested distance; it does not touch the target; it is parallel to its source edge at 0° and 37°; `length_ratio` scales it; a zero width or non-positive distance produces no feature; polarity is `"assist"`.

---

## Task 2: Collision and keep-out resolution

**Interfaces:**
- `Rejection(feature_id, reason, detail)` — frozen.
- `resolve_collisions(features, target, tech, target_keepout_um, sraf_keepout_um) -> tuple[list[Feature], list[Rejection]]`

Rules, applied in feature-id order for determinism:

1. An assist feature closer than `target_keepout_um` to any target polygon is rejected — reason `"target_keepout"`.
2. An assist feature closer than `sraf_keepout_um` to an already-kept assist feature is rejected — reason `"sraf_keepout"`.
3. Non-assist features pass through untouched; M5's corrections are meant to touch the target.

Key tests: a bar inside the keep-out is rejected naming the reason; two bars too close leave exactly one survivor; which one survives is stable across runs; a comfortably spaced pair both survive; corrections are never rejected; the rejection carries the measured distance so a deck author can see by how much.

---

## Task 3: Integration and acceptance

`DecorateResult` gains `srafs: list[Polygon]` and `rejected: list[Rejection]`; `DecorateReport` gains `srafs_placed` and `srafs_rejected`. `decorate` routes assist features through `resolve_collisions`, writes survivors to the `SRAF` layer, and writes rejected candidates to `DEBUG_MARKERS`.

Key tests: SRAFs land on layer 12 and markers on 201; `POST_OPC` area is unchanged by SRAF placement (they are not merged); the report counts placed and rejected; a deck with an aggressive SRAF rule produces rejections rather than overlapping geometry; the whole thing round-trips to GDS with five distinct layers.

---

## Self-Review

**Spec coverage.** M6's acceptance is "rule-constrained SRAFs, collision and keep-out resolution". Task 1 makes them rule-constrained through the same deck mechanism as every other correction. Task 2 resolves collisions and enforces keep-out with reported reasons. Task 3 wires them into the pipeline on their own layer.

**Known risk.** Keep-out is measured between polygon boundaries, so a SRAF *inside* a ring-shaped target would measure a comfortable distance to the boundary while sitting in a place no assist feature belongs. The V1 pattern set has no such geometry, and the check is honest for everything it does cover — but it is a distance test, not a containment test, and that distinction will matter when donut-shaped targets appear.
