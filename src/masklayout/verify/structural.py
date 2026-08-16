"""Structural checks: geometry that is malformed regardless of any rule."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.normalize import is_simple, signed_area
from masklayout.model.geometry import Polygon
from masklayout.verify.violation import Violation


def check_grid_alignment(polygons: Sequence[Polygon], tech: TechConfig) -> list[Violation]:
    """Every vertex must be an exact multiple of the design grid.

    Model coordinates are int64 DBU, so this holds by construction; the check
    exists to catch a polygon built by some path that bypassed compilation.
    """
    violations: list[Violation] = []
    for index, polygon in enumerate(polygons):
        if polygon.points.dtype != np.int64:
            violations.append(
                Violation(
                    check="grid_alignment",
                    severity="error",
                    message=(
                        f"polygon {index} has {polygon.points.dtype} coordinates, "
                        "not int64 design database units"
                    ),
                    polygons=[polygon],
                    detail={"polygon_index": index, "dtype": str(polygon.points.dtype)},
                )
            )
    return violations


def check_simple(polygons: Sequence[Polygon], tech: TechConfig) -> list[Violation]:
    """No polygon may self-intersect."""
    violations: list[Violation] = []
    for index, polygon in enumerate(polygons):
        if not is_simple(polygon.points.astype(np.float64) * tech.precision_um):
            violations.append(
                Violation(
                    check="self_intersection",
                    severity="error",
                    message=f"polygon {index} self-intersects",
                    polygons=[polygon],
                    detail={"polygon_index": index},
                )
            )
    return violations


def check_min_area(polygons: Sequence[Polygon], tech: TechConfig) -> list[Violation]:
    """Reject slivers below the configured minimum area."""
    violations: list[Violation] = []
    for index, polygon in enumerate(polygons):
        area_nm2 = abs(signed_area(polygon.points)) * (tech.design_grid_nm**2)
        if area_nm2 < tech.min_polygon_area_nm2:
            violations.append(
                Violation(
                    check="min_area",
                    severity="error",
                    message=(
                        f"polygon {index} has area {area_nm2:.3f} nm^2, "
                        f"below the {tech.min_polygon_area_nm2} nm^2 minimum"
                    ),
                    polygons=[polygon],
                    detail={"polygon_index": index, "area_nm2": area_nm2},
                )
            )
    return violations


def check_min_edge_length(polygons: Sequence[Polygon], tech: TechConfig) -> list[Violation]:
    """Reject edges shorter than the configured minimum.

    A finely tessellated curve legitimately has short chords, so this fires
    on a curve whose ``min_segment_length_nm`` disagrees with
    ``min_edge_length_nm``. That is a configuration inconsistency, not a
    geometry defect, and the message says so.
    """
    violations: list[Violation] = []
    for index, polygon in enumerate(polygons):
        points = polygon.points.astype(np.float64)
        lengths = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
        lengths_nm = lengths * tech.design_grid_nm
        short = lengths_nm[lengths_nm < tech.min_edge_length_nm]
        if short.size:
            violations.append(
                Violation(
                    check="min_edge_length",
                    severity="warning",
                    message=(
                        f"polygon {index} has {short.size} edge(s) shorter than "
                        f"{tech.min_edge_length_nm} nm (shortest {short.min():.3f} nm); "
                        "on a tessellated curve this indicates min_segment_length_nm "
                        "and min_edge_length_nm disagree"
                    ),
                    polygons=[polygon],
                    detail={
                        "polygon_index": index,
                        "short_edges": int(short.size),
                        "shortest_nm": float(short.min()),
                    },
                )
            )
    return violations


def check_vertex_limit(polygons: Sequence[Polygon], tech: TechConfig) -> list[Violation]:
    """No polygon may exceed the export vertex limit."""
    violations: list[Violation] = []
    for index, polygon in enumerate(polygons):
        if polygon.vertex_count > tech.fracture_vertex_limit:
            violations.append(
                Violation(
                    check="vertex_limit",
                    severity="error",
                    message=(
                        f"polygon {index} has {polygon.vertex_count} vertices, "
                        f"above the {tech.fracture_vertex_limit} limit; fracture first"
                    ),
                    polygons=[polygon],
                    detail={
                        "polygon_index": index,
                        "vertex_count": polygon.vertex_count,
                    },
                )
            )
    return violations


def run_structural_checks(polygons: Sequence[Polygon], tech: TechConfig) -> list[Violation]:
    """Every structural check, in a stable order."""
    violations: list[Violation] = []
    for check in (
        check_grid_alignment,
        check_simple,
        check_min_area,
        check_min_edge_length,
        check_vertex_limit,
    ):
        violations.extend(check(polygons, tech))
    return violations
