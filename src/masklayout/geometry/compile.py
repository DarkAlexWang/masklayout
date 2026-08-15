"""Compile float polylines into grid-aligned integer polygons."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from masklayout.config import TechConfig
from masklayout.geometry.normalize import is_simple, normalize_polyline
from masklayout.geometry.report import TessellationReport
from masklayout.model.geometry import Polygon

#: Worst-case displacement when snapping a point to a square grid: half the
#: cell diagonal.
_HALF_GRID_DIAGONAL = math.sqrt(2.0) / 2.0


def compile_polyline(
    points_um: NDArray[np.float64],
    tech: TechConfig,
    layer: int,
    datatype: int = 0,
) -> tuple[Polygon, TessellationReport]:
    """Quantize, clean, and validate a float polyline into a model polygon.

    Order matters: normalize in float first so collinear removal uses true
    positions, then quantize once. Quantizing first would let grid noise turn
    genuinely collinear points into false corners.
    """
    source = np.asarray(points_um, dtype=np.float64)
    if not is_simple(source):
        raise ValueError("polyline self-intersects and cannot be compiled into a valid polygon")

    cleaned = np.asarray(
        normalize_polyline(
            source,
            duplicate_tolerance=tech.precision_um / 2.0,
            collinear_tolerance=tech.remove_collinear_tolerance_um,
        ),
        dtype=np.float64,
    )

    quantized = np.round(cleaned / tech.precision_um).astype(np.int64)

    displacement_um = np.linalg.norm(
        (quantized.astype(np.float64) * tech.precision_um) - cleaned, axis=1
    )
    measured_displacement_nm = float(displacement_um.max()) * 1000.0 if len(cleaned) else 0.0
    grid_error_nm = tech.design_grid_nm * _HALF_GRID_DIAGONAL

    polygon = Polygon(points=quantized, layer=layer, datatype=datatype)
    report = TessellationReport(
        vertex_count=polygon.vertex_count,
        # Whatever the measured displacement exceeds the grid's own worst case
        # is attributable to the incoming tessellation, not to snapping.
        tessellation_error_nm=max(measured_displacement_nm - grid_error_nm, 0.0),
        grid_error_nm=grid_error_nm,
        budget_nm=tech.max_chord_error_nm + grid_error_nm,
    )
    return polygon, report
