"""Wire PCells built by offsetting a centerline along its normals.

shapely's buffer would stroke a constant-width line, but cannot express a
taper and approximates joins on its own terms rather than against our
chord-error budget. Offsetting along per-point normals handles varying width
directly and keeps the budget in one place.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import bezier_um
from masklayout.model.geometry import Polygon
from masklayout.pcells.base import PCellParams, register

MIN_CENTERLINE_POINTS = 2


def _unit_tangents(centerline_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Tangent at each point: central differences inside, one-sided at the ends."""
    tangents = np.empty_like(centerline_um)
    tangents[1:-1] = centerline_um[2:] - centerline_um[:-2]
    tangents[0] = centerline_um[1] - centerline_um[0]
    tangents[-1] = centerline_um[-1] - centerline_um[-2]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents / np.where(lengths > 0.0, lengths, 1.0)


def offset_centerline_um(
    centerline_um: NDArray[np.float64], widths_um: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Build the closed ring of a wire of varying width about a centerline."""
    centre = np.asarray(centerline_um, dtype=np.float64)
    widths = np.asarray(widths_um, dtype=np.float64)
    if len(centre) < MIN_CENTERLINE_POINTS:
        raise ValueError(f"a centerline needs at least {MIN_CENTERLINE_POINTS} points")
    if len(widths) != len(centre):
        raise ValueError(
            f"need one width per centerline point: {len(widths)} widths for {len(centre)} points"
        )
    if np.any(widths <= 0.0):
        raise ValueError("every width must be positive")

    tangents = _unit_tangents(centre)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    half = (widths / 2.0)[:, None]
    left = centre + normals * half
    right = centre - normals * half
    return np.vstack((left, right[::-1]))


class BezierWireParams(PCellParams):
    """A constant-width wire following a Bezier centerline."""

    control_points_um: tuple[tuple[float, float], ...]
    width_um: float = Field(gt=0)


@register("bezier_wire", BezierWireParams)
def build_bezier_wire(
    params: BezierWireParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    centerline = bezier_um(
        np.array(params.control_points_um, dtype=np.float64),
        tech.max_chord_error_nm / 1000.0,
    )
    widths = np.full(len(centerline), params.width_um, dtype=np.float64)
    polygon, _ = compile_polyline(offset_centerline_um(centerline, widths), tech, layer, datatype)
    return [polygon]


class TaperedWireParams(PCellParams):
    """A wire whose width varies linearly along a Bezier centerline."""

    control_points_um: tuple[tuple[float, float], ...]
    start_width_um: float = Field(gt=0)
    end_width_um: float = Field(gt=0)


@register("tapered_wire", TaperedWireParams)
def build_tapered_wire(
    params: TaperedWireParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    centerline = bezier_um(
        np.array(params.control_points_um, dtype=np.float64),
        tech.max_chord_error_nm / 1000.0,
    )
    widths = np.linspace(
        params.start_width_um, params.end_width_um, len(centerline), dtype=np.float64
    )
    polygon, _ = compile_polyline(offset_centerline_um(centerline, widths), tech, layer, datatype)
    return [polygon]
