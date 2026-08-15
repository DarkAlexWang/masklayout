"""Shape PCells: rounded rectangles and line ends."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import arc_um, rounded_rect_um
from masklayout.model.geometry import Polygon
from masklayout.pcells.base import PCellParams, register


def rotate_um(
    points_um: NDArray[np.float64], angle_rad: float, origin_um: tuple[float, float]
) -> NDArray[np.float64]:
    """Rotate points about an origin. Used to place edge-local geometry."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    shifted = points_um - np.array(origin_um, dtype=np.float64)
    rotated = np.column_stack(
        (
            shifted[:, 0] * cos_a - shifted[:, 1] * sin_a,
            shifted[:, 0] * sin_a + shifted[:, 1] * cos_a,
        )
    )
    return rotated + np.array(origin_um, dtype=np.float64)


class RoundedRectParams(PCellParams):
    """An axis-aligned rectangle with circular corner fillets."""

    lower_um: tuple[float, float]
    upper_um: tuple[float, float]
    radius_um: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_extent(self) -> RoundedRectParams:
        width = self.upper_um[0] - self.lower_um[0]
        height = self.upper_um[1] - self.lower_um[1]
        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"degenerate rectangle {self.lower_um} to {self.upper_um}: "
                f"width={width}, height={height}"
            )
        return self


@register("rounded_rect", RoundedRectParams)
def build_rounded_rect(
    params: RoundedRectParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    points = rounded_rect_um(
        params.lower_um,
        params.upper_um,
        params.radius_um,
        tech.max_chord_error_nm / 1000.0,
    )
    polygon, _ = compile_polyline(points, tech, layer, datatype)
    return [polygon]


class LineEndParams(PCellParams):
    """The terminating cap of a drawn line, in edge-local coordinates.

    The cap extends forward from ``centre_um`` along ``angle_rad`` by
    ``extension_um``, spanning ``width_um`` across. Building it edge-local and
    rotating into place is what lets one PCell serve any line angle.
    """

    centre_um: tuple[float, float]
    width_um: float = Field(gt=0)
    extension_um: float = Field(gt=0)
    angle_rad: float = 0.0
    corner_radius_um: float = Field(default=0.0, ge=0)


@register("line_end", LineEndParams)
def build_line_end(
    params: LineEndParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    budget_um = tech.max_chord_error_nm / 1000.0
    half_width = params.width_um / 2.0
    radius = params.corner_radius_um

    if radius > 0.0:
        if 2.0 * radius > min(params.width_um, params.extension_um):
            raise ValueError(
                f"corner radius {radius} does not fit a line end "
                f"{params.width_um} wide by {params.extension_um} long"
            )
        outer = params.extension_um
        local = np.vstack(
            [
                np.array([[0.0, -half_width]]),
                np.array([[outer - radius, -half_width]]),
                arc_um(
                    (outer - radius, -half_width + radius),
                    radius,
                    -math.pi / 2,
                    0.0,
                    budget_um,
                ),
                arc_um(
                    (outer - radius, half_width - radius),
                    radius,
                    0.0,
                    math.pi / 2,
                    budget_um,
                ),
                np.array([[0.0, half_width]]),
            ]
        )
    else:
        local = np.array(
            [
                [0.0, -half_width],
                [params.extension_um, -half_width],
                [params.extension_um, half_width],
                [0.0, half_width],
            ],
            dtype=np.float64,
        )

    placed = rotate_um(local, params.angle_rad, (0.0, 0.0)) + np.array(
        params.centre_um, dtype=np.float64
    )
    polygon, _ = compile_polyline(placed, tech, layer, datatype)
    return [polygon]
