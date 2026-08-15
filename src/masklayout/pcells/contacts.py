"""Contact PCell and hierarchical array placement."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import rounded_rect_um
from masklayout.model.cell import Cell, RectangularRepetition, Reference
from masklayout.model.geometry import Polygon
from masklayout.model.layout import Layout, UnknownCellError
from masklayout.pcells.base import PCellParams, register


class ContactParams(PCellParams):
    """A rectangular contact or via, optionally with rounded corners."""

    centre_um: tuple[float, float]
    size_um: tuple[float, float]
    corner_radius_um: float = Field(default=0.0, ge=0)


@register("contact", ContactParams)
def build_contact(
    params: ContactParams, tech: TechConfig, layer: int, datatype: int
) -> list[Polygon]:
    half_x = params.size_um[0] / 2.0
    half_y = params.size_um[1] / 2.0
    lower = (params.centre_um[0] - half_x, params.centre_um[1] - half_y)
    upper = (params.centre_um[0] + half_x, params.centre_um[1] + half_y)

    if params.corner_radius_um > 0.0:
        if 2.0 * params.corner_radius_um > min(params.size_um):
            raise ValueError(
                f"corner radius {params.corner_radius_um} does not fit a contact "
                f"of size {params.size_um}"
            )
        points = rounded_rect_um(
            lower, upper, params.corner_radius_um, tech.max_chord_error_nm / 1000.0
        )
    else:
        points = np.array(
            [
                [lower[0], lower[1]],
                [upper[0], lower[1]],
                [upper[0], upper[1]],
                [lower[0], upper[1]],
            ],
            dtype=np.float64,
        )

    polygon, _ = compile_polyline(points, tech, layer, datatype)
    return [polygon]


def place_contact_array(
    layout: Layout,
    parent_cell: str,
    contact_cell_name: str,
    params: ContactParams,
    columns: int,
    rows: int,
    pitch_um: tuple[float, float],
    layer: int,
    origin_um: tuple[float, float] = (0.0, 0.0),
    datatype: int = 0,
) -> Reference:
    """Place a contact array as hierarchy: one cell, one repeated reference.

    The design forbids implicit flattening, so this emits a Reference carrying
    a RectangularRepetition rather than N copies of the contact geometry.
    """
    if parent_cell not in layout.cells:
        raise UnknownCellError(
            f"unknown parent cell {parent_cell!r}; known cells: {sorted(layout.cells)}"
        )
    if contact_cell_name not in layout.cells:
        cell = layout.add(Cell(name=contact_cell_name))
        cell.polygons.extend(build_contact(params, layout.tech, layer, datatype))

    precision_um = layout.tech.precision_um
    reference = Reference(
        cell_name=contact_cell_name,
        origin_dbu=(
            round(origin_um[0] / precision_um),
            round(origin_um[1] / precision_um),
        ),
        repetition=RectangularRepetition(
            columns=columns,
            rows=rows,
            spacing_dbu=(
                round(pitch_um[0] / precision_um),
                round(pitch_um[1] / precision_um),
            ),
        ),
    )
    layout.cells[parent_cell].references.append(reference)
    return reference
