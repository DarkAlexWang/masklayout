"""Conversion between the typed model and gdstk.

This is one of exactly two modules permitted to import gdstk; see the design
document, section "The gdstk boundary". Nothing here may leak a gdstk type
into a public signature.
"""

from __future__ import annotations

import gdstk
import numpy as np
from numpy.typing import NDArray

from masklayout.config import TechConfig
from masklayout.io.errors import (
    GridMismatchError,
    OffGridCoordinateError,
    UnsupportedEntityError,
)
from masklayout.io.report import ReadReport
from masklayout.model.cell import (
    Cell,
    ExplicitRepetition,
    RectangularRepetition,
    Reference,
    Repetition,
)
from masklayout.model.geometry import Label, Polygon
from masklayout.model.layers import LayerMap
from masklayout.model.layout import Layout

#: Maximum acceptable deviation, in grid units, before a coordinate counts as
#: off-grid. Generous enough for float64 division noise, tight enough to catch
#: a genuine half-grid coordinate.
_ON_GRID_TOLERANCE = 1e-6

#: GDSII user unit expected by this toolkit: 1 micrometre.
_EXPECTED_UNIT_M = 1e-6


def um_to_dbu(points_um: NDArray[np.float64], precision_um: float) -> NDArray[np.int64]:
    """Convert float micrometres to integer design database units."""
    values = np.asarray(points_um, dtype=np.float64)
    scaled = values / precision_um
    rounded = np.round(scaled)
    residue = np.abs(scaled - rounded)
    if residue.size and residue.max() > _ON_GRID_TOLERANCE:
        worst = np.unravel_index(int(np.argmax(residue)), residue.shape)
        raise OffGridCoordinateError(
            f"coordinate {float(values[worst])!r} um is not a multiple of the design "
            f"grid {precision_um} um (off by {float(residue.max())} grid units)"
        )
    return rounded.astype(np.int64)


def dbu_to_um(points_dbu: NDArray[np.int64], precision_um: float) -> NDArray[np.float64]:
    """Convert integer design database units to float micrometres."""
    return np.asarray(points_dbu, dtype=np.float64) * precision_um


def check_library_grid(unit: float, precision_m: float, tech: TechConfig) -> None:
    """Reject a stream whose grid differs from the configured design grid.

    The design forbids silently adopting a file's grid or resampling onto
    ours: target geometry is immutable, so a mismatch is an error.
    """
    if not np.isclose(unit, _EXPECTED_UNIT_M, rtol=0.0, atol=1e-18):
        raise GridMismatchError(
            f"unsupported user unit {unit!r} m; masklayout expects {_EXPECTED_UNIT_M!r} m"
        )
    if not np.isclose(precision_m, tech.precision_m, rtol=1e-12, atol=0.0):
        raise GridMismatchError(
            f"file database precision {precision_m!r} m does not match the configured "
            f"design grid {tech.precision_m!r} m "
            f"(design_grid_nm={tech.design_grid_nm}); "
            "set design_grid_nm to match the file, or convert the file"
        )


def _points_to_polygon(gpoly: gdstk.Polygon, precision_um: float) -> Polygon:
    return Polygon(
        points=um_to_dbu(np.asarray(gpoly.points), precision_um),
        layer=int(gpoly.layer),
        datatype=int(gpoly.datatype),
    )


def _repetition_to_model(
    repetition: gdstk.Repetition | None, precision_um: float
) -> Repetition | None:
    """Convert a gdstk repetition.

    gdstk's Repetition has no ``type`` attribute; the kind is inferred from
    which fields are populated.
    """
    if repetition is None:
        return None
    if (
        repetition.columns is not None
        and repetition.rows is not None
        and repetition.spacing is not None
    ):
        spacing = um_to_dbu(np.array([repetition.spacing], dtype=np.float64), precision_um)[0]
        return RectangularRepetition(
            columns=int(repetition.columns),
            rows=int(repetition.rows),
            spacing_dbu=(int(spacing[0]), int(spacing[1])),
        )
    offsets = np.asarray(repetition.get_offsets(), dtype=np.float64)
    return ExplicitRepetition(offsets_dbu_array=um_to_dbu(offsets, precision_um))


def library_to_layout(
    library: gdstk.Library,
    tech: TechConfig,
    layers: LayerMap,
    source: str,
) -> tuple[Layout, ReadReport]:
    """Convert a gdstk library into the typed model.

    Paths are converted to polygons and counted. Labels are preserved.
    A grid mismatch is an error, never a silent regrid.
    """
    check_library_grid(library.unit, library.precision, tech)
    precision_um = tech.precision_um

    layout = Layout(name=library.name, tech=tech, layers=layers)
    paths_converted = 0
    label_count = 0
    reference_count = 0

    for gcell in library.cells:
        if not isinstance(gcell, gdstk.Cell):
            raise UnsupportedEntityError(
                f"{source}: cell {gcell!r} is a raw cell and cannot be modelled"
            )
        cell = Cell(name=gcell.name)

        for gpoly in gcell.polygons:
            cell.polygons.append(_points_to_polygon(gpoly, precision_um))

        for gpath in gcell.paths:
            paths_converted += 1
            for gpoly in gpath.to_polygons():
                cell.polygons.append(_points_to_polygon(gpoly, precision_um))

        for glabel in gcell.labels:
            label_count += 1
            origin = um_to_dbu(np.array([glabel.origin], dtype=np.float64), precision_um)[0]
            cell.labels.append(
                Label(
                    text=glabel.text,
                    origin_dbu=(int(origin[0]), int(origin[1])),
                    layer=int(glabel.layer),
                    datatype=int(glabel.texttype),
                )
            )

        for gref in gcell.references:
            reference_count += 1
            origin = um_to_dbu(np.array([gref.origin], dtype=np.float64), precision_um)[0]
            cell.references.append(
                Reference(
                    cell_name=gref.cell.name,
                    origin_dbu=(int(origin[0]), int(origin[1])),
                    rotation_rad=float(gref.rotation),
                    magnification=float(gref.magnification),
                    x_reflection=bool(gref.x_reflection),
                    repetition=_repetition_to_model(gref.repetition, precision_um),
                )
            )

        layout.add(cell)

    report = ReadReport(
        source=source,
        cell_count=len(layout.cells),
        polygon_count=layout.polygon_count(),
        label_count=label_count,
        reference_count=reference_count,
        paths_converted=paths_converted,
        file_precision_m=float(library.precision),
        top_cells=tuple(layout.top_cells()),
    )
    return layout, report
