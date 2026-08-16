"""The single boundary between masklayout and gdstk.

This is the only module in the package permitted to import gdstk, enforced
by tests/test_architecture.py.

It exists because gdstk's defaults silently override configuration:
``boolean`` and ``offset`` default to ``precision=1e-3``, and ``write_gds``
defaults to ``max_points=199``. Calling gdstk directly anywhere else would
quietly substitute those for the configured grid and fracture limit.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import gdstk
import numpy as np

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon as ModelPolygon

BooleanOperation = Literal["or", "and", "xor", "not"]
#: Join styles accepted by ``gdstk.offset``. Narrower than FlexPath's join
#: styles: "natural" and "smooth" are path joins and are rejected here.
JoinStyle = Literal["miter", "bevel", "round"]

#: Default pinned timestamp. GDSII embeds a header timestamp that would
#: otherwise make output non-reproducible between runs.
PINNED_TIMESTAMP = datetime.datetime(1970, 1, 1)

#: GDSII user unit: 1 micrometre.
USER_UNIT_M = 1e-6


class GeomContext:
    """Carries the configured precision and fracture limit into every gdstk call."""

    def __init__(self, tech: TechConfig, timestamp: datetime.datetime | None = None) -> None:
        self._tech = tech
        self._timestamp = PINNED_TIMESTAMP if timestamp is None else timestamp

    @property
    def tech(self) -> TechConfig:
        return self._tech

    @property
    def precision_um(self) -> float:
        """The design grid, in micrometres."""
        return self._tech.precision_um

    def boolean(
        self,
        operand1: gdstk.Polygon | list[gdstk.Polygon],
        operand2: gdstk.Polygon | list[gdstk.Polygon],
        operation: BooleanOperation,
    ) -> list[gdstk.Polygon]:
        """Boolean operation at the configured grid.

        Because precision equals the design grid, the result is grid-aligned
        by construction and needs no separate snapping pass.
        """
        return gdstk.boolean(operand1, operand2, operation, precision=self.precision_um)

    def offset(
        self,
        polygons: gdstk.Polygon | list[gdstk.Polygon],
        distance_um: float,
        join: JoinStyle = "miter",
        tolerance: int = 2,
        use_union: bool = True,
    ) -> list[gdstk.Polygon]:
        """Dilate (positive distance) or erode (negative) at the configured grid."""
        return gdstk.offset(
            polygons,
            distance_um,
            join=join,
            tolerance=tolerance,
            precision=self.precision_um,
            use_union=use_union,
        )

    def fracture(self, polygon: gdstk.Polygon) -> list[gdstk.Polygon]:
        """Split a polygon to the configured vertex limit."""
        return polygon.fracture(
            max_points=self._tech.fracture_vertex_limit,
            precision=self.precision_um,
        )

    def boolean_polygons(
        self,
        operand1: Sequence[ModelPolygon],
        operand2: Sequence[ModelPolygon],
        operation: BooleanOperation,
        layer: int,
        datatype: int = 0,
    ) -> list[ModelPolygon]:
        """Boolean on model polygons, returning model polygons.

        Lets callers outside the gdstk allowlist combine geometry. Because
        precision equals the design grid, the result is grid-aligned by
        construction and no separate snap is needed.
        """
        precision_um = self._tech.precision_um

        def to_gdstk(polygons: Sequence[ModelPolygon]) -> list[gdstk.Polygon]:
            return [
                gdstk.Polygon(
                    cast(
                        "Sequence[tuple[float, float]]",
                        polygon.points.astype(np.float64) * precision_um,
                    )
                )
                for polygon in polygons
            ]

        result = gdstk.boolean(
            to_gdstk(operand1), to_gdstk(operand2), operation, precision=precision_um
        )
        return [
            ModelPolygon(
                points=np.round(np.asarray(piece.points) / precision_um).astype(np.int64),
                layer=layer,
                datatype=datatype,
            )
            for piece in result
        ]

    def offset_polygons(
        self,
        polygons: Sequence[ModelPolygon],
        distance_um: float,
        layer: int,
        datatype: int = 0,
        join: JoinStyle = "miter",
        tolerance: int = 2,
    ) -> list[ModelPolygon]:
        """Dilate or erode model polygons, returning model polygons.

        Lets callers outside the gdstk allowlist do morphological work. Uses
        use_union=True so a set of polygons erodes as one shape rather than
        each in isolation.
        """
        precision_um = self._tech.precision_um
        source = [
            gdstk.Polygon(
                cast(
                    "Sequence[tuple[float, float]]",
                    polygon.points.astype(np.float64) * precision_um,
                )
            )
            for polygon in polygons
        ]
        if not source:
            return []
        result = gdstk.offset(
            source,
            distance_um,
            join=join,
            tolerance=tolerance,
            precision=precision_um,
            use_union=True,
        )
        return [
            ModelPolygon(
                points=np.round(np.asarray(piece.points) / precision_um).astype(np.int64),
                layer=layer,
                datatype=datatype,
            )
            for piece in result
            if len(piece.points) >= 3
        ]

    def new_library(self, name: str) -> gdstk.Library:
        """A library whose database precision matches the design grid."""
        return gdstk.Library(name, unit=USER_UNIT_M, precision=self._tech.precision_m)

    def read_gds(self, path: Path | str) -> gdstk.Library:
        """Read a GDSII stream into a gdstk library."""
        return gdstk.read_gds(path)

    def read_oas(self, path: Path | str) -> gdstk.Library:
        """Read an OASIS stream into a gdstk library."""
        return gdstk.read_oas(path)

    def write_gds(self, library: gdstk.Library, path: Path | str) -> None:
        """Write GDSII with the configured vertex limit and a pinned timestamp."""
        library.write_gds(
            path,
            max_points=self._tech.fracture_vertex_limit,
            timestamp=self._timestamp,
        )

    def write_oas(self, library: gdstk.Library, path: Path | str) -> None:
        """Write OASIS.

        OASIS carries no timestamp, so output is reproducible without one.
        """
        library.write_oas(path)
