"""Integer-coordinate geometry types.

All coordinates are int64 in design database units (DBU). No float
coordinate exists at this layer; see the design document, section
"Units and coordinate model".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _check_integer_points(points: NDArray[np.int64]) -> None:
    if points.dtype != np.int64:
        raise TypeError(f"coordinates must be int64 design database units, got {points.dtype}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got shape {points.shape}")


@dataclass(frozen=True, eq=False)
class Polygon:
    """A closed polygon in design database units."""

    points: NDArray[np.int64]
    layer: int
    datatype: int = 0

    def __post_init__(self) -> None:
        _check_integer_points(self.points)

    @property
    def vertex_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def bounds_dbu(self) -> tuple[int, int, int, int]:
        """(min_x, min_y, max_x, max_y)."""
        low = self.points.min(axis=0)
        high = self.points.max(axis=0)
        return (int(low[0]), int(low[1]), int(high[0]), int(high[1]))


@dataclass(frozen=True)
class Label:
    """A text annotation. Carried through I/O but never used as geometry."""

    text: str
    origin_dbu: tuple[int, int]
    layer: int
    datatype: int = 0
