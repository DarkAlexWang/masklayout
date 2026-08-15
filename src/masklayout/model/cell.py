"""Cells, references, and array repetitions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from masklayout.model.geometry import Label, Polygon


@dataclass(frozen=True)
class RectangularRepetition:
    """A regular grid of placements, as GDSII AREF expresses it."""

    columns: int
    rows: int
    spacing_dbu: tuple[int, int]

    def __post_init__(self) -> None:
        if self.columns < 1:
            raise ValueError(f"columns must be >= 1, got {self.columns}")
        if self.rows < 1:
            raise ValueError(f"rows must be >= 1, got {self.rows}")

    def offsets_dbu(self) -> NDArray[np.int64]:
        """Every placement offset, column-major, including (0, 0)."""
        dx, dy = self.spacing_dbu
        cols = np.arange(self.columns, dtype=np.int64) * dx
        rows = np.arange(self.rows, dtype=np.int64) * dy
        grid_x, grid_y = np.meshgrid(cols, rows, indexing="ij")
        return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.int64)


@dataclass(frozen=True, eq=False)
class ExplicitRepetition:
    """An arbitrary list of placement offsets."""

    offsets_dbu_array: NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.offsets_dbu_array.dtype != np.int64:
            raise TypeError(f"offsets must be int64 DBU, got {self.offsets_dbu_array.dtype}")

    def offsets_dbu(self) -> NDArray[np.int64]:
        return self.offsets_dbu_array


Repetition = RectangularRepetition | ExplicitRepetition


@dataclass(frozen=True)
class Reference:
    """A placement of another cell.

    ``rotation_rad`` and ``magnification`` are floats because they are not
    coordinates; the integer-coordinate rule applies to positions only.
    """

    cell_name: str
    origin_dbu: tuple[int, int]
    rotation_rad: float = 0.0
    magnification: float = 1.0
    x_reflection: bool = False
    repetition: Repetition | None = None


@dataclass
class Cell:
    """A named container of geometry and placements."""

    name: str
    polygons: list[Polygon] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
