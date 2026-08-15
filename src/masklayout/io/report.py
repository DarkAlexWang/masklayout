"""Structured reports describing what a read actually did."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadReport:
    """What was found, converted, and preserved when reading a stream.

    ``paths_converted`` exists so that path-to-polygon conversion is always
    visible: geometry may change representation on import, but never silently.
    """

    source: str
    cell_count: int
    polygon_count: int
    label_count: int
    reference_count: int
    paths_converted: int
    file_precision_m: float
    top_cells: tuple[str, ...]

    def summary(self) -> str:
        parts = [
            f"{self.cell_count} cells",
            f"{self.polygon_count} polygons",
            f"{self.reference_count} references",
            f"{self.label_count} labels",
        ]
        if self.paths_converted:
            parts.append(f"{self.paths_converted} paths converted to polygons")
        return f"{self.source}: " + ", ".join(parts)
