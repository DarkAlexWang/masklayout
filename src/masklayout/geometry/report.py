"""Reports describing what compilation actually produced."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TessellationReport:
    """Measured deviation of a compiled polygon from its source polyline.

    The two error terms are kept apart deliberately. Tessellation error comes
    from replacing a curve with chords; grid error comes from snapping each
    vertex onto the design grid. They compose, so the acceptance budget is
    their sum and neither may hide inside the other.
    """

    vertex_count: int
    tessellation_error_nm: float
    grid_error_nm: float
    budget_nm: float

    @property
    def total_error_nm(self) -> float:
        return self.tessellation_error_nm + self.grid_error_nm

    @property
    def within_budget(self) -> bool:
        return self.total_error_nm <= self.budget_nm
