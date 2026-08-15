"""Extracted sites: the things a rule can select."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SiteKind = Literal["edge", "line_end", "convex_corner", "concave_corner"]
CornerType = Literal["convex", "concave", "none"]


@dataclass(frozen=True)
class Site:
    """One location on target geometry, with its measured geometry.

    Positional attributes are float micrometres because a site is a
    measurement about the model, not geometry stored in it.
    """

    kind: SiteKind
    polygon_index: int
    vertex_index: int
    midpoint_um: tuple[float, float]
    outward_normal_um: tuple[float, float]
    edge_length_um: float
    angle_deg: float
    corner_type: CornerType
    curvature_1_per_um: float

    @property
    def site_id(self) -> str:
        """Stable identifier, assigned after normalization (design section 7)."""
        return f"{self.polygon_index}#{self.vertex_index}:{self.kind}"
