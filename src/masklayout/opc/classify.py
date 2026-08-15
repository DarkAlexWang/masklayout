"""Measure the closed selector vocabulary at each site.

This module is the contract described in the design document, section
"Compile pipeline": whatever is measured here is exactly what a rule can
select on, and nothing else. Extending the vocabulary later is additive;
a rule can never reach past it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from masklayout.config import TechConfig
from masklayout.geometry.index import SpatialIndex
from masklayout.model.geometry import Polygon
from masklayout.opc.sites import Site

#: The closed selector vocabulary. A deck naming anything outside this set
#: fails at load.
SELECTOR_KEYS = frozenset(
    {
        "site",
        "width_nm",
        "space_nm",
        "edge_length_nm",
        "angle_deg",
        "corner_type",
        "curvature_1_per_um",
        "local_density",
    }
)

#: Site kinds that lie along an edge, and so have a width and a space.
_EDGE_KINDS = ("edge", "line_end")


@dataclass(frozen=True)
class SiteMeasurement:
    """A site with the full selector vocabulary measured at it."""

    site: Site
    width_nm: float | None
    space_nm: float | None
    edge_length_nm: float
    angle_deg: float
    corner_type: str
    curvature_1_per_um: float
    local_density: float

    def as_selector_values(self) -> dict[str, Any]:
        """Exactly the selector vocabulary, for matching against a rule."""
        return {
            "site": self.site.kind,
            "width_nm": self.width_nm,
            "space_nm": self.space_nm,
            "edge_length_nm": self.edge_length_nm,
            "angle_deg": self.angle_deg,
            "corner_type": self.corner_type,
            "curvature_1_per_um": self.curvature_1_per_um,
            "local_density": self.local_density,
        }


def classify_sites(
    sites: Sequence[Site],
    polygons: Sequence[Polygon],
    tech: TechConfig,
    max_probe_um: float = 2.0,
    density_window_um: float = 1.0,
) -> list[SiteMeasurement]:
    """Measure the selector vocabulary at every site.

    Width casts a ray inward from the edge midpoint to the far side of the
    same polygon; space casts outward to the nearest other polygon. Both are
    exact at any angle, which is why no Manhattan special case exists here.

    ``None`` means "not measurable at this site" -- a corner has no width, and
    an isolated feature has no space. A selector constraining a None value
    never matches, rather than matching vacuously.

    ``local_density`` is pattern density: covered area over window area. An
    earlier draft used a count of neighbours over the total polygon count,
    which is degenerate -- it reports 1.0 for any layout where every polygon
    is near the site, regardless of how much area they cover.
    """
    index = SpatialIndex(polygons, tech.precision_um)
    measurements: list[SiteMeasurement] = []

    for site in sites:
        on_edge = site.kind in _EDGE_KINDS
        width_um = None
        space_um = None
        if on_edge:
            inward = (-site.outward_normal_um[0], -site.outward_normal_um[1])
            width_um = index.nearest_distance_um(
                site.midpoint_um, inward, max_probe_um, exclude=None
            )
            space_um = index.nearest_distance_um(
                site.midpoint_um,
                site.outward_normal_um,
                max_probe_um,
                exclude=site.polygon_index,
            )

        half = density_window_um / 2.0
        density = index.covered_fraction(
            site.midpoint_um[0] - half,
            site.midpoint_um[1] - half,
            site.midpoint_um[0] + half,
            site.midpoint_um[1] + half,
        )

        measurements.append(
            SiteMeasurement(
                site=site,
                width_nm=None if width_um is None else width_um * 1000.0,
                space_nm=None if space_um is None else space_um * 1000.0,
                edge_length_nm=site.edge_length_um * 1000.0,
                angle_deg=site.angle_deg,
                corner_type=site.corner_type,
                curvature_1_per_um=site.curvature_1_per_um,
                local_density=density,
            )
        )

    return measurements
