"""Placement derived from a site, and rejection of rule overrides."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.extract import extract_sites
from masklayout.opc.placement import (
    PLACEMENT_KEYS,
    PlacementOverrideError,
    merge_params,
    placement_for,
)
from masklayout.opc.sites import Site


def _site(normal: tuple[float, float], midpoint: tuple[float, float] = (1.0, 2.0)) -> Site:
    return Site(
        kind="line_end",
        polygon_index=0,
        vertex_index=1,
        midpoint_um=midpoint,
        outward_normal_um=normal,
        edge_length_um=0.1,
        angle_deg=0.0,
        corner_type="none",
        curvature_1_per_um=0.0,
    )


def test_placement_angle_follows_the_outward_normal() -> None:
    assert placement_for(_site((1.0, 0.0)))["angle_rad"] == pytest.approx(0.0)
    assert placement_for(_site((0.0, 1.0)))["angle_rad"] == pytest.approx(math.pi / 2)
    assert placement_for(_site((-1.0, 0.0)))["angle_rad"] == pytest.approx(math.pi)


def test_placement_centre_is_the_site_midpoint() -> None:
    assert placement_for(_site((1.0, 0.0), midpoint=(3.5, -1.25)))["centre_um"] == (3.5, -1.25)


def test_placement_supplies_exactly_the_placement_keys() -> None:
    assert set(placement_for(_site((1.0, 0.0)))) == PLACEMENT_KEYS


def test_merge_params_combines_placement_and_shape() -> None:
    merged = merge_params(placement_for(_site((1.0, 0.0))), {"extension_um": 0.028})
    assert merged["extension_um"] == 0.028
    assert merged["angle_rad"] == pytest.approx(0.0)


@pytest.mark.parametrize("key", ["centre_um", "angle_rad"])
def test_a_rule_may_not_override_placement(key: str) -> None:
    """Where a correction goes is a geometric fact, not an authoring choice."""
    with pytest.raises(PlacementOverrideError) as excinfo:
        merge_params(placement_for(_site((1.0, 0.0))), {key: 0.0, "extension_um": 0.028})
    assert key in str(excinfo.value)


def test_placement_is_derived_correctly_on_rotated_geometry() -> None:
    """A 37 degree bar's line ends must point along the bar, not along an axis."""
    tech = TechConfig()
    angle = math.radians(37.0)
    pts = np.array([[0, 0], [2000, 0], [2000, 100], [0, 100]], dtype=np.float64)
    rotated = np.column_stack(
        (
            pts[:, 0] * math.cos(angle) - pts[:, 1] * math.sin(angle),
            pts[:, 0] * math.sin(angle) + pts[:, 1] * math.cos(angle),
        )
    )
    bar = Polygon(points=np.round(rotated).astype(np.int64), layer=10)

    line_ends = [s for s in extract_sites([bar], tech.precision_um, 0.5) if s.kind == "line_end"]
    assert len(line_ends) == 2

    angles = sorted(math.degrees(placement_for(s)["angle_rad"]) % 360.0 for s in line_ends)
    # The two ends point in opposite directions along the bar: 37 and 217.
    assert angles[0] == pytest.approx(37.0, abs=1.0)
    assert angles[1] == pytest.approx(217.0, abs=1.0)
