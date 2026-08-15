"""The eight selector measurements."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.classify import SELECTOR_KEYS, SiteMeasurement, classify_sites
from masklayout.opc.extract import extract_sites


def _bar(length_nm: int, width_nm: int, angle_deg: float, dy_nm: int = 0) -> Polygon:
    pts = np.array(
        [
            [0, dy_nm],
            [length_nm, dy_nm],
            [length_nm, dy_nm + width_nm],
            [0, dy_nm + width_nm],
        ],
        dtype=np.float64,
    )
    a = math.radians(angle_deg)
    rotated = np.column_stack(
        (
            pts[:, 0] * math.cos(a) - pts[:, 1] * math.sin(a),
            pts[:, 0] * math.sin(a) + pts[:, 1] * math.cos(a),
        )
    )
    return Polygon(points=np.round(rotated).astype(np.int64), layer=10)


def _measure(polygons: list[Polygon]) -> list[SiteMeasurement]:
    tech = TechConfig()
    sites = extract_sites(polygons, tech.precision_um, line_end_ratio=0.5)
    return classify_sites(sites, polygons, tech, max_probe_um=2.0, density_window_um=1.0)


def test_selector_keys_are_exactly_the_documented_vocabulary() -> None:
    assert (
        frozenset(
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
        == SELECTOR_KEYS
    )


@pytest.mark.parametrize("angle_deg", [0.0, 17.0, 37.0, 45.0, 73.0])
def test_width_is_measured_exactly_at_any_angle(angle_deg: float) -> None:
    measurements = _measure([_bar(2000, 100, angle_deg)])
    long_edges = [m for m in measurements if m.site.kind == "edge" and m.edge_length_nm > 1000]
    assert long_edges
    for measurement in long_edges:
        assert measurement.width_nm == pytest.approx(100.0, abs=1.5)


@pytest.mark.parametrize("angle_deg", [0.0, 37.0])
def test_space_to_a_neighbour_is_measured_at_any_angle(angle_deg: float) -> None:
    measurements = _measure([_bar(2000, 100, angle_deg), _bar(2000, 100, angle_deg, dy_nm=160)])
    spaces = [
        m.space_nm
        for m in measurements
        if m.site.kind == "edge" and m.edge_length_nm > 1000 and m.space_nm is not None
    ]
    assert spaces
    assert min(spaces) == pytest.approx(60.0, abs=1.5)


def test_space_is_infinite_for_an_isolated_feature() -> None:
    """Isolated means unbounded space, not absent space.

    A rule saying ``space_nm: {min: 120}`` is the definition of isolated, so
    an isolated edge must satisfy it. Reporting None here would make the
    feature fail the very selector that describes it.
    """
    edges = [m for m in _measure([_bar(2000, 100, 0.0)]) if m.site.kind in ("edge", "line_end")]
    assert edges
    assert all(m.space_nm == math.inf for m in edges)


def test_edge_length_is_reported_in_nanometres() -> None:
    lengths = {round(m.edge_length_nm) for m in _measure([_bar(2000, 100, 0.0)])}
    assert 2000 in lengths
    assert 100 in lengths


def test_local_density_is_the_covered_area_fraction() -> None:
    # One 100 nm bar crossing a 1 um window covers 1.0 x 0.1 um^2 of 1.0 um^2.
    isolated = _measure([_bar(2000, 100, 0.0)])
    assert max(m.local_density for m in isolated) == pytest.approx(0.1, abs=1e-6)

    # Three such bars cover three times as much.
    dense = _measure(
        [
            _bar(2000, 100, 0.0),
            _bar(2000, 100, 0.0, dy_nm=160),
            _bar(2000, 100, 0.0, dy_nm=-160),
        ]
    )
    assert max(m.local_density for m in dense) == pytest.approx(0.3, abs=1e-6)
    assert max(m.local_density for m in dense) > max(m.local_density for m in isolated)


def test_local_density_never_exceeds_one_when_polygons_overlap() -> None:
    # Two identical bars stacked: unioned, not double-counted.
    overlapping = _measure([_bar(2000, 100, 0.0), _bar(2000, 100, 0.0)])
    assert max(m.local_density for m in overlapping) == pytest.approx(0.1, abs=1e-6)


def test_every_measurement_exposes_the_full_vocabulary() -> None:
    for measurement in _measure([_bar(2000, 100, 0.0)]):
        assert set(measurement.as_selector_values()) == SELECTOR_KEYS


def test_corner_measurements_carry_no_width_or_space() -> None:
    """A corner's width and space are None -- not applicable, not unbounded."""
    for measurement in _measure([_bar(2000, 100, 0.0)]):
        if measurement.site.kind.endswith("corner"):
            assert measurement.width_nm is None
            assert measurement.space_nm is None


def test_line_ends_are_measured_like_edges() -> None:
    line_ends = [m for m in _measure([_bar(2000, 100, 0.0)]) if m.site.kind == "line_end"]
    assert len(line_ends) == 2
    for measurement in line_ends:
        # A line end's inward cast runs the length of the bar, not its width.
        assert measurement.width_nm == pytest.approx(2000.0, abs=2.0)
