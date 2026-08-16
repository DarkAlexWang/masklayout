"""SRAF placement."""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.classify import SiteMeasurement, classify_sites
from masklayout.opc.extract import extract_sites
from masklayout.opc.feature import Feature
from masklayout.opc.generate import generate_feature, registered_kinds
from masklayout.opc.match import Match

TECH = TechConfig()


def _bar(length_nm: int = 2000, width_nm: int = 100, angle_deg: float = 0.0) -> Polygon:
    pts = np.array([[0, 0], [length_nm, 0], [length_nm, width_nm], [0, width_nm]], dtype=np.float64)
    a = math.radians(angle_deg)
    rotated = np.column_stack(
        (
            pts[:, 0] * math.cos(a) - pts[:, 1] * math.sin(a),
            pts[:, 0] * math.sin(a) + pts[:, 1] * math.cos(a),
        )
    )
    return Polygon(points=np.round(rotated).astype(np.int64), layer=10)


def _long_edges(polygon: Polygon) -> list[SiteMeasurement]:
    sites = extract_sites([polygon], TECH.precision_um, line_end_ratio=0.5)
    measured = classify_sites(sites, [polygon], TECH, max_probe_um=2.0, density_window_um=1.0)
    return [m for m in measured if m.site.kind == "edge" and m.edge_length_nm > 1000]


def _match(**params: object) -> Match:
    return Match(
        site_id="0#0:edge",
        rule_id="sraf1",
        kind="sraf_bar",
        pcell="line_end",
        params=dict(params),
        deck_id="d",
        deck_version="1.0.0",
        deck_hash="hash",
    )


def test_sraf_bar_is_registered() -> None:
    assert "sraf_bar" in registered_kinds()


def test_an_sraf_is_an_assist_feature_not_a_correction() -> None:
    measurement = _long_edges(_bar())[0]
    feature = generate_feature(measurement, _match(distance_um=0.08, width_um=0.03), TECH)
    assert feature is not None
    assert feature.polarity == "assist"


def test_the_bar_sits_outside_the_target_at_the_requested_distance() -> None:
    bar = _bar()
    shape = ShapelyPolygon(bar.points.astype(np.float64) * TECH.precision_um)
    measurement = _long_edges(bar)[0]
    feature = generate_feature(measurement, _match(distance_um=0.08, width_um=0.03), TECH)
    assert feature is not None

    sraf = ShapelyPolygon(feature.polygons[0].points.astype(np.float64) * TECH.precision_um)
    assert not sraf.intersects(shape), "an SRAF must not touch the target"
    # distance_um is measured edge-to-near-side, so the gap is that value.
    assert shape.distance(sraf) == pytest.approx(0.08, abs=0.002)


@pytest.mark.parametrize("angle_deg", [0.0, 37.0, 90.0])
def test_the_bar_runs_parallel_to_its_source_edge(angle_deg: float) -> None:
    bar = _bar(angle_deg=angle_deg)
    shape = ShapelyPolygon(bar.points.astype(np.float64) * TECH.precision_um)
    for measurement in _long_edges(bar):
        feature = generate_feature(measurement, _match(distance_um=0.08, width_um=0.03), TECH)
        assert feature is not None
        sraf = ShapelyPolygon(feature.polygons[0].points.astype(np.float64) * TECH.precision_um)
        # Parallel and offset: every point of the bar is roughly the same
        # distance from the target edge it was placed against.
        assert not sraf.intersects(shape)
        assert shape.distance(sraf) == pytest.approx(0.08, abs=0.003)


def test_length_ratio_scales_the_bar() -> None:
    measurement = _long_edges(_bar())[0]
    full = generate_feature(measurement, _match(distance_um=0.08, width_um=0.03), TECH)
    half = generate_feature(
        measurement, _match(distance_um=0.08, width_um=0.03, length_ratio=0.5), TECH
    )
    assert full is not None and half is not None

    def longest(feature: Feature) -> int:
        x0, y0, x1, y1 = feature.polygons[0].bounds_dbu
        return max(x1 - x0, y1 - y0)

    assert longest(half) == pytest.approx(longest(full) / 2, abs=2)


def test_a_zero_width_or_distance_produces_no_feature() -> None:
    measurement = _long_edges(_bar())[0]
    assert generate_feature(measurement, _match(distance_um=0.0, width_um=0.03), TECH) is None
    assert generate_feature(measurement, _match(distance_um=0.08, width_um=0.0), TECH) is None


def test_missing_parameters_fail_naming_them() -> None:
    measurement = _long_edges(_bar())[0]
    with pytest.raises(ValueError, match="distance_um"):
        generate_feature(measurement, _match(width_um=0.03), TECH)
    with pytest.raises(ValueError, match="width_um"):
        generate_feature(measurement, _match(distance_um=0.08), TECH)


def test_the_sraf_carries_provenance() -> None:
    measurement = _long_edges(_bar())[0]
    feature = generate_feature(measurement, _match(distance_um=0.08, width_um=0.03), TECH)
    assert feature is not None
    assert feature.provenance()["rule_id"] == "sraf1"
    assert feature.provenance()["polarity"] == "assist"
