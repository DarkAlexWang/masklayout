"""Correction generators."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.classify import SiteMeasurement, classify_sites
from masklayout.opc.extract import extract_sites
from masklayout.opc.generate import (
    UnknownCorrectionKindError,
    generate_feature,
    registered_kinds,
)
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


def _line_ends(polygon: Polygon) -> list[SiteMeasurement]:
    sites = extract_sites([polygon], TECH.precision_um, line_end_ratio=0.5)
    measured = classify_sites(sites, [polygon], TECH, max_probe_um=2.0, density_window_um=1.0)
    return [m for m in measured if m.site.kind == "line_end"]


def _match(kind: str, pcell: str, **params: object) -> Match:
    return Match(
        site_id="0#1:line_end",
        rule_id="r1",
        kind=kind,
        pcell=pcell,
        params=dict(params),
        deck_id="d",
        deck_version="1.0.0",
        deck_hash="hash",
    )


def test_the_expected_kinds_are_registered() -> None:
    for kind in ("hammerhead", "line_end_extension"):
        assert kind in registered_kinds()


def test_an_unknown_kind_lists_the_known_ones() -> None:
    measurement = _line_ends(_bar())[0]
    with pytest.raises(UnknownCorrectionKindError) as excinfo:
        generate_feature(measurement, _match("no_such_kind", "line_end"), TECH)
    assert "hammerhead" in str(excinfo.value)


def test_a_hammerhead_is_wider_than_its_line() -> None:
    measurement = _line_ends(_bar(width_nm=100))[0]
    feature = generate_feature(
        measurement,
        _match("hammerhead", "line_end", extension_um=0.028, head_width_ratio=1.4),
        TECH,
    )
    assert feature is not None
    _, y0, _, y1 = feature.polygons[0].bounds_dbu
    # The bar runs along x, so the head's extent across y is its width.
    assert (y1 - y0) == pytest.approx(140, abs=2)


def test_a_line_end_extension_is_no_wider_than_its_line() -> None:
    measurement = _line_ends(_bar(width_nm=100))[0]
    feature = generate_feature(
        measurement, _match("line_end_extension", "line_end", extension_um=0.028), TECH
    )
    assert feature is not None
    _, y0, _, y1 = feature.polygons[0].bounds_dbu
    assert (y1 - y0) == pytest.approx(100, abs=2)


def test_the_correction_extends_outward_never_inward() -> None:
    bar = _bar(length_nm=2000, width_nm=100)
    for measurement in _line_ends(bar):
        feature = generate_feature(
            measurement,
            _match("hammerhead", "line_end", extension_um=0.028, head_width_ratio=1.4),
            TECH,
        )
        assert feature is not None
        hx0, _, hx1, _ = feature.polygons[0].bounds_dbu
        bx0, _, bx1, _ = bar.bounds_dbu
        # The head must stick out past one end of the bar, not sit inside it.
        assert hx0 < bx0 or hx1 > bx1


@pytest.mark.parametrize("angle_deg", [0.0, 37.0, 90.0])
def test_generation_works_at_any_angle(angle_deg: float) -> None:
    measurements = _line_ends(_bar(angle_deg=angle_deg))
    assert len(measurements) == 2
    for measurement in measurements:
        feature = generate_feature(
            measurement,
            _match("hammerhead", "line_end", extension_um=0.028, head_width_ratio=1.4),
            TECH,
        )
        assert feature is not None
        assert feature.polygons[0].points.dtype == np.int64


def test_a_missing_required_parameter_fails_naming_it() -> None:
    measurement = _line_ends(_bar())[0]
    with pytest.raises(ValueError, match="extension_um"):
        generate_feature(measurement, _match("hammerhead", "line_end"), TECH)


def test_a_rule_may_not_override_placement() -> None:
    from masklayout.opc.placement import PlacementOverrideError

    measurement = _line_ends(_bar())[0]
    with pytest.raises(PlacementOverrideError, match="centre_um"):
        generate_feature(
            measurement,
            _match("hammerhead", "line_end", extension_um=0.028, centre_um=(0.0, 0.0)),
            TECH,
        )


def test_the_feature_carries_full_provenance() -> None:
    measurement = _line_ends(_bar())[0]
    feature = generate_feature(
        measurement, _match("hammerhead", "line_end", extension_um=0.028), TECH
    )
    assert feature is not None
    provenance = feature.provenance()
    assert provenance["rule_id"] == "r1"
    assert provenance["deck_hash"] == "hash"
    assert provenance["source_site_id"] == measurement.site.site_id
    assert provenance["kind"] == "hammerhead"
    assert provenance["parameters"]["extension_um"] == 0.028


def test_a_zero_extension_produces_no_feature() -> None:
    """Nothing to add is not the same as a degenerate polygon."""
    measurement = _line_ends(_bar())[0]
    assert (
        generate_feature(measurement, _match("hammerhead", "line_end", extension_um=0.0), TECH)
        is None
    )


def test_features_are_additive_by_default() -> None:
    measurement = _line_ends(_bar())[0]
    feature = generate_feature(
        measurement, _match("hammerhead", "line_end", extension_um=0.028), TECH
    )
    assert feature is not None
    assert feature.polarity == "add"
