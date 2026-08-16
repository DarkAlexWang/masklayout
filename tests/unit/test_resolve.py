"""Keep-out and collision resolution."""

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.feature import Feature
from masklayout.opc.resolve import resolve_collisions

TECH = TechConfig()


def _rect(x0: int, y0: int, x1: int, y1: int) -> Polygon:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int64)
    return Polygon(points=pts, layer=12)


def _feature(fid: str, polygon: Polygon, polarity: str = "assist") -> Feature:
    return Feature(
        id=fid,
        kind="sraf_bar" if polarity == "assist" else "hammerhead",
        polygons=[polygon],
        source_site_id="0#0:edge",
        rule_id="r",
        deck_id="d",
        deck_version="1.0.0",
        deck_hash="h",
        polarity=polarity,  # type: ignore[arg-type]
    )


#: A 2000 x 100 nm bar at the origin.
TARGET = [_rect(0, 0, 2000, 100)]


def test_a_comfortably_spaced_assist_survives() -> None:
    far = _feature("a", _rect(0, 200, 2000, 230))  # 100 nm clear of the target
    kept, rejected = resolve_collisions([far], TARGET, TECH, target_keepout_um=0.02)
    assert [f.id for f in kept] == ["a"]
    assert rejected == []


def test_an_assist_inside_the_target_keepout_is_rejected() -> None:
    close = _feature("a", _rect(0, 105, 2000, 135))  # only 5 nm clear
    kept, rejected = resolve_collisions([close], TARGET, TECH, target_keepout_um=0.02)
    assert kept == []
    assert len(rejected) == 1
    assert rejected[0].reason == "target_keepout"
    assert rejected[0].feature_id == "a"


def test_the_rejection_reports_the_measured_distance() -> None:
    close = _feature("a", _rect(0, 105, 2000, 135))
    _, rejected = resolve_collisions([close], TARGET, TECH, target_keepout_um=0.02)
    assert "5.0 nm" in rejected[0].detail
    assert "20.0 nm" in rejected[0].detail


def test_two_assists_too_close_leave_exactly_one() -> None:
    first = _feature("a", _rect(0, 200, 2000, 230))
    second = _feature("b", _rect(0, 235, 2000, 265))  # only 5 nm from the first
    kept, rejected = resolve_collisions(
        [first, second], TARGET, TECH, target_keepout_um=0.02, sraf_keepout_um=0.02
    )
    assert len(kept) == 1
    assert len(rejected) == 1
    assert rejected[0].reason == "sraf_keepout"


def test_which_assist_survives_is_stable() -> None:
    first = _feature("a", _rect(0, 200, 2000, 230))
    second = _feature("b", _rect(0, 235, 2000, 265))
    # Feed them in both orders; resolution sorts by id, so "a" always wins.
    for order in ([first, second], [second, first]):
        kept, _ = resolve_collisions(order, TARGET, TECH, sraf_keepout_um=0.02)
        assert [f.id for f in kept] == ["a"]


def test_well_separated_assists_all_survive() -> None:
    features = [
        _feature("a", _rect(0, 200, 2000, 230)),
        _feature("b", _rect(0, 300, 2000, 330)),
        _feature("c", _rect(0, 400, 2000, 430)),
    ]
    kept, rejected = resolve_collisions(features, TARGET, TECH, sraf_keepout_um=0.02)
    assert sorted(f.id for f in kept) == ["a", "b", "c"]
    assert rejected == []


def test_corrections_are_never_rejected() -> None:
    """A hammerhead is supposed to touch the target."""
    touching = _feature("hh", _rect(2000, 0, 2030, 100), polarity="add")
    kept, rejected = resolve_collisions([touching], TARGET, TECH, target_keepout_um=0.02)
    assert [f.id for f in kept] == ["hh"]
    assert rejected == []


def test_an_empty_target_rejects_nothing_on_target_grounds() -> None:
    assist = _feature("a", _rect(0, 200, 2000, 230))
    kept, rejected = resolve_collisions([assist], [], TECH)
    assert [f.id for f in kept] == ["a"]
    assert rejected == []


def test_a_rejection_keeps_its_geometry_for_marker_output() -> None:
    close = _feature("a", _rect(0, 105, 2000, 135))
    _, rejected = resolve_collisions([close], TARGET, TECH, target_keepout_um=0.02)
    assert rejected[0].polygons
    assert rejected[0].as_record()["reason"] == "target_keepout"


def test_keepout_thresholds_are_respected_exactly() -> None:
    # Exactly at the keep-out distance is acceptable; a nanometre less is not.
    at_limit = _feature("a", _rect(0, 120, 2000, 150))  # exactly 20 nm clear
    inside = _feature("a", _rect(0, 119, 2000, 149))  # 19 nm clear
    kept, _ = resolve_collisions([at_limit], TARGET, TECH, target_keepout_um=0.02)
    assert len(kept) == 1
    kept, rejected = resolve_collisions([inside], TARGET, TECH, target_keepout_um=0.02)
    assert kept == []
    assert rejected[0].reason == "target_keepout"


def test_resolution_is_reproducible() -> None:
    features = [
        _feature("a", _rect(0, 200, 2000, 230)),
        _feature("b", _rect(0, 235, 2000, 265)),
        _feature("c", _rect(0, 400, 2000, 430)),
    ]
    first = resolve_collisions(features, TARGET, TECH, sraf_keepout_um=0.02)
    second = resolve_collisions(features, TARGET, TECH, sraf_keepout_um=0.02)
    assert [f.id for f in first[0]] == [f.id for f in second[0]]
    assert [r.feature_id for r in first[1]] == [r.feature_id for r in second[1]]


def test_a_feature_with_no_geometry_is_skipped_quietly() -> None:
    empty = Feature(
        id="empty",
        kind="sraf_bar",
        polygons=[],
        source_site_id="s",
        rule_id="r",
        deck_id="d",
        deck_version="1",
        deck_hash="h",
        polarity="assist",
    )
    kept, rejected = resolve_collisions([empty], TARGET, TECH)
    assert kept == []
    assert rejected == []


def test_pytest_approx_sanity_on_distance_reporting() -> None:
    close = _feature("a", _rect(0, 110, 2000, 140))  # 10 nm clear
    _, rejected = resolve_collisions([close], TARGET, TECH, target_keepout_um=0.02)
    assert float(rejected[0].detail.split()[0]) == pytest.approx(10.0, abs=0.5)
