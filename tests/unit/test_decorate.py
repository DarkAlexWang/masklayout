"""Decorating a target: merge, overlays, immutability."""

import copy
from typing import Any

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.deck import RuleDeck, load_deck_from_mapping
from masklayout.opc.decorate import decorate

TECH = TechConfig()


def _bar(x0: int = 0, y0: int = 0, length_nm: int = 2000, width_nm: int = 100) -> Polygon:
    pts = np.array(
        [
            [x0, y0],
            [x0 + length_nm, y0],
            [x0 + length_nm, y0 + width_nm],
            [x0, y0 + width_nm],
        ],
        dtype=np.int64,
    )
    return Polygon(points=pts, layer=10)


def _area(polygons: list[Polygon]) -> float:
    from masklayout.geometry.normalize import signed_area

    return sum(abs(signed_area(p.points)) for p in polygons)


def _deck(*rules: dict[str, Any]) -> RuleDeck:
    return load_deck_from_mapping({"deck": {"id": "d", "version": "1.0.0"}, "rules": list(rules)})


_HAMMERHEAD: dict[str, Any] = {
    "id": "hh",
    "priority": 10,
    "kind": "hammerhead",
    "when": {"site": "line_end"},
    "apply": {"pcell": "line_end", "params": {"extension_um": 0.03, "head_width_ratio": 1.4}},
}
_SHRINK: dict[str, Any] = {
    "id": "shrink",
    "priority": 10,
    "kind": "edge_bias",
    "when": {"site": "edge"},
    "apply": {"pcell": "line_end", "params": {"bias_um": -0.005}},
}


def test_the_target_is_never_modified() -> None:
    target = [_bar()]
    before = copy.deepcopy(target[0].points)
    decorate(target, _deck(_HAMMERHEAD), TECH)
    assert np.array_equal(target[0].points, before)
    assert len(target) == 1


def test_additive_corrections_grow_the_area() -> None:
    target = [_bar()]
    result = decorate(target, _deck(_HAMMERHEAD), TECH)
    assert _area(result.post_opc) > _area(target)


def test_additive_corrections_populate_only_overlay_add() -> None:
    result = decorate([_bar()], _deck(_HAMMERHEAD), TECH)
    assert result.overlay_add
    assert result.overlay_remove == []


def test_subtractive_corrections_populate_only_overlay_remove() -> None:
    result = decorate([_bar()], _deck(_SHRINK), TECH)
    assert result.overlay_remove
    assert result.overlay_add == []
    assert _area(result.post_opc) < _area([_bar()])


def test_post_opc_lands_on_the_configured_layer() -> None:
    result = decorate([_bar()], _deck(_HAMMERHEAD), TECH)
    assert all(p.layer == 11 for p in result.post_opc)
    assert all(p.layer == 202 for p in result.overlay_add)


def test_post_opc_is_grid_aligned() -> None:
    result = decorate([_bar()], _deck(_HAMMERHEAD), TECH)
    for polygon in result.post_opc:
        assert polygon.points.dtype == np.int64


def test_features_carry_provenance() -> None:
    result = decorate([_bar()], _deck(_HAMMERHEAD), TECH)
    assert result.features
    for feature in result.features:
        assert feature.rule_id == "hh"
        assert feature.deck_id == "d"
        assert feature.provenance()["parameters"]["extension_um"] == 0.03


def test_the_report_counts_what_happened() -> None:
    result = decorate([_bar()], _deck(_HAMMERHEAD), TECH)
    assert result.report.sites > 0
    assert result.report.features_generated == 2  # two line ends
    assert result.report.by_kind == {"hammerhead": 2}
    assert "hammerhead" in result.report.summary()


def test_decorating_twice_is_identical() -> None:
    first = decorate([_bar()], _deck(_HAMMERHEAD), TECH)
    second = decorate([_bar()], _deck(_HAMMERHEAD), TECH)
    assert len(first.post_opc) == len(second.post_opc)
    for a, b in zip(first.post_opc, second.post_opc, strict=True):
        assert np.array_equal(a.points, b.points)


def test_an_empty_deck_leaves_the_geometry_unchanged() -> None:
    target = [_bar()]
    result = decorate(target, _deck(), TECH)
    assert result.features == []
    assert result.overlay_add == []
    assert result.overlay_remove == []
    assert _area(result.post_opc) == pytest.approx(_area(target), rel=1e-9)


def test_an_unimplemented_kind_is_loud_by_default() -> None:
    from masklayout.opc.generate import UnknownCorrectionKindError

    jog: dict[str, Any] = {
        "id": "jog",
        "priority": 10,
        "kind": "jog",
        "when": {"site": "edge"},
        "apply": {"pcell": "line_end", "params": {}},
    }
    with pytest.raises(UnknownCorrectionKindError, match="jog"):
        decorate([_bar()], _deck(jog), TECH)


def test_an_unimplemented_kind_can_be_skipped_and_is_reported() -> None:
    jog: dict[str, Any] = {
        "id": "jog",
        "priority": 10,
        "kind": "jog",
        "when": {"site": "edge"},
        "apply": {"pcell": "line_end", "params": {}},
    }
    result = decorate([_bar()], _deck(jog), TECH, skip_unknown_kinds=True)
    assert result.report.unknown_kinds == ("jog",)
    assert result.features == []
    assert "jog" in result.report.summary()
