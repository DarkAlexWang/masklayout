"""Matching sites against a rule deck."""

from typing import Any

import pytest

from masklayout.opc.deck import load_deck_from_mapping
from masklayout.opc.match import match_sites
from masklayout.opc.sites import Site


def _site(kind: str = "line_end", index: int = 0) -> Site:
    return Site(
        kind=kind,  # type: ignore[arg-type]
        polygon_index=index,
        vertex_index=0,
        midpoint_um=(0.0, 0.0),
        outward_normal_um=(0.0, 1.0),
        edge_length_um=0.1,
        angle_deg=0.0,
        corner_type="none",
        curvature_1_per_um=0.0,
    )


class _FakeMeasurement:
    """A stand-in exposing just what the matcher consumes."""

    def __init__(self, site: Site, **values: Any) -> None:
        self.site = site
        self._values = {
            "site": site.kind,
            "width_nm": None,
            "space_nm": None,
            "edge_length_nm": site.edge_length_um * 1000.0,
            "angle_deg": site.angle_deg,
            "corner_type": site.corner_type,
            "curvature_1_per_um": 0.0,
            "local_density": 0.0,
        }
        self._values.update(values)

    def as_selector_values(self) -> dict[str, Any]:
        return self._values


def _deck(*rules: dict[str, Any]) -> Any:
    return load_deck_from_mapping({"deck": {"id": "d", "version": "1.0.0"}, "rules": list(rules)})


def test_a_matching_rule_produces_a_match() -> None:
    deck = _deck(
        {
            "id": "r1",
            "priority": 10,
            "kind": "hammerhead",
            "when": {"site": "line_end"},
            "apply": {"pcell": "line_end", "params": {"extension_um": 0.028}},
        }
    )
    matches, report = match_sites([_FakeMeasurement(_site())], deck)
    assert len(matches) == 1
    assert matches[0].rule_id == "r1"
    assert matches[0].kind == "hammerhead"
    assert matches[0].pcell == "line_end"
    assert matches[0].params == {"extension_um": 0.028}
    assert report.matched == 1
    assert report.unmatched == 0


def test_first_match_wins_within_a_kind() -> None:
    deck = _deck(
        {
            "id": "specific",
            "priority": 10,
            "kind": "hammerhead",
            "when": {"site": "line_end"},
            "apply": {"pcell": "line_end"},
        },
        {
            "id": "general",
            "priority": 20,
            "kind": "hammerhead",
            "when": {},
            "apply": {"pcell": "line_end"},
        },
    )
    matches, _ = match_sites([_FakeMeasurement(_site())], deck)
    assert [m.rule_id for m in matches] == ["specific"]


def test_different_kinds_compose_on_one_site() -> None:
    deck = _deck(
        {
            "id": "hh",
            "priority": 10,
            "kind": "hammerhead",
            "when": {"site": "line_end"},
            "apply": {"pcell": "line_end"},
        },
        {
            "id": "bias",
            "priority": 20,
            "kind": "edge_bias",
            "when": {"site": "line_end"},
            "apply": {"pcell": "rounded_rect"},
        },
    )
    matches, _ = match_sites([_FakeMeasurement(_site())], deck)
    assert sorted(m.kind for m in matches) == ["edge_bias", "hammerhead"]


def test_a_site_matching_nothing_is_counted_as_unmatched() -> None:
    deck = _deck(
        {
            "id": "corners_only",
            "priority": 10,
            "kind": "serif",
            "when": {"site": "convex_corner"},
            "apply": {"pcell": "rounded_rect"},
        }
    )
    matches, report = match_sites([_FakeMeasurement(_site())], deck)
    assert matches == []
    assert report.considered == 1
    assert report.matched == 0
    assert report.unmatched == 1


def test_an_unmeasurable_attribute_never_satisfies_a_constraint() -> None:
    # width_nm is None at a corner; a width constraint must not match it.
    deck = _deck(
        {
            "id": "needs_width",
            "priority": 10,
            "kind": "serif",
            "when": {"site": "convex_corner", "width_nm": {"min": 1}},
            "apply": {"pcell": "rounded_rect"},
        }
    )
    corner = _FakeMeasurement(_site(kind="convex_corner"), width_nm=None)
    matches, _ = match_sites([corner], deck)
    assert matches == []


def test_infinite_space_satisfies_an_isolation_rule() -> None:
    import math

    deck = _deck(
        {
            "id": "isolated",
            "priority": 10,
            "kind": "hammerhead",
            "when": {"site": "line_end", "space_nm": {"min": 120}},
            "apply": {"pcell": "line_end"},
        }
    )
    isolated = _FakeMeasurement(_site(), space_nm=math.inf)
    matches, _ = match_sites([isolated], deck)
    assert [m.rule_id for m in matches] == ["isolated"]


def test_by_rule_counts_are_reported() -> None:
    deck = _deck(
        {
            "id": "r1",
            "priority": 10,
            "kind": "hammerhead",
            "when": {"site": "line_end"},
            "apply": {"pcell": "line_end"},
        }
    )
    measurements = [_FakeMeasurement(_site(index=i)) for i in range(3)]
    _, report = match_sites(measurements, deck)
    assert report.by_rule == {"r1": 3}


def test_output_order_is_deterministic() -> None:
    deck = _deck(
        {
            "id": "b",
            "priority": 20,
            "kind": "edge_bias",
            "when": {},
            "apply": {"pcell": "p"},
        },
        {
            "id": "a",
            "priority": 10,
            "kind": "hammerhead",
            "when": {},
            "apply": {"pcell": "p"},
        },
    )
    measurements = [_FakeMeasurement(_site(index=i)) for i in range(3)]
    first = [(m.site_id, m.rule_id) for m in match_sites(measurements, deck)[0]]
    second = [(m.site_id, m.rule_id) for m in match_sites(measurements, deck)[0]]
    assert first == second


def test_matches_carry_the_deck_identity_for_provenance() -> None:
    deck = _deck(
        {
            "id": "r1",
            "priority": 10,
            "kind": "hammerhead",
            "when": {},
            "apply": {"pcell": "p"},
        }
    )
    matches, _ = match_sites([_FakeMeasurement(_site())], deck)
    assert matches[0].deck_id == "d"
    assert matches[0].deck_version == "1.0.0"
    assert matches[0].deck_hash == deck.content_hash


def test_an_empty_deck_matches_nothing() -> None:
    matches, report = match_sites([_FakeMeasurement(_site())], _deck())
    assert matches == []
    assert report.unmatched == 1


@pytest.mark.parametrize("kind", ["edge", "line_end", "convex_corner", "concave_corner"])
def test_site_kind_selector_is_exact(kind: str) -> None:
    deck = _deck(
        {
            "id": "only_this",
            "priority": 10,
            "kind": "k",
            "when": {"site": kind},
            "apply": {"pcell": "p"},
        }
    )
    matching = _FakeMeasurement(_site(kind=kind))
    assert len(match_sites([matching], deck)[0]) == 1
    other = "edge" if kind != "edge" else "line_end"
    assert match_sites([_FakeMeasurement(_site(kind=other))], deck)[0] == []
