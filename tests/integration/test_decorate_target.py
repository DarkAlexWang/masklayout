"""M4 acceptance: extract, classify, and match a real target against the shipped deck."""

import math
from pathlib import Path
from typing import Any

import numpy as np

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.classify import classify_sites
from masklayout.opc.deck import load_deck
from masklayout.opc.extract import extract_sites
from masklayout.opc.match import Match, match_sites

_DECK = Path(__file__).parents[2] / "src" / "masklayout" / "decks" / "generic_hammerhead_v1.yaml"


def _bar(x0: int, y0: int, length_nm: int, width_nm: int) -> Polygon:
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


def _run(polygons: list[Polygon]) -> tuple[list[Match], dict[str, Any]]:
    tech = TechConfig()
    sites = extract_sites(polygons, tech.precision_um, line_end_ratio=0.5)
    measurements = classify_sites(sites, polygons, tech, max_probe_um=2.0, density_window_um=1.0)
    deck = load_deck(_DECK)
    matches, report = match_sites(measurements, deck)
    return matches, {"report": report, "measurements": measurements}


def test_an_isolated_line_end_takes_the_isolated_rule() -> None:
    # One bar, nothing nearby: its line ends see unbounded space.
    matches, _ = _run([_bar(0, 0, 2000, 100)])

    hammerheads = [m for m in matches if m.kind == "hammerhead"]
    assert hammerheads, "the isolated line ends should have matched a hammerhead rule"
    assert {m.rule_id for m in hammerheads} == {"hh_isolated_line_end"}
    assert all(m.params["extension_um"] == 0.028 for m in hammerheads)


def test_a_dense_line_end_takes_the_dense_rule() -> None:
    # Three bars end-to-end with 60 nm gaps: the inner line ends are dense.
    polygons = [
        _bar(0, 0, 2000, 100),
        _bar(2060, 0, 2000, 100),
        _bar(4120, 0, 2000, 100),
    ]
    matches, _ = _run(polygons)

    dense = [m for m in matches if m.rule_id == "hh_dense_line_end"]
    assert dense, "line ends facing a 60 nm gap should take the dense rule"
    assert all(m.params["extension_um"] == 0.014 for m in dense)


def test_isolation_and_density_select_different_rules_in_one_run() -> None:
    """The whole point of the vocabulary: context changes the correction."""
    polygons = [
        _bar(0, 0, 2000, 100),  # left bar: outer end isolated, inner end dense
        _bar(2060, 0, 2000, 100),
    ]
    matches, _ = _run(polygons)
    rule_ids = {m.rule_id for m in matches if m.kind == "hammerhead"}
    assert "hh_isolated_line_end" in rule_ids
    assert "hh_dense_line_end" in rule_ids


def test_every_match_carries_deck_provenance() -> None:
    matches, _ = _run([_bar(0, 0, 2000, 100)])
    deck = load_deck(_DECK)
    assert matches
    for match in matches:
        assert match.deck_id == "generic_hammerhead_v1"
        assert match.deck_version == "1.0.0"
        assert match.deck_hash == deck.content_hash


def test_no_site_takes_two_rules_of_the_same_kind() -> None:
    polygons = [_bar(0, 0, 2000, 100), _bar(2060, 0, 2000, 100)]
    matches, _ = _run(polygons)
    seen: set[tuple[str, str]] = set()
    for match in matches:
        key = (match.site_id, match.kind)
        assert key not in seen, f"site {match.site_id} took two {match.kind} rules"
        seen.add(key)


def test_matching_is_reproducible() -> None:
    polygons = [_bar(0, 0, 2000, 100), _bar(2060, 0, 2000, 100)]
    first = [(m.site_id, m.rule_id) for m in _run(polygons)[0]]
    second = [(m.site_id, m.rule_id) for m in _run(polygons)[0]]
    assert first == second


def test_isolated_bar_reports_infinite_space_at_its_line_ends() -> None:
    _, extra = _run([_bar(0, 0, 2000, 100)])
    line_ends = [m for m in extra["measurements"] if m.site.kind == "line_end"]
    assert line_ends
    assert all(m.space_nm == math.inf for m in line_ends)
