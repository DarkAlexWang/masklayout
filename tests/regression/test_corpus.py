"""The golden regression corpus.

Regenerate with:
    MASKLAYOUT_REGENERATE_GOLDENS=1 uv run pytest tests/regression/
"""

from pathlib import Path

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.normalize import is_simple
from masklayout.opc.deck import RuleDeck, load_deck
from tests.regression.golden import (
    REGENERATE_ENV,
    compare,
    golden_path,
    load_golden,
    regenerating,
    summarise,
    write_golden,
)
from tests.regression.patterns import ALL_PATTERNS

DECK_PATH = (
    Path(__file__).parents[2] / "src" / "masklayout" / "decks" / "generic_hammerhead_v1.yaml"
)

#: The pattern classes the design's corpus section requires.
REQUIRED_COVERAGE = {
    "isolated_line",
    "dense_lines",
    "semi_isolated_lines",
    "line_ends_manhattan",
    "line_ends_45",
    "line_ends_arbitrary",
    "convex_and_concave_corners",
    "acute_corner",
    "narrow_neck",
    "contact_array",
    "curvilinear_wire",
    "pathological_sliver",
}


def _deck() -> RuleDeck:
    return load_deck(DECK_PATH)


def test_the_corpus_covers_every_required_pattern_class() -> None:
    """Adding a class to the design without a fixture must fail here."""
    assert set(ALL_PATTERNS) == REQUIRED_COVERAGE


@pytest.mark.parametrize("name", sorted(ALL_PATTERNS))
def test_every_pattern_produces_valid_geometry(name: str) -> None:
    tech = TechConfig()
    polygons = ALL_PATTERNS[name]()
    assert polygons, f"{name} produced no geometry"
    for polygon in polygons:
        assert polygon.points.dtype == np.int64
        assert polygon.vertex_count >= 3
        assert is_simple(polygon.points.astype(np.float64) * tech.precision_um)


@pytest.mark.parametrize("name", sorted(ALL_PATTERNS))
def test_pattern_matches_its_golden(name: str) -> None:
    tech = TechConfig()
    summary = summarise(name, ALL_PATTERNS[name](), tech, _deck())

    if regenerating():
        write_golden(name, summary)
        pytest.skip(f"regenerated golden for {name}")

    expected = load_golden(name)
    assert expected is not None, (
        f"no golden for {name!r}. Create it with:\n"
        f"  {REGENERATE_ENV}=1 uv run pytest tests/regression/ -k {name}"
    )

    differences = compare(summary, expected)
    assert not differences, f"{name} differs from its golden:\n  " + "\n  ".join(differences)


@pytest.mark.parametrize("name", sorted(ALL_PATTERNS))
def test_every_pattern_has_a_committed_golden(name: str) -> None:
    assert golden_path(name).exists(), f"{name} has no committed golden"


def test_no_fixture_polygons_unexpectedly_touch() -> None:
    """Patterns that describe a gap must actually have one.

    An earlier `line_ends_manhattan` passed `_rect(2060, 0, 2000, 100)` as
    though the signature were (x0, length, width), producing a stub abutting
    its neighbour instead of a bar across a gap. Nothing caught it until the
    corrected geometry merged into one polygon.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    separated = {
        "line_ends_manhattan",
        "dense_lines",
        "semi_isolated_lines",
        "contact_array",
    }
    tech = TechConfig()
    for name in separated:
        shapes = [
            ShapelyPolygon(p.points.astype(np.float64) * tech.precision_um)
            for p in ALL_PATTERNS[name]()
        ]
        for i, first in enumerate(shapes):
            for second in shapes[i + 1 :]:
                assert not first.intersects(second), (
                    f"{name}: two fixture polygons touch; this pattern is "
                    "supposed to have separated geometry"
                )


def test_summarising_is_reproducible() -> None:
    tech = TechConfig()
    polygons = ALL_PATTERNS["line_ends_manhattan"]()
    first = summarise("x", polygons, tech, _deck())
    second = summarise("x", polygons, tech, _deck())
    assert compare(first, second) == []


def test_compare_names_the_field_that_differs() -> None:
    baseline = {"features": {"hammerhead": 4}, "geometry": {"TARGET": {"polygons": 2}}}
    changed = {"features": {"hammerhead": 3}, "geometry": {"TARGET": {"polygons": 2}}}
    differences = compare(changed, baseline)
    assert len(differences) == 1
    assert "features.hammerhead" in differences[0]
    assert "4 -> 3" in differences[0]


def test_compare_reports_missing_and_unexpected_keys() -> None:
    differences = compare({"b": 1}, {"a": 1})
    joined = "\n".join(differences)
    assert "a: missing" in joined
    assert "b: unexpected" in joined


def test_a_perturbed_summary_is_caught() -> None:
    """The corpus must actually detect a change, not just pass."""
    tech = TechConfig()
    summary = summarise("dense_lines", ALL_PATTERNS["dense_lines"](), tech, _deck())
    perturbed = dict(summary)
    perturbed["features"] = {**summary["features"], "hammerhead": 999}
    assert compare(perturbed, summary)
