"""M5 acceptance: decorate a target and write a viewable POST_OPC layout."""

from pathlib import Path

import numpy as np

from masklayout.config import TechConfig
from masklayout.io.streams import read_gds, write_gds
from masklayout.model.cell import Cell
from masklayout.model.geometry import Polygon
from masklayout.model.layers import LayerMap
from masklayout.model.layout import Layout
from masklayout.opc.deck import load_deck
from masklayout.opc.decorate import decorate

DECK = Path(__file__).parents[2] / "src" / "masklayout" / "decks" / "generic_hammerhead_v1.yaml"


def _bar(x0: int, length_nm: int = 2000, width_nm: int = 100) -> Polygon:
    pts = np.array(
        [[x0, 0], [x0 + length_nm, 0], [x0 + length_nm, width_nm], [x0, width_nm]],
        dtype=np.int64,
    )
    return Polygon(points=pts, layer=10)


def _target() -> list[Polygon]:
    """Two collinear bars with a 60 nm gap: two isolated ends, two dense."""
    return [_bar(0), _bar(2060)]


def test_decorating_produces_corrections_from_the_shipped_deck() -> None:
    result = decorate(_target(), load_deck(DECK), TechConfig())
    assert result.report.features_generated > 0
    assert "hammerhead" in result.report.by_kind


def test_isolated_and_dense_ends_get_different_corrections() -> None:
    """The whole chain, end to end: context selects the correction geometry."""
    result = decorate(_target(), load_deck(DECK), TechConfig())
    hammerheads = [f for f in result.features if f.kind == "hammerhead"]
    rules = {f.rule_id for f in hammerheads}
    assert "hh_isolated_line_end" in rules
    assert "hh_dense_line_end" in rules

    by_rule = {f.rule_id: f for f in hammerheads}
    isolated = by_rule["hh_isolated_line_end"]
    dense = by_rule["hh_dense_line_end"]
    # The isolated rule extends 28 nm, the dense one 14 nm.
    assert isolated.parameters["extension_um"] > dense.parameters["extension_um"]


def test_post_opc_is_larger_than_the_target() -> None:
    from masklayout.geometry.normalize import signed_area

    target = _target()
    result = decorate(target, load_deck(DECK), TechConfig())
    area = sum(abs(signed_area(p.points)) for p in result.post_opc)
    original = sum(abs(signed_area(p.points)) for p in target)
    assert area > original


def test_a_decorated_layout_round_trips_with_all_four_layers(tmp_path: Path) -> None:
    tech = TechConfig()
    layers = LayerMap.default()
    target = _target()
    result = decorate(target, load_deck(DECK), tech, layers=layers)

    layout = Layout(name="DECORATED", tech=tech, layers=layers)
    top = layout.add(Cell(name="TOP"))
    top.polygons.extend(target)  # TARGET, layer 10
    top.polygons.extend(result.post_opc)  # POST_OPC, layer 11
    top.polygons.extend(result.overlay_add)  # OVERLAY_ADD, layer 202
    top.polygons.extend(result.overlay_remove)  # OVERLAY_REMOVE, layer 203

    path = tmp_path / "post_opc.gds"
    write_gds(layout, path)
    restored, _ = read_gds(path)

    present = {p.layer for p in restored.cells["TOP"].polygons}
    assert layers["TARGET"].number in present
    assert layers["POST_OPC"].number in present
    assert layers["OVERLAY_ADD"].number in present
    # Only additive corrections fire, so OVERLAY_REMOVE is legitimately empty.
    assert layers["OVERLAY_REMOVE"].number not in present


def test_every_generated_feature_names_the_deck_that_produced_it() -> None:
    deck = load_deck(DECK)
    result = decorate(_target(), deck, TechConfig())
    assert result.features
    for feature in result.features:
        assert feature.deck_id == "generic_hammerhead_v1"
        assert feature.deck_hash == deck.content_hash
        assert feature.source_site_id


def test_decorating_is_reproducible() -> None:
    first = decorate(_target(), load_deck(DECK), TechConfig())
    second = decorate(_target(), load_deck(DECK), TechConfig())
    assert len(first.post_opc) == len(second.post_opc)
    for a, b in zip(first.post_opc, second.post_opc, strict=True):
        assert np.array_equal(a.points, b.points)
