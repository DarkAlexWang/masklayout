"""M6 acceptance: SRAF placement, keep-out enforcement, and layer separation."""

from pathlib import Path
from typing import Any

import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.io.streams import read_gds, write_gds
from masklayout.model.cell import Cell
from masklayout.model.geometry import Polygon
from masklayout.model.layers import LayerMap
from masklayout.model.layout import Layout
from masklayout.opc.deck import RuleDeck, load_deck_from_mapping
from masklayout.opc.decorate import decorate

TECH = TechConfig()


def _bar(x0: int, length_nm: int = 2000, width_nm: int = 100) -> Polygon:
    pts = np.array(
        [[x0, 0], [x0 + length_nm, 0], [x0 + length_nm, width_nm], [x0, width_nm]],
        dtype=np.int64,
    )
    return Polygon(points=pts, layer=10)


def _sraf_rule(distance_um: float, width_um: float = 0.03) -> dict[str, Any]:
    return {
        "id": "sraf",
        "priority": 10,
        "kind": "sraf_bar",
        "when": {"site": "edge", "edge_length_nm": {"min": 1000}},
        "apply": {
            "pcell": "line_end",
            "params": {"distance_um": distance_um, "width_um": width_um},
        },
    }


def _deck(*rules: dict[str, Any]) -> RuleDeck:
    return load_deck_from_mapping(
        {"deck": {"id": "sraf_deck", "version": "1.0.0"}, "rules": list(rules)}
    )


def test_srafs_are_placed_on_their_own_layer() -> None:
    result = decorate([_bar(0)], _deck(_sraf_rule(0.08)), TECH)
    assert result.srafs
    assert all(p.layer == 12 for p in result.srafs)


def test_srafs_are_not_merged_into_post_opc() -> None:
    """The design keeps them distinct: tone inversion is FIELD - (POST_OPC | SRAF)."""
    target = [_bar(0)]
    with_srafs = decorate(target, _deck(_sraf_rule(0.08)), TECH)
    without = decorate(target, _deck(), TECH)

    area_with = sum(abs(signed_area(p.points)) for p in with_srafs.post_opc)
    area_without = sum(abs(signed_area(p.points)) for p in without.post_opc)
    assert area_with == area_without
    assert with_srafs.srafs  # they exist, just not in POST_OPC


def test_an_sraf_too_close_to_the_target_is_rejected_and_reported() -> None:
    # 5 nm from the target, well inside a 20 nm keep-out.
    result = decorate([_bar(0)], _deck(_sraf_rule(0.005)), TECH, target_keepout_um=0.02)
    assert result.srafs == []
    assert result.rejected
    assert all(r.reason == "target_keepout" for r in result.rejected)
    assert result.report.srafs_rejected == len(result.rejected)


def test_rejected_srafs_become_debug_markers() -> None:
    result = decorate([_bar(0)], _deck(_sraf_rule(0.005)), TECH, target_keepout_um=0.02)
    assert result.markers
    assert all(p.layer == 201 for p in result.markers)


def test_the_report_counts_placed_and_rejected() -> None:
    placed = decorate([_bar(0)], _deck(_sraf_rule(0.08)), TECH)
    assert placed.report.srafs_placed > 0
    assert placed.report.srafs_rejected == 0
    assert "SRAFs placed" in placed.report.summary()

    blocked = decorate([_bar(0)], _deck(_sraf_rule(0.005)), TECH)
    assert blocked.report.srafs_placed == 0
    assert blocked.report.srafs_rejected > 0


def test_srafs_between_two_bars_respect_keepout_from_both() -> None:
    """A narrow gap cannot hold an assist feature; the engine must say so."""
    target = [_bar(0), _bar(0)]  # identical bars, so the gap is nil
    result = decorate(target, _deck(_sraf_rule(0.005)), TECH, target_keepout_um=0.02)
    assert result.srafs == []
    assert result.rejected


def test_a_decorated_layout_with_srafs_round_trips(tmp_path: Path) -> None:
    layers = LayerMap.default()
    target = [_bar(0)]
    result = decorate(target, _deck(_sraf_rule(0.08)), TECH, layers=layers)

    layout = Layout(name="WITH_SRAF", tech=TECH, layers=layers)
    top = layout.add(Cell(name="TOP"))
    top.polygons.extend(target)
    top.polygons.extend(result.post_opc)
    top.polygons.extend(result.srafs)

    path = tmp_path / "sraf.gds"
    write_gds(layout, path)
    restored, _ = read_gds(path)

    present = {p.layer for p in restored.cells["TOP"].polygons}
    assert layers["TARGET"].number in present
    assert layers["POST_OPC"].number in present
    assert layers["SRAF"].number in present


def test_sraf_placement_is_reproducible() -> None:
    first = decorate([_bar(0)], _deck(_sraf_rule(0.08)), TECH)
    second = decorate([_bar(0)], _deck(_sraf_rule(0.08)), TECH)
    assert len(first.srafs) == len(second.srafs)
    for a, b in zip(first.srafs, second.srafs, strict=True):
        assert np.array_equal(a.points, b.points)
