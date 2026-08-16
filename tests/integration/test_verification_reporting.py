"""M7 acceptance: verify, report, and render a decorated layout."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from masklayout import __version__
from masklayout.config import TechConfig
from masklayout.io.manifest import build_manifest, write_manifest
from masklayout.model.geometry import Polygon
from masklayout.model.layers import LayerMap
from masklayout.opc.deck import RuleDeck, load_deck_from_mapping
from masklayout.opc.decorate import decorate
from masklayout.render.svg import render_svg
from masklayout.verify.mrc import check_min_space, check_min_width
from masklayout.verify.structural import run_structural_checks

TECH = TechConfig()


def _bar(x0: int, length_nm: int = 2000, width_nm: int = 100) -> Polygon:
    pts = np.array(
        [[x0, 0], [x0 + length_nm, 0], [x0 + length_nm, width_nm], [x0, width_nm]],
        dtype=np.int64,
    )
    return Polygon(points=pts, layer=10)


def _deck() -> RuleDeck:
    rule: dict[str, Any] = {
        "id": "hh",
        "priority": 10,
        "kind": "hammerhead",
        "when": {"site": "line_end"},
        "apply": {
            "pcell": "line_end",
            "params": {"extension_um": 0.028, "head_width_ratio": 1.4},
        },
    }
    return load_deck_from_mapping({"deck": {"id": "d", "version": "1.0.0"}, "rules": [rule]})


def _run() -> tuple[Any, list[Any], dict[str, list[Polygon]]]:
    target = [_bar(0), _bar(2060)]
    deck = _deck()
    result = decorate(target, deck, TECH, layers=LayerMap.default())
    violations = run_structural_checks(result.post_opc, TECH)
    violations += check_min_width(result.post_opc, 20.0, TECH)
    violations += check_min_space(result.post_opc, 20.0, TECH)
    geometry = {
        "TARGET": target,
        "POST_OPC": result.post_opc,
        "OVERLAY_ADD": result.overlay_add,
    }
    return result, violations, geometry


def test_decorated_geometry_passes_structural_checks() -> None:
    result, _, _ = _run()
    assert run_structural_checks(result.post_opc, TECH) == []


def test_the_manifest_records_every_feature_and_the_deck_hash(tmp_path: Path) -> None:
    result, violations, geometry = _run()
    deck = _deck()
    manifest = build_manifest(
        TECH,
        tool_version=__version__,
        features=result.features,
        violations=violations,
        layer_geometry=geometry,
        deck_id=deck.id,
        deck_version=deck.version,
        deck_hash=deck.content_hash,
        mrc_ran=True,
    )
    path = tmp_path / "m.json"
    write_manifest(path, manifest)

    loaded = json.loads(path.read_text())
    assert loaded["deck"]["content_hash"] == deck.content_hash
    assert len(loaded["features"]) == len(result.features)
    assert loaded["tool"]["name"] == "masklayout"
    assert loaded["statistics"]["POST_OPC"]["polygon_count"] > 0
    assert loaded["statistics"]["POST_OPC"]["area_nm2"] > 0


def test_the_manifest_states_the_mrc_sensitivity_floor(tmp_path: Path) -> None:
    """A report that does not say what it can miss overstates its coverage."""
    manifest = build_manifest(TECH, tool_version=__version__, mrc_ran=True)
    assert "sensitivity floor" in manifest["mrc_sensitivity"]

    without = build_manifest(TECH, tool_version=__version__, mrc_ran=False)
    assert "mrc_sensitivity" not in without


def test_the_manifest_is_byte_reproducible(tmp_path: Path) -> None:
    result, violations, geometry = _run()
    deck = _deck()

    def emit(name: str) -> bytes:
        manifest = build_manifest(
            TECH,
            tool_version=__version__,
            features=result.features,
            violations=violations,
            layer_geometry=geometry,
            deck_id=deck.id,
            deck_version=deck.version,
            deck_hash=deck.content_hash,
        )
        path = tmp_path / name
        write_manifest(path, manifest)
        return path.read_bytes()

    assert emit("a.json") == emit("b.json")


def test_the_svg_renders_every_populated_layer(tmp_path: Path) -> None:
    _, _, geometry = _run()
    path = tmp_path / "preview.svg"
    render_svg(path, geometry, TECH)

    text = path.read_text()
    assert text.startswith("<svg")
    assert text.rstrip().endswith("</svg>")
    for name in ("TARGET", "POST_OPC", "OVERLAY_ADD"):
        assert f'id="{name}"' in text
    assert text.count("<polygon") >= 4


def test_the_svg_omits_empty_layers(tmp_path: Path) -> None:
    _, _, geometry = _run()
    geometry["OVERLAY_REMOVE"] = []
    path = tmp_path / "preview.svg"
    render_svg(path, geometry, TECH)
    assert 'id="OVERLAY_REMOVE"' not in path.read_text()
