"""Decorate a target layout: extract, classify, match, and generate corrections.

Run:
    uv run python examples/decorate_target.py

Runs the full OPC path as it exists today and writes a POST_OPC GDS into
examples/out/ with TARGET, POST_OPC, and overlay layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.io.streams import write_gds
from masklayout.model.cell import Cell
from masklayout.model.geometry import Polygon
from masklayout.model.layers import LayerMap
from masklayout.model.layout import Layout
from masklayout.opc.classify import classify_sites
from masklayout.opc.deck import load_deck
from masklayout.opc.decorate import decorate
from masklayout.opc.extract import extract_sites
from masklayout.opc.match import match_sites

DECK = Path(__file__).parents[1] / "src" / "masklayout" / "decks" / "generic_hammerhead_v1.yaml"
OUT = Path(__file__).parent / "out"


def bar(x0_nm: int, y0_nm: int, length_nm: int, width_nm: int) -> Polygon:
    """An axis-aligned bar, in integer database units."""
    pts = np.array(
        [
            [x0_nm, y0_nm],
            [x0_nm + length_nm, y0_nm],
            [x0_nm + length_nm, y0_nm + width_nm],
            [x0_nm, y0_nm + width_nm],
        ],
        dtype=np.int64,
    )
    return Polygon(points=pts, layer=10)


def main() -> None:
    tech = TechConfig()

    # Two collinear bars with a 60 nm gap between them. The outer line ends
    # see nothing; the inner two face each other.
    target = [bar(0, 0, 2000, 100), bar(2060, 0, 2000, 100)]

    deck = load_deck(DECK)
    print(f"deck            : {deck.id} v{deck.version}")
    print(f"content hash    : {deck.content_hash[:16]}...")
    print(f"rules           : {[r.id for r in deck.rules_in_priority_order()]}")
    print()

    sites = extract_sites(target, tech.precision_um, line_end_ratio=0.5)
    kinds: dict[str, int] = {}
    for site in sites:
        kinds[site.kind] = kinds.get(site.kind, 0) + 1
    print(f"extracted       : {len(sites)} sites from {len(target)} polygons")
    for kind in sorted(kinds):
        print(f"  {kind:<16} {kinds[kind]}")
    print()

    measurements = classify_sites(sites, target, tech, max_probe_um=2.0, density_window_um=1.0)
    print("measured line ends (the closed selector vocabulary):")
    header = f"  {'x (um)':>8} {'width':>8} {'space':>8} {'density':>8}"
    print(header)
    for m in measurements:
        if m.site.kind != "line_end":
            continue
        space = "inf" if m.space_nm == float("inf") else f"{m.space_nm:.0f}"
        print(
            f"  {m.site.midpoint_um[0]:>8.3f} {m.width_nm:>8.0f} {space:>8} {m.local_density:>8.3f}"
        )
    print()

    matches, report = match_sites(measurements, deck)
    print(f"matched         : {report.matched} sites, {report.unmatched} unmatched")
    print(f"by rule         : {report.by_rule}")
    print()
    print("hammerhead decisions:")
    for match in matches:
        if match.kind != "hammerhead":
            continue
        print(f"  site {match.site_id:<22} -> {match.rule_id:<22} {match.params}")
    print()
    print("The outer line ends see unbounded space and take the isolated rule;")
    print("the inner two face a 60 nm gap and take the dense rule. That is the")
    print("whole point of the vocabulary: context selects the correction.")
    print()

    # --- M5: turn those decisions into geometry ---------------------------
    result = decorate(target, deck, tech)
    print(f"decorated       : {result.report.summary()}")
    print()
    print("generated features:")
    for feature in result.features:
        print(f"  {feature.id:<28} {feature.polarity:<9} {feature.vertex_count} vertices")
    print()

    target_area = sum(abs(signed_area(p.points)) for p in target)
    post_area = sum(abs(signed_area(p.points)) for p in result.post_opc)
    growth_pct = 100.0 * (post_area / target_area - 1.0)
    print(f"target area     : {target_area / 1e6:.6f} um^2")
    print(f"post-OPC area   : {post_area / 1e6:.6f} um^2  (+{growth_pct:.2f}%)")
    print(f"overlay ADD     : {len(result.overlay_add)} polygon(s)")
    print(f"overlay REMOVE  : {len(result.overlay_remove)} polygon(s)")
    print()

    OUT.mkdir(exist_ok=True)
    layers = LayerMap.default()
    layout = Layout(name="DECORATED", tech=tech, layers=layers)
    top = layout.add(Cell(name="TOP"))
    top.polygons.extend(target)
    top.polygons.extend(result.post_opc)
    top.polygons.extend(result.overlay_add)
    top.polygons.extend(result.overlay_remove)

    path = OUT / "post_opc.gds"
    write_gds(layout, path)
    print(f"wrote {path.name} ({path.stat().st_size} bytes) with layers:")
    for name in ("TARGET", "POST_OPC", "OVERLAY_ADD", "OVERLAY_REMOVE"):
        layer = layers[name]
        count = sum(1 for p in top.polygons if p.layer == layer.number)
        print(f"  {name:<15} {layer.number}/{layer.datatype}  {count} polygon(s)")
    print()
    print("Open it in KLayout to see the corrections against the target.")


if __name__ == "__main__":
    main()
