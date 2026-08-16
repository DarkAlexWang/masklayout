"""Decorate a target layout: extract sites, classify them, match a rule deck.

Run:
    uv run python examples/decorate_target.py

This is the OPC path as far as it exists today. It stops at MATCHING:
M4 decides which rule fires where, and M5 will turn those decisions into
correction geometry. Nothing is written to a mask here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.classify import classify_sites
from masklayout.opc.deck import load_deck
from masklayout.opc.extract import extract_sites
from masklayout.opc.match import match_sites

DECK = Path(__file__).parents[1] / "src" / "masklayout" / "decks" / "generic_hammerhead_v1.yaml"


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
    print("NOT YET IMPLEMENTED: turning these decisions into geometry (M5).")


if __name__ == "__main__":
    main()
