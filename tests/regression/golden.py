"""Summarise a pipeline run into a comparable golden record.

A golden records **counts and statistics, not coordinates**. A golden full
of vertices fails on every legitimate improvement and teaches the reader to
regenerate without looking, which is worse than no golden at all.

When a golden fails, the diff must answer "what changed" — 4 hammerheads
became 3, or a new min-width violation appeared.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.model.geometry import Polygon
from masklayout.opc.classify import classify_sites
from masklayout.opc.deck import RuleDeck
from masklayout.opc.decorate import decorate
from masklayout.opc.extract import extract_sites
from masklayout.opc.match import match_sites
from masklayout.verify.mrc import check_min_space, check_min_width
from masklayout.verify.structural import run_structural_checks

GOLDENS = Path(__file__).parent / "goldens"

#: Set to regenerate rather than compare. Never automatic: a corpus that
#: silently rewrites its own expectations tests nothing.
REGENERATE_ENV = "MASKLAYOUT_REGENERATE_GOLDENS"


def _counts(values: Sequence[str]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for value in values:
        counted[value] = counted.get(value, 0) + 1
    return dict(sorted(counted.items()))


def _stats(polygons: Sequence[Polygon], tech: TechConfig) -> dict[str, Any]:
    return {
        "polygons": len(polygons),
        "vertices": sum(p.vertex_count for p in polygons),
        # Rounded to the nearest nm^2: a shift big enough to matter still shows,
        # while float noise does not churn the golden.
        "area_nm2": round(
            sum(abs(signed_area(p.points)) for p in polygons) * (tech.design_grid_nm**2)
        ),
    }


def summarise(
    name: str,
    polygons: Sequence[Polygon],
    tech: TechConfig,
    deck: RuleDeck,
    min_width_nm: float = 20.0,
    min_space_nm: float = 40.0,
) -> dict[str, Any]:
    """Run the pipeline and record what it produced."""
    sites = extract_sites(polygons, tech.precision_um, line_end_ratio=0.5)
    measurements = classify_sites(sites, polygons, tech, max_probe_um=2.0, density_window_um=1.0)
    _, match_report = match_sites(measurements, deck)
    result = decorate(polygons, deck, tech)

    violations = run_structural_checks(result.post_opc, tech)
    violations += check_min_width(result.post_opc, min_width_nm, tech)
    violations += check_min_space(result.post_opc, min_space_nm, tech)

    return {
        "pattern": name,
        "deck": {"id": deck.id, "version": deck.version},
        "sites": _counts([s.kind for s in sites]),
        "matches": {
            "matched": match_report.matched,
            "unmatched": match_report.unmatched,
            "by_rule": dict(sorted(match_report.by_rule.items())),
        },
        "features": _counts([f.kind for f in result.features]),
        "geometry": {
            "TARGET": _stats(polygons, tech),
            "POST_OPC": _stats(result.post_opc, tech),
            "OVERLAY_ADD": _stats(result.overlay_add, tech),
            "OVERLAY_REMOVE": _stats(result.overlay_remove, tech),
            "SRAF": _stats(result.srafs, tech),
        },
        "violations": _counts([v.check for v in violations]),
    }


def golden_path(name: str) -> Path:
    return GOLDENS / f"{name}.json"


def load_golden(name: str) -> dict[str, Any] | None:
    path = golden_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def write_golden(name: str, summary: dict[str, Any]) -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    golden_path(name).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def regenerating() -> bool:
    return os.environ.get(REGENERATE_ENV) == "1"


def compare(actual: Any, expected: Any, path: str = "") -> list[str]:
    """Human-readable differences, naming the field rather than dumping both."""
    differences: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            here = f"{path}.{key}" if path else key
            if key not in actual:
                differences.append(f"{here}: missing (expected {expected[key]!r})")
            elif key not in expected:
                differences.append(f"{here}: unexpected {actual[key]!r}")
            else:
                differences.extend(compare(actual[key], expected[key], here))
    elif actual != expected:
        differences.append(f"{path}: {expected!r} -> {actual!r}")
    return differences
