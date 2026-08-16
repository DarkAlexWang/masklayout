"""The JSON manifest: the durable record of what a run did.

Provenance is only useful if it survives the process that produced it, so
everything needed to explain an output polygon goes here — the tool and
config that made it, the deck that decided it, and every check that ran.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.model.geometry import Polygon
from masklayout.opc.feature import Feature
from masklayout.verify.mrc import MRC_SENSITIVITY_NOTE
from masklayout.verify.violation import Violation

#: Manifest schema version, so a consumer can tell what shape to expect.
MANIFEST_VERSION = "1"


def geometric_statistics(polygons: Sequence[Polygon], tech: TechConfig) -> dict[str, Any]:
    """Counts, area, and perimeter for a set of polygons."""
    if not polygons:
        return {"polygon_count": 0, "vertex_count": 0, "area_nm2": 0.0, "perimeter_nm": 0.0}

    area_nm2 = sum(abs(signed_area(p.points)) for p in polygons) * (tech.design_grid_nm**2)
    perimeter_nm = 0.0
    for polygon in polygons:
        points = polygon.points.astype(np.float64)
        edges = np.diff(np.vstack([points, points[:1]]), axis=0)
        perimeter_nm += float(np.linalg.norm(edges, axis=1).sum()) * tech.design_grid_nm

    return {
        "polygon_count": len(polygons),
        "vertex_count": sum(p.vertex_count for p in polygons),
        "area_nm2": area_nm2,
        "perimeter_nm": perimeter_nm,
    }


def build_manifest(
    tech: TechConfig,
    tool_version: str,
    features: Sequence[Feature] = (),
    violations: Sequence[Violation] = (),
    layer_geometry: Mapping[str, Sequence[Polygon]] | None = None,
    deck_id: str | None = None,
    deck_version: str | None = None,
    deck_hash: str | None = None,
    mrc_ran: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest as a plain dict.

    No timestamp is recorded. The design requires reproducible output, and a
    wall-clock stamp would make two identical runs differ — the same reason
    GDS writes use a pinned timestamp.
    """
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "tool": {"name": "masklayout", "version": tool_version},
        "technology": tech.model_dump(mode="json"),
        "deck": {"id": deck_id, "version": deck_version, "content_hash": deck_hash},
        "features": [feature.provenance() for feature in features],
        "violations": [violation.as_record() for violation in violations],
        "statistics": {
            name: geometric_statistics(polygons, tech)
            for name, polygons in sorted((layer_geometry or {}).items())
        },
    }
    if mrc_ran:
        manifest["mrc_sensitivity"] = MRC_SENSITIVITY_NOTE
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(path: Path | str, manifest: dict[str, Any]) -> None:
    """Write the manifest as canonical JSON.

    Keys are sorted so two runs of the same input produce byte-identical
    files, which is what makes a manifest diffable in review.
    """
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")
