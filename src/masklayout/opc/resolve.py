"""Collision and keep-out resolution for assist features.

Every rejection is reported with a reason and the measured distance.
Silently dropping candidates would make a sparse result indistinguishable
from a working one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.opc.feature import Feature

#: Slack on keep-out comparisons, in micrometres. A gap of exactly the
#: keep-out distance must pass, but subtracting float coordinates makes it
#: land just under: 0.12 - 0.1 is 0.019999999999999997, which is 1e-17 below
#: a 20 nm keep-out. Without this, a deck specifying exactly the keep-out
#: would have its assist features silently rejected. One femtometre is
#: astronomically below any physical grid and astronomically above the noise.
_DISTANCE_EPSILON_UM = 1e-9


@dataclass(frozen=True)
class Rejection:
    """One assist feature that could not be placed, and why."""

    feature_id: str
    reason: str
    detail: str
    polygons: list[Polygon]

    def as_record(self) -> dict[str, str]:
        return {"feature_id": self.feature_id, "reason": self.reason, "detail": self.detail}


def _shapely(polygons: Sequence[Polygon], precision_um: float) -> ShapelyPolygon | None:
    shapes = [
        ShapelyPolygon(polygon.points.astype(np.float64) * precision_um) for polygon in polygons
    ]
    if not shapes:
        return None
    return unary_union(shapes)


def resolve_collisions(
    features: Sequence[Feature],
    target: Sequence[Polygon],
    tech: TechConfig,
    target_keepout_um: float = 0.02,
    sraf_keepout_um: float = 0.02,
) -> tuple[list[Feature], list[Rejection]]:
    """Drop assist features that violate keep-out, reporting each one.

    Corrections pass through untouched: they are *meant* to touch the target.
    Assist features are considered in feature-id order, so which of two
    conflicting candidates survives is stable across runs.
    """
    precision_um = tech.precision_um
    target_shape = _shapely(target, precision_um)

    kept: list[Feature] = []
    rejected: list[Rejection] = []
    placed_shapes: list[ShapelyPolygon] = []

    corrections = [f for f in features if f.polarity != "assist"]
    assists = sorted((f for f in features if f.polarity == "assist"), key=lambda f: f.id)
    kept.extend(corrections)

    for feature in assists:
        shape = _shapely(feature.polygons, precision_um)
        if shape is None:
            continue

        if target_shape is not None:
            distance = float(target_shape.distance(shape))
            if distance < target_keepout_um - _DISTANCE_EPSILON_UM:
                rejected.append(
                    Rejection(
                        feature_id=feature.id,
                        reason="target_keepout",
                        detail=(
                            f"{distance * 1000:.1f} nm from the target, "
                            f"below the {target_keepout_um * 1000:.1f} nm keep-out"
                        ),
                        polygons=list(feature.polygons),
                    )
                )
                continue

        conflict = None
        for other in placed_shapes:
            distance = float(other.distance(shape))
            if distance < sraf_keepout_um - _DISTANCE_EPSILON_UM:
                conflict = distance
                break
        if conflict is not None:
            rejected.append(
                Rejection(
                    feature_id=feature.id,
                    reason="sraf_keepout",
                    detail=(
                        f"{conflict * 1000:.1f} nm from an already-placed assist "
                        f"feature, below the {sraf_keepout_um * 1000:.1f} nm keep-out"
                    ),
                    polygons=list(feature.polygons),
                )
            )
            continue

        kept.append(feature)
        placed_shapes.append(shape)

    return kept, rejected
