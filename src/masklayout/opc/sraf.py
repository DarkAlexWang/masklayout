"""SRAF placement: sub-resolution assist features.

An assist feature is not part of the main pattern. It is written to the
SRAF layer and never merged into POST_OPC — the design treats the two as
distinct geometry, since tone inversion is ``FIELD - (POST_OPC | SRAF)``.
"""

from __future__ import annotations

import math
from typing import Any

from masklayout.config import TechConfig
from masklayout.opc.classify import SiteMeasurement
from masklayout.opc.feature import Feature
from masklayout.opc.generate import (
    DEFAULT_CORRECTION_LAYER,
    _require,
    feature_id,
    register_generator,
)
from masklayout.opc.match import Match
from masklayout.opc.placement import merge_params, placement_for
from masklayout.pcells.base import build_pcell, params_model_for


@register_generator("sraf_bar")
def generate_sraf_bar(
    measurement: SiteMeasurement, match: Match, tech: TechConfig
) -> Feature | None:
    """A bar placed parallel to an edge, offset outward into the space beside it.

    ``distance_um`` is measured from the source edge to the **near side** of
    the bar, not to its centre, because that is the number a rule deck
    actually specifies. The generator converts it to a centre offset.
    """
    distance_um = _require(match.params, "distance_um", "sraf_bar")
    width_um = _require(match.params, "width_um", "sraf_bar")
    if distance_um <= 0.0 or width_um <= 0.0:
        return None

    edge_length_um = measurement.site.edge_length_um
    if edge_length_um <= 0.0:
        return None

    length_ratio = float(match.params.get("length_ratio", 1.0))
    if length_ratio <= 0.0:
        return None

    site = measurement.site
    # `line_end` anchors at a point and extends along an angle, which maps
    # exactly onto "start `distance_um` out from the edge and be `width_um`
    # deep". Anchoring at the near side is why distance is measured there and
    # not to the bar's centre: it is the number a deck actually specifies.
    anchor_um = (
        site.midpoint_um[0] + site.outward_normal_um[0] * distance_um,
        site.midpoint_um[1] + site.outward_normal_um[1] * distance_um,
    )

    shape: dict[str, Any] = {
        key: value
        for key, value in match.params.items()
        if key not in ("distance_um", "width_um", "length_ratio")
    }
    shape["extension_um"] = width_um  # depth of the bar, away from the edge
    shape["width_um"] = edge_length_um * length_ratio  # length, along the edge

    accepted = params_model_for(match.pcell).model_fields
    placement = {
        "centre_um": anchor_um,
        "angle_rad": math.atan2(site.outward_normal_um[1], site.outward_normal_um[0]),
    }
    params = merge_params(placement, shape, accepted_keys=accepted)
    polygons = build_pcell(match.pcell, params, tech, layer=DEFAULT_CORRECTION_LAYER, datatype=0)
    return Feature(
        id=feature_id(match),
        kind="sraf_bar",
        polygons=polygons,
        source_site_id=site.site_id,
        rule_id=match.rule_id,
        deck_id=match.deck_id,
        deck_version=match.deck_version,
        deck_hash=match.deck_hash,
        parameters=dict(match.params),
        polarity="assist",
    )


__all__ = ["generate_sraf_bar", "placement_for"]
