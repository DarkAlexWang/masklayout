"""Placement derived from a site.

The milestone's central idea: **the site supplies placement, the rule
supplies shape**. Where a correction goes is a geometric fact read off the
target, not something a deck author chooses, so a rule that tries to set a
placement key is rejected rather than silently honoured.
"""

from __future__ import annotations

import math
from typing import Any

from masklayout.opc.sites import Site

#: Parameters derived from geometry. A rule may not supply these.
PLACEMENT_KEYS = frozenset({"centre_um", "angle_rad"})


class PlacementOverrideError(ValueError):
    """A rule tried to set a parameter that the site determines."""


def placement_for(site: Site) -> dict[str, Any]:
    """Placement parameters for a correction at this site.

    The angle is the direction of the outward normal, so a shape built in
    edge-local coordinates extends away from the target rather than into it.
    This is what lets one generator serve any line angle.
    """
    return {
        "centre_um": site.midpoint_um,
        "angle_rad": math.atan2(site.outward_normal_um[1], site.outward_normal_um[0]),
    }


def merge_params(placement: dict[str, Any], rule_params: dict[str, Any]) -> dict[str, Any]:
    """Combine derived placement with a rule's shape parameters."""
    conflicts = PLACEMENT_KEYS & set(rule_params)
    if conflicts:
        raise PlacementOverrideError(
            f"rule supplies placement parameter(s) {sorted(conflicts)}, which are "
            f"determined by the site geometry; a rule may only supply shape "
            f"parameters. Placement keys are {sorted(PLACEMENT_KEYS)}"
        )
    return {**rule_params, **placement}
