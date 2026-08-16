"""Mask rule checks: minimum width and minimum space.

Both are computed morphologically — erode then dilate for width, the
reverse for space — and both **require a deburr**, without which they are
unusable on non-Manhattan geometry.

The erode/dilate round trip accumulates roughly half a grid step of
quantization error along every edge, and that noise scales with perimeter.
Measured on a 31-degree rotated bar during design, a *clean* bar reported
723 nm^2 of violation and a genuinely defective one 777 nm^2 — the clean
bar looked worse. ``join="miter"`` does not help; the residue lies along
edges, not only at corners.

Eroding each candidate violation by half the design grid removes the noise
and leaves real defects. See design section 6.2.
"""

from __future__ import annotations

from collections.abc import Sequence

from masklayout.config import TechConfig
from masklayout.geometry.context import GeomContext
from masklayout.model.geometry import Polygon
from masklayout.verify.violation import Violation

#: The limit of what these checks can see, stated so a report cannot
#: overstate its own coverage.
MRC_SENSITIVITY_NOTE = (
    "Morphological MRC on quantized geometry has a sensitivity floor of "
    "roughly one design grid step: a violation within about that much of the "
    "rule value may not be reported. The deburr that makes these checks "
    "usable on non-Manhattan geometry is what imposes the floor."
)


def _open_close(
    polygons: Sequence[Polygon],
    radius_um: float,
    context: GeomContext,
    layer: int,
    close: bool,
) -> list[Polygon]:
    """Morphological open (close=False) or close (close=True)."""
    first, second = (radius_um, -radius_um) if close else (-radius_um, radius_um)
    stage = context.offset_polygons(polygons, first, layer)
    if not stage:
        return []
    return context.offset_polygons(stage, second, layer)


def _deburr(
    candidates: Sequence[Polygon], context: GeomContext, tech: TechConfig, layer: int
) -> list[Polygon]:
    """Erode candidate violations by half the grid to strip quantization noise."""
    deburr_um = tech.effective_mrc_deburr_nm / 1000.0
    if deburr_um <= 0.0 or not candidates:
        return list(candidates)
    return context.offset_polygons(candidates, -deburr_um, layer)


def check_min_width(
    polygons: Sequence[Polygon],
    min_width_nm: float,
    tech: TechConfig,
    marker_layer: int = 201,
    deburr: bool = True,
) -> list[Violation]:
    """Report regions narrower than ``min_width_nm``.

    ``deburr=False`` exists only so a test can demonstrate that the check is
    unusable without it. Production callers leave it on.
    """
    if not polygons:
        return []
    context = GeomContext(tech)
    radius_um = (min_width_nm / 2.0) / 1000.0

    opened = _open_close(polygons, radius_um, context, marker_layer, close=False)
    raw = (
        context.boolean_polygons(polygons, opened, "not", marker_layer)
        if opened
        else list(polygons)
    )
    regions = _deburr(raw, context, tech, marker_layer) if deburr else raw
    if not regions:
        return []
    return [
        Violation(
            check="min_width",
            severity="error",
            message=(
                f"{len(regions)} region(s) narrower than {min_width_nm} nm"
                + ("" if deburr else " (deburr disabled — expect quantization noise)")
            ),
            polygons=list(regions),
            detail={"min_width_nm": min_width_nm, "regions": len(regions), "deburr": deburr},
        )
    ]


def check_min_space(
    polygons: Sequence[Polygon],
    min_space_nm: float,
    tech: TechConfig,
    marker_layer: int = 201,
    deburr: bool = True,
) -> list[Violation]:
    """Report gaps narrower than ``min_space_nm``."""
    if not polygons:
        return []
    context = GeomContext(tech)
    radius_um = (min_space_nm / 2.0) / 1000.0

    closed = _open_close(polygons, radius_um, context, marker_layer, close=True)
    if not closed:
        return []
    raw = context.boolean_polygons(closed, polygons, "not", marker_layer)
    regions = _deburr(raw, context, tech, marker_layer) if deburr else raw
    if not regions:
        return []
    return [
        Violation(
            check="min_space",
            severity="error",
            message=f"{len(regions)} gap(s) narrower than {min_space_nm} nm",
            polygons=list(regions),
            detail={"min_space_nm": min_space_nm, "regions": len(regions), "deburr": deburr},
        )
    ]
