"""Mask-data export: scale to reticle, apply tone, re-verify.

A separate path from the engineering export. Scaling is a pure integer
multiply, because ``TechConfig`` validates at load that
``magnification * design_grid_nm`` is an exact multiple of ``mask_grid_nm``
— that decision, taken at the very start of the design, is what keeps this
module free of re-snapping and second quantization.

Tone inversion swaps width and space, so mask rule checks run *again* on
the inverted geometry rather than being trusted from the 1x pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.context import GeomContext
from masklayout.model.cell import Cell
from masklayout.model.geometry import Polygon
from masklayout.model.layers import Layer, LayerMap
from masklayout.model.layout import Layout
from masklayout.verify.mrc import check_min_space, check_min_width
from masklayout.verify.structural import run_structural_checks
from masklayout.verify.violation import Violation


class MaskExportError(ValueError):
    """The layout cannot be turned into mask data as configured."""


class FieldMissingError(MaskExportError):
    """Tone inversion was requested without a FIELD polygon."""


class GeometryOutsideFieldError(MaskExportError):
    """Geometry lies outside the declared field extent."""


def scale_to_mask(polygons: Sequence[Polygon], tech: TechConfig) -> list[Polygon]:
    """Scale 1x design geometry to mask scale.

    An exact integer multiply. The config validation guarantees the result
    lands on the mask grid, so nothing is re-snapped here.
    """
    return [
        Polygon(
            points=polygon.points * tech.magnification,
            layer=polygon.layer,
            datatype=polygon.datatype,
        )
        for polygon in polygons
    ]


def on_mask_grid(polygons: Sequence[Polygon], tech: TechConfig) -> bool:
    """Whether every vertex lies on the mask-writer grid."""
    step = tech.mask_grid_nm / tech.design_grid_nm
    for polygon in polygons:
        scaled = polygon.points.astype(np.float64) / step
        if not np.allclose(scaled, np.round(scaled), rtol=0.0, atol=1e-9):
            return False
    return True


def validate_field(field: Sequence[Polygon], geometry: Sequence[Polygon], tech: TechConfig) -> None:
    """The field must exist and must contain every polygon.

    Nothing is inferred and nothing is clipped: the design requires the
    boundary to be declared, and geometry outside it is an error rather than
    a silent truncation.
    """
    if not field:
        raise FieldMissingError(
            "tone inversion requires a FIELD polygon; none was supplied. "
            "Draw the field extent on the FIELD layer and pass it to export_mask"
        )
    if not geometry:
        return
    outside = GeomContext(tech).boolean_polygons(geometry, field, "not", layer=0)
    if outside:
        raise GeometryOutsideFieldError(
            f"{len(outside)} polygon region(s) lie outside the declared FIELD extent; "
            "geometry is never silently clipped to the field"
        )


def invert_tone(
    field: Sequence[Polygon], geometry: Sequence[Polygon], tech: TechConfig, layer: Layer
) -> list[Polygon]:
    """FIELD minus geometry: the dark-tone written pattern."""
    return GeomContext(tech).boolean_polygons(field, geometry, "not", layer.number, layer.datatype)


def apply_tone(
    field: Sequence[Polygon],
    post_opc: Sequence[Polygon],
    srafs: Sequence[Polygon],
    tech: TechConfig,
    layer: Layer,
) -> list[Polygon]:
    """Written geometry for the configured tone.

    Clear tone writes the pattern as drawn. Dark tone writes
    ``FIELD - (POST_OPC | SRAF)`` — assist features are subtracted along with
    the main pattern, not left as holes inside holes.
    """
    context = GeomContext(tech)
    drawn = context.boolean_polygons(
        list(post_opc), list(srafs), "or", layer.number, layer.datatype
    )
    if tech.tone == "clear":
        return drawn
    validate_field(field, drawn, tech)
    return invert_tone(field, drawn, tech, layer)


@dataclass(frozen=True)
class MaskExportResult:
    """Mask-scale geometry with the checks that ran on it."""

    geometry: list[Polygon]
    violations: list[Violation] = dataclass_field(default_factory=list)
    statistics: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)


def export_mask(
    post_opc: Sequence[Polygon],
    srafs: Sequence[Polygon],
    field: Sequence[Polygon],
    tech: TechConfig,
    layers: LayerMap | None = None,
    mask_layer_name: str = "POST_OPC",
    min_width_nm: float | None = None,
    min_space_nm: float | None = None,
) -> MaskExportResult:
    """Apply tone, scale to mask, and re-verify.

    MRC runs *again* here because inversion swaps width and space: a design
    that passes min-space at 1x can fail min-width once inverted. Violations
    from this pass are labelled so they are never confused with the 1x ones.
    """
    layers = layers or LayerMap.default()
    mask_layer = layers[mask_layer_name]

    written = apply_tone(field, post_opc, srafs, tech, mask_layer)
    scaled = scale_to_mask(written, tech)

    violations = list(run_structural_checks(scaled, tech))
    if min_width_nm is not None:
        violations.extend(check_min_width(scaled, min_width_nm * tech.magnification, tech))
    if min_space_nm is not None:
        violations.extend(check_min_space(scaled, min_space_nm * tech.magnification, tech))
    for violation in violations:
        violation.detail["pass"] = "post_inversion"

    return MaskExportResult(
        geometry=scaled,
        violations=violations,
        statistics={
            "tone": tech.tone,
            "magnification": tech.magnification,
            "polygon_count": len(scaled),
            "vertex_count": sum(p.vertex_count for p in scaled),
            "on_mask_grid": on_mask_grid(scaled, tech),
        },
    )


def write_mask_gds(
    path: Path | str,
    result: MaskExportResult,
    tech: TechConfig,
    layers: LayerMap | None = None,
    cell_name: str = "MASK",
) -> None:
    """Write mask-scale geometry as GDSII.

    Carries the production layer only: no debug layers, no overlays. This is
    reticle data, not an engineering view.
    """
    from masklayout.io.streams import write_gds

    layout = Layout(name="MASK", tech=tech, layers=layers or LayerMap.default())
    cell = layout.add(Cell(name=cell_name))
    cell.polygons.extend(result.geometry)
    write_gds(layout, path)
