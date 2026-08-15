"""Technology configuration.

Every value here is a software default. None of them are foundry,
mask-writer, or process-node rules. All are configurable and are recorded
in generated manifests.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Tone = Literal["clear", "dark"]

_MULTIPLE_TOLERANCE = 1e-9


class TechConfig(BaseModel):
    """Geometry, grid, and export configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = "generic_mask_geometry_v1"

    # Grid and mask scale.
    design_grid_nm: float = Field(default=1.0, gt=0)
    mask_grid_nm: float = Field(default=0.5, gt=0)
    magnification: int = Field(default=4, ge=1)
    tone: Tone = "clear"

    # Curve tessellation.
    max_chord_error_nm: float = Field(default=1.0, gt=0)
    max_segment_length_nm: float = Field(default=10.0, gt=0)
    min_segment_length_nm: float = Field(default=1.0, gt=0)
    max_vertices_per_polygon: int = Field(default=4000, ge=4)

    # Cleanup thresholds.
    min_polygon_area_nm2: float = Field(default=4.0, ge=0)
    min_edge_length_nm: float = Field(default=1.0, ge=0)
    remove_collinear_tolerance_nm: float = Field(default=0.001, ge=0)

    # Export.
    fracture_vertex_limit: int = Field(default=4000, ge=4)

    # Verification. None means "derive from design_grid_nm".
    mrc_deburr_nm: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_mask_grid_multiple(self) -> TechConfig:
        """Scaling to mask scale must be an exact integer multiply.

        If this holds, export needs no re-snap and introduces no second
        grid-error term.
        """
        scaled = self.magnification * self.design_grid_nm
        ratio = scaled / self.mask_grid_nm
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=_MULTIPLE_TOLERANCE):
            raise ValueError(
                "magnification * design_grid_nm must be an exact multiple of mask_grid_nm: "
                f"magnification={self.magnification}, "
                f"design_grid_nm={self.design_grid_nm}, "
                f"mask_grid_nm={self.mask_grid_nm} "
                f"(gives {scaled} / {self.mask_grid_nm} = {ratio})"
            )
        return self

    @model_validator(mode="after")
    def _validate_segment_lengths(self) -> TechConfig:
        if self.min_segment_length_nm > self.max_segment_length_nm:
            raise ValueError(
                "min_segment_length_nm must not exceed max_segment_length_nm: "
                f"min_segment_length_nm={self.min_segment_length_nm}, "
                f"max_segment_length_nm={self.max_segment_length_nm}"
            )
        return self

    @property
    def effective_mrc_deburr_nm(self) -> float:
        """Deburr radius used by MRC.

        Compensates the quantization error accumulated by the erode/dilate
        round trip, which is a function of the grid — so it derives from the
        grid unless explicitly overridden.
        """
        if self.mrc_deburr_nm is None:
            return self.design_grid_nm / 2.0
        return self.mrc_deburr_nm

    @property
    def precision_um(self) -> float:
        """Design grid in micrometres, for gdstk boolean/offset/fracture."""
        return self.design_grid_nm / 1000.0

    @property
    def precision_m(self) -> float:
        """Design grid in metres, for the gdstk Library precision."""
        return self.design_grid_nm * 1e-9

    @property
    def remove_collinear_tolerance_um(self) -> float:
        """Collinearity tolerance in micrometres."""
        return self.remove_collinear_tolerance_nm / 1000.0
