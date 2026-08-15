"""Conversion between the typed model and gdstk.

This is one of exactly two modules permitted to import gdstk; see the design
document, section "The gdstk boundary". Nothing here may leak a gdstk type
into a public signature.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from masklayout.config import TechConfig
from masklayout.io.errors import GridMismatchError, OffGridCoordinateError

#: Maximum acceptable deviation, in grid units, before a coordinate counts as
#: off-grid. Generous enough for float64 division noise, tight enough to catch
#: a genuine half-grid coordinate.
_ON_GRID_TOLERANCE = 1e-6

#: GDSII user unit expected by this toolkit: 1 micrometre.
_EXPECTED_UNIT_M = 1e-6


def um_to_dbu(points_um: NDArray[np.float64], precision_um: float) -> NDArray[np.int64]:
    """Convert float micrometres to integer design database units."""
    values = np.asarray(points_um, dtype=np.float64)
    scaled = values / precision_um
    rounded = np.round(scaled)
    residue = np.abs(scaled - rounded)
    if residue.size and residue.max() > _ON_GRID_TOLERANCE:
        worst = np.unravel_index(int(np.argmax(residue)), residue.shape)
        raise OffGridCoordinateError(
            f"coordinate {float(values[worst])!r} um is not a multiple of the design "
            f"grid {precision_um} um (off by {float(residue.max())} grid units)"
        )
    return rounded.astype(np.int64)


def dbu_to_um(points_dbu: NDArray[np.int64], precision_um: float) -> NDArray[np.float64]:
    """Convert integer design database units to float micrometres."""
    return np.asarray(points_dbu, dtype=np.float64) * precision_um


def check_library_grid(unit: float, precision_m: float, tech: TechConfig) -> None:
    """Reject a stream whose grid differs from the configured design grid.

    The design forbids silently adopting a file's grid or resampling onto
    ours: target geometry is immutable, so a mismatch is an error.
    """
    if not np.isclose(unit, _EXPECTED_UNIT_M, rtol=0.0, atol=1e-18):
        raise GridMismatchError(
            f"unsupported user unit {unit!r} m; masklayout expects {_EXPECTED_UNIT_M!r} m"
        )
    if not np.isclose(precision_m, tech.precision_m, rtol=1e-12, atol=0.0):
        raise GridMismatchError(
            f"file database precision {precision_m!r} m does not match the configured "
            f"design grid {tech.precision_m!r} m "
            f"(design_grid_nm={tech.design_grid_nm}); "
            "set design_grid_nm to match the file, or convert the file"
        )
