"""Coordinate conversion between float micrometres and integer DBU."""

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.io._gdstk_bridge import check_library_grid, dbu_to_um, um_to_dbu
from masklayout.io.errors import GridMismatchError, OffGridCoordinateError


def test_um_to_dbu_is_exact_on_grid() -> None:
    points = np.array([[0.0, 0.001], [1.5, -0.25]], dtype=np.float64)
    result = um_to_dbu(points, precision_um=0.001)
    assert result.dtype == np.int64
    assert result.tolist() == [[0, 1], [1500, -250]]


def test_um_to_dbu_tolerates_float_representation_error() -> None:
    # 0.3 / 0.001 evaluates to 299.99999999999994 in IEEE 754.
    result = um_to_dbu(np.array([[0.3, 0.7]], dtype=np.float64), precision_um=0.001)
    assert result.tolist() == [[300, 700]]


def test_um_to_dbu_rejects_genuinely_off_grid_coordinates() -> None:
    with pytest.raises(OffGridCoordinateError, match=r"0\.0005"):
        um_to_dbu(np.array([[0.0005, 0.0]], dtype=np.float64), precision_um=0.001)


def test_dbu_to_um_round_trips() -> None:
    original = np.array([[0, 1], [1500, -250]], dtype=np.int64)
    back = um_to_dbu(dbu_to_um(original, precision_um=0.001), precision_um=0.001)
    assert np.array_equal(back, original)


def test_check_library_grid_accepts_matching_precision() -> None:
    check_library_grid(unit=1e-6, precision_m=1e-9, tech=TechConfig(design_grid_nm=1.0))


def test_check_library_grid_rejects_mismatch_naming_both() -> None:
    with pytest.raises(GridMismatchError) as excinfo:
        check_library_grid(unit=1e-6, precision_m=2.5e-10, tech=TechConfig(design_grid_nm=1.0))
    message = str(excinfo.value)
    assert "2.5e-10" in message
    assert "1e-09" in message


def test_check_library_grid_rejects_unexpected_unit() -> None:
    with pytest.raises(GridMismatchError, match="unit"):
        check_library_grid(unit=1e-3, precision_m=1e-9, tech=TechConfig(design_grid_nm=1.0))
