"""TechConfig validation."""

import pytest
from pydantic import ValidationError

from masklayout.config import TechConfig


def test_defaults_are_valid() -> None:
    tech = TechConfig()
    assert tech.design_grid_nm == 1.0
    assert tech.mask_grid_nm == 0.5
    assert tech.magnification == 4
    assert tech.tone == "clear"
    assert tech.fracture_vertex_limit == 4000


def test_config_is_frozen() -> None:
    tech = TechConfig()
    with pytest.raises(ValidationError):
        tech.design_grid_nm = 2.0


def test_precision_properties_convert_grid_correctly() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    assert tech.precision_um == pytest.approx(0.001)
    assert tech.precision_m == pytest.approx(1e-9)


def test_mrc_deburr_derives_from_half_the_design_grid() -> None:
    assert TechConfig(design_grid_nm=1.0).effective_mrc_deburr_nm == pytest.approx(0.5)
    # A finer grid needs a mask grid it divides exactly: 4 * 0.4 / 0.4 == 4.
    finer = TechConfig(design_grid_nm=0.4, mask_grid_nm=0.4)
    assert finer.effective_mrc_deburr_nm == pytest.approx(0.2)


def test_mrc_deburr_override_is_respected() -> None:
    tech = TechConfig(design_grid_nm=1.0, mrc_deburr_nm=0.25)
    assert tech.effective_mrc_deburr_nm == pytest.approx(0.25)


def test_mask_grid_multiple_violation_names_all_three_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TechConfig(design_grid_nm=1.0, mask_grid_nm=0.3, magnification=4)
    message = str(excinfo.value)
    assert "magnification" in message
    assert "design_grid_nm" in message
    assert "mask_grid_nm" in message


def test_mask_grid_multiple_tolerates_float_representation_error() -> None:
    # 4 * 0.1 / 0.05 evaluates to 8.000000000000002 in IEEE 754.
    # This is a legal configuration and must not be rejected.
    tech = TechConfig(design_grid_nm=0.1, mask_grid_nm=0.05, magnification=4)
    assert tech.design_grid_nm == 0.1


def test_segment_length_ordering_is_enforced() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TechConfig(min_segment_length_nm=20.0, max_segment_length_nm=10.0)
    assert "min_segment_length_nm" in str(excinfo.value)


def test_non_positive_grid_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechConfig(design_grid_nm=0.0)
