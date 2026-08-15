"""PCell registry and parameter validation."""

import numpy as np
import pytest
from pydantic import ValidationError

from masklayout.config import TechConfig
from masklayout.pcells.base import (
    UnknownPCellError,
    build_pcell,
    registered_names,
)


def test_registry_exposes_the_m3_pcells() -> None:
    names = registered_names()
    for expected in ("bezier_wire", "tapered_wire", "line_end", "contact", "rounded_rect"):
        assert expected in names


def test_registered_names_is_sorted_for_determinism() -> None:
    assert registered_names() == sorted(registered_names())


def test_build_by_name_accepts_a_plain_params_dict() -> None:
    # This is exactly the path M4's rule deck will take.
    polygons = build_pcell(
        "rounded_rect",
        {"lower_um": (0.0, 0.0), "upper_um": (2.0, 1.0), "radius_um": 0.2},
        TechConfig(),
        layer=10,
    )
    assert polygons
    assert polygons[0].points.dtype == np.int64
    assert polygons[0].layer == 10


def test_unknown_pcell_lists_the_known_ones() -> None:
    with pytest.raises(UnknownPCellError) as excinfo:
        build_pcell("no_such_pcell", {}, TechConfig(), layer=10)
    assert "rounded_rect" in str(excinfo.value)


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_pcell(
            "rounded_rect",
            {
                "lower_um": (0.0, 0.0),
                "upper_um": (2.0, 1.0),
                "radius_um": 0.2,
                "bogus": 1,
            },
            TechConfig(),
            layer=10,
        )


def test_params_are_frozen() -> None:
    from masklayout.pcells.shapes import RoundedRectParams

    params = RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(2.0, 1.0), radius_um=0.2)
    with pytest.raises(ValidationError):
        params.radius_um = 0.5


def test_params_round_trip_through_json() -> None:
    from masklayout.pcells.shapes import RoundedRectParams

    original = RoundedRectParams(lower_um=(0.0, 0.0), upper_um=(2.0, 1.0), radius_um=0.2)
    assert RoundedRectParams.model_validate_json(original.model_dump_json()) == original
