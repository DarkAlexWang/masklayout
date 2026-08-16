"""Structural and mask-rule checks."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon
from masklayout.verify.mrc import MRC_SENSITIVITY_NOTE, check_min_space, check_min_width
from masklayout.verify.structural import (
    check_grid_alignment,
    check_min_area,
    check_min_edge_length,
    check_simple,
    check_vertex_limit,
    run_structural_checks,
)

TECH = TechConfig()


def _rect(x0: int, y0: int, x1: int, y1: int) -> Polygon:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int64)
    return Polygon(points=pts, layer=11)


def _rotated(points: np.ndarray, deg: float = 31.0) -> Polygon:
    a = math.radians(deg)
    rotated = np.column_stack(
        (
            points[:, 0] * math.cos(a) - points[:, 1] * math.sin(a),
            points[:, 0] * math.sin(a) + points[:, 1] * math.cos(a),
        )
    )
    return Polygon(points=np.round(rotated).astype(np.int64), layer=11)


def _clean_bar() -> Polygon:
    return _rotated(np.array([[0, 0], [1000, 0], [1000, 100], [0, 100]], dtype=np.float64))


def _necked_bar() -> Polygon:
    """A bar with an 8 nm neck — well below a 20 nm minimum width."""
    return _rotated(
        np.array(
            [[0, 0], [1000, 0], [1000, 100], [520, 100], [500, 8], [480, 100], [0, 100]],
            dtype=np.float64,
        )
    )


class TestStructural:
    def test_a_good_polygon_passes_everything(self) -> None:
        assert run_structural_checks([_rect(0, 0, 2000, 100)], TECH) == []

    def test_grid_alignment_accepts_integer_coordinates(self) -> None:
        assert check_grid_alignment([_rect(0, 0, 100, 100)], TECH) == []

    def test_self_intersection_is_caught(self) -> None:
        bowtie = Polygon(
            points=np.array([[0, 0], [100, 100], [100, 0], [0, 100]], dtype=np.int64), layer=11
        )
        violations = check_simple([bowtie], TECH)
        assert len(violations) == 1
        assert violations[0].check == "self_intersection"
        assert violations[0].polygons == [bowtie]

    def test_a_sliver_below_min_area_is_caught(self) -> None:
        sliver = _rect(0, 0, 1, 1)  # 1 nm^2, below the 4 nm^2 default
        violations = check_min_area([sliver], TECH)
        assert len(violations) == 1
        assert violations[0].detail["area_nm2"] == pytest.approx(1.0)

    def test_a_healthy_polygon_passes_min_area(self) -> None:
        assert check_min_area([_rect(0, 0, 100, 100)], TECH) == []

    def test_a_short_edge_is_flagged_as_a_warning(self) -> None:
        # A 4-sided shape with one sub-nanometre edge is impossible on an
        # integer grid, so use a polygon with a repeated-ish short run.
        pts = np.array([[0, 0], [1000, 0], [1000, 100], [999, 100], [0, 100]], dtype=np.int64)
        tech = TechConfig(min_edge_length_nm=5.0)
        violations = check_min_edge_length([Polygon(points=pts, layer=11)], tech)
        assert len(violations) == 1
        assert violations[0].severity == "warning"
        assert "min_segment_length_nm" in violations[0].message

    def test_the_vertex_limit_is_enforced_at_the_configured_value(self) -> None:
        tech = TechConfig(fracture_vertex_limit=10)
        big = Polygon(
            points=np.column_stack(
                (np.arange(20, dtype=np.int64) * 10, np.zeros(20, dtype=np.int64))
            ),
            layer=11,
        )
        violations = check_vertex_limit([big], tech)
        assert len(violations) == 1
        assert violations[0].detail["vertex_count"] == 20

    def test_a_polygon_at_the_limit_passes(self) -> None:
        tech = TechConfig(fracture_vertex_limit=4)
        assert check_vertex_limit([_rect(0, 0, 100, 100)], tech) == []


class TestMRC:
    """The half-grid deburr, both halves.

    A test that only checked "the defect is found" would pass on the
    unusable version, where a clean bar also reports violations.
    """

    def test_a_clean_rotated_bar_reports_no_width_violation(self) -> None:
        assert check_min_width([_clean_bar()], 20.0, TECH) == []

    def test_a_necked_bar_reports_a_width_violation(self) -> None:
        violations = check_min_width([_necked_bar()], 20.0, TECH)
        assert violations
        assert violations[0].check == "min_width"
        assert violations[0].polygons

    def test_without_the_deburr_the_clean_bar_falsely_fails(self) -> None:
        """This is why the deburr exists; deleting it must break a test.

        Design section 6.2 measured a clean 31-degree bar reporting MORE
        violation area than a genuinely defective one.
        """
        noisy = check_min_width([_clean_bar()], 20.0, TECH, deburr=False)
        assert noisy, "without the deburr, quantization noise must show up"
        assert sum(len(v.polygons) for v in noisy) > 0

    def test_the_deburr_removes_the_noise_but_keeps_the_defect(self) -> None:
        clean_regions = sum(len(v.polygons) for v in check_min_width([_clean_bar()], 20.0, TECH))
        neck_regions = sum(len(v.polygons) for v in check_min_width([_necked_bar()], 20.0, TECH))
        assert clean_regions == 0
        assert neck_regions >= 1

    def test_two_bars_closer_than_min_space_are_flagged(self) -> None:
        close = [_rect(0, 0, 2000, 100), _rect(0, 110, 2000, 210)]  # 10 nm gap
        violations = check_min_space(close, 20.0, TECH)
        assert violations
        assert violations[0].check == "min_space"

    def test_comfortably_spaced_bars_are_not_flagged(self) -> None:
        far = [_rect(0, 0, 2000, 100), _rect(0, 400, 2000, 500)]  # 300 nm gap
        assert check_min_space(far, 20.0, TECH) == []

    def test_empty_input_produces_no_violations(self) -> None:
        assert check_min_width([], 20.0, TECH) == []
        assert check_min_space([], 20.0, TECH) == []

    def test_the_sensitivity_floor_is_stated_not_implied(self) -> None:
        assert "sensitivity floor" in MRC_SENSITIVITY_NOTE
        assert "grid" in MRC_SENSITIVITY_NOTE
