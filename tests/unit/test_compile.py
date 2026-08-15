"""Compiling tessellated curves into grid-aligned integer polygons."""

import math

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.compile import compile_polyline
from masklayout.geometry.curves import circle_um
from masklayout.geometry.normalize import signed_area
from masklayout.model.geometry import Polygon


def test_compiled_polygon_is_grid_aligned_and_integer() -> None:
    tech = TechConfig(design_grid_nm=1.0, max_chord_error_nm=1.0)
    pts = circle_um((0.0, 0.0), 1.0, max_chord_error_um=tech.max_chord_error_nm / 1000.0)
    poly, _ = compile_polyline(pts, tech, layer=10)

    assert isinstance(poly, Polygon)
    assert poly.points.dtype == np.int64  # integer DBU is the grid guarantee


def test_report_separates_tessellation_error_from_grid_error() -> None:
    tech = TechConfig(design_grid_nm=1.0, max_chord_error_nm=1.0)
    pts = circle_um((0.0, 0.0), 5.0, max_chord_error_um=0.001)
    _, report = compile_polyline(pts, tech, layer=10)

    assert report.grid_error_nm == pytest.approx(math.sqrt(2.0) / 2.0)
    assert report.tessellation_error_nm >= 0.0
    assert report.total_error_nm == pytest.approx(
        report.tessellation_error_nm + report.grid_error_nm
    )


def test_report_budget_accounts_for_both_error_terms() -> None:
    # A budget of only max_chord_error_nm would be wrong: quantization adds up
    # to half the grid diagonal on top of it.
    tech = TechConfig(design_grid_nm=1.0, max_chord_error_nm=1.0)
    pts = circle_um((0.0, 0.0), 5.0, max_chord_error_um=0.001)
    _, report = compile_polyline(pts, tech, layer=10)

    assert report.budget_nm == pytest.approx(1.0 + math.sqrt(2.0) / 2.0)
    assert report.within_budget


def test_compile_removes_collinear_points() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    square_with_midpoints = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    poly, report = compile_polyline(square_with_midpoints, tech, layer=10)
    assert poly.vertex_count == 4
    assert report.vertex_count == 4


def test_compile_orients_counterclockwise() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    clockwise = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    poly, _ = compile_polyline(clockwise, tech, layer=10)
    assert signed_area(poly.points) > 0


def test_compile_rejects_a_self_intersecting_ring() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    bowtie = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="self-intersect"):
        compile_polyline(bowtie, tech, layer=10)


def test_compile_preserves_layer_and_datatype() -> None:
    tech = TechConfig(design_grid_nm=1.0)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    poly, _ = compile_polyline(square, tech, layer=11, datatype=5)
    assert (poly.layer, poly.datatype) == (11, 5)


@pytest.mark.parametrize("radius_um", [0.5, 2.0, 10.0, 50.0])
def test_compiled_circle_stays_within_budget_of_the_true_curve(radius_um: float) -> None:
    """The M2 acceptance criterion, measured against the analytic circle.

    This is the only test that closes the loop: tessellate, compile, then
    compare the finished integer polygon against the curve it came from.

    It also pins down why the budget is a sum. Measured deviation here runs
    1.18-1.50 nm at a 1.0 nm chord-error setting, so asserting against
    max_chord_error_nm alone would fail at every radius.
    """
    tech = TechConfig(design_grid_nm=1.0, max_chord_error_nm=1.0)
    pts = circle_um((0.0, 0.0), radius_um, max_chord_error_um=tech.max_chord_error_nm / 1000.0)
    poly, report = compile_polyline(pts, tech, layer=10)

    in_um = poly.points.astype(np.float64) * tech.precision_um
    midpoints = (in_um + np.roll(in_um, -1, axis=0)) / 2.0
    deviation_nm = float(np.max(np.abs(radius_um - np.linalg.norm(midpoints, axis=1)))) * 1000.0

    assert deviation_nm <= report.budget_nm
    assert deviation_nm > tech.max_chord_error_nm, (
        "if this ever passes, quantization stopped contributing and the "
        "two-term budget should be revisited"
    )
