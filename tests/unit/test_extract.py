"""Site extraction from target geometry."""

import math

import numpy as np
import pytest
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon

from masklayout.geometry.curves import circle_um
from masklayout.model.geometry import Polygon
from masklayout.opc.extract import extract_sites, vertex_curvature_1_per_um


def _bar_dbu(length_nm: int = 2000, width_nm: int = 100, angle_deg: float = 0.0) -> Polygon:
    pts = np.array([[0, 0], [length_nm, 0], [length_nm, width_nm], [0, width_nm]], dtype=np.float64)
    a = math.radians(angle_deg)
    rotated = np.column_stack(
        (
            pts[:, 0] * math.cos(a) - pts[:, 1] * math.sin(a),
            pts[:, 0] * math.sin(a) + pts[:, 1] * math.cos(a),
        )
    )
    return Polygon(points=np.round(rotated).astype(np.int64), layer=10)


def test_a_rectangle_yields_four_edges_and_four_convex_corners() -> None:
    sites = extract_sites([_bar_dbu()], precision_um=0.001, line_end_ratio=0.5)
    edges = [s for s in sites if s.kind in ("edge", "line_end")]
    corners = [s for s in sites if s.kind.endswith("corner")]
    assert len(edges) == 4
    assert len(corners) == 4
    assert all(s.corner_type == "convex" for s in corners)


def test_the_short_edges_of_a_bar_are_line_ends() -> None:
    sites = extract_sites([_bar_dbu()], precision_um=0.001, line_end_ratio=0.5)
    line_ends = [s for s in sites if s.kind == "line_end"]
    assert len(line_ends) == 2
    for site in line_ends:
        assert site.edge_length_um == pytest.approx(0.1)


@pytest.mark.parametrize("angle_deg", [0.0, 17.0, 37.0, 45.0])
def test_line_end_detection_is_angle_independent(angle_deg: float) -> None:
    sites = extract_sites([_bar_dbu(angle_deg=angle_deg)], precision_um=0.001, line_end_ratio=0.5)
    assert len([s for s in sites if s.kind == "line_end"]) == 2


def test_edge_angle_follows_rotation() -> None:
    sites = extract_sites([_bar_dbu(angle_deg=30.0)], precision_um=0.001, line_end_ratio=0.5)
    long_edges = sorted(
        (s for s in sites if s.kind in ("edge", "line_end")),
        key=lambda s: s.edge_length_um,
        reverse=True,
    )[:2]
    angles = sorted(a.angle_deg % 180.0 for a in long_edges)
    assert angles[0] == pytest.approx(30.0, abs=0.5)


def test_outward_normal_points_away_from_the_polygon() -> None:
    polygon = _bar_dbu()
    shape = ShapelyPolygon(polygon.points.astype(np.float64) * 0.001)
    for site in extract_sites([polygon], precision_um=0.001, line_end_ratio=0.5):
        if site.kind not in ("edge", "line_end"):
            continue
        probe = (
            site.midpoint_um[0] + site.outward_normal_um[0] * 1e-4,
            site.midpoint_um[1] + site.outward_normal_um[1] * 1e-4,
        )
        assert not shape.contains(Point(probe))


def test_concave_corners_are_detected() -> None:
    # An L shape has exactly one concave corner.
    pts = np.array(
        [[0, 0], [3000, 0], [3000, 1000], [1000, 1000], [1000, 3000], [0, 3000]],
        dtype=np.int64,
    )
    sites = extract_sites([Polygon(points=pts, layer=10)], precision_um=0.001, line_end_ratio=0.5)
    assert len([s for s in sites if s.corner_type == "concave"]) == 1


def test_curvature_of_a_circle_is_one_over_its_radius() -> None:
    radius = 2.0
    points = circle_um((0.0, 0.0), radius, max_chord_error_um=0.0005)
    curvature = vertex_curvature_1_per_um(points)
    assert float(np.median(curvature)) == pytest.approx(1.0 / radius, rel=0.02)


def test_curvature_of_a_straight_run_is_zero() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert float(np.max(np.abs(vertex_curvature_1_per_um(points)[1:-1]))) < 1e-9


def test_site_ids_are_unique_and_stable() -> None:
    polygons = [_bar_dbu(), _bar_dbu(angle_deg=20.0)]
    first = [s.site_id for s in extract_sites(polygons, 0.001, 0.5)]
    second = [s.site_id for s in extract_sites(polygons, 0.001, 0.5)]
    assert first == second
    assert len(set(first)) == len(first)
