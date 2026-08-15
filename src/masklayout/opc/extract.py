"""Walk target polygons into sites: edges, corners, and line ends."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from masklayout.geometry.normalize import signed_area
from masklayout.model.geometry import Polygon
from masklayout.opc.sites import CornerType, Site, SiteKind

#: Two edges count as antiparallel when their directions oppose within this
#: many degrees. Used to recognise the two long sides flanking a line end.
_ANTIPARALLEL_TOLERANCE_DEG = 30.0


def vertex_curvature_1_per_um(points_um: NDArray[np.float64]) -> NDArray[np.float64]:
    """Curvature at each vertex from the circumcircle of it and its neighbours.

    The design takes curvature numerically from the polyline rather than
    re-fitting an analytic curve (decision 10); this is that measurement.
    Reciprocal of the circumradius, zero where the three points are collinear.
    """
    values = np.asarray(points_um, dtype=np.float64)
    previous = np.roll(values, 1, axis=0)
    following = np.roll(values, -1, axis=0)

    a = np.linalg.norm(values - previous, axis=1)
    b = np.linalg.norm(following - values, axis=1)
    c = np.linalg.norm(following - previous, axis=1)

    cross = (values[:, 0] - previous[:, 0]) * (following[:, 1] - previous[:, 1]) - (
        values[:, 1] - previous[:, 1]
    ) * (following[:, 0] - previous[:, 0])
    area = np.abs(cross) / 2.0

    denominator = a * b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.where(denominator > 0.0, 4.0 * area / denominator, 0.0)
    return np.nan_to_num(curvature, nan=0.0, posinf=0.0)


def _corner_types(points_um: NDArray[np.float64]) -> list[CornerType]:
    """Convex or concave at each vertex, for a counterclockwise ring."""
    previous = np.roll(points_um, 1, axis=0)
    following = np.roll(points_um, -1, axis=0)
    incoming = points_um - previous
    outgoing = following - points_um
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    return ["convex" if value > 0 else "concave" if value < 0 else "none" for value in cross]


def _is_line_end(
    lengths: NDArray[np.float64],
    directions: NDArray[np.float64],
    index: int,
    ratio: float,
) -> bool:
    """True when edge ``index`` is short and flanked by two long antiparallel edges.

    This is a heuristic, not a definition: a chamfer between two long edges can
    satisfy it. ``ratio`` is a parameter so the misfire is tunable.
    """
    count = len(lengths)
    before = (index - 1) % count
    after = (index + 1) % count
    if lengths[index] > ratio * min(lengths[before], lengths[after]):
        return False
    dot = float(np.clip(np.dot(directions[before], directions[after]), -1.0, 1.0))
    return math.degrees(math.acos(dot)) >= 180.0 - _ANTIPARALLEL_TOLERANCE_DEG


def extract_sites(
    polygons: Sequence[Polygon],
    precision_um: float,
    line_end_ratio: float = 0.5,
) -> list[Site]:
    """Extract every edge, corner, and line end from the given polygons."""
    sites: list[Site] = []

    for polygon_index, polygon in enumerate(polygons):
        points = polygon.points.astype(np.float64) * precision_um
        if signed_area(points) < 0:
            points = points[::-1]

        following = np.roll(points, -1, axis=0)
        segments = following - points
        lengths = np.linalg.norm(segments, axis=1)
        safe = np.where(lengths > 0.0, lengths, 1.0)[:, None]
        directions = segments / safe
        # Right normal of a counterclockwise ring points out of the polygon.
        normals = np.column_stack((directions[:, 1], -directions[:, 0]))
        midpoints = (points + following) / 2.0
        curvature = vertex_curvature_1_per_um(points)
        corners = _corner_types(points)

        for i in range(len(points)):
            angle_deg = math.degrees(math.atan2(segments[i, 1], segments[i, 0])) % 360.0
            edge_kind: SiteKind = (
                "line_end" if _is_line_end(lengths, directions, i, line_end_ratio) else "edge"
            )
            corner_kind: SiteKind = "convex_corner" if corners[i] == "convex" else "concave_corner"
            sites.append(
                Site(
                    kind=edge_kind,
                    polygon_index=polygon_index,
                    vertex_index=i,
                    midpoint_um=(float(midpoints[i, 0]), float(midpoints[i, 1])),
                    outward_normal_um=(float(normals[i, 0]), float(normals[i, 1])),
                    edge_length_um=float(lengths[i]),
                    angle_deg=angle_deg,
                    corner_type="none",
                    curvature_1_per_um=0.0,
                )
            )
            sites.append(
                Site(
                    kind=corner_kind,
                    polygon_index=polygon_index,
                    vertex_index=i,
                    midpoint_um=(float(points[i, 0]), float(points[i, 1])),
                    outward_normal_um=(float(normals[i, 0]), float(normals[i, 1])),
                    edge_length_um=0.0,
                    angle_deg=angle_deg,
                    corner_type=corners[i],
                    curvature_1_per_um=float(curvature[i]),
                )
            )

    return sites
