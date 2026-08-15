"""Spatial index over model polygons.

gdstk provides no spatial index, so this wraps shapely's STRtree. It owns the
float-micrometre shapely mirror of the integer model so no other module has
to convert between the two representations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from masklayout.model.geometry import Polygon

#: Ignore intersections closer than this to the ray origin; the origin sits on
#: the source edge, so it would otherwise register as a zero-distance hit.
_ORIGIN_EPSILON_UM = 1e-9


class SpatialIndex:
    """Bounding-box index with ray queries, in float micrometres."""

    def __init__(self, polygons: Sequence[Polygon], precision_um: float) -> None:
        self._precision_um = precision_um
        self._geometries = [
            ShapelyPolygon(polygon.points.astype(np.float64) * precision_um) for polygon in polygons
        ]
        self._tree = STRtree(self._geometries) if self._geometries else None

    @property
    def polygon_count(self) -> int:
        return len(self._geometries)

    def shapely_geometry(self, index: int) -> ShapelyPolygon:
        return self._geometries[index]

    def _query(self, geometry: LineString | ShapelyPolygon) -> list[int]:
        if self._tree is None:
            return []
        return [int(i) for i in self._tree.query(geometry)]

    def query_bbox(
        self, minx_um: float, miny_um: float, maxx_um: float, maxy_um: float
    ) -> list[int]:
        """Indices of polygons whose bounding boxes meet the given box."""
        return self._query(self._box(minx_um, miny_um, maxx_um, maxy_um))

    def query_ray(
        self,
        origin_um: tuple[float, float],
        direction_um: tuple[float, float],
        length_um: float,
    ) -> list[int]:
        """Indices of polygons whose bounding boxes meet the ray."""
        return self._query(self._ray(origin_um, direction_um, length_um))

    def covered_fraction(
        self, minx_um: float, miny_um: float, maxx_um: float, maxy_um: float
    ) -> float:
        """Fraction of the window's area covered by indexed polygons.

        This is pattern density in its usual sense — covered area over window
        area. Overlapping polygons are unioned first so coverage cannot exceed
        one by double-counting.
        """
        window = self._box(minx_um, miny_um, maxx_um, maxy_um)
        if window.area <= 0.0:
            return 0.0
        pieces = [window.intersection(self._geometries[index]) for index in self._query(window)]
        pieces = [piece for piece in pieces if not piece.is_empty]
        if not pieces:
            return 0.0
        return float(unary_union(pieces).area / window.area)

    @staticmethod
    def _box(minx_um: float, miny_um: float, maxx_um: float, maxy_um: float) -> ShapelyPolygon:
        return ShapelyPolygon(
            [
                (minx_um, miny_um),
                (maxx_um, miny_um),
                (maxx_um, maxy_um),
                (minx_um, maxy_um),
            ]
        )

    @staticmethod
    def _ray(
        origin_um: tuple[float, float],
        direction_um: tuple[float, float],
        length_um: float,
    ) -> LineString:
        norm = float(np.hypot(direction_um[0], direction_um[1]))
        if norm == 0.0:
            raise ValueError("ray direction must be non-zero")
        unit = (direction_um[0] / norm, direction_um[1] / norm)
        end = (origin_um[0] + unit[0] * length_um, origin_um[1] + unit[1] * length_um)
        return LineString([origin_um, end])

    def nearest_distance_um(
        self,
        origin_um: tuple[float, float],
        direction_um: tuple[float, float],
        max_length_um: float,
        exclude: int | None,
    ) -> float | None:
        """Distance along a ray to the first polygon boundary it meets.

        Returns None when the ray reaches ``max_length_um`` without hitting
        anything, which callers read as "no neighbour within range".
        """
        ray = self._ray(origin_um, direction_um, max_length_um)
        origin = Point(origin_um)
        best: float | None = None

        for candidate in self._query(ray):
            if candidate == exclude:
                continue
            crossing = ray.intersection(self._geometries[candidate].exterior)
            if crossing.is_empty:
                continue
            parts = crossing.geoms if hasattr(crossing, "geoms") else [crossing]
            for part in parts:
                distance = origin.distance(part)
                if distance <= _ORIGIN_EPSILON_UM:
                    continue
                if best is None or distance < best:
                    best = distance
        return best
