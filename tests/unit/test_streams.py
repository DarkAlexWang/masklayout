"""Stream reading, writing, and round-trip fidelity."""

import math

import gdstk
import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.io._gdstk_bridge import library_to_layout
from masklayout.io.errors import GridMismatchError
from masklayout.model.cell import RectangularRepetition
from masklayout.model.layers import LayerMap


def _hierarchical_library() -> gdstk.Library:
    lib = gdstk.Library("LIB", unit=1e-6, precision=1e-9)
    leaf = lib.new_cell("LEAF")
    leaf.add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5), layer=10, datatype=0))
    leaf.add(gdstk.Label("tag", (0.5, 0.25), layer=12))
    top = lib.new_cell("TOP")
    top.add(gdstk.Reference(leaf, (5.0, 5.0), rotation=math.pi / 4, magnification=2.0))
    top.add(gdstk.Reference(leaf, (0.0, 0.0), columns=3, rows=2, spacing=(10.0, 10.0)))
    return lib


def test_read_preserves_hierarchy_and_reports_counts() -> None:
    layout, report = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    assert sorted(layout.cells) == ["LEAF", "TOP"]
    assert layout.top_cells() == ["TOP"]
    assert layout.dependencies("TOP") == {"LEAF"}
    assert report.cell_count == 2
    assert report.polygon_count == 1
    assert report.label_count == 1
    assert report.reference_count == 2
    assert report.paths_converted == 0


def test_read_converts_polygon_coordinates_to_dbu() -> None:
    layout, _ = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    poly = layout.cells["LEAF"].polygons[0]
    assert poly.points.dtype == np.int64
    assert poly.bounds_dbu == (0, 0, 1000, 500)
    assert poly.layer == 10


def test_read_preserves_reference_transform() -> None:
    layout, _ = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    ref = layout.cells["TOP"].references[0]
    assert ref.cell_name == "LEAF"
    assert ref.origin_dbu == (5000, 5000)
    assert ref.rotation_rad == pytest.approx(math.pi / 4)
    assert ref.magnification == pytest.approx(2.0)


def test_read_preserves_rectangular_repetition() -> None:
    layout, _ = library_to_layout(
        _hierarchical_library(), TechConfig(), LayerMap.default(), source="memory"
    )
    rep = layout.cells["TOP"].references[1].repetition
    assert isinstance(rep, RectangularRepetition)
    assert (rep.columns, rep.rows) == (3, 2)
    assert rep.spacing_dbu == (10000, 10000)
    assert rep.offsets_dbu().shape == (6, 2)


def test_read_converts_paths_to_polygons_and_counts_them() -> None:
    # gdstk writes FlexPath as BOUNDARY, so this must NOT go through a file.
    lib = gdstk.Library("LIB", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("WITH_PATH")
    cell.add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5), layer=10))
    cell.add(gdstk.FlexPath([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)], 0.2, layer=11))
    assert len(cell.paths) == 1, "in-memory cell must hold a real path"

    layout, report = library_to_layout(lib, TechConfig(), LayerMap.default(), source="memory")

    assert report.paths_converted == 1
    assert len(layout.cells["WITH_PATH"].polygons) == 2
    assert {p.layer for p in layout.cells["WITH_PATH"].polygons} == {10, 11}


def test_read_rejects_a_library_whose_grid_differs() -> None:
    lib = gdstk.Library("LIB", unit=1e-6, precision=2.5e-10)
    lib.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5), layer=10))
    with pytest.raises(GridMismatchError, match="design grid"):
        library_to_layout(lib, TechConfig(design_grid_nm=1.0), LayerMap.default(), "memory")
