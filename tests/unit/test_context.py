"""GeomContext behaviour.

Test files may import gdstk directly; the import boundary applies to src/ only.
"""

import datetime
import hashlib
import math
from pathlib import Path

import gdstk
import numpy as np

from masklayout.config import TechConfig
from masklayout.geometry.context import GeomContext


def test_boolean_output_lands_exactly_on_the_design_grid() -> None:
    ctx = GeomContext(TechConfig(design_grid_nm=1.0))
    first = gdstk.rectangle((0.0, 0.0), (1.0, 0.2))
    first.rotate(math.radians(37))
    second = gdstk.rectangle((0.5, 0.0), (1.5, 0.2))
    second.rotate(math.radians(37))

    result = ctx.boolean(first, second, "or")

    assert result
    for polygon in result:
        scaled = np.asarray(polygon.points) / ctx.precision_um
        residue = np.abs(scaled - np.round(scaled)).max()
        assert residue < 1e-9, f"vertex off the design grid by {residue} grid units"


def test_write_gds_honours_the_configured_fracture_limit(tmp_path: Path) -> None:
    # Regression test: gdstk's write_gds defaults to max_points=199, which would
    # silently fracture this polygon regardless of fracture_vertex_limit.
    tech = TechConfig(fracture_vertex_limit=4000)
    ctx = GeomContext(tech)
    library = ctx.new_library("TOP")
    cell = library.new_cell("TOP")

    circle = gdstk.ellipse((0.0, 0.0), 10.0, tolerance=1e-4)
    assert len(circle.points) > 199, "test is meaningless unless it exceeds gdstk's default"
    assert len(circle.points) <= tech.fracture_vertex_limit
    cell.add(circle)

    out = tmp_path / "circle.gds"
    ctx.write_gds(library, out)

    read_back = gdstk.read_gds(out)
    cell_back = read_back.cells[0]
    assert isinstance(cell_back, gdstk.Cell)
    assert len(cell_back.polygons) == 1, (
        "polygon was fractured; gdstk's max_points=199 default leaked"
    )


def test_write_gds_is_byte_reproducible(tmp_path: Path) -> None:
    digests = []
    for name in ("first.gds", "second.gds"):
        ctx = GeomContext(TechConfig())
        library = ctx.new_library("TOP")
        cell = library.new_cell("TOP")
        cell.add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
        path = tmp_path / name
        ctx.write_gds(library, path)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert digests[0] == digests[1]


def test_write_gds_uses_the_pinned_timestamp(tmp_path: Path) -> None:
    pinned = datetime.datetime(2001, 2, 3, 4, 5, 6)
    ctx = GeomContext(TechConfig(), timestamp=pinned)
    library = ctx.new_library("TOP")
    library.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
    path = tmp_path / "stamped.gds"
    ctx.write_gds(library, path)

    other = GeomContext(TechConfig(), timestamp=datetime.datetime(1999, 1, 1))
    other_library = other.new_library("TOP")
    other_library.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
    other_path = tmp_path / "other.gds"
    other.write_gds(other_library, other_path)

    assert path.read_bytes() != other_path.read_bytes()


def test_write_oas_round_trips(tmp_path: Path) -> None:
    ctx = GeomContext(TechConfig())
    library = ctx.new_library("TOP")
    library.new_cell("TOP").add(gdstk.rectangle((0.0, 0.0), (1.0, 0.5)))
    path = tmp_path / "out.oas"
    ctx.write_oas(library, path)

    read_back = gdstk.read_oas(path)
    assert [cell.name for cell in read_back.cells] == ["TOP"]
    cell_back = read_back.cells[0]
    assert isinstance(cell_back, gdstk.Cell)
    assert len(cell_back.polygons) == 1


def test_fracture_uses_the_configured_limit() -> None:
    ctx = GeomContext(TechConfig(fracture_vertex_limit=100))
    circle = gdstk.ellipse((0.0, 0.0), 10.0, tolerance=1e-4)
    assert len(circle.points) > 100

    pieces = ctx.fracture(circle)

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece.points) <= 100


def test_new_library_uses_the_configured_precision() -> None:
    ctx = GeomContext(TechConfig(design_grid_nm=1.0))
    library = ctx.new_library("TOP")
    assert library.precision == 1e-9
    assert library.unit == 1e-6
