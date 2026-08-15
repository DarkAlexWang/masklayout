"""M3 acceptance: author a layout from scratch and round-trip it."""

from pathlib import Path

from masklayout.config import TechConfig
from masklayout.io.streams import read_gds, write_gds
from masklayout.model.cell import Cell
from masklayout.model.layout import Layout
from masklayout.pcells.contacts import ContactParams, place_contact_array
from masklayout.pcells.shapes import LineEndParams, build_line_end
from masklayout.pcells.wires import BezierWireParams, build_bezier_wire


def test_authored_layout_round_trips_through_gds(tmp_path: Path) -> None:
    tech = TechConfig()
    layout = Layout(name="AUTHORED", tech=tech)
    top = layout.add(Cell(name="TOP"))

    top.polygons.extend(
        build_bezier_wire(
            BezierWireParams(
                control_points_um=((0.0, 0.0), (2.0, 3.0), (6.0, -3.0), (8.0, 0.0)),
                width_um=0.4,
            ),
            tech,
            10,
            0,
        )
    )
    top.polygons.extend(
        build_line_end(
            LineEndParams(centre_um=(8.0, 0.0), width_um=0.4, extension_um=0.05),
            tech,
            10,
            0,
        )
    )
    place_contact_array(
        layout,
        parent_cell="TOP",
        contact_cell_name="CONTACT",
        params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
        columns=3,
        rows=3,
        pitch_um=(0.5, 0.5),
        layer=12,
        origin_um=(10.0, 0.0),
    )

    path = tmp_path / "authored.gds"
    write_gds(layout, path)
    restored, report = read_gds(path)

    assert sorted(restored.cells) == ["CONTACT", "TOP"]
    assert restored.top_cells() == ["TOP"]
    assert restored.dependencies("TOP") == {"CONTACT"}
    # Nine placements, but one contact polygon: the array stayed hierarchical.
    assert report.polygon_count == 3
