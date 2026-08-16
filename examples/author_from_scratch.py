"""Author a layout from scratch and export it.

Run:
    uv run python examples/author_from_scratch.py

Writes engineering GDSII and OASIS into examples/out/ and prints a summary.
Everything here works today; no OPC correction geometry is involved.
"""

from __future__ import annotations

from pathlib import Path

from masklayout.config import TechConfig
from masklayout.io.streams import read_gds, write_gds, write_oas
from masklayout.model.cell import Cell
from masklayout.model.layout import Layout
from masklayout.pcells.contacts import ContactParams, place_contact_array
from masklayout.pcells.shapes import LineEndParams, build_line_end
from masklayout.pcells.wires import (
    BezierWireParams,
    TaperedWireParams,
    build_bezier_wire,
    build_tapered_wire,
)

OUT = Path(__file__).parent / "out"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tech = TechConfig()
    print(f"technology      : {tech.name}")
    print(f"design grid     : {tech.design_grid_nm} nm")
    print(f"mask grid x mag : {tech.mask_grid_nm} nm x {tech.magnification}")
    print(f"chord error     : {tech.max_chord_error_nm} nm")
    print(f"MRC deburr      : {tech.effective_mrc_deburr_nm} nm (derived from the grid)")
    print()

    layout = Layout(name="AUTHORED", tech=tech)
    top = layout.add(Cell(name="TOP"))

    # A non-Manhattan curvilinear wire.
    wire = build_bezier_wire(
        BezierWireParams(
            control_points_um=((0.0, 0.0), (2.0, 3.0), (6.0, -3.0), (8.0, 0.0)),
            width_um=0.4,
        ),
        tech,
        layer=10,
        datatype=0,
    )
    top.polygons.extend(wire)
    print(f"bezier wire     : {wire[0].vertex_count} vertices")

    # A wire that narrows along its length.
    taper = build_tapered_wire(
        TaperedWireParams(
            control_points_um=((0.0, 2.0), (3.0, 2.0), (7.0, 2.0), (10.0, 2.0)),
            start_width_um=0.8,
            end_width_um=0.2,
        ),
        tech,
        layer=10,
        datatype=0,
    )
    top.polygons.extend(taper)
    _, y0, _, y1 = taper[0].bounds_dbu
    print(f"tapered wire    : {taper[0].vertex_count} vertices, {y1 - y0} nm tall at its widest")

    # A line-end cap, built edge-local so it works at any angle.
    cap = build_line_end(
        LineEndParams(centre_um=(8.0, 0.0), width_um=0.4, extension_um=0.05),
        tech,
        layer=10,
        datatype=0,
    )
    top.polygons.extend(cap)
    print(f"line end        : {cap[0].vertex_count} vertices")

    # A contact array, placed as hierarchy rather than flattened copies.
    place_contact_array(
        layout,
        parent_cell="TOP",
        contact_cell_name="CONTACT",
        params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
        columns=4,
        rows=3,
        pitch_um=(0.5, 0.5),
        layer=12,
        origin_um=(12.0, 0.0),
    )
    print("contact array   : 4 x 3 = 12 placements, stored as 1 cell + 1 reference")
    print()

    gds_path = OUT / "authored.gds"
    oas_path = OUT / "authored.oas"
    write_gds(layout, gds_path)
    write_oas(layout, oas_path)
    print(f"wrote {gds_path.name}  ({gds_path.stat().st_size} bytes)")
    print(f"wrote {oas_path.name}  ({oas_path.stat().st_size} bytes)")
    print()

    restored, report = read_gds(gds_path)
    print("read back:")
    print(f"  {report.summary()}")
    print(f"  top cells    : {restored.top_cells()}")
    print(f"  hierarchy    : TOP depends on {sorted(restored.dependencies('TOP'))}")
    print(f"  depth        : {restored.depth()}")
    print()
    print("Note the polygon count: the 12-placement array contributes ONE polygon,")
    print("because it round-tripped as hierarchy rather than being flattened.")


if __name__ == "__main__":
    main()
