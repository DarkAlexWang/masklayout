"""Contact PCell and hierarchical array placement."""

import pytest

from masklayout.config import TechConfig
from masklayout.model.cell import Cell, RectangularRepetition
from masklayout.model.layout import Layout
from masklayout.pcells.contacts import ContactParams, build_contact, place_contact_array


def test_contact_is_centred_on_its_origin() -> None:
    polys = build_contact(
        ContactParams(centre_um=(1.0, 1.0), size_um=(0.2, 0.2)), TechConfig(), 12, 0
    )
    assert polys[0].bounds_dbu == (900, 900, 1100, 1100)


def test_contact_with_a_corner_radius_has_more_vertices() -> None:
    tech = TechConfig()
    square = build_contact(ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)), tech, 12, 0)
    rounded = build_contact(
        ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2), corner_radius_um=0.05),
        tech,
        12,
        0,
    )
    assert rounded[0].vertex_count > square[0].vertex_count


def test_contact_rejects_a_radius_larger_than_half_its_size() -> None:
    with pytest.raises(ValueError, match="radius"):
        build_contact(
            ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2), corner_radius_um=0.2),
            TechConfig(),
            12,
            0,
        )


def test_array_placement_preserves_hierarchy() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP"))

    reference = place_contact_array(
        layout,
        parent_cell="TOP",
        contact_cell_name="CONTACT",
        params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
        columns=4,
        rows=3,
        pitch_um=(0.5, 0.5),
        layer=12,
    )

    # A cell plus a placement, not 12 flattened copies.
    assert "CONTACT" in layout.cells
    assert len(layout.cells["CONTACT"].polygons) == 1
    assert layout.cells["TOP"].polygons == []
    assert layout.cells["TOP"].references == [reference]
    assert layout.dependencies("TOP") == {"CONTACT"}


def test_array_repetition_describes_the_grid() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP"))
    reference = place_contact_array(
        layout,
        parent_cell="TOP",
        contact_cell_name="CONTACT",
        params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
        columns=4,
        rows=3,
        pitch_um=(0.5, 0.25),
        layer=12,
    )
    rep = reference.repetition
    assert isinstance(rep, RectangularRepetition)
    assert (rep.columns, rep.rows) == (4, 3)
    assert rep.spacing_dbu == (500, 250)
    assert rep.offsets_dbu().shape == (12, 2)


def test_array_reuses_an_existing_contact_cell() -> None:
    layout = Layout(name="LIB")
    layout.add(Cell(name="TOP"))
    params = ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2))
    for _ in range(2):
        place_contact_array(
            layout,
            parent_cell="TOP",
            contact_cell_name="CONTACT",
            params=params,
            columns=2,
            rows=2,
            pitch_um=(0.5, 0.5),
            layer=12,
        )
    assert len(layout.cells) == 2  # TOP and CONTACT, not TOP and two contacts
    assert len(layout.cells["TOP"].references) == 2


def test_array_rejects_an_unknown_parent_cell() -> None:
    layout = Layout(name="LIB")
    with pytest.raises(KeyError, match="NOPE"):
        place_contact_array(
            layout,
            parent_cell="NOPE",
            contact_cell_name="CONTACT",
            params=ContactParams(centre_um=(0.0, 0.0), size_um=(0.2, 0.2)),
            columns=2,
            rows=2,
            pitch_um=(0.5, 0.5),
            layer=12,
        )
