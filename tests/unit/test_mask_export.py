"""Mask-data export: scaling, tone inversion, and re-verification."""

from pathlib import Path

import numpy as np
import pytest

from masklayout.config import TechConfig
from masklayout.geometry.normalize import signed_area
from masklayout.io.mask import (
    FieldMissingError,
    GeometryOutsideFieldError,
    apply_tone,
    export_mask,
    invert_tone,
    on_mask_grid,
    scale_to_mask,
    validate_field,
    write_mask_gds,
)
from masklayout.io.streams import read_gds
from masklayout.model.geometry import Polygon
from masklayout.model.layers import LayerMap

LAYERS = LayerMap.default()
POST_OPC = LAYERS["POST_OPC"]


def _rect(x0: int, y0: int, x1: int, y1: int, layer: int = 11) -> Polygon:
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int64)
    return Polygon(points=pts, layer=layer)


def _field() -> list[Polygon]:
    return [_rect(-1000, -1000, 6000, 2000, layer=20)]


def _bar() -> list[Polygon]:
    return [_rect(0, 0, 2000, 100)]


class TestScaling:
    def test_a_100nm_feature_becomes_400nm_at_4x(self) -> None:
        scaled = scale_to_mask(_bar(), TechConfig(magnification=4))
        _, y0, _, y1 = scaled[0].bounds_dbu
        assert (y1 - y0) == 400

    def test_scaling_is_an_exact_integer_multiply(self) -> None:
        tech = TechConfig(magnification=4)
        source = _bar()
        scaled = scale_to_mask(source, tech)
        assert np.array_equal(scaled[0].points, source[0].points * 4)
        assert scaled[0].points.dtype == np.int64

    def test_the_scaled_result_lands_on_the_mask_grid(self) -> None:
        """Guaranteed by the config validation, asserted here anyway."""
        tech = TechConfig()  # 1 nm design, 0.5 nm mask, 4x
        assert on_mask_grid(scale_to_mask(_bar(), tech), tech)

    def test_scaling_preserves_layer_and_datatype(self) -> None:
        scaled = scale_to_mask([_rect(0, 0, 10, 10, layer=11)], TechConfig())
        assert (scaled[0].layer, scaled[0].datatype) == (11, 0)


class TestField:
    def test_a_missing_field_is_rejected_naming_the_layer(self) -> None:
        with pytest.raises(FieldMissingError, match="FIELD"):
            validate_field([], _bar(), TechConfig())

    def test_geometry_inside_the_field_validates(self) -> None:
        validate_field(_field(), _bar(), TechConfig())

    def test_geometry_outside_the_field_is_an_error_not_a_clip(self) -> None:
        far = [_rect(50000, 50000, 52000, 50100)]
        with pytest.raises(GeometryOutsideFieldError, match="outside"):
            validate_field(_field(), far, TechConfig())

    def test_an_empty_layout_needs_no_containment_check(self) -> None:
        validate_field(_field(), [], TechConfig())


class TestTone:
    def test_clear_tone_writes_the_pattern_as_drawn(self) -> None:
        tech = TechConfig(tone="clear")
        written = apply_tone(_field(), _bar(), [], tech, POST_OPC)
        assert sum(abs(signed_area(p.points)) for p in written) == pytest.approx(
            sum(abs(signed_area(p.points)) for p in _bar())
        )

    def test_dark_tone_writes_the_field_minus_the_pattern(self) -> None:
        tech = TechConfig(tone="dark")
        written = apply_tone(_field(), _bar(), [], tech, POST_OPC)
        field_area = sum(abs(signed_area(p.points)) for p in _field())
        bar_area = sum(abs(signed_area(p.points)) for p in _bar())
        written_area = sum(abs(signed_area(p.points)) for p in written)
        assert written_area == pytest.approx(field_area - bar_area, rel=1e-9)

    def test_inverting_twice_returns_the_original(self) -> None:
        tech = TechConfig()
        once = invert_tone(_field(), _bar(), tech, POST_OPC)
        twice = invert_tone(_field(), once, tech, POST_OPC)
        assert sum(abs(signed_area(p.points)) for p in twice) == pytest.approx(
            sum(abs(signed_area(p.points)) for p in _bar()), rel=1e-9
        )

    def test_srafs_are_subtracted_with_the_main_pattern(self) -> None:
        """Not left as holes inside holes."""
        tech = TechConfig(tone="dark")
        srafs = [_rect(0, 300, 2000, 330, layer=12)]
        with_srafs = apply_tone(_field(), _bar(), srafs, tech, POST_OPC)
        without = apply_tone(_field(), _bar(), [], tech, POST_OPC)
        assert sum(abs(signed_area(p.points)) for p in with_srafs) < sum(
            abs(signed_area(p.points)) for p in without
        )

    def test_dark_tone_without_a_field_fails(self) -> None:
        with pytest.raises(FieldMissingError):
            apply_tone([], _bar(), [], TechConfig(tone="dark"), POST_OPC)

    def test_the_inverted_result_is_grid_aligned(self) -> None:
        written = apply_tone(_field(), _bar(), [], TechConfig(tone="dark"), POST_OPC)
        assert all(p.points.dtype == np.int64 for p in written)


class TestExport:
    def test_a_clear_tone_export_scales_and_verifies(self) -> None:
        result = export_mask(_bar(), [], _field(), TechConfig(tone="clear"))
        assert result.geometry
        assert result.statistics["magnification"] == 4
        assert result.statistics["on_mask_grid"] is True
        assert result.clean

    def test_a_dark_tone_export_inverts(self) -> None:
        clear = export_mask(_bar(), [], _field(), TechConfig(tone="clear"))
        dark = export_mask(_bar(), [], _field(), TechConfig(tone="dark"))
        clear_area = sum(abs(signed_area(p.points)) for p in clear.geometry)
        dark_area = sum(abs(signed_area(p.points)) for p in dark.geometry)
        assert dark_area > clear_area  # the field minus a small bar

    def test_post_inversion_violations_are_labelled(self) -> None:
        result = export_mask(_bar(), [], _field(), TechConfig(tone="dark"), min_width_nm=20.0)
        for violation in result.violations:
            assert violation.detail["pass"] == "post_inversion"

    def test_inversion_turns_a_space_problem_into_a_width_problem(self) -> None:
        """The design's reason for re-running MRC, demonstrated.

        Two bars 30 nm apart: at 1x that gap is a min-SPACE concern. After
        inversion the gap becomes drawn material 30 nm wide, so the same
        geometry is now a min-WIDTH concern.
        """
        tech_clear = TechConfig(tone="clear")
        tech_dark = TechConfig(tone="dark")
        pair = [_rect(0, 0, 2000, 100), _rect(0, 130, 2000, 230)]

        # Clear tone: the 30 nm gap is space, and min-width is satisfied.
        clear = export_mask(pair, [], _field(), tech_clear, min_width_nm=20.0)
        assert clear.clean

        # Dark tone: that same 30 nm gap is now a 30 nm wide written feature.
        # Check it against a rule it cannot meet.
        dark = export_mask(pair, [], _field(), tech_dark, min_width_nm=200.0)
        assert not dark.clean
        assert any(v.check == "min_width" for v in dark.violations)

    def test_the_mask_stream_carries_only_the_production_layer(self, tmp_path: Path) -> None:
        result = export_mask(_bar(), [], _field(), TechConfig(tone="clear"))
        path = tmp_path / "mask.gds"
        write_mask_gds(path, result, TechConfig())
        restored, _ = read_gds(path)
        layers_present = {p.layer for p in restored.cells["MASK"].polygons}
        assert layers_present == {11}
        assert 202 not in layers_present  # no overlays
        assert 201 not in layers_present  # no debug markers

    def test_the_mask_gds_is_byte_reproducible(self, tmp_path: Path) -> None:
        import hashlib

        digests = []
        for name in ("a.gds", "b.gds"):
            result = export_mask(_bar(), [], _field(), TechConfig(tone="dark"))
            path = tmp_path / name
            write_mask_gds(path, result, TechConfig())
            digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        assert digests[0] == digests[1]
