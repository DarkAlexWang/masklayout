"""Layer map behaviour."""

import pytest
from pydantic import ValidationError

from masklayout.model.layers import Layer, LayerMap


def test_default_map_contains_the_engineering_layers() -> None:
    lm = LayerMap.default()
    assert lm["TARGET"] == Layer(number=10, datatype=0, name="TARGET")
    assert lm["POST_OPC"] == Layer(number=11, datatype=0, name="POST_OPC")
    assert lm["SRAF"] == Layer(number=12, datatype=0, name="SRAF")
    assert lm["DEBUG_SOURCE"] == Layer(number=200, datatype=0, name="DEBUG_SOURCE")
    assert lm["DEBUG_MARKERS"] == Layer(number=201, datatype=0, name="DEBUG_MARKERS")
    assert lm["OVERLAY_ADD"] == Layer(number=202, datatype=0, name="OVERLAY_ADD")
    assert lm["OVERLAY_REMOVE"] == Layer(number=203, datatype=0, name="OVERLAY_REMOVE")


def test_default_map_contains_field_layer_for_tone_inversion() -> None:
    # FIELD is required by design decision 5: tone inversion needs an explicit extent.
    assert LayerMap.default()["FIELD"] == Layer(number=20, datatype=0, name="FIELD")


def test_layer_is_frozen() -> None:
    layer = Layer(number=10, datatype=0, name="TARGET")
    with pytest.raises(ValidationError):
        layer.number = 11


def test_layer_number_out_of_gds_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Layer(number=70000, datatype=0, name="TOO_BIG")
    with pytest.raises(ValidationError):
        Layer(number=-1, datatype=0, name="NEGATIVE")


def test_duplicate_layer_datatype_pair_is_rejected_naming_both() -> None:
    with pytest.raises(ValidationError) as excinfo:
        LayerMap(
            layers={
                "TARGET": Layer(number=10, datatype=0, name="TARGET"),
                "SHADOW": Layer(number=10, datatype=0, name="SHADOW"),
            }
        )
    message = str(excinfo.value)
    assert "TARGET" in message
    assert "SHADOW" in message
    assert "10/0" in message


def test_unknown_layer_lookup_lists_known_layers() -> None:
    with pytest.raises(KeyError) as excinfo:
        LayerMap.default()["NOPE"]
    assert "TARGET" in str(excinfo.value)
