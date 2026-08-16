"""Generated features and their provenance."""

import json

import numpy as np
import pytest

from masklayout.model.geometry import Polygon
from masklayout.opc.feature import Feature


def _square(size: int = 100) -> Polygon:
    pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.int64)
    return Polygon(points=pts, layer=11)


def _feature(**overrides: object) -> Feature:
    defaults: dict[str, object] = {
        "id": "hammerhead@0#1:line_end",
        "kind": "hammerhead",
        "polygons": [_square()],
        "source_site_id": "0#1:line_end",
        "rule_id": "hh_isolated_line_end",
        "deck_id": "generic_hammerhead_v1",
        "deck_version": "1.0.0",
        "deck_hash": "abc123",
        "parameters": {"extension_um": 0.028},
    }
    defaults.update(overrides)
    return Feature(**defaults)  # type: ignore[arg-type]


def test_feature_reports_its_vertex_count() -> None:
    assert _feature().vertex_count == 4


def test_feature_defaults_to_additive_polarity() -> None:
    assert _feature().polarity == "add"


def test_feature_can_be_subtractive() -> None:
    assert _feature(polarity="subtract").polarity == "subtract"


def test_provenance_names_the_source_rule_and_deck() -> None:
    provenance = _feature().provenance()
    assert provenance["source_site_id"] == "0#1:line_end"
    assert provenance["rule_id"] == "hh_isolated_line_end"
    assert provenance["deck_id"] == "generic_hammerhead_v1"
    assert provenance["deck_version"] == "1.0.0"
    assert provenance["deck_hash"] == "abc123"
    assert provenance["parameters"] == {"extension_um": 0.028}


def test_provenance_is_json_serialisable() -> None:
    """It has to survive into the manifest, which is JSON."""
    restored = json.loads(json.dumps(_feature().provenance()))
    assert restored["kind"] == "hammerhead"


def test_provenance_carries_no_geometry() -> None:
    """Geometry belongs on a layer; provenance is a record about it."""
    assert "polygons" not in _feature().provenance()


def test_feature_is_frozen() -> None:
    feature = _feature()
    with pytest.raises(AttributeError):
        feature.kind = "serif"  # type: ignore[misc]
