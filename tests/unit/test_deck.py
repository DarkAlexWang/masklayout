"""Rule deck schema, loading, validation, and hashing."""

import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from masklayout.opc.classify import SELECTOR_KEYS
from masklayout.opc.deck import (
    Range,
    Selector,
    UnknownSelectorError,
    load_deck,
    load_deck_from_mapping,
)

_MINIMAL: dict[str, Any] = {
    "deck": {"id": "test_deck", "version": "1.0.0"},
    "rules": [
        {
            "id": "hh_isolated",
            "priority": 10,
            "kind": "hammerhead",
            "when": {
                "site": "line_end",
                "width_nm": {"min": 18, "max": 26},
                "space_nm": {"min": 120},
            },
            "apply": {"pcell": "line_end", "params": {"extension_um": 0.028}},
        }
    ],
}


def test_selector_fields_are_exactly_the_classify_vocabulary() -> None:
    """The deck and the classifier must agree on what is selectable."""
    assert set(Selector.model_fields) == SELECTOR_KEYS


def test_a_minimal_deck_loads() -> None:
    deck = load_deck_from_mapping(_MINIMAL)
    assert deck.id == "test_deck"
    assert deck.version == "1.0.0"
    assert len(deck.rules) == 1
    assert deck.rules[0].apply.pcell == "line_end"


def test_an_unknown_selector_key_is_rejected_listing_the_valid_ones() -> None:
    bad = {
        "deck": {"id": "d", "version": "1.0.0"},
        "rules": [
            {
                "id": "r",
                "priority": 1,
                "kind": "hammerhead",
                "when": {"site": "line_end", "widht_nm": {"min": 1}},
                "apply": {"pcell": "line_end"},
            }
        ],
    }
    with pytest.raises(UnknownSelectorError) as excinfo:
        load_deck_from_mapping(bad)
    message = str(excinfo.value)
    assert "widht_nm" in message
    assert "width_nm" in message


def test_duplicate_rule_ids_are_rejected() -> None:
    duplicated = {
        "deck": {"id": "d", "version": "1.0.0"},
        "rules": [
            {"id": "same", "priority": 1, "kind": "a", "when": {}, "apply": {"pcell": "p"}},
            {"id": "same", "priority": 2, "kind": "b", "when": {}, "apply": {"pcell": "p"}},
        ],
    }
    with pytest.raises(ValidationError, match="same"):
        load_deck_from_mapping(duplicated)


def test_rules_are_returned_in_priority_order() -> None:
    mapping = {
        "deck": {"id": "d", "version": "1.0.0"},
        "rules": [
            {"id": "late", "priority": 30, "kind": "a", "when": {}, "apply": {"pcell": "p"}},
            {"id": "early", "priority": 10, "kind": "a", "when": {}, "apply": {"pcell": "p"}},
            {"id": "mid", "priority": 20, "kind": "a", "when": {}, "apply": {"pcell": "p"}},
        ],
    }
    ordered = [r.id for r in load_deck_from_mapping(mapping).rules_in_priority_order()]
    assert ordered == ["early", "mid", "late"]


def test_equal_priorities_break_ties_by_id_for_determinism() -> None:
    mapping = {
        "deck": {"id": "d", "version": "1.0.0"},
        "rules": [
            {"id": "zebra", "priority": 10, "kind": "a", "when": {}, "apply": {"pcell": "p"}},
            {"id": "alpha", "priority": 10, "kind": "a", "when": {}, "apply": {"pcell": "p"}},
        ],
    }
    ordered = [r.id for r in load_deck_from_mapping(mapping).rules_in_priority_order()]
    assert ordered == ["alpha", "zebra"]


def test_content_hash_changes_when_a_threshold_changes() -> None:
    baseline = load_deck_from_mapping(_MINIMAL).content_hash
    altered = {
        "deck": _MINIMAL["deck"],
        "rules": [
            {
                **_MINIMAL["rules"][0],
                "when": {
                    "site": "line_end",
                    "width_nm": {"min": 19, "max": 26},
                    "space_nm": {"min": 120},
                },
            }
        ],
    }
    assert load_deck_from_mapping(altered).content_hash != baseline


def test_content_hash_is_stable_across_key_order() -> None:
    reordered = {
        "rules": _MINIMAL["rules"],
        "deck": _MINIMAL["deck"],
    }
    assert load_deck_from_mapping(reordered).content_hash == (
        load_deck_from_mapping(_MINIMAL).content_hash
    )


def test_the_shipped_deck_loads_from_disk() -> None:
    path = Path(__file__).parents[2] / "src" / "masklayout" / "decks" / "generic_hammerhead_v1.yaml"
    deck = load_deck(path)
    assert deck.id == "generic_hammerhead_v1"
    assert len(deck.rules) >= 2
    assert deck.content_hash


class TestRange:
    def test_min_only(self) -> None:
        assert Range(min=10).contains(10.0)
        assert Range(min=10).contains(1000.0)
        assert not Range(min=10).contains(9.9)

    def test_max_only(self) -> None:
        assert Range(max=10).contains(9.9)
        assert not Range(max=10).contains(10.1)

    def test_infinity_satisfies_a_minimum_but_not_a_maximum(self) -> None:
        # An isolated feature reports inf space; "isolated" is a min constraint.
        assert Range(min=120).contains(math.inf)
        assert not Range(max=60).contains(math.inf)

    def test_none_never_matches_a_constraint(self) -> None:
        # None means the measurement does not apply, e.g. width at a corner.
        assert not Range(min=1).contains(None)
        assert not Range(max=1).contains(None)

    def test_an_empty_range_matches_any_real_value(self) -> None:
        assert Range().contains(5.0)
        assert not Range().contains(None)


class TestSelector:
    def test_an_empty_selector_matches_everything(self) -> None:
        values: dict[str, Any] = dict.fromkeys(SELECTOR_KEYS)
        values["site"] = "edge"
        assert Selector().matches(values)

    def test_a_categorical_key_must_match_exactly(self) -> None:
        values: dict[str, Any] = dict.fromkeys(SELECTOR_KEYS)
        values["site"] = "edge"
        assert Selector(site="edge").matches(values)
        assert not Selector(site="line_end").matches(values)

    def test_all_constraints_must_hold(self) -> None:
        values: dict[str, Any] = dict.fromkeys(SELECTOR_KEYS)
        values.update({"site": "line_end", "width_nm": 20.0, "space_nm": math.inf})
        selector = Selector(
            site="line_end", width_nm=Range(min=18, max=26), space_nm=Range(min=120)
        )
        assert selector.matches(values)

        narrower = Selector(site="line_end", width_nm=Range(min=30))
        assert not narrower.matches(values)
