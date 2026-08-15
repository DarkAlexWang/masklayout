"""Rule deck: schema, loading, validation, and content hashing.

A deck is data, not code. It is validated on load, hashed by content, and
carries its own version — so a generated feature's provenance can name the
exact rule set that produced it (design section "Decisions", rule 6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from masklayout.opc.classify import SELECTOR_KEYS


class UnknownSelectorError(ValueError):
    """A rule selects on a key outside the closed vocabulary."""


class Range(BaseModel):
    """An inclusive numeric bound. Either end may be omitted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min: float | None = None
    max: float | None = None

    def contains(self, value: float | None) -> bool:
        """Whether a measurement satisfies this bound.

        ``None`` means the measurement does not apply at this site, so it
        never satisfies a constraint. ``inf`` means the measurement applies
        but found nothing in range, so it satisfies a minimum and fails a
        maximum — which is exactly how an isolated feature should behave.
        """
        if value is None:
            return False
        if self.min is not None and value < self.min:
            return False
        return not (self.max is not None and value > self.max)


class Selector(BaseModel):
    """The conditions under which a rule fires.

    Field names are exactly the closed vocabulary measured by ``classify``.
    An omitted key is unconstrained.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    site: str | None = None
    corner_type: str | None = None
    width_nm: Range | None = None
    space_nm: Range | None = None
    edge_length_nm: Range | None = None
    angle_deg: Range | None = None
    curvature_1_per_um: Range | None = None
    local_density: Range | None = None

    def matches(self, values: dict[str, Any]) -> bool:
        """True when every constraint in this selector holds."""
        if self.site is not None and values.get("site") != self.site:
            return False
        if self.corner_type is not None and values.get("corner_type") != self.corner_type:
            return False
        for key in (
            "width_nm",
            "space_nm",
            "edge_length_nm",
            "angle_deg",
            "curvature_1_per_um",
            "local_density",
        ):
            bound: Range | None = getattr(self, key)
            if bound is not None and not bound.contains(values.get(key)):
                return False
        return True


class Apply(BaseModel):
    """What to build when a rule fires.

    ``pcell`` is resolved against the PCell registry at build time, not at
    load time, so a deck can be validated without every PCell existing yet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pcell: str
    params: dict[str, Any] = Field(default_factory=dict)


class Rule(BaseModel):
    """One selector paired with one action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    priority: int
    kind: str = Field(min_length=1)
    when: Selector
    apply: Apply


class RuleDeck(BaseModel):
    """A validated, hashable set of rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    rules: tuple[Rule, ...]

    @model_validator(mode="after")
    def _reject_duplicate_ids(self) -> RuleDeck:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id!r} in deck {self.id!r}")
            seen.add(rule.id)
        return self

    @property
    def content_hash(self) -> str:
        """SHA-256 over the canonical JSON form.

        Stable across YAML formatting and key order, so it changes when the
        rules change and not when the file is merely reformatted.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def rules_in_priority_order(self) -> list[Rule]:
        """Rules sorted by priority, ties broken by id for determinism."""
        return sorted(self.rules, key=lambda rule: (rule.priority, rule.id))


def _check_selector_keys(mapping: dict[str, Any], rule_id: str) -> None:
    unknown = set(mapping) - SELECTOR_KEYS
    if unknown:
        raise UnknownSelectorError(
            f"rule {rule_id!r} selects on unknown key(s) {sorted(unknown)}; "
            f"the vocabulary is {sorted(SELECTOR_KEYS)}"
        )


def load_deck_from_mapping(mapping: dict[str, Any]) -> RuleDeck:
    """Build a deck from an already-parsed mapping."""
    header = mapping.get("deck", {})
    raw_rules = mapping.get("rules", [])

    for index, raw in enumerate(raw_rules):
        _check_selector_keys(raw.get("when", {}), raw.get("id", f"<rule {index}>"))

    return RuleDeck(
        id=header.get("id", ""),
        version=header.get("version", ""),
        rules=tuple(Rule.model_validate(raw) for raw in raw_rules),
    )


def load_deck(path: Path | str) -> RuleDeck:
    """Load and validate a deck from a YAML file."""
    with Path(path).open(encoding="utf-8") as handle:
        return load_deck_from_mapping(yaml.safe_load(handle))
