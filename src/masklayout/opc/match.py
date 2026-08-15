"""Match classified sites against a rule deck.

A match is a decision, not geometry. Turning a match into corrected polygons
is M5's job; keeping that boundary means the selector vocabulary can be
validated before anything depends on the shapes it drives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from masklayout.opc.deck import RuleDeck


class Measured(Protocol):
    """What the matcher needs from a classified site."""

    @property
    def site(self) -> Any: ...

    def as_selector_values(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Match:
    """One rule firing at one site.

    Carries the deck's identity, version, and content hash so a generated
    feature can name the exact rule set that produced it, as the design's
    provenance requirements demand.
    """

    site_id: str
    rule_id: str
    kind: str
    pcell: str
    params: dict[str, Any]
    deck_id: str
    deck_version: str
    deck_hash: str


@dataclass(frozen=True)
class MatchReport:
    """What matching did, for the manifest and for CI to assert on."""

    considered: int
    matched: int
    unmatched: int
    by_rule: dict[str, int] = field(default_factory=dict)


def match_sites(
    measurements: Sequence[Measured], deck: RuleDeck
) -> tuple[list[Match], MatchReport]:
    """Pair sites with rules by priority, first match winning per kind.

    A site may take one rule of each kind — a bias and a hammerhead can both
    apply — but never two rules of the same kind. Rules are evaluated in
    priority order with ties broken by id, so output is deterministic for
    identical input.
    """
    ordered_rules = deck.rules_in_priority_order()
    deck_hash = deck.content_hash

    matches: list[Match] = []
    by_rule: dict[str, int] = {}
    matched_sites = 0

    for measurement in measurements:
        values = measurement.as_selector_values()
        taken_kinds: set[str] = set()
        site_matched = False

        for rule in ordered_rules:
            if rule.kind in taken_kinds:
                continue
            if not rule.when.matches(values):
                continue
            taken_kinds.add(rule.kind)
            site_matched = True
            by_rule[rule.id] = by_rule.get(rule.id, 0) + 1
            matches.append(
                Match(
                    site_id=measurement.site.site_id,
                    rule_id=rule.id,
                    kind=rule.kind,
                    pcell=rule.apply.pcell,
                    params=dict(rule.apply.params),
                    deck_id=deck.id,
                    deck_version=deck.version,
                    deck_hash=deck_hash,
                )
            )

        if site_matched:
            matched_sites += 1

    report = MatchReport(
        considered=len(measurements),
        matched=matched_sites,
        unmatched=len(measurements) - matched_sites,
        by_rule=by_rule,
    )
    return matches, report
