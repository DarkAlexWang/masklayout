"""Generated correction features and their provenance.

Every feature this toolkit produces is traceable data, not anonymous
geometry: it names the site it came from, the rule that fired, and the
exact deck that rule belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from masklayout.model.geometry import Polygon

#: Whether a correction adds material or removes it. Negative edge bias is
#: the first correction that removes, so polarity cannot be assumed.
Polarity = Literal["add", "subtract"]


@dataclass(frozen=True, eq=False)
class Feature:
    """Correction geometry plus the record of why it exists."""

    id: str
    kind: str
    polygons: list[Polygon]
    source_site_id: str
    rule_id: str
    deck_id: str
    deck_version: str
    deck_hash: str
    parameters: dict[str, Any] = field(default_factory=dict)
    polarity: Polarity = "add"

    @property
    def vertex_count(self) -> int:
        return sum(polygon.vertex_count for polygon in self.polygons)

    def provenance(self) -> dict[str, Any]:
        """A JSON-ready record of this feature's origin.

        Deliberately carries no geometry: the shapes belong on a layer, and
        this is the record *about* them that goes into the manifest.
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "polarity": self.polarity,
            "source_site_id": self.source_site_id,
            "rule_id": self.rule_id,
            "deck_id": self.deck_id,
            "deck_version": self.deck_version,
            "deck_hash": self.deck_hash,
            "parameters": dict(self.parameters),
            "vertex_count": self.vertex_count,
        }
