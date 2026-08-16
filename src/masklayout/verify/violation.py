"""Verification findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from masklayout.model.geometry import Polygon

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True, eq=False)
class Violation:
    """One geometric check that failed, with the geometry that failed it.

    Carries its polygons so the same finding can be written twice: as a
    marker a human can see in a layout viewer, and as a record CI can assert
    on. A finding that exists only in a log is one nobody acts on.
    """

    check: str
    severity: Severity
    message: str
    polygons: list[Polygon] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """JSON-ready, without geometry — the shapes go on a marker layer."""
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "detail": dict(self.detail),
            "marker_count": len(self.polygons),
        }
