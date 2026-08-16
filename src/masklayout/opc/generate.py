"""Turn a matched site into correction geometry.

Each correction kind has a generator keyed by the rule's ``kind``. A
generator reads placement from the site, takes shape parameters from the
rule, and calls the PCell registry — the shapes themselves were built in
M3 and are not re-implemented here.

Returning ``None`` means "this rule fired but produces nothing", which is
different from failing: a zero-extension hammerhead is a no-op, not a
degenerate polygon.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from masklayout.config import TechConfig
from masklayout.opc.classify import SiteMeasurement
from masklayout.opc.feature import Feature
from masklayout.opc.match import Match
from masklayout.opc.placement import merge_params, placement_for
from masklayout.pcells.base import build_pcell

#: Layer that generated corrections are written to before merging.
DEFAULT_CORRECTION_LAYER = 11


class UnknownCorrectionKindError(KeyError):
    """A rule names a correction kind that has no generator."""


GeneratorFn = Callable[[SiteMeasurement, Match, TechConfig], Feature | None]
GeneratorFnT = TypeVar("GeneratorFnT", bound=GeneratorFn)

_GENERATORS: dict[str, GeneratorFn] = {}


def register_generator(kind: str) -> Callable[[GeneratorFnT], GeneratorFnT]:
    """Register a generator for a correction kind."""

    def decorate(generator: GeneratorFnT) -> GeneratorFnT:
        if kind in _GENERATORS:
            raise ValueError(f"a generator for kind {kind!r} is already registered")
        _GENERATORS[kind] = generator
        return generator

    return decorate


def registered_kinds() -> list[str]:
    """Every correction kind with a generator, sorted."""
    return sorted(_GENERATORS)


def _require(params: dict[str, Any], name: str, kind: str) -> float:
    if name not in params:
        raise ValueError(
            f"correction kind {kind!r} requires parameter {name!r}; "
            f"the rule supplied {sorted(params)}"
        )
    return float(params[name])


def feature_id(match: Match) -> str:
    """Stable id: which rule fired where."""
    return f"{match.kind}@{match.site_id}"


def generate_feature(
    measurement: SiteMeasurement, match: Match, tech: TechConfig
) -> Feature | None:
    """Build the correction geometry for one match, or None if it is a no-op."""
    try:
        generator = _GENERATORS[match.kind]
    except KeyError:
        raise UnknownCorrectionKindError(
            f"unknown correction kind {match.kind!r}; registered: {registered_kinds()}"
        ) from None
    return generator(measurement, match, tech)


def _build_line_end_cap(
    measurement: SiteMeasurement, match: Match, tech: TechConfig, kind: str
) -> Feature | None:
    """Shared by hammerhead and line_end_extension.

    They differ only in ``head_width_ratio``: 1.0 is a plain extension, above
    1.0 flares into a hammerhead. The width is a ratio of the *measured* line
    width, so a correction scales with the line it corrects rather than being
    a fixed absolute that silently mis-fits.
    """
    extension_um = _require(match.params, "extension_um", kind)
    if extension_um <= 0.0:
        return None

    line_width_nm = measurement.width_nm
    if line_width_nm is None or line_width_nm <= 0.0:
        return None

    ratio = float(match.params.get("head_width_ratio", 1.0))
    shape: dict[str, Any] = {
        key: value for key, value in match.params.items() if key not in ("head_width_ratio",)
    }
    shape["width_um"] = (line_width_nm / 1000.0) * ratio

    params = merge_params(placement_for(measurement.site), shape)
    polygons = build_pcell(match.pcell, params, tech, layer=DEFAULT_CORRECTION_LAYER, datatype=0)
    return Feature(
        id=feature_id(match),
        kind=kind,
        polygons=polygons,
        source_site_id=measurement.site.site_id,
        rule_id=match.rule_id,
        deck_id=match.deck_id,
        deck_version=match.deck_version,
        deck_hash=match.deck_hash,
        parameters=dict(match.params),
        polarity="add",
    )


@register_generator("hammerhead")
def generate_hammerhead(
    measurement: SiteMeasurement, match: Match, tech: TechConfig
) -> Feature | None:
    return _build_line_end_cap(measurement, match, tech, "hammerhead")


@register_generator("line_end_extension")
def generate_line_end_extension(
    measurement: SiteMeasurement, match: Match, tech: TechConfig
) -> Feature | None:
    return _build_line_end_cap(measurement, match, tech, "line_end_extension")
