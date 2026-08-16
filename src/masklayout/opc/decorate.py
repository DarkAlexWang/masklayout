"""The target decorator: target geometry in, corrected geometry out.

Runs the whole pipeline — extract, classify, match, generate, merge — and
returns POST_OPC with overlay layers and every feature's provenance.

The target is never modified. Corrections are new polygons; the input list
comes back unchanged, which a test asserts directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from masklayout.config import TechConfig
from masklayout.geometry.context import GeomContext
from masklayout.model.geometry import Polygon
from masklayout.model.layers import LayerMap
from masklayout.opc.classify import classify_sites
from masklayout.opc.deck import RuleDeck
from masklayout.opc.extract import extract_sites
from masklayout.opc.feature import Feature
from masklayout.opc.generate import UnknownCorrectionKindError, generate_feature
from masklayout.opc.match import match_sites
from masklayout.opc.resolve import Rejection, resolve_collisions


@dataclass(frozen=True)
class DecorateReport:
    """What decorating did, for the manifest and for CI to assert on."""

    sites: int
    matched: int
    features_generated: int
    features_skipped: int
    srafs_placed: int = 0
    srafs_rejected: int = 0
    unknown_kinds: tuple[str, ...] = ()
    by_kind: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"{self.sites} sites",
            f"{self.matched} matched",
            f"{self.features_generated} features",
        ]
        if self.by_kind:
            detail = ", ".join(f"{kind} x{count}" for kind, count in sorted(self.by_kind.items()))
            parts.append(f"({detail})")
        if self.srafs_placed or self.srafs_rejected:
            parts.append(f"{self.srafs_placed} SRAFs placed, {self.srafs_rejected} rejected")
        if self.features_skipped:
            parts.append(f"{self.features_skipped} no-ops")
        if self.unknown_kinds:
            parts.append(f"unknown kinds {list(self.unknown_kinds)}")
        return ", ".join(parts)


@dataclass(frozen=True)
class DecorateResult:
    """Corrected geometry, its provenance, and the overlays that show the delta."""

    post_opc: list[Polygon]
    features: list[Feature]
    overlay_add: list[Polygon]
    overlay_remove: list[Polygon]
    report: DecorateReport
    srafs: list[Polygon] = field(default_factory=list)
    markers: list[Polygon] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


def decorate(
    target: Sequence[Polygon],
    deck: RuleDeck,
    tech: TechConfig,
    layers: LayerMap | None = None,
    line_end_ratio: float = 0.5,
    max_probe_um: float = 2.0,
    density_window_um: float = 1.0,
    skip_unknown_kinds: bool = False,
    target_keepout_um: float = 0.02,
    sraf_keepout_um: float = 0.02,
) -> DecorateResult:
    """Extract, classify, match, generate, and merge.

    ``skip_unknown_kinds`` lets a deck reference corrections this version
    cannot build yet — ``jog``, for instance — and records them in the report
    instead of failing. It defaults to False so an unimplemented kind is loud.
    """
    layers = layers or LayerMap.default()
    post_opc_layer = layers["POST_OPC"]
    overlay_add_layer = layers["OVERLAY_ADD"]
    overlay_remove_layer = layers["OVERLAY_REMOVE"]

    sites = extract_sites(target, tech.precision_um, line_end_ratio=line_end_ratio)
    measurements = classify_sites(
        sites, target, tech, max_probe_um=max_probe_um, density_window_um=density_window_um
    )
    matches, match_report = match_sites(measurements, deck)

    by_site = {m.site.site_id: m for m in measurements}
    features: list[Feature] = []
    skipped = 0
    unknown: list[str] = []
    by_kind: dict[str, int] = {}

    for match in matches:
        measurement = by_site.get(match.site_id)
        if measurement is None:
            continue
        try:
            feature = generate_feature(measurement, match, tech)
        except UnknownCorrectionKindError:
            if not skip_unknown_kinds:
                raise
            if match.kind not in unknown:
                unknown.append(match.kind)
            continue
        if feature is None:
            skipped += 1
            continue
        features.append(feature)
        by_kind[feature.kind] = by_kind.get(feature.kind, 0) + 1

    features, rejections = resolve_collisions(
        features, target, tech, target_keepout_um, sraf_keepout_um
    )

    context = GeomContext(tech)
    additive = [p for f in features if f.polarity == "add" for p in f.polygons]
    subtractive = [p for f in features if f.polarity == "subtract" for p in f.polygons]
    assists = [f for f in features if f.polarity == "assist"]

    merged = list(target)
    if additive:
        merged = context.boolean_polygons(
            merged, additive, "or", post_opc_layer.number, post_opc_layer.datatype
        )
    if subtractive:
        merged = context.boolean_polygons(
            merged, subtractive, "not", post_opc_layer.number, post_opc_layer.datatype
        )
    if not additive and not subtractive:
        merged = context.boolean_polygons(
            merged, [], "or", post_opc_layer.number, post_opc_layer.datatype
        )

    overlay_add = context.boolean_polygons(
        merged, list(target), "not", overlay_add_layer.number, overlay_add_layer.datatype
    )
    overlay_remove = context.boolean_polygons(
        list(target), merged, "not", overlay_remove_layer.number, overlay_remove_layer.datatype
    )

    sraf_layer = layers["SRAF"]
    srafs = [
        Polygon(points=p.points, layer=sraf_layer.number, datatype=sraf_layer.datatype)
        for f in assists
        for p in f.polygons
    ]
    marker_layer = layers["DEBUG_MARKERS"]
    markers = [
        Polygon(points=p.points, layer=marker_layer.number, datatype=marker_layer.datatype)
        for rejection in rejections
        for p in rejection.polygons
    ]

    report = DecorateReport(
        sites=len(sites),
        matched=match_report.matched,
        features_generated=len(features),
        features_skipped=skipped,
        srafs_placed=len(assists),
        srafs_rejected=len(rejections),
        unknown_kinds=tuple(unknown),
        by_kind=by_kind,
    )
    return DecorateResult(
        post_opc=merged,
        features=features,
        overlay_add=overlay_add,
        overlay_remove=overlay_remove,
        report=report,
        srafs=srafs,
        markers=markers,
        rejected=rejections,
    )
