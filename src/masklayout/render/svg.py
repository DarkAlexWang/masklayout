"""SVG preview.

SVG is text, so this needs no dependency: a standalone file with one group
per logical layer, a distinct colour each, and a legend. PNG is not
produced — rasterising would add a dependency for no capability V1 needs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon

#: A colour per logical layer. Distinct in hue and in lightness, so the
#: rendering survives being printed or viewed without colour.
LAYER_STYLE: dict[str, tuple[str, float]] = {
    "TARGET": ("#4c78a8", 0.35),
    "POST_OPC": ("#f58518", 0.45),
    "SRAF": ("#54a24b", 0.55),
    "OVERLAY_ADD": ("#e45756", 0.70),
    "OVERLAY_REMOVE": ("#b279a2", 0.70),
    "DEBUG_MARKERS": ("#eeca3b", 0.85),
    "DEBUG_SOURCE": ("#9d755d", 0.50),
}
_DEFAULT_STYLE = ("#808080", 0.4)


def _bounds(groups: Mapping[str, Sequence[Polygon]]) -> tuple[float, float, float, float]:
    everything = [p for polygons in groups.values() for p in polygons]
    if not everything:
        return (0.0, 0.0, 1.0, 1.0)
    stacked = np.vstack([p.points for p in everything]).astype(np.float64)
    return (
        float(stacked[:, 0].min()),
        float(stacked[:, 1].min()),
        float(stacked[:, 0].max()),
        float(stacked[:, 1].max()),
    )


def render_svg(
    path: Path | str,
    layer_geometry: Mapping[str, Sequence[Polygon]],
    tech: TechConfig,
    width_px: int = 1000,
    margin_frac: float = 0.06,
) -> None:
    """Write a standalone SVG preview, one group per logical layer."""
    populated = {name: list(polys) for name, polys in layer_geometry.items() if polys}
    min_x, min_y, max_x, max_y = _bounds(populated)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    margin = max(span_x, span_y) * margin_frac
    view_w = span_x + 2 * margin
    view_h = span_y + 2 * margin
    height_px = max(int(width_px * view_h / view_w), 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" '
        f'viewBox="{min_x - margin:.3f} {min_y - margin:.3f} {view_w:.3f} {view_h:.3f}">',
        # GDS y grows upward, SVG y grows downward; flip so the preview
        # matches what a layout viewer shows.
        f'<g transform="translate(0,{2 * min_y + span_y:.3f}) scale(1,-1)">',
        f'<rect x="{min_x - margin:.3f}" y="{min_y - margin:.3f}" '
        f'width="{view_w:.3f}" height="{view_h:.3f}" fill="#ffffff"/>',
    ]

    for name in sorted(populated):
        colour, opacity = LAYER_STYLE.get(name, _DEFAULT_STYLE)
        lines.append(f'<g id="{name}" fill="{colour}" fill-opacity="{opacity}" stroke="none">')
        for polygon in populated[name]:
            coords = " ".join(f"{x},{y}" for x, y in polygon.points.tolist())
            lines.append(f'<polygon points="{coords}"/>')
        lines.append("</g>")

    lines.append("</g>")

    # Legend, outside the flipped group so the text is not mirrored.
    legend_y = 18
    for name in sorted(populated):
        colour, opacity = LAYER_STYLE.get(name, _DEFAULT_STYLE)
        lines.append(
            f'<rect x="10" y="{legend_y - 10}" width="12" height="12" '
            f'fill="{colour}" fill-opacity="{opacity}"/>'
        )
        lines.append(
            f'<text x="28" y="{legend_y}" font-family="monospace" font-size="12" '
            f'fill="#222222">{name} ({len(populated[name])})</text>'
        )
        legend_y += 18

    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
