"""Public stream I/O: GDSII and OASIS."""

from __future__ import annotations

import datetime
from pathlib import Path

from masklayout.config import TechConfig
from masklayout.geometry.context import GeomContext
from masklayout.io._gdstk_bridge import layout_to_library, library_to_layout
from masklayout.io.report import ReadReport
from masklayout.model.layers import LayerMap
from masklayout.model.layout import Layout


def read_gds(
    path: Path | str,
    tech: TechConfig | None = None,
    layers: LayerMap | None = None,
) -> tuple[Layout, ReadReport]:
    """Read a GDSII file into the typed model."""
    tech = tech or TechConfig()
    library = GeomContext(tech).read_gds(path)
    return library_to_layout(library, tech, layers or LayerMap.default(), str(path))


def read_oas(
    path: Path | str,
    tech: TechConfig | None = None,
    layers: LayerMap | None = None,
) -> tuple[Layout, ReadReport]:
    """Read an OASIS file into the typed model."""
    tech = tech or TechConfig()
    library = GeomContext(tech).read_oas(path)
    return library_to_layout(library, tech, layers or LayerMap.default(), str(path))


def write_gds(
    layout: Layout,
    path: Path | str,
    timestamp: datetime.datetime | None = None,
) -> None:
    """Write the layout as GDSII, with a pinned timestamp for reproducibility."""
    context = GeomContext(layout.tech, timestamp=timestamp)
    context.write_gds(layout_to_library(layout), path)


def write_oas(layout: Layout, path: Path | str) -> None:
    """Write the layout as OASIS.

    OASIS embeds no timestamp, so output is reproducible without pinning one.
    """
    GeomContext(layout.tech).write_oas(layout_to_library(layout), path)
