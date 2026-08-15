"""I/O error types."""

from __future__ import annotations


class GridMismatchError(ValueError):
    """A file's database grid does not match the configured design grid."""


class OffGridCoordinateError(ValueError):
    """A coordinate does not lie on the design grid."""


class UnsupportedEntityError(ValueError):
    """A stream contains an entity kind this version cannot represent."""
