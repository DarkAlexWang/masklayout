"""The top-level layout container and hierarchy inspection."""

from __future__ import annotations

from dataclasses import dataclass, field

from masklayout.config import TechConfig
from masklayout.model.cell import Cell
from masklayout.model.layers import LayerMap


class UnknownCellError(KeyError):
    """A reference names a cell that is not in the layout."""


@dataclass
class Layout:
    """A library of cells sharing one technology configuration."""

    name: str
    cells: dict[str, Cell] = field(default_factory=dict)
    tech: TechConfig = field(default_factory=TechConfig)
    layers: LayerMap = field(default_factory=LayerMap.default)

    def add(self, cell: Cell) -> Cell:
        if cell.name in self.cells:
            raise ValueError(f"cell {cell.name!r} already exists in layout {self.name!r}")
        self.cells[cell.name] = cell
        return cell

    def referenced_names(self) -> set[str]:
        """Every cell name that appears as a reference target."""
        return {ref.cell_name for cell in self.cells.values() for ref in cell.references}

    def top_cells(self) -> list[str]:
        """Cells that nothing else references, sorted for determinism."""
        referenced = self.referenced_names()
        return sorted(name for name in self.cells if name not in referenced)

    def dependencies(self, cell_name: str) -> set[str]:
        """Every cell reachable from ``cell_name``, excluding itself."""
        if cell_name not in self.cells:
            raise UnknownCellError(f"unknown cell {cell_name!r}; known cells: {sorted(self.cells)}")
        seen: set[str] = set()
        stack = [cell_name]
        while stack:
            current = stack.pop()
            for ref in self.cells[current].references:
                if ref.cell_name in seen:
                    continue
                if ref.cell_name not in self.cells:
                    raise UnknownCellError(
                        f"cell {current!r} references unknown cell {ref.cell_name!r}"
                    )
                seen.add(ref.cell_name)
                stack.append(ref.cell_name)
        seen.discard(cell_name)
        return seen

    def depth(self) -> int:
        """Longest reference chain. A flat layout has depth 0."""
        memo: dict[str, int] = {}

        def walk(name: str, path: frozenset[str]) -> int:
            if name in path:
                raise ValueError(f"reference cycle detected at cell {name!r}")
            if name in memo:
                return memo[name]
            refs = self.cells[name].references
            best = 0 if not refs else 1 + max(walk(ref.cell_name, path | {name}) for ref in refs)
            memo[name] = best
            return best

        return max((walk(name, frozenset()) for name in self.cells), default=0)

    def polygon_count(self) -> int:
        return sum(len(cell.polygons) for cell in self.cells.values())
