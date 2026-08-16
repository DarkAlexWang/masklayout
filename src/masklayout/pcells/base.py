"""PCell parameter models, the build protocol, and the name registry.

The registry exists because M4's rule deck references a PCell by name and
supplies parameters as data. Building that path here means the deck reuses
one validated mechanism rather than introducing a second one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from masklayout.config import TechConfig
from masklayout.model.geometry import Polygon


class UnknownPCellError(KeyError):
    """A PCell was requested by a name that is not registered."""


class PCellParams(BaseModel):
    """Base for every PCell's parameters.

    Frozen so a built cell cannot be silently re-parameterized, and strict so
    a typo in a recipe fails loudly instead of being ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


BuildFn = Callable[[Any, TechConfig, int, int], list[Polygon]]

#: Bound to BuildFn so ``register`` returns the *concrete* decorated function
#: rather than the bare Callable alias. Returning the alias would erase
#: parameter names, and callers writing the natural
#: ``build_contact(params, tech, layer=12)`` would fail type checking even
#: though it runs fine.
BuildFnT = TypeVar("BuildFnT", bound=BuildFn)


class PCellBuilder(Protocol):
    """What a registered PCell provides."""

    params_model: type[PCellParams]

    def __call__(
        self, params: Any, tech: TechConfig, layer: int, datatype: int
    ) -> list[Polygon]: ...


_REGISTRY: dict[str, tuple[type[PCellParams], BuildFn]] = {}


def register(name: str, params_model: type[PCellParams]) -> Callable[[BuildFnT], BuildFnT]:
    """Register a build function under a name, with its params model."""

    def decorate(build: BuildFnT) -> BuildFnT:
        if name in _REGISTRY:
            raise ValueError(f"PCell {name!r} is already registered")
        _REGISTRY[name] = (params_model, build)
        return build

    return decorate


def registered_names() -> list[str]:
    """Every registered PCell name, sorted for determinism."""
    return sorted(_REGISTRY)


def params_model_for(name: str) -> type[PCellParams]:
    """The parameters model a PCell accepts.

    Callers use this to inject only the placement keys a given PCell
    understands: a contact takes a centre but no angle, while a line end
    takes both, and params models forbid unknown keys.
    """
    try:
        return _REGISTRY[name][0]
    except KeyError:
        raise UnknownPCellError(
            f"unknown PCell {name!r}; registered: {registered_names()}"
        ) from None


def build_pcell(
    name: str,
    params: PCellParams | dict[str, Any],
    tech: TechConfig,
    layer: int,
    datatype: int = 0,
) -> list[Polygon]:
    """Build a PCell by name from either a params model or a plain dict."""
    try:
        params_model, build = _REGISTRY[name]
    except KeyError:
        raise UnknownPCellError(
            f"unknown PCell {name!r}; registered: {registered_names()}"
        ) from None
    validated = params if isinstance(params, params_model) else params_model.model_validate(params)
    return build(validated, tech, layer, datatype)
