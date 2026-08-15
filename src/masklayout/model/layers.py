"""Logical layer definitions and the layer map.

Layer numbers are configurable. The defaults below are the engineering
convention from the V1 design, section "Layer policy".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

GDS_MAX = 65535

DEFAULT_LAYERS: dict[str, tuple[int, int]] = {
    "TARGET": (10, 0),
    "POST_OPC": (11, 0),
    "SRAF": (12, 0),
    "FIELD": (20, 0),
    "DEBUG_SOURCE": (200, 0),
    "DEBUG_MARKERS": (201, 0),
    "OVERLAY_ADD": (202, 0),
    "OVERLAY_REMOVE": (203, 0),
}


class Layer(BaseModel):
    """A GDSII layer/datatype pair with a logical name."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=0, le=GDS_MAX)
    datatype: int = Field(ge=0, le=GDS_MAX)
    name: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"{self.name}({self.number}/{self.datatype})"


class LayerMap(BaseModel):
    """Maps logical layer names onto GDSII layer/datatype pairs."""

    model_config = ConfigDict(frozen=True)

    layers: dict[str, Layer]

    @model_validator(mode="after")
    def _reject_duplicate_pairs(self) -> LayerMap:
        seen: dict[tuple[int, int], str] = {}
        for name, layer in self.layers.items():
            key = (layer.number, layer.datatype)
            if key in seen:
                raise ValueError(
                    f"layer/datatype {key[0]}/{key[1]} is assigned to both "
                    f"{seen[key]!r} and {name!r}; logical layers must be distinct"
                )
            seen[key] = name
        return self

    @classmethod
    def default(cls) -> LayerMap:
        """The engineering-convention layer map."""
        return cls(
            layers={
                name: Layer(number=number, datatype=datatype, name=name)
                for name, (number, datatype) in DEFAULT_LAYERS.items()
            }
        )

    def __getitem__(self, name: str) -> Layer:
        try:
            return self.layers[name]
        except KeyError:
            raise KeyError(
                f"unknown logical layer {name!r}; known layers: {sorted(self.layers)}"
            ) from None
