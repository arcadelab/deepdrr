from typing import TypeVar
from .volume import Volume, MetalVolume
from .kwire import KWire

AnyVolume = TypeVar("AnyVolume", bound=Volume)

__all__ = ["Volume", "MetalVolume", "KWire", "AnyVolume"]
