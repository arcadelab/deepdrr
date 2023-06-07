from typing import TypeVar
from .volume import Volume, MetalVolume
from .kwire import KWire
from .mesh import Mesh
from .primitive import Primitive

AnyVolume = TypeVar("AnyVolume", bound=Volume)

__all__ = ["Volume", "MetalVolume", "KWire", "AnyVolume", "Mesh", "Primitive"]
