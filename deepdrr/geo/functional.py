from __future__ import annotations
import logging
from typing import Optional, overload
import numpy as np

from . import core
from .typing import P, V

log = logging.getLogger(__name__)


@overload
def distance(p: P, q: P) -> float:
    ...


@overload
def distance(p: core.Point3D, q: core.Line3D) -> float:
    ...


@overload
def distance(p: core.Point3D, q: core.Plane3D) -> float:
    ...


def distance(a, q) -> float:
    """Get the distance between two geometric options, which have location."""
    if isinstance(a, core.Point) and isinstance(q, core.Point):
        return (a - q).norm()
    else:
        raise NotImplementedError(
            f"Distance between {type(a)} and {type(q)} is not implemented. Use the object's distance method."
        )
