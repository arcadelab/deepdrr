from typing import TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    from .core import Primitive, PointOrVector, Point, Vector
    from .hyperplane import Line, Plane
    from .segment import Segment
    from .ray import Ray


Pr = TypeVar("Pr", bound="Primitive")
PV = TypeVar("PV", bound="PointOrVector")
P = TypeVar("P", bound="Point")
V = TypeVar("V", bound="Vector")
L = TypeVar("L", bound="Line")
Pl = TypeVar("Pl", bound="Plane")
R = TypeVar("R", bound="Ray")
S = TypeVar("S", bound="Segment")
