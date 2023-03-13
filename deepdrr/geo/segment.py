from __future__ import annotations
import numpy as np
from typing import Tuple, Union, overload, Type, TypeVar, TYPE_CHECKING
import logging
from abc import ABC, abstractmethod

from .exceptions import JoinError, MeetError
from .core import (
    Primitive,
    Meetable,
    Joinable,
    Vector,
    Vector2D,
    Vector3D,
    Point,
    Point2D,
    Point3D,
    point,
    vector,
    HasLocationAndDirection,
    HasProjection,
)

from .typing import P, V, L, Pl, S
from .utils import _array

if TYPE_CHECKING:
    from .hyperplane import Line, Line2D, Line3D, Plane

log = logging.getLogger(__name__)


class Segment(HasLocationAndDirection, Meetable):
    def __init__(self, data: np.ndarray) -> None:
        """Initialize the segment.

        Args:
            data (np.ndarray): [dim + 1, 2] array of homogeneous 2D points (in the columns).

        """
        assert data.shape == (
            self.dim + 1,
            2,
        ), f"invalid shape {data.shape}, looking for {(self.dim + 1, 2)} in class {self.__class__.__name__}"
        super().__init__(data)

        if np.isclose(self.data[self.dim, :], 0).any():
            raise ValueError("segment is degenerate")
        self.data[:, 0] /= self.data[self.dim, 0]
        self.data[:, 1] /= self.data[self.dim, 1]

    @classmethod
    def from_pq(cls: Type[S], p: Point, q: Point) -> S:
        """Initialize the segment containing two points.

        Args:
            p (Point): The first point.
            q (Point): The second point.

        Returns:
            Segment: The segment.

        """
        p = point(p)
        q = point(q)
        return cls(np.stack([p.data, q.data], axis=1))

    @classmethod
    def from_point_direction(cls: Type[S], p: Point, n: Vector) -> S:
        """Initialize the segment with a poind and a direction.

        Args:
            p (Point): The first point.
            n (Vector): The direction vector.

        Returns:
            Segment: The segment.

        """
        p = point(p)
        n = vector(n)
        return cls.from_pq(p, p + n)

    @classmethod
    def from_pn(cls: Type[S], p: Point, n: Vector) -> S:
        """Initialize the segment with a poind and a direction.

        Args:
            p (Point): The first point.
            n (Vector): The direction vector.

        Returns:
            Segment: The segment.

        """
        return cls.from_point_direction(p, n)

    @property
    def p(self) -> Point:
        """Get the first point of the segment.

        Returns:
            Point2D: The first point of the segment.

        """
        return point(self.data[: self.dim, 0])

    @p.setter
    def p(self, value: Point2D) -> None:
        """Set the first point of the segment.

        Args:
            value (Point2D): The new first point of the segment.

        """
        self.data[:, 0] = point(value).data

    @property
    def q(self) -> Point:
        """Get the second point of the segment.

        Returns:
            Point2D: The second point of the segment.

        """
        return point(self.data[: self.dim, 1])

    @q.setter
    def q(self, value: Point) -> None:
        """Set the second point of the segment.

        Args:
            value (Point2D): The new second point of the segment.

        """
        self.data[:, 1] = point(value).data

    def length(self) -> float:
        """Get the length of the segment.

        Returns:
            float: The length of the segment.

        """
        return (self.p - self.q).norm()

    def get_point(self) -> Point:
        return self.p

    def get_direction(self) -> Vector:
        return self.q - self.p

    @overload
    def line(self: Segment2D) -> Line2D:
        ...

    @overload
    def line(self: Segment3D) -> Line3D:
        ...

    def line(self) -> Line:
        return self.p.join(self.q)

    def midpoint(self) -> Point:
        return self.p + (self.q - self.p) / 2


class Segment2D(Segment):
    """Represents a line segment in 2D."""

    dim = 2

    def meet(self, other: Union[Line2D, Segment2D]) -> Point2D:
        """Get the point of intersection between this segment and another line.

        Args:
            other (Line2D): The other line.

        Returns:
            Point2D: The point of intersection.

        """
        p = super().meet(other)
        from .hyperplane import Line2D

        if isinstance(other, Line2D):
            return other.meet(self)
        elif isinstance(other, Segment2D):
            # Checks intersections of one segment with the other line and vice versa.
            # MeetError is raised if there is no intersection.
            self.meet(other.line())
            return other.meet(self.line())
        else:
            raise TypeError()


class Segment3D(Segment, Joinable, HasProjection):
    """Represents a segment in 3D."""

    dim = 3

    @classmethod
    def projection_type(cls) -> Type[Segment2D]:
        return Segment2D

    def join(self, other: Point3D) -> Plane:
        if isinstance(other, Point3D):
            return self.line().join(other)
        else:
            raise TypeError()

    def meet(self, other: Plane) -> Point3D:
        """Get the point of intersection between this segment and a plane.

        TODO: check if the intersection is on the segment.

        """
        return self.line().meet(other)


@overload
def segment(s: S) -> S:
    ...


@overload
def segment(p: Point2D, q: Point2D) -> Segment2D:
    ...


@overload
def segment(p: Point3D, q: Point3D) -> Segment3D:
    ...


@overload
def segment(p: Point2D, n: Vector2D) -> Segment2D:
    ...


@overload
def segment(p: Point3D, n: Vector3D) -> Segment3D:
    ...


@overload
def segment(a: float, b: float, c: float, d: float) -> Segment2D:
    ...


@overload
def segment(a: float, b: float, c: float, d: float, e: float, f: float) -> Segment3D:
    ...


@overload
def segment(x: np.ndarray) -> Segment:
    ...


def segment(*args):
    """More flexible method for creating a segment."""
    if len(args) == 1 and isinstance(args[0], Segment):
        return args[0]
    elif len(args) == 2:
        if isinstance(args[0], Point2D) and isinstance(args[1], Point2D):
            return Segment2D.from_pq(args[0], args[1])
        elif isinstance(args[0], Point3D) and isinstance(args[1], Point3D):
            return Segment3D.from_pq(args[0], args[1])
        elif isinstance(args[0], Point2D) and isinstance(args[1], Vector2D):
            return Segment2D.from_point_direction(args[0], args[1])
        elif isinstance(args[0], Point3D) and isinstance(args[1], Vector3D):
            return Segment3D.from_point_direction(args[0], args[1])

    r = _array(args)
    if r.shape == (4,):
        return Segment2D.from_pq(r[:2], r[2:])
    elif r.shape == (6,):
        return Segment3D.from_pq(r[:3], r[3:])
    elif r.shape == (2, 2):
        return Segment3D.from_pq(r[0], r[1])
    elif r.shape == (2, 3):
        return Segment3D.from_pq(r[0], r[1])
    elif r.shape == (2, 2):
        return Segment2D.from_pq(r[:, 0], r[:, 1])
    elif r.shape == (3, 2):
        return Segment3D.from_pq(r[:, 0], r[:, 1])
    else:
        raise ValueError(f"invalid data for ray: {r}")
