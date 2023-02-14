from __future__ import annotations
import numpy as np
from typing import Tuple, Union, overload, Type, TypeVar, Iterator, TYPE_CHECKING
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


from .segment import Segment, Segment2D, Segment3D
from .utils import _array

if TYPE_CHECKING:
    from .hyperplane import Line, Line2D, Line3D, Plane

R = TypeVar("R", bound="Ray")


class Ray(HasLocationAndDirection, Meetable):
    def __init__(self, data: np.ndarray) -> None:
        """Initialize the ray.

        A ray is defined by a point and a direction. The magnitude of the direction is not preserved.

        Args:
            data (np.ndarray): [dim+1, 2] array with a homogeneous point and a vector in the columns.

        """
        assert data.shape == (
            self.dim + 1,
            2,
        ), f"data must be [dim+1, 2], got {data.shape}"
        super().__init__(data)

        if np.isclose(self.data[self.dim, 0], 0):
            raise ValueError("point is at infinity")
        if not np.isclose(self.data[self.dim, 1], 0):
            raise ValueError("direction is not at infinity")

        if not np.isclose(self.data[self.dim, 0], 1):
            self.data[:, 0] /= self.data[self.dim, 0]

    @classmethod
    def from_pn(cls: Type[R], p: Point, d: Vector) -> R:
        """Create a ray from a point and a direction."""
        p = point(p)
        d = vector(d).hat()
        return cls(np.stack([p.data, d.data], axis=1))

    @classmethod
    def from_point_direction(cls: Type[Ray], p: Point, d: Vector) -> Ray:
        """Create a ray from a point and a direction."""
        return cls.from_pn(p, d)

    @classmethod
    def from_pq(cls: Type[R], p: Point, q: Point) -> R:
        """Create a ray from two points.

        The point q is not preserved in the ray.

        Args:
            p (Point3D): The origin of the ray.
            q (Point3D): A point on the ray.
        """
        return cls.from_pn(p, q - p)

    @property
    def p(self) -> Point3D:
        return Point3D(self.data[:, 0])

    @p.setter
    def p(self, p: Union[Point, np.ndarray]) -> None:
        self.data[:, 0] = point(p).data

    @property
    def n(self) -> Vector:
        return vector(self.data[: self.dim, 1])

    @n.setter
    def n(self, n: Union[Vector, np.ndarray]) -> None:
        self.data[:, 1] = vector(n).data

    def get_direction(self) -> Vector3D:
        return self.n

    def get_point(self) -> Point3D:
        return self.p


class Ray2D(Ray):
    dim = 2

    def meet(self, other: Union[Line2D, Segment2D]) -> Point2D:
        """Get the point of intersection between this ray and another line.

        Args:
            other (Line2D): The other line.

        Returns:
            Point2D: The point of intersection.

        """
        raise NotImplementedError()


class Ray3D(Ray, Joinable, HasProjection):
    """A homogeneous representation of a ray.

    This is just a (4,2) array with the homogeneous coordinates of the
    origin and the direction, respectively.

    """

    dim = 3

    @classmethod
    def projection_type(cls) -> Type[Ray2D]:
        return Ray2D

    def join(self, other: Point3D) -> Plane:
        l = self.p.join(self.p + self.n)
        return l.join(other)

    def meet(self, other: Plane) -> Point3D:
        # TODO: depending on direction, ray may not intersect plane. Sort of the whole point.
        l = self.p.join(self.p + self.n)
        return l.meet(other)

    def __iter__(self) -> Iterator[Union[Point3D, Vector3D]]:
        yield self.p
        yield self.n


@overload
def ray(r: R) -> R:
    ...


@overload
def ray(l: Line2D) -> Ray2D:
    ...


@overload
def ray(l: Line3D) -> Ray3D:
    ...


@overload
def ray(p: Point2D, n: Vector2D) -> Ray2D:
    ...


@overload
def ray(p: Point3D, n: Vector3D) -> Ray3D:
    ...


@overload
def ray(p: Point3D, q: Point3D) -> Ray3D:
    ...


@overload
def ray(a: float, b: float, c: float, d: float) -> Ray2D:
    ...


@overload
def ray(a: float, b: float, c: float, d: float, e: float, f: float) -> Ray3D:
    ...


@overload
def ray(x: np.ndarray) -> Ray:
    ...


def ray(*args):
    """More flexible method for creating a ray."""
    if len(args) == 1 and isinstance(args[0], Ray):
        return args[0]
    elif len(args) == 1 and isinstance(args[0], HasLocationAndDirection):
        return Ray2D.from_pn(args[0].get_point(), args[0].get_direction())
    elif len(args) == 2:
        if isinstance(args[0], Point2D) and isinstance(args[1], Vector2D):
            return Ray2D.from_pn(args[0], args[1])
        elif isinstance(args[0], Point3D) and isinstance(args[1], Vector3D):
            return Ray3D.from_pn(args[0], args[1])
        elif isinstance(args[0], Point2D) and isinstance(args[1], Point2D):
            return Ray2D.from_pq(args[0], args[1])
        elif isinstance(args[0], Point3D) and isinstance(args[1], Point3D):
            return Ray3D.from_pq(args[0], args[1])

    r = _array(args)
    if r.shape == (4,):
        return Ray2D.from_pn(r[:2], r[2:])
    elif r.shape == (6,):
        return Ray3D.from_pn(r[:3], r[3:])
    elif r.shape == (2, 2):
        return Ray3D.from_pn(r[0], r[1])
    elif r.shape == (2, 3):
        return Ray3D.from_pn(r[0], r[1])
    elif r.shape == (2, 2):
        return Ray2D.from_pn(r[:, 0], r[:, 1])
    elif r.shape == (3, 2):
        return Ray3D.from_pn(r[:, 0], r[:, 1])
    else:
        raise ValueError(f"invalid data for ray: {r}")
