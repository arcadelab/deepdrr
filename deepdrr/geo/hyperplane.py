from __future__ import annotations
import numpy as np
from typing import Tuple, Type, Union, overload, TYPE_CHECKING, Any
import logging
from abc import ABC, abstractmethod

from .exceptions import JoinError, MeetError
from .core import (
    HasProjection,
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
    _array,
    HasLocationAndDirection,
    HasProjection,
)
from .segment import Segment, Segment2D, Segment3D
from .typing import R, P, V, S, Pl
from .ray import Ray, Ray2D, Ray3D

if TYPE_CHECKING:
    from .core import CameraProjection


log = logging.getLogger(__name__)


class HyperPlane(HasLocationAndDirection, Meetable):
    """Represents a hyperplane in 2D (a line) or 3D (a plane).

    Hyperplanes can be intersected with other hyperplanes or lower dimensional objects, but they are
    not joinable.

    """

    def __init__(self, data: np.ndarray) -> None:
        assert len(data) == self.dim + 1, f"data has shape {data.shape}"
        super().__init__(data)

    @property
    def a(self) -> float:
        """Get the coefficient of the first variable.

        Returns:
            float: The coefficient of the first variable.

        """
        return self.data[0]

    @property
    def b(self) -> float:
        """Get the coefficient of the second variable.

        Returns:
            float: The coefficient of the second variable.

        """
        return self.data[1]

    @property
    def c(self) -> float:
        """Get the coefficient of the third variable.

        Returns:
            float: The coefficient of the third variable.

        """
        return self.data[2]

    @property
    def d(self) -> float:
        """Get the constant term.

        Returns:
            float: The constant term.

        """
        if self.dim < 3:
            raise ValueError("2D lines have no constant term")
        return self.data[3]

    def evaluate(self, p: Point) -> float:
        """Evaluate the hyperplane at the given point.

        The sign of this value tells you which side of the hyperplane the point is on.

        Args:
            p (Point): the point to evaluate at.

        Returns:
            float: the value of the hyperplane at the given point.

        """
        assert self.dim == p.dim, f"dimension mismatch: {self.dim} != {p.dim}"
        return self.data @ p.data

    def get_normal(self) -> Vector3D:
        """Get the normal vector of the plane.

        Returns:
            Vector3D: The normal vector of the plane.

        """
        return vector(self.data[: self.dim])

    def normal(self) -> Vector3D:
        return self.get_normal()

    @property
    def n(self) -> Vector3D:
        return self.get_normal()

    def signed_distance(self, p: Point) -> float:
        """Get the signed distance from the given point to the hyperplane.

        Args:
            p (Point): the point to measure the distance from.

        Returns:
            float: the signed distance from the point to the hyperplane.

        """
        p = point(p)
        if self.dim != p.dim:
            raise ValueError(f"dimension mismatch: {self.dim} != {p.dim}")
        return -self.evaluate(p) / (p.w * self.n.norm())

    def distance(self, p: Point) -> float:
        """Get the distance of the point to the hyperplane.

        Args:
            p (Point): the point to evaluate at.

        Returns:
            float: the distance of the point to the hyperplane.

        """
        return abs(self.signed_distance(p))

    def project(self, p: P) -> P:
        """Get the closest point on the hyperplane to p.

        Args:
            p (Point): The point to project.

        Returns:
            Point: The closest point on the hyperplane to p.

        """
        p = point(p)  # guaranteed to be homogenized
        assert np.isclose(p.w, 1), f"point is not homogenized: {p}"
        d = self.signed_distance(p)
        return p + d * self.n


class Line(HasLocationAndDirection, Meetable):
    """Abstract parent class for lines and line-like objects."""

    @overload
    def as_points(self: Line2D) -> Tuple[Point2D, Point2D]:
        ...

    @overload
    def as_points(self: Line3D) -> Tuple[Point3D, Point3D]:
        ...

    def as_points(self):
        """Get two points on the line.

        Returns:
            Tuple[Point, Point]: Two points on the line.

        """
        return self.get_point(), self.get_point() + self.get_direction()

    @overload
    def project(self: Line2D, other: Point2D) -> Point2D:
        ...

    @overload
    def project(self: Line3D, other: Point3D) -> Point3D:
        ...

    def project(self, other):
        """Get the closest point on the line to another point.

        Args:
            other (Point): The point to which the closest point is sought.

        Returns:
            Point: The closest point on the line to the other point.

        """
        p = self.get_point()
        v = self.get_direction()
        other = point(other)
        d = other - p
        return p + v.dot(d) * v

    def distance(self, other: Point) -> float:
        """Get the distance from the line to another point.

        Args:
            other (Point): The point to which the distance is sought.

        Returns:
            float: The distance from the line to the other point.

        """
        other = point(other)
        p = self.get_point()
        v = self.get_direction()
        diff = other - p
        return (diff - v.dot(diff) * v).norm()

    def angle(self, other: Union[Line, Vector]) -> float:
        """Get the acute angle between the two lines."""
        assert other.dim == self.dim
        d1 = self.get_direction()
        if isinstance(other, Vector):
            d2 = other
        elif isinstance(other, Line):
            d2 = other.get_direction()
        else:
            TypeError

        if d1.dot(d2) < 0:
            d2 = -d2
        return d1.angle(d2)

    @classmethod
    def from_point_direction(cls, p: Point, v: Vector) -> Line:
        """Construct a line from a point and a direction vector.

        Args:
            p (Point): The point on the line.
            v (Vector): The direction vector.

        Returns:
            Line: The line through the point in the direction of the vector.

        """
        log.debug(f"constructing line from point {p} and direction {v}")
        p = point(p)
        v = vector(v)
        return p.join(p + v)


class Line2D(Line, HyperPlane):
    """Represents a line in 2D.

    Consists of a 3-vector :math:`\mathbf{p} = [a, b, c]` such that the line is all the points (x,y)
    such that :math:`ax + by + c = 0` or, alternatively, all the homogeneous points
    :math:`\mathbf{x} = [x,y,w]` such that :math:`p^T x = 0`.

    """

    dim = 2

    @overload
    def meet(self, other: Line2D) -> Point2D:
        ...

    @overload
    def meet(self, other: Segment2D) -> Point2D:
        ...

    def meet(self, other):
        if type(other) is Line2D:
            m = np.cross(self.data, other.data)
            if np.isclose(m[2], 0):
                raise MeetError("lines are parallel")
            else:
                return Point2D(m)
        elif isinstance(other, Segment2D):
            r = self.meet(other.line())
            v = other.q - other.p
            if 0 <= v.dot(r - other.p) / v.normsqr() <= 1:
                return r
            else:
                raise MeetError("line does not meet segment")
        else:
            raise TypeError(f"unrecognized type for meet: {type(other)}")

    def backproject(self, index_from_world: CameraProjection) -> Plane:
        """Get the plane containing all the points that `P` projects onto this line.

        Args:
            P (Transform): A so-called `index_from_world` projection transform.

        Returns:
            Plane:
        """
        assert index_from_world.shape == (3, 4), "P is not a projective transformation"
        return Plane(index_from_world.data.T @ self.data)

    def get_direction(self) -> Vector2D:
        """Get the direction of the line.

        Returns:
            Vector2D: The unit-length direction of the line.

        """
        # If a x + b y + c = 0, then for all w,
        # a (x + wb) + b (y - wa) + c = ax + awb + by - bwa + c = 0
        return vector(self.b, -self.a).hat()

    def get_point(self) -> Point:
        """Get an arbitrary point on the line.

        Returns:
            Point: A point on the line.

        """
        return Point2D([0, -self.c / self.b, 1])


class Plane(HyperPlane):
    """Represents a plane in 3D"""

    dim = 3

    @classmethod
    def from_point_direction(cls, r: Point3D, d: Vector3D):
        """Make a plane from a point and a direction vector.

        Args:
            r (Point3D): The point on the plane.
            d (Vector3D): The direction vector of the plane.

        Returns:
            Plane: The plane.
        """
        r = point(r)
        d = vector(d)
        a, b, c = d
        d = -(a * r.x + b * r.y + c * r.z)
        return cls(np.array([a, b, c, d]))

    @classmethod
    def from_point_normal(cls, r: Point3D, n: Vector3D):
        """Make a plane from a point and a normal vector.

        Args:
            r (Point3D): The point on the plane.
            n (Vector3D): The normal vector of the plane.

        Returns:
            Plane: The plane.
        """
        return cls.from_point_direction(r, n)

    @classmethod
    def from_points(cls, a: Point3D, b: Point3D, c: Point3D) -> None:
        """Initialize the plane containing three points.

        Args:
            a (Point3D): a point on the plane.
            b (Point3D): a point on the plane.
            c (Point3D): a point on the plane.

        Returns:
            Plane: The plane.
        """
        a = point(a)
        b = point(b)
        c = point(c)

        assert a.dim == 3 and b.dim == 3 and c.dim == 3, "points must be 3D"

        return a.join(b).join(c)

    def get_point(self) -> Point3D:
        """Get an arbitrary point on the plane.

        Returns:
            Point3D: A point on the plane.

        """
        return Point3D([0, 0, -self.d / self.c, 1])

    def get_direction(self) -> Vector3D:
        """Get the direction of the plane.

        Returns:
            Vector3D: The unit-length direction of the plane.

        """
        return self.get_normal()

    @overload
    def meet(self, other: Plane) -> Line3D:
        ...

    @overload
    def meet(self, other: Line3D) -> Point3D:
        ...

    def meet(self, other):
        if isinstance(other, Plane):
            # Intersection of two planes in P^3.
            a1, b1, c1, d1 = self.data
            a2, b2, c2, d2 = other.data
            l = np.array(
                [
                    -(a1 * b2 - a2 * b1),  # p
                    a1 * c2 - a2 * c1,  # q
                    -(a1 * d2 - a2 * d1),  # r
                    -(b1 * c2 - b2 * c1),  # s
                    b1 * d2 - b2 * d1,  # t
                    -(c1 * d2 - c2 * d1),  # u
                ]
            )
            return Line3D(l)
        elif isinstance(other, Line3D):
            p = other.K @ self
            if np.all(np.isclose(p, 0)):
                raise MeetError("Plane and line are parallel")
            return Point3D(p)
        else:
            raise TypeError(f"unrecognized type for meet: {type(other)}")


class Line3D(Line, Primitive, Joinable, Meetable, HasProjection):
    """Represents a line in 3D as a 6-vector (p,q,r,s,t,u).

    Based on https://dl.acm.org/doi/pdf/10.1145/965141.563900.

    """

    dim = 3

    def __init__(self, data: np.ndarray) -> None:
        assert data.shape == (6,)
        # TODO: assert the necessary line conditions
        super().__init__(data)

    @classmethod
    def projection_type(cls) -> Type[Line2D]:
        return Line2D

    @classmethod
    def from_primal(cls, lp: np.ndarray) -> Line3D:
        assert lp.shape == (4, 4)
        data = np.array([lp[0, 1], -lp[0, 2], lp[0, 3], lp[1, 2], -lp[1, 3], lp[2, 3]])
        return cls(data)

    @classmethod
    def from_dual(cls, lk: np.ndarray) -> Line3D:
        assert lk.shape == (4, 4)
        data = np.array([lk[3, 2], lk[3, 1], lk[2, 1], lk[3, 0], lk[2, 0], lk[1, 0]])
        return cls(data)

    def primal(self) -> np.ndarray:
        """Get the primal matrix of the line."""
        p, q, r, s, t, u = self.data

        return np.array(
            [
                [0, p, -q, r],
                [-p, 0, s, -t],
                [q, -s, 0, u],
                [-r, t, -u, 0],
            ]
        )

    @property
    def L(self) -> np.ndarray:
        """Get the primal matrix of the line."""
        return self.primal()

    def dual(self) -> np.ndarray:
        """Get the dual form of the line."""
        p, q, r, s, t, u = self

        return np.array(
            [
                [0, -u, -t, -s],
                [u, 0, -r, -q],
                [t, r, 0, -p],
                [s, q, p, 0],
            ]
        )

    @property
    def K(self) -> np.ndarray:
        """Get the dual form of the line."""
        return self.dual()

    @property
    def p(self) -> float:
        """Get the first parameter of the line."""
        return self.data[0]

    @property
    def q(self) -> float:
        """Get the second parameter of the line."""
        return self.data[1]

    @property
    def r(self) -> float:
        """Get the third parameter of the line."""
        return self.data[2]

    @property
    def s(self) -> float:
        """Get the fourth parameter of the line."""
        return self.data[3]

    @property
    def t(self) -> float:
        """Get the fifth parameter of the line."""
        return self.data[4]

    @property
    def u(self) -> float:
        """Get the sixth parameter of the line."""
        return self.data[5]

    def join(self, other: Point3D) -> Plane:
        return other.join(self)

    def meet(self, other: Plane) -> Point3D:
        return other.meet(self)

    def get_direction(self) -> Vector3D:
        """Get the direction of the line."""
        d = vector(self.s, self.q, self.p)
        return d.hat()

    def get_point(self) -> Point3D:
        """Get a point on the line."""
        d = self.get_direction()
        return d.as_plane().meet(self)


@overload
def line(l: Line2D) -> Line2D:
    ...


@overload
def line(l: Line3D) -> Line3D:
    ...


@overload
def line(r: Ray2D) -> Line2D:
    ...


@overload
def line(r: Ray3D) -> Line3D:
    ...


@overload
def line(s: Segment2D) -> Line2D:
    ...


@overload
def line(s: Segment3D) -> Line3D:
    ...


@overload
def line(a: float, b: float, c: float) -> Line2D:
    ...


@overload
def line(l: np.ndarray) -> Line:
    ...


@overload
def line(p: float, q: float, r: float, s: float, t: float, u: float) -> Line3D:
    ...


@overload
def line(x: Point2D, y: Point2D) -> Line2D:
    ...


@overload
def line(x: Point3D, y: Point3D) -> Line3D:
    ...


@overload
def line(a: Plane, b: Plane) -> Line3D:
    ...


@overload
def line(x: Point2D, v: Vector2D) -> Line2D:
    ...


@overload
def line(*args: Any) -> Line:
    ...


def line(*args):
    """The preferred method for creating a line.

    Can create a line using one of the following methods:
    - Pass the coordinates as separate arguments. For instance, `line(1, 2, 3)` returns the 2D homogeneous line `1x + 2y + 3 = 0`.
    - Pass a numpy array with the homogeneous coordinates (NOTE THE DIFFERENCE WITH `point` and `vector`).
    - Pass a Line2D or Line3D instance, in which case `line()` is a no-op.
    - Pass two points of the same dimension, in which case `line()` returns the line through the points.
    - Pass two planes, in which case `line()` returns the line of intersection of the planes.

    """

    if len(args) == 1 and isinstance(args[0], Line):
        return args[0]
    elif len(args) == 1 and isinstance(args[0], Ray):
        return line(args[0].p, args[0].n)
    elif len(args) == 1 and isinstance(args[0], Segment):
        return line(args[0].p, args[0].q)
    elif len(args) == 2 and isinstance(args[0], Point) and isinstance(args[1], Point):
        return args[0].join(args[1])
    elif len(args) == 2 and isinstance(args[0], Plane) and isinstance(args[1], Plane):
        return args[0].meet(args[1])
    elif len(args) == 2 and isinstance(args[0], Point) and isinstance(args[1], Vector):
        x: Point = args[0]
        v: Vector = args[1]
        return x.join(x + v)

    l = _array(args)
    if l.shape == (3,):
        return Line2D(l)
    elif l.shape == (6,):
        return Line3D(l)
    elif l.shape == (4, 4):
        raise ValueError(
            f"cannot create line from matrix form. Use Line3D.from_dual() or Line3D.from_primal() instead."
        )
    else:
        raise ValueError(f"invalid data for line: {l}")


@overload
def plane(p: Plane) -> Plane:
    ...


@overload
def plane(r: Ray3D) -> Ray3D:
    ...


@overload
def plane(a: float, b: float, c: float, d: float) -> Plane:
    ...


@overload
def plane(x: np.ndarray) -> Plane:
    ...


@overload
def plane(p: Point3D, n: Vector3D) -> Plane:
    ...


def plane(*args):
    """The preferred method for creating a plane.

    Can create a plane using one of the following methods:
    - Pass the coordinates as separate arguments. For instance, `plane(1, 2, 3, 4)` returns the 2D homogeneous plane `1x + 2y + 3z + 4 = 0`.
    - Pass a numpy array with the homogeneous coordinates.
    - Pass a Plane instance, in which case `plane()` is a no-op.
    - Pass a Point3D and Vector3D instance, in which case `plane(p, n)` returns the plane corresponding to
    - Pass a ray, which defines r, n as above.
    """
    if len(args) == 1 and isinstance(args[0], Plane):
        return args[0]
    elif len(args) == 1 and isinstance(args[0], Ray3D):
        return Plane.from_point_normal(args[0].p, args[0].n)
    elif (
        len(args) == 2
        and isinstance(args[0], Point3D)
        and isinstance(args[1], Vector3D)
    ):
        r: Point3D = args[0]
        n: Vector3D = args[1]
        return Plane.from_point_normal(r, n)

    p = _array(args)
    if p.shape == (4,):
        return Plane(p)
    else:
        raise ValueError(f"invalid data for plane: {p}")


def l(*args):
    return line(*args)


def pl(*args):
    return plane(*args)
