#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Homogeneous geometry library. 

Copyright (c) 2021, Benjamin D. Killeen. MIT License.

KNOWN ISSUES: 

- When multiplying vectors by scalars it is safer to put the vector on the left. This is because
  your float or int may actually by a numpy scalar, in which case numpy will greedily convert the
  vector (which has an __array__ methodz) to a numpy array, so the multiplication will return an
  np.ndarray and not a geo.Vector. It will still be the *correct* result, just the wrong type (and
  no longer homogeneous).


"""

from __future__ import annotations
import traceback

from typing import (
    Any,
    Union,
    Tuple,
    Optional,
    Type,
    List,
    TypeVar,
    TYPE_CHECKING,
    overload,
)
import logging
from abc import ABC, abstractmethod
from typing_extensions import Self
import numpy as np
import scipy.spatial.distance
from scipy.spatial.transform import Rotation
from copy import deepcopy

if TYPE_CHECKING:
    from ..vol import Volume

from .exceptions import MeetError, JoinError
from .. import utils


Pr = TypeVar("Pr", bound="Primitive")
PV = TypeVar("PV", bound="PointOrVector")
P = TypeVar("P", bound="Point")
V = TypeVar("V", bound="Vector")
L = TypeVar("L", bound="Line")
PL = TypeVar("PL", bound="Plane")


log = logging.getLogger(__name__)


def _to_homogeneous(x: np.ndarray, is_point: bool = True) -> np.ndarray:
    """Convert an array to homogeneous points or vectors.

    Args:
        x (np.ndarray): array with objects on the last axis.
        is_point (bool, optional): if True, the array represents a point, otherwise it represents a vector. Defaults to True.

    Returns:
        np.ndarray: array containing the homogeneous point/vector(s).
    """
    if is_point:
        return np.concatenate([x, np.ones_like(x[..., -1:])], axis=-1)
    else:
        return np.concatenate([x, np.zeros_like(x[..., -1:])], axis=-1)


def _from_homogeneous(x: np.ndarray, is_point: bool = True) -> np.ndarray:
    """Convert array containing homogeneous data to raw form.

    Args:
        x (np.ndarray): array containing homogenous
        is_point (bool, optional): whether the objects are points (true) or vectors (False). Defaults to True.

    Returns:
        np.ndarray: the raw data representing the point/vector(s).
    """
    if is_point:
        return (x / x[..., -1:])[..., :-1]
    else:
        assert np.all(np.isclose(x[..., -1], 0)), f"not a homogeneous vector: {x}"
        return x[..., :-1]


T = TypeVar("T")


class HomogeneousObject(ABC):
    """Any of the objects that rely on homogeneous transforms, all of which wrap a single array called `data`."""

    dtype = np.float32
    data: np.ndarray

    def __init__(
        self,
        data: np.ndarray,
    ) -> None:
        """Create a HomogeneousObject.

        If data is already a Homogeneous object, just uses the data inside it. This ensures that
        calling Point3D(x), for example, where x is already a Point object, doesn't cause an error.

        Args:
            data (np.ndarray): the numpy array with the data.
        """
        data = data.data if isinstance(data, HomogeneousObject) else np.array(data)
        self.data = data.astype(self.dtype)

    @classmethod
    def from_array(
        cls: Type[T],
        x: np.ndarray,
    ) -> T:
        """Create a homogeneous object from its non-homogeous representation as an array."""
        return cls(x)

    @property
    @abstractmethod
    def dim(self) -> int:
        """Get the dimension of the space the object lives in. For transforms, this is the OUTPUT dim."""
        pass

    def tolist(self) -> List:
        """Get a json-save list with the data in this object."""
        return self.data.tolist()

    def __array__(self, *args, **kwargs):
        """Get the non-homogeneous representation of the object.

        For points, this removes the is_point indicator at the bottom (added 1 or 0).
        For transforms and other primitives, this simply returns the data without modifying it.
        """
        return np.array(self.data, *args, **kwargs)

    def __str__(self):
        if self.data.ndim < 2:
            return f"{self.__class__.__name__[0]}{np.array_str(self.data, suppress_small=True)}"
        else:
            return f"{self.__class__.__name__[0]}(\n{np.array_str(self.data, suppress_small=True)}\n)"

    def __repr__(self):
        if self.data.ndim < 2:
            s = np.array_str(self.data, suppress_small=True)
            return f"{self.__class__.__name__}({s})"
        else:
            s = "  " + str(np.array_str(self.data, suppress_small=True)).replace(
                "\n", "\n  "
            )
            return f"{self.__class__.__name__}({s})"

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __iter__(self):
        return iter(np.array(self).tolist())

    def get_data(self) -> np.ndarray:
        return self.data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def get_config(self) -> dict:
        """Get a config dict with the data in this object."""
        return {"data": self.data.tolist()}

    def copy(self) -> Self:
        return self.__class__(**self.get_config())


def get_data(x: Union[HomogeneousObject, List[HomogeneousObject]]) -> np.ndarray:
    if isinstance(x, HomogeneousObject):
        return x.get_data()
    elif isinstance(x, list):
        return np.array([get_data(x_) for x_ in x])
    else:
        raise TypeError


class Primitive(HomogeneousObject):
    """Abstract class for geometric primitives.

    Primitives are the objects contained in a homogeneous frame, like points, vectors, lines, shapes, etc.

    """

    pass


class Joinable(ABC):
    """Abstract class for objects that can be joined together."""

    @abstractmethod
    def join(self, other: Joinable) -> Primitive:
        """Join two objects.

        For example, given two points, get the line that connects them.

        Args:
            other (Primitive): the other primitive.

        Returns:
            Primitive: the joined primitive.
        """
        pass


class Meetable(ABC):
    """Abstract class for objects that can be intersected."""

    @abstractmethod
    def meet(self, other: Meetable) -> Primitive:
        """Get the intersection of two objects.

        For example, given two lines, get the line that is the intersection of them.

        Args:
            other (Primitive): the other primitive.

        Returns:
            Primitive: the intersection of `self` and `other`.
        """
        pass


class PointOrVector(Primitive):
    """A Homogeneous point or vector in any dimension."""

    def __init__(
        self,
        data: np.ndarray,
    ) -> None:
        """Instantiate the homogeneous point or vector and check its dimension."""
        super().__init__(data)

        if self.data.shape != (self.dim + 1,):
            raise ValueError(
                f"invalid shape for {self.dim}D object in homogeneous coordinates: {self.data.shape}"
            )

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Return non-homogeneous numpy representation of object."""
        return np.array(
            _from_homogeneous(self.data, is_point=bool(self.data[-1])), *args, **kwargs
        )

    def normsqr(self, order: int = 2) -> float:
        """Get the squared L-order norm of the vector."""
        return float(np.power(self.data, order).sum())

    def norm(self, *args, **kwargs) -> float:
        """Get the norm of the vector. Pass any arguments to `np.linalg.norm`."""
        return float(np.linalg.norm(self, *args, **kwargs))

    def __div__(self, other: float) -> Self:
        return self * (1 / other)

    def __truediv__(self, other: float) -> Self:
        return self * (1 / other)

    @property
    def x(self) -> float:
        return self.data[0]

    @property
    def y(self) -> float:
        assert self.dim >= 2
        return float(self.data[1])

    @property
    def z(self) -> float:
        assert self.dim >= 3
        return float(self.data[2])

    @property
    def w(self) -> float:
        # Should always be 1 or 0.
        return float(self.data[-1])


class Point(PointOrVector, Joinable):
    def __init__(self, data: np.ndarray) -> None:
        assert not np.isclose(data[-1], 0), "cannot create a point with 0 for w"
        if data[-1] != 1:
            # TODO: relax this constraint internally, and just divide by w when needed
            # NOTE: if we do that, adding/subtracting points with points or vectors should
            # be done with the same w
            data /= data[-1]

        super().__init__(data)

    @classmethod
    def from_array(
        cls: Type[T],
        x: np.ndarray,
    ) -> T:
        x = np.array(x).astype(cls.dtype)
        data = _to_homogeneous(x, is_point=True)
        return cls(data)

    @classmethod
    def from_any(
        cls: Type[Point],
        other: Union[np.ndarray, Point],
    ):
        """If other is not a point, make it one."""
        return other if issubclass(type(other), Point) else cls.from_array(other)

    @overload
    def __sub__(self, other: Point2D) -> Vector2D:
        ...

    @overload
    def __sub__(self, other: Point3D) -> Vector3D:
        ...

    @overload
    def __sub__(self, other: Point) -> Vector:
        ...

    @overload
    def __sub__(self, other: Vector2D) -> Point2D:
        ...

    @overload
    def __sub__(self, other: Vector3D) -> Point3D:
        ...

    @overload
    def __sub__(self, other: Vector) -> Point:
        ...

    def __sub__(self, other):
        """Subtract from a point.

        Note that arrays are not allowed.
        """
        if isinstance(other, Point) and self.dim == other.dim:
            assert np.isclose(
                self.w, other.w
            ), "cannot subtract points with different w"
            if self.dim == 2:
                return Vector2D(self.data - other.data)
            elif self.dim == 3:
                return Vector3D(self.data - other.data)
            else:
                raise NotImplementedError(
                    f"subtraction of points of dimension {self.dim}"
                )
        elif isinstance(other, Vector):
            return type(self)(self.data - other.data)
        elif isinstance(other, np.ndarray):
            raise TypeError(
                f"ambiguous subtraction of {self} and {other}. Can't determine if point or vector."
            )
        else:
            raise TypeError(f"cannot do {other} - {self}")

    def __rsub__(self, other):
        """Means other - self was called."""
        return -self + other

    def __add__(self, other: Union[Vector, np.ndarray]) -> Self:
        """Can add a vector to a point, but cannot add two points."""
        if isinstance(other, Vector):
            if self.dim != other.dim:
                raise ValueError(f"cannot add {self.dim}D point to {other.dim}D vector")
            return type(self)(self.data + other.data)
        elif isinstance(other, Point):
            # TODO: should points be allowed to be added together?
            log.warning(
                f"cannot add two points together: {self} + {other}. This will raise an error in the future."
            )
            traceback.print_stack()
            return point(np.array(self) + np.array(other))
        elif isinstance(other, np.ndarray):
            return self + vector(other)
        else:
            raise TypeError(f"cannot add {type(other)} to a point")

    def __radd__(self, other: Vector) -> Point:
        return self + other

    def __mul__(self, other: Union[int, float]) -> Vector:
        log.warning(
            f"cannot multiply a point by a scalar: {self} * {other}. This will raise an error in the future."
        )
        traceback.print_stack()
        if isinstance(other, (int, float, np.number)) or np.isscalar(other):
            return point(float(other) * np.array(self))
        else:
            return NotImplemented

    def __rmul__(self, other: Union[int, float]) -> Vector:
        return self * other

    def __neg__(self):
        # TODO: this shouldn't be allowed.
        return self * (-1)

    def lerp(self, other: Point, alpha: float = 0.5) -> Self:
        """Linearly interpolate between one point and another.

        Args:
            other (Point): other point.
            alpha (float): fraction of the distance from self to other to travel. Defaults to 0.5 (the midpoint).

        Returns:
            Point: the point that is `alpha` of the way between self and other.
        """
        alpha = float(alpha)
        diff = other - self
        return self + diff * alpha

    def as_vector(self) -> Vector:
        """Get the vector with the same numerical representation as this point."""
        return vector(np.array(self))


class Vector(PointOrVector):
    def __init__(self, data: np.ndarray) -> None:
        if np.isclose(data[-1], 0):
            data[-1] = 0
        assert data[-1] == 0, f"cannot create a vector with non-zero w: {data[-1]}"
        super().__init__(data)

    @classmethod
    def from_array(
        cls: Type[Vector],
        v: np.ndarray,
    ) -> T:
        v = np.array(v).astype(cls.dtype)
        data = _to_homogeneous(v, is_point=False)
        return cls(data)

    @classmethod
    def from_any(
        cls: Type[Vector],
        other: Union[np.ndarray, Vector],
    ):
        """If other is not a Vector, make it one."""
        return other if issubclass(type(other), Vector) else cls.from_array(other)

    def __mul__(self, other: Union[int, float]) -> Self:
        """Vectors can be multiplied by scalars."""
        if isinstance(other, (int, float, np.number)) or np.isscalar(other):
            return type(self)(float(other) * self.data)
        else:
            return NotImplemented

    def __rmul__(self, other: Union[int, float]) -> Self:
        return self.__mul__(other)

    def __matmul__(self, other: Vector) -> float:
        """Inner product between two Vectors."""
        other = self.from_any(other)
        return float(np.dot(self.data, other.data))

    def __add__(self, other: Vector) -> Self:
        """Two vectors can be added to make another vector."""
        if isinstance(other, Vector):
            if self.dim != other.dim:
                raise ValueError(
                    f"cannot add {self.dim}D vector to {other.dim}D vector"
                )
            return type(self)(self.data + other.data)
        elif isinstance(other, np.ndarray):
            return self + vector(other)
        else:
            return NotImplemented

    def __radd__(self, other: Vector):
        return self + other

    def __neg__(self) -> Vector:
        return self.__mul__(-1)

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __rsub__(self, other: Vector):
        return self.__neg__().__add__(other)

    def normalized(self) -> Self:
        return self * (1 / self.norm())

    def hat(self) -> Self:
        return self.normalized()

    def dot(self, other) -> float:
        if isinstance(other, Vector) and self.dim == other.dim:
            return float(np.dot(self.data, other.data))
        else:
            return NotImplemented

    def cross(self, other: Vector) -> Vector3D:
        if isinstance(other, Vector) and self.dim == other.dim:
            return vector(np.cross(self, other))
        else:
            raise TypeError(f"unrecognized type for cross product: {type(other)}")

    def angle(self, other: Vector) -> float:
        """Get the angle between self and other in radians."""
        other = vector(other)
        num = self.dot(other)
        den = self.norm() * other.norm()
        cos_theta = num / den
        if np.isclose(cos_theta, 1):
            return 0
        else:
            return np.arccos(cos_theta)

    def rotation(self, other: Vector) -> FrameTransform:
        """Get the rotation F such that `self || F @ other`.

        NOTE: not tested with 2D vectors.

        Args:
            other (Vector): the vector to rotate to.

        Returns:
            FrameTransform: the rotation that rotates other to self.
        """
        v = self.cross(other)
        if np.isclose(v.norm(), 0):
            return FrameTransform.identity(self.dim)
        v = v.hat()
        theta = self.angle(other)
        rot = Rotation.from_rotvec(v * theta)
        return FrameTransform.from_rotation(rot)

    def rotate(self, n: Vector, theta: Optional[float] = None) -> Vector:
        """Rotate self by the given vector.

        Args:
            n (Vector): the axis of rotation. If theta is None, the magnitude of this vector is
                used. Otherwise, it is ignored.
            theta (float, optional): the angle of rotation. Defaults to None.

        Returns:
            Vector: the rotated vector.
        """
        if theta is None:
            theta = n.norm()

        rot = Rotation.from_rotvec(n.hat() * theta)
        return vector(rot.apply(self))

    def cosine_distance(self, other: Vector) -> float:
        """Get the cosine distance between the angles.

        Args:
            other (Vector): the other vector.

        Returns:
            float: `1 - cos(angle)`, where `angle` is between self and other.
        """
        return float(scipy.spatial.distance.cosine(np.array(self), np.array(other)))

    def as_point(self) -> Point:
        """Gets the point with the same numerical representation as this vector."""
        return point(np.array(self))


class Point2D(Point):
    """Homogeneous point in 2D, represented as an array with [x, y, 1]"""

    dim = 2

    @overload
    def join(self, other: Point2D) -> Line2D:
        ...

    @overload
    def join(self, other: Line2D) -> Vector2D:
        ...

    def join(self, other):
        if isinstance(other, Point2D):
            return Line2D(np.cross(self.data, other.data))
        elif isinstance(other, Line2D):
            raise NotImplementedError("TODO: get vector from point to line")
        else:
            raise TypeError(f"unrecognized type for join: {type(other)}")

    def backproject(self, index_from_world: CameraProjection) -> Line3D:
        """Backproject this point into a line.

        Args:
            index_from_world (Transform): The transform from the world to the index.

        Returns:
            Line3D: The line in 3D space through the source of `index_from_world` and self.

        """
        s = index_from_world.get_center()
        v = index_from_world.inv @ self
        return line(s, v)


class Vector2D(Vector):
    """Homogeneous vector in 2D, represented as an array with [x, y, 0]"""

    dim = 2

    def perpendicular(self, random: bool = False) -> Vector2D:
        """Find an arbitrary perpendicular vector to self.

        Args:
            random: Whether to randomize the vector's direction in
                the perpendicular plane, drawing from [0, 2pi).
                Defaults to False.

        Returns:
            Vector3D: A vector in 3D space, perpendicular
                to the original.

        """
        # TODO: if the vector is 2D, return one of the other vectors in the plane, to keep it 2D.

        v = vector(-self.y, self.x)
        if random and np.random.rand() < 0.5:
            v = -v
        return v


class Point3D(Point):
    """Homogeneous point in 3D, represented as an array with [x, y, z, 1]"""

    dim = 3

    @overload
    def join(self, other: Point3D) -> Line3D:
        ...

    @overload
    def join(self, other: Line3D) -> Plane:
        ...

    def join(self, other):
        if isinstance(other, Point3D):
            # Line joining two points in P^3.
            ax, ay, az, aw = self.data
            bx, by, bz, bw = other.data
            l = np.array(
                [
                    az * bw - aw * bz,  # p
                    ay * bw - aw * by,  # q
                    ay * bz - az * by,  # r
                    ax * bw - aw * bx,  # s
                    ax * bz - az * bx,  # t
                    ax * by - ay * bx,  # u
                ]
            )
            return Line3D(l)
        elif isinstance(other, Line3D):
            return Plane(self.data.T @ other.L)
        elif isinstance(other, Plane):
            raise NotImplementedError("TODO: get vector from point to plane")
        else:
            raise TypeError(f"unrecognized type for join: {type(other)}")


class Vector3D(Vector):
    """Homogeneous vector in 3D, represented as an array with [x, y, z, 0]"""

    dim = 3

    def as_plane(self) -> Plane:
        """Get the plane through the origin with this vector as its normal."""
        return Plane(self.data)

    def perpendicular(self, random: bool = False) -> Vector3D:
        """Find an arbitrary perpendicular vector to self.

        Args:
            random: Whether to randomize the vector's direction in
                the perpendicular plane, drawing from [0, 2pi).
                Defaults to False.

        Returns:
            Vector3D: A vector in 3D space, perpendicular
                to the original.

        """
        # TODO: if the vector is 2D, return one of the other vectors in the plane, to keep it 2D.

        if self.x == self.y == self.z == 0:
            raise ValueError("zero-vector")

        # If one dimension is zero, this can be solved by setting that to
        # non-zero and the others to zero. Example: (4, 2, 0) lies in the
        # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
        if self.x == 0:
            return vector(1, 0, 0)
        if self.y == 0:
            return vector(0, 1, 0)
        if self.z == 0:
            return vector(0, 0, 1)

        # arbitrarily set a = b = 1
        # then the equation simplifies to
        #     c = -(x + y)/z
        v = vector(1, 1, -1.0 * (self.x + self.y) / self.z).hat()

        if random:
            angle = np.random.uniform(0, 2 * np.pi)
            v = vector(Rotation.from_rotvec(angle * self.hat()).apply(v))

        return v


class HyperPlane(Primitive, Meetable):
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


class Line(Primitive, Meetable):
    """Abstract parent class for lines."""

    @abstractmethod
    def get_direction(self) -> Vector:
        """Get the direction of the line.

        Returns:
            Vector: The unit-length direction of the line.

        """
        pass

    @abstractmethod
    def get_point(self) -> Point:
        """Get an arbitrary point on the line.

        Returns:
            Point: A point on the line.

        """
        pass

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

    def meet(self, other):
        if isinstance(other, Line2D):
            return Point2D(np.cross(self.data, other.data))
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
    def from_point_normal(cls, r: Point3D, n: Vector3D):
        """Make a plane from a point and a normal vector.

        Args:
            r (Point3D): The point on the plane.
            n (Vector3D): The normal vector of the plane.

        Returns:
            Plane: The plane.
        """
        r = point(r)
        n = vector(n)
        a, b, c = n
        d = -(a * r.x + b * r.y + c * r.z)
        return cls(np.array([a, b, c, d]))

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


class Line3D(Line, Primitive, Joinable, Meetable):
    """Represents a line in 3D as a 6-vector (p,q,r,s,t,u).

    Based on https://dl.acm.org/doi/pdf/10.1145/965141.563900.

    """

    dim = 3

    def __init__(self, data: np.ndarray) -> None:
        assert data.shape == (6,)
        # TODO: assert the necessary line conditions
        super().__init__(data)

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


### convenience functions for instantiating primitive objects ###


def _array(x: Union[List[np.ndarray], List[float]]) -> np.ndarray:
    # TODO: this is a little sketchy
    if len(x) == 1:
        return np.array(x[0])
    else:
        if isinstance(x[0], np.ndarray):
            log.warning(f"got unusual args for array: {x}")
            traceback.print_stack()
        return np.array(x)


@overload
def point(p: P) -> P:
    ...


@overload
def point(v: Vector2D) -> Point2D:
    ...


@overload
def point(v: Vector3D) -> Point3D:
    ...


@overload
def point(x: float, y: float) -> Point2D:
    ...


@overload
def point(x: float, y: float, z: float) -> Point3D:
    ...


@overload
def point(x: np.ndarray) -> Point:
    ...


@overload
def point(*args: Any) -> Point:
    ...


def point(*args):
    """The preferred method for creating a point.

    There are three ways to create a point using `point()`.
    - Pass the coordinates as separate arguments. For instance, `point(0, 0)` returns the 2D homogeneous point for the origin `Point2D([0, 0, 1])`.
    - Pass a numpy array containing the non-homogeneous representation of the point. For example `point(np.ndarray([0, 1, 2]))` is the 3D homogeneous point `Point3D([0, 1, 2, 1])`.
    - Pass a Point2D or Point3D instance, in which case `point()` just returns the first argument.

    `point()` shoud NOT be given a numpy array containing the homogeneous data. In this case, use the `Point2D` and `Point3D` constructors directly.

    Raises:
        ValueError: if arguments cannot be parsed as data for a point.

    Returns:
        Union[Point2D, Point3D]: Point2D or Point3D.
    """
    if len(args) == 1 and isinstance(args[0], Point):
        return args[0]

    x = _array(args)
    if x.shape == (2,):
        return Point2D.from_array(x)
    elif x.shape == (3,):
        return Point3D.from_array(x)
    else:
        raise ValueError(f"invalid data for point: {x}")


@overload
def vector(v: V) -> V:
    ...


@overload
def vector(p: Point2D) -> Vector2D:
    ...


@overload
def vector(p: Point3D) -> Vector3D:
    ...


@overload
def vector(x: float, y: float) -> Vector2D:
    ...


@overload
def vector(x: float, y: float, z: float) -> Vector3D:
    ...


@overload
def vector(x: np.ndarray) -> Vector:
    ...


@overload
def vector(*args: Any) -> Vector:
    ...


def vector(*args):
    """The preferred method for creating a vector.

    There are three ways to create a point using `vector()`.

    - Pass the coordinates as separate arguments. For instance, `vector(0, 0)` returns the 2D homogeneous vector `Vector2D([0, 0, 0])`.
    - Pass a numpy array containing the non-homogeneous representation of the vector.
      For example `vector(np.ndarray([0, 1, 2]))` is the 3D homogeneous veector `Vector3D([0, 1, 2, 0])`.
    - Pass a Vector2D or Vector3D instance, in which case `vector()` just returns the first argument.

    `point()` should NOT be given a numpy array containing the homogeneous data. In this case, use the `Vector2D` and `Vector3D` constructors directly.

    Raises:
        ValueError: if arguments cannot be parsed as data for a point.

    Returns:
        Union[Point2D, Point3D]: Point2D or Point3D.
    """
    if len(args) == 1 and isinstance(args[0], Vector):
        return args[0]

    v = _array(args)
    if v.shape == (2,):
        return Vector2D.from_array(v)
    elif v.shape == (3,):
        return Vector3D.from_array(v)
    else:
        raise ValueError(f"invalid data for vector: {v}")


@overload
def line(l: Line2D) -> Line2D:
    ...


@overload
def line(l: Line3D) -> Line3D:
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
def line(x: Point3D, v: Vector3D) -> Line3D:
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
def plane(a: float, b: float, c: float, d: float) -> Plane:
    ...


@overload
def plane(x: np.ndarray) -> Plane:
    ...


@overload
def plane(r: Point3D, n: Vector3D) -> Plane:
    ...


def plane(*args):
    """The preferred method for creating a plane.

    Can create a plane using one of the following methods:
    - Pass the coordinates as separate arguments. For instance, `plane(1, 2, 3, 4)` returns the 2D homogeneous plane `1x + 2y + 3z + 4 = 0`.
    - Pass a numpy array with the homogeneous coordinates.
    - Pass a Plane instance, in which case `plane()` is a no-op.
    - Pass a Point3D and Vector3D instance, in which case `plane(r, n)` returns the plane corresponding to
    """
    if len(args) == 1 and isinstance(args[0], Plane):
        return args[0]
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


def _point_or_vector(data: np.ndarray):
    """Convert a point where the "homogeneous" element may not be 1."""

    if bool(data[-1]):
        return point(data[:-1] / data[-1])
    else:
        return vector(data[:-1])


### aliases ###
def p(*args):
    return point(*args)


def v(*args):
    return vector(*args)


def l(*args):
    return line(*args)


def pl(*args):
    return plane(*args)


"""
Transforms
"""


class Transform(HomogeneousObject):
    def __init__(self, data: np.ndarray, _inv: Optional[np.ndarray] = None) -> None:
        """A geometric transform represented as a matrix multiplication on homogeneous points or vectors.

        The Transform class should not generally be used directly. Rather it arises from composing different transforms together.

        Args:
            data (np.ndarray): the numpy representation of the matrix.
            _inv (Optional[np.ndarray], optional): the matrix representation of the inverse of the transformation.
                This is only necessary when `_inv` is not overriden by subclasses. Defaults to None.
        """
        super().__init__(data)
        self._inv = _inv if _inv is not None else np.linalg.pinv(data)

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """Output the transform as a non-homogeneous matrix.

        The convention here is that "nonhomegenous" transforms would still have the last column,
        so it would take in homogeneous objects, but it doesn't have the last row, so it outputs non-homogeneous objects.

        If someone wants the data array, they can access it directly.

        Returns:
            np.ndarray: the non-homogeneous array
        """

        return np.array(self.data, *args, **kwargs)

    @classmethod
    def from_array(cls, array: np.ndarray) -> Transform:
        """Convert non-homogeneous matrix to homogeneous transform.

        Usually, one would instantiate Transforms directly from the homogeneous matrix `data` or using one of the other classmethods.

        Args:
            array (np.ndarray): transformation matrix.

        Returns:
            Transform: the transform.
        """
        bottom_row = np.zeros((1, array.shape[1]))
        bottom_row[0, -1] = 1
        data = np.concatenate([array, bottom_row], axis=0)
        return cls(data)

    @overload
    def __matmul__(self: FrameTransform, other: FrameTransform) -> FrameTransform:
        ...

    @overload
    def __matmul__(self: FrameTransform, other: PV) -> PV:
        ...

    @overload
    def __matmul__(self: FrameTransform, other: List[Pr]) -> List[Pr]:
        ...

    @overload
    def __matmul__(self: FrameTransform, other: L) -> L:
        ...

    @overload
    def __matmul__(self: FrameTransform, other: Plane) -> Plane:
        ...

    @overload
    def __matmul__(
        self: FrameTransform, other: CameraIntrinsicTransform
    ) -> CameraIntrinsicTransform:
        ...

    @overload
    def __matmul__(self: FrameTransform, other: CameraProjection) -> CameraProjection:
        ...

    @overload
    def __matmul__(self: CameraProjection, other: FrameTransform) -> CameraProjection:
        ...

    @overload
    def __matmul__(self: CameraProjection, other: Point3D) -> Point2D:
        ...

    @overload
    def __matmul__(self: CameraProjection, other: List[Point3D]) -> List[Point2D]:
        ...

    # @overload
    # def __matmul__(self: CameraProjection, other: Vector3D) -> Vector2D:
    #     ...
    # TODO: this is not actually well defined unless the vector has a location.

    @overload
    def __matmul__(self: CameraProjection, other: Line3D) -> Line2D:
        ...

    @overload
    def __matmul__(self, other: Primitive) -> Primitive:
        ...

    def __matmul__(
        self,
        other: Union[Transform, PointOrVector],
    ) -> Union[Transform, PointOrVector]:
        def check_dim():
            assert (
                self.input_dim == other.dim
            ), f"dimensions must match. Got {self.input_dim} and {other.dim}"

        if isinstance(other, PointOrVector):
            check_dim()
            out = self.data @ other.data
            return _point_or_vector(self.data @ other.data)
        # elif isinstance(self, FrameTransform) and isinstance(other, Line2D):
        #     check_dim()
        #     l0, l1, l2 = other.data
        #     r00, r01, r10, r11 = self.R.flat
        #     p0, p1 = self.t
        #     l0_ = r11 * l0 - r10 * l1
        #     l1_ = -r01 * l0 + r00 * l1
        #     l2_ = np.linalg.det([[l0, l1, l2], [r00, r01, p0], [r10, r11, p1]])
        #     return line(l0_, l1_, l2_)
        elif isinstance(self, (FrameTransform, CameraProjection)) and isinstance(
            other, list
        ):
            # TODO: handle fiducials elegantly, so the transform just affects their
            # world_from_anatomical. It think we need to change the design so that matmul is
            # implemented by the right side object as well, because this function is getting quite
            # bulky.
            return [self @ x for x in other]
        elif isinstance(self, FrameTransform) and isinstance(other, (Line2D, Line3D)):
            p = other.get_point()
            v = other.get_direction()
            p_ = self @ p
            v_ = self @ v
            return line(p_, v_)
        elif isinstance(self, FrameTransform) and isinstance(other, Plane):
            p = other.get_point()
            n = other.get_normal()
            p_ = self @ p
            n_ = self @ n
            return plane(p_, n_)
        elif isinstance(self, CameraProjection) and isinstance(other, Line3D):
            p1 = other.get_point()
            v = other.get_direction()
            p2 = p1 + v
            p1_ = self @ p1
            p2_ = self @ p2
            return line(p1_, p2_)
        elif isinstance(self, FrameTransform) and isinstance(
            other, CameraIntrinsicTransform
        ):
            return CameraIntrinsicTransform(
                self.data @ other.data, other._sensor_height, other._sensor_width
            )
        elif isinstance(self, CameraProjection) and isinstance(other, FrameTransform):
            return CameraProjection(self.intrinsic.copy(), self.extrinsic @ other)
        elif isinstance(self, FrameTransform) and isinstance(other, CameraProjection):
            return CameraProjection(self @ other.intrinsic, other.extrinsic.copy())
        elif isinstance(self, FrameTransform) and isinstance(other, FrameTransform):
            # very common case of composing FrameTransforms.
            return FrameTransform(self.data @ other.data)
        elif isinstance(other, Transform):
            # if other is a Transform, then compose their inverses as well to store that.
            check_dim()
            _inv = other.inv.data @ self.inv.data
            return Transform(self.data @ other.data, _inv=_inv)
        else:
            return NotImplemented

    @property
    def dim(self):
        """The output dimension of the transformation."""
        return self.data.shape[0] - 1

    @property
    def input_dim(self):
        """The input dimension of the transformation."""
        return self.data.shape[1] - 1

    def __call__(
        self,
        other: PointOrVector,
    ) -> PointOrVector:
        return self @ other

    @property
    def inv(self) -> Self:
        """Get the inverse of the Transform.

        Returns:
            (Transform): a Transform (or subclass) that is well-defined as the inverse of this transform.

        Raises:
            NotImplementedError: if _inv is None and method is not overriden.
        """
        if self._inv is None:
            raise NotImplementedError(
                "inverse operation not well-defined except when overridden by subclasses, unless _inv provided"
            )

        return Transform(self._inv, _inv=self.data)

    @overload
    def inverse(self: FrameTransform) -> FrameTransform:
        ...

    @overload
    def inverse(self: Transform) -> Transform:
        ...

    def inverse(self):
        """Get the inverse of the Transform.

        Returns:
            (Transform): a Transform (or subclass) that is well-defined as the inverse of this transform.

        Raises:
            NotImplementedError: if _inv is None and method is not overriden.
        """
        return self.inv

    def get_center(self) -> Point3D:
        """If the transform is a projection, get the center of the projection.

        Returns:
            (Point3D): the center of the projection.

        Raises:
            ValueError: if the transform is not a projection.

        """
        if self.shape != (3, 4):
            raise ValueError("transform must be a projection")

        p1 = plane(self.data[0, :])
        p2 = plane(self.data[1, :])
        p3 = plane(self.data[2, :])
        return p1.meet(p2).meet(p3)


class FrameTransform(Transform):
    def __init__(
        self,
        data: np.ndarray,  # the homogeneous frame transformation matrix
    ) -> None:
        """Defines a rigid (affine) transformation from one frame to another.

        So that, for a point `x` in world-coordinates `F(x)` (or `F @ x`) is the same point in `F`'s
        coordinates. Note that if `x` is a numpy array, it is assumed to be a point.

        In order to maximize readability, the suggested naming convention for frames should be as follows.
        As an example, if there is a volume with an IJK index frame (indices into the volume), an anatomical frame (e.g. LPS),
        both of which are situated somewhere in world-space, `F_world_lps` should be the LPS frame, `F_lps_ijk`
        should be the index frame in the LPS system. In this setup, then, given an index-space point [i,j,k],
        the corresponding world-space representation is `[x,y,z] = F_world_lps @ F_lps_ijk @ [i,j,k]`.

        In this setup, an inverse simply flips the two subscripted frames, so one would denote `F_lps_world = F_world_lps.inv`.
        Thus, if `[x,y,z]` is a world-space representation of a point, `F_lps_ijk.inv @ F_world_lps.inv @ [x,y,z]`
        is the point's representation in index-space.

        The idea here is that the frame being transformed to comes first, so that (if no inverses are present) one can glance
        at the outermost frame to see what frame the point is in. This also allows one to easily verify that frametransforms rightly
        go next to one another by checking whether the inner frames match.

        The "F2_to_F1" convention for naming frames is confusing and should be avoided. Instead, this would be `F_F1_F2` (hence the confusion).

        Helpful resources:
        - https://nipy.org/nibabel/coordinate_systems.html

        Args:
            data (np.ndarray): the homogeneous matrix of the transformation.

        """
        super().__init__(data)

    @property
    def dim(self):
        return self.data.shape[0] - 1

    @classmethod
    def from_rt(
        cls,
        rotation: Optional[Union[Rotation, np.ndarray]] = None,
        translation: Optional[Union[Point3D, Vector3D, np.ndarray]] = None,
        dim: Optional[int] = None,
    ) -> FrameTransform:
        """Make a frame translation from a rotation and translation, as [R,t], where x' = Rx + t.

        Args:
            rotation (Optional[np.ndarray], optional): Rotation matrix. If None, uses the identity. Defaults to None.
            translation: Optional[Union[Point3D, np.ndarray]]: Translation of the transformation. If None, no translation. Defaults to None.
            dim (Optional[int], optional): Must be provided if both  Defaults to None.

        If both args are None,

        Returns:
            FrameTransform: The transformation `F` such that `F(x) = rotation @ x + translation`
        """
        # Process weird rotation shapes
        if isinstance(rotation, Rotation):
            rotation = rotation.as_matrix()
        elif isinstance(rotation, np.ndarray) and rotation.shape == (4,):
            rotation = Rotation.from_quat(rotation).as_matrix()

        if rotation is not None:
            dim = np.array(rotation).shape[0]
        elif translation is not None:
            dim = np.array(translation).shape[0]
        else:
            return cls.identity(dim)

        R = np.eye(dim) if rotation is None else np.array(rotation)
        t = np.zeros(dim) if translation is None else np.array(translation)

        assert t.shape[0] == dim
        assert R.shape == (dim, dim), f"{dim} does not match R.shape {R.shape}"

        data = np.concatenate(
            [
                np.concatenate([R, t[:, np.newaxis]], axis=1),
                np.concatenate([np.zeros((1, dim)), [[1]]], axis=1),
            ],
            axis=0,
        )

        return cls(data)

    @classmethod
    def from_scaling(
        cls: Type[FrameTransform],
        scaling: Union[int, float],
        dim: int = 3,
    ) -> FrameTransform:
        """Create a frame based on scaling dimensions uniformly.

        Args:
            cls (Type[FrameTransform]): the class.
            dim (int, optional): the dimension of the frame. Defaults to 3.

        Returns:
            FrameTransform:
        """
        F = np.eye(dim + 1)
        F[dim, dim] = 1 / scaling
        return cls(F)

    @classmethod
    def from_translation(
        cls,
        translation: np.ndarray,
    ) -> FrameTransform:
        """Wrapper around from_rt."""
        return cls.from_rt(translation=translation)

    @classmethod
    def from_rotation(
        cls,
        rotation: Union[Rotation, np.ndarray],
    ) -> FrameTransform:
        """Wrapper around from_rt."""
        return cls.from_rt(rotation=rotation)

    @classmethod
    def identity(
        cls: Type[FrameTransform],
        dim: int = 3,
    ) -> FrameTransform:
        """Get the identity FrameTransform."""
        return cls.from_rt(np.identity(dim), np.zeros(dim))

    @classmethod
    def from_origin(
        cls,
        origin: Point,
    ) -> FrameTransform:
        """Make a transfrom to a frame knowing the origin.

        Suppose `origin` is point where frame `B` has its origin, as a point
        in frame `A`. Make the `B_from_A` transform.
        This just negates `origin`, but this is often counterintuitive.

        Args:
            origin (Point): origin of the target frame in the world frame

        Returns:
            FrameTransform: the B_from_A transform.
        """
        origin = point(origin)
        return cls.from_rt(translation=-origin)

    @classmethod
    def from_point_correspondence(
        cls,
        points_B: Union[List[Point], np.ndarray],
        points_A: Union[List[Point], np.ndarray],
    ):
        """Create a (rigid) frame transform from a known point correspondence.

        Args:
            points_B: a list of N corresponding points in the B frame.
            points_A: a list of N points in the A frame (or an array with shape [N, 3]).

        Returns:
            FrameTransform: the `B_from_A` transform that minimizes the mean squared distance
                between matching points.

        """
        a = np.array(points_A)
        b = np.array(points_B)

        if a.shape != b.shape:
            raise ValueError(
                f"unmatched shapes for point correspondence: {a.shape}, {b.shape}"
            )

        N = a.shape[0]

        # get centroids
        a_m = a.mean(axis=0)
        b_m = b.mean(axis=0)

        # get points in centroid frames
        a_q = a - a_m
        b_q = b - b_m

        # solve the optimization with SVD
        H = np.sum([np.outer(a_q[i], b_q[i]) for i in range(N)], axis=0)
        U, S, VT = np.linalg.svd(H)
        V = VT.T
        R = V @ U.T
        d = np.linalg.det(R)
        if d < 0:
            V[:, 2] *= -1
            R = V @ U.T
            d = np.linalg.det(R)

        t = b_m - R @ a_m

        if not np.isclose(d, 1):
            raise RuntimeError(f"det(R) = {d}, should be +1 for rotation matrices.")

        return cls.from_rt(rotation=R, translation=t)

    @classmethod
    def from_points(
        cls,
        points_B: Union[List[Point], np.ndarray],
        points_A: Union[List[Point], np.ndarray],
    ):
        """Create a (rigid) frame transform from a known point correspondence.

        Args:
            points_B: a list of N corresponding points in the B frame.
            points_A: a list of N points in the A frame (or an array with shape [N, 3]).

        Returns:
            FrameTransform: the `B_from_A` transform that minimizes the mean squared distance
                between matching points.

        """
        # TODO: either generalize this to not require correspondence, or do `from_pointclouds`
        return cls.from_point_correspondence(points_B, points_A)

    @classmethod
    def from_line_segments(
        cls,
        x_B: Point3D,
        y_B: Point3D,
        x_A: Point3D,
        y_A: Point3D,
    ) -> FrameTransform:
        """Get the `B_from_A` frame transform that aligns the line segments, given by endpoints.

        Perfectly aligns the two line segments, so there is possibly some scaling.

        Args:
            x_B (Point3D): The first endpoint, in frame B.
            y_B (Point3D): The second endpoint, in frame B.
            x_A (Point3D): The first endpoint, in frame A.
            y_A (Point3D): The second endpoint, in frame A.

        Returns:
            FrameTransform: A `B_from_A` transform that aligns the points.
                Note that this is not unique, due to rotation about the axis between the points.
        """
        x_B = point(x_B)
        y_B = point(y_B)
        x_A = point(x_A)
        y_A = point(y_A)

        # First, get the vectors pointing from x to y in each frames.
        x2y_A = y_A - x_A
        x2y_B = y_B - x_B

        # Second, get the rotation between the vectors.
        rotvec = x2y_A.cross(x2y_B).hat()
        rotvec = rotvec * x2y_A.angle(x2y_B)
        rot = Rotation.from_rotvec(np.array(rotvec))

        return (
            cls.from_translation(x_B)
            @ cls.from_scaling(x2y_B.norm() / x2y_A.norm())
            @ cls.from_rotation(rot)
            @ cls.from_translation(-x_A.as_vector())
        )

    @classmethod
    def from_pointdir(
        cls: Type[FrameTransform],
        origin: Point3D,
        direction: Vector3D,
        axis: Union[str, Vector3D] = "z",
    ) -> FrameTransform:
        """Get the pose of the coordinate frame given its origin and direction of the given axis.

        Args:
            origin (Point3D): The origin of the frame.
            direction (Vector3D): The direction of the axis.
            axis (Union[str, Vector3D], optional): The axis to align with the direction. Defaults to "z".

        """
        ax = (
            dict(x=vector(1, 0, 0), y=vector(0, 1, 0), z=vector(0, 0, 1))[axis.lower()]
            if isinstance(axis, str)
            else vector(axis)
        )

        rotvec = ax.cross(direction).hat()
        rotvec = rotvec * ax.angle(direction)
        rot = Rotation.from_rotvec(np.array(rotvec))
        return cls.from_rt(rotation=rot, translation=origin)

    @property
    def R(self):
        return self.data[0 : self.dim, 0 : self.dim]

    @R.setter
    def R(self, R):
        self.data[0 : self.dim, 0 : self.dim] = R

    @property
    def t(self):
        return self.data[0 : self.dim, self.dim]

    @t.setter
    def t(self, t):
        self.data[0 : self.dim, self.dim] = t

    @property
    def inv(self):
        # This is necessary because we use FrameTransform to represent affine as well as rigid transforms.
        # TODO: separate?
        R_inv = np.linalg.inv(self.R)
        return FrameTransform.from_rt(R_inv, -(R_inv @ self.t))

    @property
    def o(self):
        """If this is the A_from_B transform, return the origin of frame B in frame A."""
        return point(self.t)

    def tostring(self):
        formatter = {"float_kind": lambda x: "%.8f" % x}
        lines = [
            np.array2string(self.data[i], separator=" ", formatter=formatter)[1:-1]
            for i in range(self.data.shape[0])
        ]
        return "\n".join(lines)

    def toarray(self):
        """Return the transform as a 3x4 numpy array.

        This is different from calling np.array() on the transform, which returns a 4x4 array.

        """
        return self.data[:-1, :].astype(np.float32).copy()

    def as_quatpos(self) -> np.ndarray:
        """Return the transform as a quaternion and position.

        Returns:
            np.ndarray: A 7-element array, with the first 4 elements being the quaternion, and the last 3 being the position.

        """
        return np.array([*Rotation.from_matrix(self.R).as_quat(), *self.t])

    @classmethod
    def from_quatpos(cls, quatpos: np.ndarray) -> FrameTransform:
        """Create a transform from a quaternion and position.

        Args:
            quatpos (np.ndarray): A 7-element array, with the first 4 elements being the quaternion, and the last 3 being the position.

        Returns:
            FrameTransform: The transform.

        """
        return cls.from_rt(
            Rotation.from_quat(quatpos[:4]).as_matrix(),
            quatpos[4:],
        )


class F(FrameTransform):
    """Alias for FrameTransform."""

    pass


def frame_transform(*args) -> FrameTransform:
    """Convenience function for creating a 3D frame transform.

    The output depends on how the function is called:
    frame_transform() -> 3D identity transform
    frame_transform(None) -> 3D identity transform
    frame_transform(scalar) -> FrameTransform.from_scaling(scalar)
    frame_transform(ft: FrameTransform) -> ft
    frame_transform(data: np.ndarray[4,4]) -> FrameTransform(data)
    frame_transform(R: Rotation | np.ndarray[3,3]) -> FrameTransform.from_rt(R)
    frame_transform(t: Point | np.ndarray[3]) -> FrameTransform.from_translation(t)
    frame_transform((R, t)) -> FrameTransform.from_rt(R, t)
    frame_transform(R, t) -> FrameTransform.from_rt(R, t)
    frame_transform([a00, a10, a20, a01, a11, a21, a02, a12, a22, a03, a13, a23]) -> FrameTransform([
        [a00, a01, a02, a03],
        [a10, a11, a12, a13],
        [a20, a21, a22, a23],
        [0, 0, 0, 1]]
    )

    R maybe be given as a (3,3) matrix or as a 9-vector. If provided as a 9-vector, column major order is assumed,
    such that (a11, a21, a31, a12, a22, a32, a13, a23, a33) is the rotation matrix.
    [[a11, a12, a13],
     [a21, a22, a23],
     [a31, a32, a33]]

    [R | t] may be given as a (12,) array-like, where the first 9 elements are the rotation in column major order,  and the last 3 are the translation.

    If a string provided, it is converted to an array with whitespace separator.

    Returns:
        FrameTransform: [description]
    """

    if len(args) == 0:
        return FrameTransform.identity()
    elif len(args) == 1:
        a = args[0]
        if a is None:
            return FrameTransform.identity()
        elif issubclass(type(a), Point):
            return FrameTransform.from_translation(a)
        elif isinstance(a, Rotation):
            return FrameTransform.from_rotation(a)
        elif isinstance(a, FrameTransform):
            return a
        elif isinstance(a, (int, float)):
            return FrameTransform.from_scaling(a)
        elif isinstance(a, np.ndarray):
            if a.shape == (4, 4):
                return FrameTransform(a)
            elif a.shape == (3, 3):
                return FrameTransform.from_rt(rotation=a)
            elif a.shape == (3, 4):
                return FrameTransform.from_array(a)
            elif a.shape == (3,) or a.shape == (1, 3):
                return FrameTransform.from_rt(translation=a)
            elif a.shape == (12,):
                # ITK-style transform, column-major order
                r = a[:9].reshape((3, 3)).T
                t = a[9:].reshape((3,))
                return FrameTransform.from_rt(r, t)
            elif a.shape == (16,):
                # Assumed to be row-major order
                return FrameTransform(a.reshape((4, 4)))
            elif a.shape == (7,):
                # Quaternion position
                return FrameTransform.from_quatpos(a)
            else:
                raise TypeError(f"couldn't convert numpy array to FrameTransform: {a}")
        elif isinstance(a, (tuple, list)):
            if len(a) == 2:
                return frame_transform(a[0], a[1])
            else:
                return frame_transform(np.array(a))
        elif isinstance(a, str):
            return frame_transform(np.fromstring(a, sep=" "))
        else:
            raise TypeError(f"couldn't convert to FrameTransform: {a}")
    elif len(args) == 2:
        if isinstance(args[0], Rotation) and isinstance(args[1], Point3D):
            return FrameTransform.from_rt(args[0], args[1])
        elif (
            isinstance(args[0], np.ndarray)
            and isinstance(args[1], np.ndarray)
            and args[0].shape == (3, 3)
            and args[1].shape == (3,)
        ):
            return FrameTransform.from_rt(rotation=args[0], translation=args[1])
        elif isinstance(args[0], (list, tuple)) and isinstance(args[1], (list, tuple)):
            r = np.array(args[0])
            if r.shape == (3, 3):
                pass
            elif r.shape == (9,):
                r = r.reshape(3, 3).T
            t = np.array(args[1])
            if r.shape != (3, 3) or t.shape != (3,):
                raise TypeError(f"couldn't convert to FrameTransform: {args}")
            return FrameTransform.from_rt(r, t)
        else:
            raise TypeError(
                f"could not parse FrameTransfrom from [R, t]: [{args[0]}, {args[1]}]"
            )
    else:
        raise TypeError(f"too many arguments: {args}")


f = frame_transform

RAS_from_LPS = FrameTransform(
    np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
)
LPS_from_RAS = RAS_from_LPS.inv

mm_from_m = FrameTransform.from_scaling(1e3)
m_from_mm = FrameTransform.from_scaling(1e-3)
cm_from_m = FrameTransform.from_scaling(1e2)
m_from_cm = FrameTransform.from_scaling(1e-2)
mm_from_cm = FrameTransform.from_scaling(1e1)
cm_from_mm = FrameTransform.from_scaling(1e-1)

# Slicer coordinates are annoying. You flip the Z axis and convert to meters.
unity_from_slicer = frame_transform(
    """
1 0 0 0
0 1 0 0
0 0 -1 0
0 0 0 1000
"""
)
slicer_from_unity = unity_from_slicer.inv


# TODO: reorganize geo so you have primitives.py and transforms.py. Have separate classes for each type of transform?


class CameraIntrinsicTransform(FrameTransform):
    dim: int = 2
    input_dim: int = 2

    """The intrinsic camera transform.
    
    It should be scaled such that the units of the matrix (including the focal length) are in pixels
    """

    def __init__(
        self,
        data: np.ndarray,
        sensor_height: Optional[int] = None,
        sensor_width: Optional[int] = None,
    ) -> None:
        super().__init__(data)
        assert self.data.shape == (3, 3), f"unrecognized shape: {self.data.shape}"
        self._sensor_height = sensor_height
        self._sensor_width = sensor_width

    def get_config(self):
        return {
            "data": self.data.tolist(),
            "sensor_height": self.sensor_height,
            "sensor_width": self.sensor_width,
        }

    @classmethod
    def from_parameters(
        cls,
        optical_center: Point2D,
        focal_length: Union[float, Tuple[float, float]] = 1,
        shear: float = 0,
        aspect_ratio: Optional[float] = None,
    ) -> CameraIntrinsicTransform:
        """The camera intrinsic matrix.

        The intrinsic matrix is fundamentally a FrameTransform in 2D, namely `index_from_camera2d`.
        It transforms to the index-space of the image (as mapped on the sensor)
        from the index-space centered on the principle ray.

        Note:
            Focal lengths are usually measured in world units (e.g. millimeters.). This function handles the conversion.

        Useful references include Szeliski's "Computer Vision"
        - https://ksimek.github.io/2013/08/13/intrinsic/

        Args:
            optical_center (Point2D): the index-space point where the isocenter (or pinhole) is centered.
            focal_length (Union[float, Tuple[float, float]]): the focal length in index units. Can be a tubple (f_x, f_y),
                or a scalar used for both, or a scalar modified by aspect_ratio, in index units.
            shear (float): the shear `s` of the camera.
            aspect_ratio (Optional[float], optional): the aspect ratio `a` (for use with one focal length). If not provided, aspect
                ratio is 1. Defaults to None.

        Returns:
            CameraIntrinsicTransform: The camera intrinsic matrix.

        """
        optical_center = point(optical_center)
        assert optical_center.dim == 2, "center point not in 2D"

        cx, cy = np.array(optical_center)

        if aspect_ratio is None:
            fx, fy = utils.tuplify(focal_length, 2)
        else:
            assert isinstance(
                focal_length, (float, int)
            ), "cannot use aspect ratio if both focal lengths provided"
            fx, fy = (focal_length, aspect_ratio * focal_length)

        data = np.array([[fx, shear, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32)

        return cls(data)

    @classmethod
    def from_sizes(
        cls,
        sensor_size: Union[int, Tuple[int, int]],
        pixel_size: Union[float, Tuple[float, float]],
        source_to_detector_distance: float,
    ) -> CameraIntrinsicTransform:
        """Generate the camera from human-readable parameters.

        This is the recommended way to create the camera. Note that although pixel_size and source_to_detector distance are measured in world units,
        the camera intrinsic matrix contains no information about the world, as these are merely used to compute the focal length in pixels.

        Args:
            sensor_size (Union[float, Tuple[float, float]]): (width, height) of the sensor, or a single value for both, in pixels.
            pixel_size (Union[float, Tuple[float, float]]): (width, height) of a pixel, or a single value for both, in world units (e.g. mm).
            source_to_detector_distance (float): distance from source to detector in world units.

        Returns:

        """
        sensor_size = utils.tuplify(sensor_size, 2)
        pixel_size = utils.tuplify(pixel_size, 2)
        fx = source_to_detector_distance / pixel_size[0]
        fy = source_to_detector_distance / pixel_size[1]
        optical_center = point(sensor_size[0] / 2, sensor_size[1] / 2)
        return cls.from_parameters(optical_center=optical_center, focal_length=(fx, fy))

    @property
    def optical_center(self) -> Point2D:
        return Point2D(self.data[:, 2])

    @property
    def fx(self) -> float:
        return self.data[0, 0]

    @property
    def fy(self) -> float:
        return self.data[1, 1]

    @property
    def aspect_ratio(self) -> float:
        """Image aspect ratio."""
        return self.fy / self.fx

    @property
    def focal_length(self) -> float:
        """Focal length in the matrix units."""
        return self.fx

    @property
    def sensor_width(self) -> int:
        """Get the sensor width in the matrix units.

        Assumes optical center is at the center of the sensor.

        Based on the convention of origin in top left, with x pointing to the right and y pointing down."""
        if self._sensor_width is None:
            return int(np.ceil(2 * self.data[0, 2]))
        else:
            return self._sensor_width

    @sensor_width.setter
    def sensor_width(self, value: int):
        self._sensor_width = value

    @property
    def sensor_height(self) -> int:
        """Get the sensor height in pixels.

        Assumes optical center is at the center of the sensor.

        Based on the convention of origin in top left, with x pointing to the right and y pointing down."""
        if self._sensor_height is None:
            return int(np.ceil(2 * self.data[1, 2]))
        else:
            return self._sensor_height

    @sensor_height.setter
    def sensor_height(self, value: int):
        self._sensor_height = value

    @property
    def sensor_size(self) -> Tuple[int, int]:
        """Tuple with the (width, height) of the sense/image, in matrix units."""
        return (self.sensor_width, self.sensor_height)


class CameraProjection(Transform):
    dim = 3
    index_from_camera2d: CameraIntrinsicTransform
    camera3d_from_world: FrameTransform

    def __init__(
        self,
        intrinsic: Union[CameraIntrinsicTransform, np.ndarray],
        extrinsic: Union[FrameTransform, np.ndarray],
    ) -> None:
        """A class for instantiating camera projections.

        The object itself contains the "index_from_world" transform, or P = K[R|t].

        A helpful resource for this is:
        - http://wwwmayr.in.tum.de/konferenzen/MB-Jass2006/courses/1/slides/h-1-5.pdf
            which specifically Taylors the discussion toward C arms.

        TODO: RQ decomposition of projection matrix to get K, R, t as another constructor.

        Args:
            intrinsic (CameraIntrinsicTransform): the camera intrinsic matrix, or a mapping to 2D image index coordinates
                from camera coordinates, i.e. index_from_camera2d, usually denoted as :math:`K`.
            extrinsic (FrameTransform): the camera extrinsic matrix, or simply a FrameTransform to camera coordinates
                 from world coordinates, i.e. camera3d_from_world, usually denoted :math:`[R | t]`.

        """
        self.index_from_camera2d = (
            intrinsic
            if isinstance(intrinsic, CameraIntrinsicTransform)
            else CameraIntrinsicTransform(intrinsic)
        )
        self.camera3d_from_world = frame_transform(extrinsic)
        index_from_world = self.index_from_camera3d @ self.camera3d_from_world

        # TODO: adding _inv here causes the inverse projection to be different, WHY?
        super().__init__(
            get_data(index_from_world), _inv=get_data(index_from_world.inv)
        )

    def get_config(self) -> dict[str, Any]:
        """Get the configuration of the camera projection.

        Returns:
            dict[str, Any]: the configuration of the camera projection.
        """
        return {
            "intrinsic": self.index_from_camera2d,
            "extrinsic": self.camera3d_from_world,
        }

    @property
    def index_from_world(self) -> Transform:
        return self

    @classmethod
    def from_krt(
        cls, K: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> "CameraProjection":
        """Create a CameraProjection from a camera intrinsic matrix and extrinsic matrix.

        Args:
            K (np.ndarray): the camera intrinsic matrix.
            R (np.ndarray): the camera extrinsic matrix.
            t (np.ndarray): the camera extrinsic translation vector.

        Returns:
            CameraProjection: the camera projection.
        """
        return cls(intrinsic=K, extrinsic=FrameTransform.from_rt(K, R, t))

    @classmethod
    def from_rtk(
        cls,
        R: np.ndarray,
        t: Point3D,
        K: Union[CameraIntrinsicTransform, np.ndarray],
    ):
        return cls(intrinsic=K, extrinsic=FrameTransform.from_rt(R, t))

    @property
    def K(self):
        return self.index_from_camera2d

    @property
    def R(self):
        return self.camera3d_from_world.R

    @property
    def t(self):
        return self.camera3d_from_world.t

    @property
    def intrinsic(self) -> CameraIntrinsicTransform:
        return self.index_from_camera2d

    @property
    def extrinsic(self) -> FrameTransform:
        return self.camera3d_from_world

    @property
    def index_from_camera3d(self) -> Transform:
        proj = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        camera2d_from_camera3d = Transform(proj, _inv=proj.T)
        return self.index_from_camera2d @ camera2d_from_camera3d

    @property
    def camera3d_from_index(self) -> Transform:
        return self.index_from_camera3d.inv

    @property
    def world_from_index(self) -> Transform:
        """Gets the world-space vector between the source in world and the given point in index space."""
        return self.index_from_world.inv

    @property
    def world_from_index_on_image_plane(self) -> FrameTransform:
        """Get the transform to points in world on the image (detector) plane from image indices.

        The point input point should still be 3D, with a 0 in the z coordinate.

        """
        proj = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
        proj = Transform(proj, _inv=proj.T)
        index_from_world_3d = proj @ self.index_from_world
        return FrameTransform(data=get_data(index_from_world_3d.inv))

    @property
    def sensor_width(self) -> int:
        return self.intrinsic.sensor_width

    @property
    def sensor_height(self) -> int:
        return self.intrinsic.sensor_height

    def get_center_in_world(self) -> Point3D:
        """Get the center of the camera (origin of camera3d frame) in world coordinates.

        That is, get the translation vector of the world_from_camera3d FrameTransform

        This is comparable to the function get_camera_center() in DeepDRR.

        Returns:
            Point3D: the center of the camera in center.
        """

        # TODO: can also get the center from the intersection of three planes formed
        # by self.data.

        world_from_camera3d = self.camera3d_from_world.inv
        return world_from_camera3d(point(0, 0, 0))

    @property
    def center_in_world(self) -> Point3D:
        return self.get_center_in_world()

    def get_center_in_volume(self, volume: Volume) -> Point3D:
        """Get the camera center in IJK-space.

        In original deepdrr, this is the `source_point` of `get_canonical_proj_matrix()`

        Args:
            volume (AnyVolume): the volume to get the camera center in.

        Returns:
            Point3D: the camera center in the volume's IJK-space.
        """
        return volume.ijk_from_world @ self.center_in_world

    def get_ray_transform(self, volume: Volume) -> Transform:
        """Get the ray transform for the camera, in IJK-space.

        ijk_from_index transformation that goes from Point2D to Vector3D, with the vector in the
        Point2D frame.

        The ray transform takes a Point2D and converts it to a Vector3D. This is the vector in
        the direction pointing between the camera center (or source) and a given index-space
        point on the detector.

        Args:
            volume (AnyVolume): the volume to get get the ray transfrom through.

        Returns:
            Transform: the `ijk_from_index` transform.
        """
        return volume.ijk_from_world @ self.world_from_index
