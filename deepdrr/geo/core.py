from __future__ import annotations

from typing import Union, Tuple, Optional, Type, List, TypeVar, TYPE_CHECKING
import logging
from abc import ABC, abstractmethod
import numpy as np
import scipy.spatial.distance
from scipy.spatial.transform import Rotation


logger = logging.getLogger(__name__)


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
        data = data.data if issubclass(type(data), HomogeneousObject) else data
        assert isinstance(data, np.ndarray)
        self.data = data.astype(self.dtype)

    @classmethod
    @abstractmethod
    def from_array(
        cls: Type[T],
        x: np.ndarray,
    ) -> T:
        """Create a homogeneous object from its non-homogeous representation as an array."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Get the dimension of the space the object lives in. For transforms, this is the OUTPUT dim."""
        pass

    @abstractmethod
    def to_array(self, is_point):
        """Get the non-homogeneous representation of the object.

        For points, this removes the is_point indicator at the bottom (added 1 or 0).
        For transforms, this simply returns the data without modifying it.
        """
        pass

    def __array__(self, dtype=None):
        return self.to_array()

    def __str__(self):
        return np.array_str(self.data, suppress_small=True)

    def __repr__(self):
        s = "  " + str(np.array_str(self.data)).replace("\n", "\n  ")
        return f"{self.__class__.__name__}(\n{s}\n)"

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    def __iter__(self):
        return iter(np.array(self).tolist())

    def get_data(self) -> np.ndarray:
        return self.data


def get_data(x: HomogeneousObject) -> np.ndarray:
    assert issubclass(type(x), HomogeneousObject)
    return x.get_data()


class HomogeneousPointOrVector(HomogeneousObject):
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

    def to_array(self) -> np.ndarray:
        """Return non-homogeneous numpy representation of object."""
        return _from_homogeneous(self.data, is_point=bool(self.data[-1]))

    def norm(self, *args, **kwargs):
        """Get the norm of the vector. Pass any arguments to `np.linalg.norm`."""
        return np.linalg.norm(self, *args, **kwargs)

    def __len__(self) -> float:
        """Return the L2 norm of the point or vector."""
        return self.norm()

    def __div__(self, other):
        return self * (1 / other)

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]


class Point(HomogeneousPointOrVector):
    def __init__(self, data: np.ndarray) -> None:
        assert data[-1] == 1
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
        cls: Type[T],
        other: Union[np.ndarray, Point],
    ):
        """If other is not a point, make it one."""
        return other if issubclass(type(other), Point) else cls.from_array(other)

    def __sub__(
        self: Point,
        other: Point,
    ) -> Vector:
        """Subtract two points, obtaining a vector."""
        other = self.from_any(other)
        return _point_or_vector(self.data - other.data)

    def __add__(self, other: Vector) -> Point:
        """Can add a vector to a point, but cannot add two points. TODO: cannot add points together?"""
        if issubclass(type(other), Vector):
            return type(self)(self.data + other.data)
        elif issubclass(type(other), Point):
            # TODO: should points be allowed to be added together?
            return point(np.array(self) + np.array(other))
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __mul__(self, other: Union[int, float]) -> Vector:
        if isinstance(other, (int, float)):
            return point(other * np.array(self))
        else:
            return NotImplemented

    def __rmul__(self, other: Union[int, float]) -> Vector:
        return self * other

    def __neg__(self):
        return self * (-1)

    def lerp(self, other: Point, alpha: float = 0.5) -> Point:
        """Linearly interpolate between one point and another.

        Args:
            other (Point): other point.
            alpha (float): fraction of the distance from self to other to travel. Defaults to 0.5 (the midpoint).

        Returns:
            Point: the point that is `alpha` of the way between self and other.
        """
        return (1 - alpha) * self + alpha * other

    def as_vector(self) -> Vector:
        """Get the vector with the same numerical representation as this point."""
        return vector(np.array(self))


class Vector(HomogeneousPointOrVector):
    def __init__(self, data: np.ndarray) -> None:
        assert data[-1] == 0
        super().__init__(data)

    @classmethod
    def from_array(
        cls: Type[T],
        v: np.ndarray,
    ) -> T:
        v = np.array(v).astype(cls.dtype)
        data = _to_homogeneous(v, is_point=False)
        return cls(data)

    @classmethod
    def from_any(
        cls: Type[T],
        other: Union[np.ndarray, Vector],
    ):
        """If other is not a Vector, make it one."""
        return other if issubclass(type(other), Vector) else cls.from_array(other)

    def __mul__(self, other: Union[int, float]):
        """Vectors can be multiplied by scalars."""
        if isinstance(other, (int, float)):
            return type(self)(other * self.data)
        else:
            return NotImplemented

    def __matmul__(self, other: Vector):
        """Inner product between two Vectors."""
        other = self.from_any(other)
        return type(self)(self.data @ other.data)

    def __add__(self, other: Vector) -> Vector:
        """Two vectors can be added to make another vector."""
        other = self.from_any(other)
        return type(self)(self.data + other.data)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other: Vector):
        return self + (-other)

    def __rmul__(self, other: Union[int, float]):
        return self * other

    def __rsub__(self, other: Vector):
        return (-self) + other

    def __radd__(self, other: Vector):
        return self + other

    def hat(self) -> Vector:
        return self * (1 / self.norm())

    def dot(self, other) -> float:
        if issubclass(type(other), Vector) and self.dim == other.dim:
            return np.dot(self, other)
        else:
            return NotImplemented

    def cross(self, other) -> Vector:
        if issubclass(type(other), Vector) and self.dim == other.dim:
            return vector(np.cross(self, other))
        else:
            return NotImplemented

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

    def angle(self, other: Vector) -> float:
        """Get the angle between self and other in radians."""
        return np.arccos(self.dot(other) / (self.norm() * other.norm()))

    def cosine_distance(self, other: Vector) -> float:
        """Get the cosine distance between the angles.

        Args:
            other (Vector): the other vector.

        Returns:
            float: `1 - cos(angle)`, where `angle` is between self and other.
        """
        return scipy.spatial.distance.cosine(np.array(self), np.array(other))

    def as_point(self) -> Point:
        """Gets the point with the same numerical representation as this vector."""
        return point(np.array(self))


class Point2D(Point):
    """Homogeneous point in 2D, represented as an array with [x, y, 1]"""

    dim = 2


class Vector2D(Vector):
    """Homogeneous vector in 2D, represented as an array with [x, y, 0]"""

    dim = 2


class Point3D(Point):
    """Homogeneous point in 3D, represented as an array with [x, y, z, 1]"""

    dim = 3


class Vector3D(Vector):
    """Homogeneous vector in 3D, represented as an array with [x, y, z, 0]"""

    dim = 3


PointOrVector = TypeVar("PointOrVector", Point2D, Point3D, Vector2D, Vector3D)
PointOrVector2D = TypeVar("PointOrVector2D", Point2D, Vector2D)
PointOrVector3D = TypeVar("PointOrVector3D", Point3D, Vector3D)


def _array(x: Union[List[np.ndarray], List[float]]) -> np.ndarray:
    """Parse args into a numpy array."""
    if len(x) == 1:
        return np.array(x[0])
    elif len(x) == 2 or len(x) == 3:
        return np.array(x)
    else:
        raise ValueError(f"could not parse point or vector arguments: {x}")


def point(*x: Union[np.ndarray, float, Point2D, Point3D]) -> Union[Point2D, Point3D]:
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
    if len(x) == 1 and isinstance(x[0], (Point2D, Point3D)):
        return x[0]

    x = _array(x)
    if x.shape == (2,):
        return Point2D.from_array(x)
    elif x.shape == (3,):
        return Point3D.from_array(x)
    else:
        raise ValueError(f"invalid data for point: {x}")


def vector(
    *v: Union[np.ndarray, float, Vector2D, Vector3D]
) -> Union[Vector2D, Vector3D]:
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
    if len(v) == 1 and isinstance(v[0], (Vector2D, Vector3D)):
        return v[0]

    v = _array(v)
    if v.shape == (2,):
        return Vector2D.from_array(v)
    elif v.shape == (3,):
        return Vector3D.from_array(v)
    else:
        raise ValueError(f"invalid data for vector: {v}")


def _point_or_vector(data: np.ndarray):
    """Convert a point where the "homogeneous" element may not be 1."""

    if bool(data[-1]):
        return point(data[:-1] / data[-1])
    else:
        return vector(data[:-1])


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
        self._inv = _inv

    def to_array(self) -> np.ndarray:
        """Output the transform as a non-homogeneous matrix.

        The convention here is that "nonhomegenous" transforms would still have the last column,
        so it would take in homogeneous objects, but it doesn't have the last row, so it outputs non-homogeneous objects.

        If someone wants the data array, they can access it directly.

        Returns:
            np.ndarray: the non-homogeneous array
        """

        return self.data[:-1, :]

    @classmethod
    def from_array(cls, array: np.ndarray) -> Transform:
        """Convert non-homogeneous matrix to homogeneous transform.

        Usually, one would instantiate Transforms directly from the homogeneous matrix `data` or using one of the other classmethods.

        Args:
            array (np.ndarray): transformation matrix.

        Returns:
            Transform: the transform.
        """
        data = np.concatenate(
            [array, np.array([0 for _ in range(array.shape[1] - 1)] + [1])], axis=0
        )
        return cls(data)

    def __matmul__(
        self,
        other: Union[Transform, PointOrVector],
    ) -> Union[Transform, PointOrVector]:
        if issubclass(type(other), HomogeneousPointOrVector):
            assert (
                self.input_dim == other.dim
            ), f"dimensions must match between other ({other.dim}) and self ({self.input_dim})"
            return _point_or_vector(self.data @ other.data)
        elif issubclass(type(other), Transform):
            # if other is a Transform, then compose their inverses as well to store that.
            assert (
                self.input_dim == other.dim
            ), f"dimensions must match between other ({other.dim}) and self ({self.input_dim})"
            _inv = other.inv.data @ self.inv.data

            if isinstance(self, FrameTransform) and isinstance(other, FrameTransform):
                # very common case of composing FrameTransforms.
                return FrameTransform(self.data @ other.data)
            else:
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
    def inv(self) -> Transform:
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
        translation: Optional[Union[Point3D, np.ndarray]] = None,
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
        if isinstance(rotation, Rotation):
            rotation = rotation.as_matrix()

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
        scaling: Union[int, float, np.ndarray],
        translation: Optional[Union[Point3D, np.ndarray]] = None,
    ) -> FrameTransform:
        """Create a frame based on scaling dimensions. Assumes dim = 3.

        Args:
            cls (Type[FrameTransform]): the class.
            scaling (Union[int, float, np.ndarray]): coefficient to scale by, or one for each dimension.

        Returns:
            FrameTransform:
        """
        scaling = np.array(scaling) * np.ones(3)
        translation = np.zeros(3) if translation is None else translation
        return cls.from_rt(np.diag(scaling), translation)

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
        points_B: Union[List[Point3D], np.ndarray],
        points_A: Union[List[Point3D], np.ndarray],
    ):
        """Create a frame transform from a known point correspondence.

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
    def from_line_segments(
        cls,
        x_B: Point3D,
        y_B: Point3D,
        x_A: Point3D,
        y_A: Point3D,
    ) -> FrameTransform:
        """Get the `B_from_A` frame transform that aligns the line segments, given by endpoints.

        Args:
            x_B (Point3D): The first endpoint, in frame B.
            y_B (Point3D): The second endpoint, in frame B.
            x_A (Point3D): The first endpoint, in frame A.
            y_A (Point3D): The second endpoint, in frame A.

        Returns:
            FrameTransform: A `B_from_A` transform that aligns the points.
                Note that this is not unique, due to rotation about the axis between the points.
        """
        # First, get the vectors pointing from x to y in each frame.
        x2y_A = y_A - x_A
        x2y_B = y_B - x_B

        # Second, get the rotation between the vectors.
        rotvec = x2y_A.cross(x2y_B).hat()
        rotvec *= x2y_A.angle(x2y_B)
        rot = Rotation.from_rotvec(rotvec)

        return (
            cls.from_translation(x_B)
            @ cls.from_scaling(x2y_B.norm() / x2y_A.norm())
            @ cls.from_rotation(rot)
            @ cls.from_translation(-x_A)
        )

    @property
    def R(self):
        return self.data[0 : self.dim, 0 : self.dim]

    @property
    def t(self):
        return self.data[0 : self.dim, self.dim]

    @property
    def inv(self):
        R_inv = np.linalg.inv(self.R)
        return FrameTransform.from_rt(R_inv, -(R_inv @ self.t))


def frame_transform(*args) -> FrameTransform:
    """Convenience function for creating a frame transform.

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
            elif a.shape == (3,) or a.shape == (1, 3):
                return FrameTransform.from_rt(translation=a)
            else:
                raise TypeError(f"couldn't convert numpy array to FrameTransform: {a}")
        elif isinstance(a, (tuple, list)) and len(a) == 2:
            return frame_transform(a[0], a[1])
    elif len(args) == 2:
        if (
            isinstance(args[0], np.ndarray)
            and isinstance(args[1], np.ndarray)
            and args[0].shape == (3, 3)
            and args[1].shape == (3,)
        ):
            return FrameTransform.from_rt(rotation=args[0], translation=args[1])
        else:
            raise TypeError(
                f"could not parse FrameTransfrom from [R, t]: [{args[0]}, {args[1]}]"
            )
    else:
        raise TypeError(f"too many arguments: {args}")


RAS_from_LPS = FrameTransform(
    np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
)

LPS_from_RAS = RAS_from_LPS.inv
