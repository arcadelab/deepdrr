"""
This file is part of DeepDRR.
Copyright (c) 2020 Benjamin D. Killeen.

DeepDRR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DEEPDRR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepDRR.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import Union, Tuple, Optional, Type, List, Generic, TypeVar

from abc import ABC, abstractmethod
import numpy as np

from . import vol
from . import utils


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
        assert np.all(np.isclose(x[..., -1], 0)), f'not a homogeneous vector: {x}'
        return x[..., :-1]


T = TypeVar('T')


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

    def __array__(self):
        return self.to_array()
            
    def __str__(self):
        return np.array_str(self.data, suppress_small=True)

    def __repr__(self):
        s = '  ' + str(np.array_str(self.data)).replace('\n', '\n  ')
        return f"{self.__class__.__name__}(\n{s}\n)"

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)



class HomogeneousPointOrVector(HomogeneousObject):
    """A Homogeneous point or vector in any dimension."""
    def __init__(
            self,
            data: np.ndarray,
    ) -> None:
        """Instantiate the homogeneous point or vector and check its dimension."""
        super().__init__(data)
        
        if self.data.shape != (self.dim + 1,):
            raise ValueError(f'invalid shape for {self.dim}D object in homogeneous coordinates: {self.data.shape}')

    def to_array(self) -> np.ndarray:
        """Return non-homogeneous numpy representation of object."""
        return _from_homogeneous(self.data, is_point=bool(self.data[-1]))

    
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
        """ If other is not a point, make it one. """
        return other if issubclass(type(other), Point) else cls.from_array(other)

    def __sub__(
            self: Point,
            other: Point,
    ) -> Union[Vector2D, Vector3D]:
        """ Subtract two points, obtaining a vector. """
        other = self.from_any(other)
        return vector(self.data - other.data)

    def __add__(self, other):
        """ Can add a vector to a point, but cannot add two points. """
        if issubclass(type(other), Vector):
            return type(self)(other.data + self.data)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return point(other * np.array(self))
        else:
            return NotImplemented

    def __neg__(self):
        return self * (-1)


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
        """ If other is not a Vector, make it one. """
        return other if issubclass(type(other), Vector) else cls.from_array(other)
    
    def __mul__(self, other: Union[int, float]):
        """ Vectors can be multiplied by scalars. """
        if isinstance(other, (int, float)):
            return type(self)(other * self.data)
        else:
            return NotImplemented

    def __matmul__(self, other: Vector):
        """ Inner product between two Vectors. """
        other = self.from_any(other)
        return type(self)(self.data @ other.data)

    def __add__(self, other: Vector) -> Vector:
        """ Two vectors can be added to make another vector. """
        other = self.from_any(other)
        return type(self)(self.data + other.data)

    def __neg__(self):
        return (-1) * self

    def __sub__(self, other: Vector):
        return self + (-other)

    def __rmul__(self, other: Vector):
        return self * other

    def __rsub__(self, other: Vector):
        return (-self) + other

    def __radd__(self, other: Vector):
        return self + other


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


PointOrVector = TypeVar('PointOrVector', Point2D, Point3D, Vector2D, Vector3D)
PointOrVector2D = TypeVar('PointOrVector2D', Point2D, Vector2D)
PointOrVector3D = TypeVar('PointOrVector3D', Point3D, Vector3D)


def _array(x: Union[List[np.ndarray], List[float]]) -> np.ndarray:
    """Parse args into a numpy array."""
    if len(x) == 1:
        return np.array(x[0])
    elif len(x) == 2 or len(x) == 3:
        return np.array(x)
    else:
        raise ValueError(f'could not parse point or vector arguments: {x}')


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
        raise ValueError(f'invalid data for point: {x}')
    

def vector(*v: Union[np.ndarray, float, Vector2D, Vector3D]) -> Union[Vector2D, Vector3D]:
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
        raise ValueError(f'invalid data for vector: {v}')


def _point_or_vector(data: np.ndarray):
    assert data.ndim == 1 and data[-1] in [0, 1], f'{data} must be a point or vector'

    if bool(data[-1]):
        return point(data[:-1])
    else:
        return vector(data[:-1])


""" 
Transforms
"""


class Transform(HomogeneousObject):
    def __init__(
        self, 
        data: np.ndarray, 
        _inv: Optional[np.ndarray] = None
    ) -> None:
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
        data = np.concatenate([
            array,
            np.array([0 for _ in range(array.shape[1] - 1)] + [1])],
            axis=0
        )
        return cls(data)

    def __matmul__(
            self,
            other: Union[Transform, PointOrVector],
    ) -> Union[Transform, PointOrVector]:
        assert self.input_dim == other.dim, f'dimensions must match between other ({other.dim}) and self ({self.input_dim})'

        if issubclass(type(other), HomogeneousPointOrVector):
            return _point_or_vector(self.data @ other.data)
        elif issubclass(type(other), Transform):
            # if other is a Transform, then compose their inverses as well to store that.
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
    def inv(self) -> Transform:
        """Get the inverse of the Transform.

        Returns:
            (Transform): a Transform (or subclass) that is well-defined as the inverse of this transform.

        Raises:
            NotImplementedError: if _inv is None and method is not overriden.
        """
        if self._inv is None:
            raise NotImplementedError("inverse operation not well-defined except when overridden by subclasses, unless _inv provided")

        return Transform(self._inv, _inv=self.data)


class FrameTransform(Transform):
    def __init__(
            self,
            data: np.ndarray,    # the homogeneous frame transformation matrix
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
        rotation: Optional[np.ndarray] = None,
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
            FrameTransform: [description]
        """

        if rotation is not None:
            dim = np.array(rotation).shape[0]
        elif translation is not None:
            dim = np.array(translation).shape[0]
        else:
            return cls.identity(dim)

        R = np.eye(dim) if rotation is None else np.array(rotation)
        t = np.zeros(dim) if translation is None else np.array(translation)

        assert t.shape[0] == dim
        assert R.shape == (dim, dim)

        data = np.concatenate(
            [
                np.concatenate([R, t[:, np.newaxis]], axis=1),
                np.concatenate([np.zeros((1, dim)), [[1]]], axis=1)
            ],
            axis=0
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
        return cls.from_rt(translation)

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
        """Suppose `origin` is point where frame B has its origin, as a point in frame A. Get the B_from_A transform.

        Just negates the origin, but this is often counterintuitive.

        For example:

        ```
            ^
            |     ^
            |     |
            |    B --- >
            |    ^
            |   /  
            |  / `origin`
            | /
            |/
          A  ----------------------- >
        ```

        Args:
            origin (Point): origin of the target frame in the world frame

        Returns:
            FrameTransform: the B_from_A transform.
        """
        origin = point(origin)
        return cls.from_rt(translation=-origin)

    @property
    def R(self):
        return self.data[0:self.dim, 0:self.dim]

    @property
    def t(self):
        return self.data[0:self.dim, self.dim]

    @property
    def inv(self):
        R_inv = np.linalg.inv(self.R)
        return FrameTransform.from_rt(R_inv, -(R_inv @ self.t))


class CameraIntrinsicTransform(FrameTransform):
    dim: int = 2
    input_dim: int = 2

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert self.data.shape == (3,3), f'unrecognized shape: {self.data.shape}'

    @classmethod
    def from_parameters(
        cls,
        optical_center: Point2D,
        focal_length: Union[float, Tuple[float, float]] = 1,
        shear: float = 0,
        aspect_ratio: Optional[float] = None,
    ) -> CameraIntrinsicTransform:
        """The camera intrinsic matrix, which is fundamentally a FrameTransform in 2D, namely `index_from_camera2d`.

        The intrinsic matrix transfroms to the index-space of the image (as mapped on the sensor) from the 

        Note that focal lengths are often measured in world units (e.g. millimeters.), but here they are in pixels.
        The conversion can be taken from the size of a pixel.

        References:
        - Szeliski's "Computer Vision."
        - https://ksimek.github.io/2013/08/13/intrinsic/

        Args:
            optical_center (Point2D): the index-space point where the isocenter (or pinhole) is centered.
            focal_length (Union[float, Tuple[float, float]]): the focal length in index units. Can be a tubple (f_x, f_y), 
                or a scalar used for both, or a scalar modified by aspect_ratio, in index units.
            shear (float): the shear `s` of the camera.
            aspect_ratio (Optional[float], optional): the aspect ratio `a` (for use with one focal length). If not provided, aspect 
                ratio is 1. Defaults to None.

        Returns:
            CameraIntrinsicTransform: The camera intrinsic matrix as
                [[f_x, s, c_x],
                 [0, f_y, c_y],
                 [0,   0,   1]]
                or
                [[f, s, c_x],
                 [0, a f, c_y],
                 [0,   0,   1]]

        """
        optical_center = point(optical_center)
        assert optical_center.dim == 2, 'center point not in 2D'

        cx, cy = np.array(optical_center)

        if aspect_ratio is None:
            fx, fy = utils.tuplify(focal_length, 2)
        else:
            assert isinstance(focal_length, (float, int)), 'cannot use aspect ratio if both focal lengths provided'
            fx, fy = (focal_length, aspect_ratio * focal_length)

        data = np.array(
            [[fx, shear, cx],
             [0, fy, cy],
             [0, 0, 1]]).astype(np.float32)

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
            source_to_detector_distance (int): distance from source to detector in world units.

        Returns:
            
        """
        sensor_size = utils.tuplify(sensor_size, 2)
        pixel_size = utils.tuplify(pixel_size, 2)
        fx = source_to_detector_distance / pixel_size[0]
        fy = source_to_detector_distance / pixel_size[1]
        optical_center = point(sensor_size[0] / 2, sensor_size[1] / 2)
        return cls.from_parameters(
            optical_center=optical_center,
            focal_length=(fx, fy))

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
        """Focal length in pixels."""
        return self.fx

    @property
    def sensor_width(self) -> int:
        """Get the sensor width in pixels.
        
        Based on the convention of origin in top left, with x pointing to the right and y pointing down."""
        return int(np.ceil(2 * self.data[0, 2]))

    @property
    def sensor_height(self) -> int:
        """Get the sensor height in pixels.
        
        Based on the convention of origin in top left, with x pointing to the right and y pointing down."""
        return int(np.ceil(2 * self.data[1, 2]))

    @property
    def sensor_size(self) -> Tuple[int, int]:
        """Tuple with the (width, height) of the sense/image, in pixels."""
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
        """A generic camera projection.

        A helpful resource for this is:
        - http://wwwmayr.in.tum.de/konferenzen/MB-Jass2006/courses/1/slides/h-1-5.pdf
            which specifically Taylors the discussion toward C arms.

        Args:
            intrinsic (CameraIntrinsicTransform): the camera intrinsic matrix, or a mapping to 2D image index coordinates 
                from camera coordinates, i.e. index_from_camera2d.
            extrinsic (FrameTransform): the camera extrinsic matrix, or simply a FrameTransform to camera coordinates
                 from world coordinates, i.e. camera3d_from_world.

        """
        self.index_from_camera2d = intrinsic if isinstance(intrinsic, CameraIntrinsicTransform) else CameraIntrinsicTransform(intrinsic)
        self.camera3d_from_world = extrinsic if isinstance(extrinsic, FrameTransform) else FrameTransform(extrinsic)

    @classmethod
    def from_rtk(
        cls,
        R: np.ndarray,
        t: Point3D,
        K: Union[CameraIntrinsicTransform, np.ndarray],
    ):
        return cls(intrinsic=K, extrinsic=FrameTransform.from_rt(R, t))
        
    @property
    def intrinsic(self) -> CameraIntrinsicTransform:
        return self.index_from_camera2d

    @property
    def extrinsic(self) -> FrameTransform:
        return self.camera3d_from_world
        
    @property
    def index_from_world(self) -> FrameTransform:
        proj = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        camera2d_from_camera3d = Transform(proj, _inv=proj.T)
        return self.index_from_camera2d @ camera2d_from_camera3d @ self.camera3d_from_world

    @property
    def world_from_index(self) -> FrameTransform:
        return self.index_from_world.inv

    @property
    def sensor_width(self) -> int:
        return self.intrinsic.sensor_width

    @property
    def sensor_height(self) -> int:
        return self.intrinsic.sensor_height

    @property
    def center_in_world(self) -> Point3D:
        """Get the center of the camera (origin of camera3d frame) in world coordinates.

        That is, get the translation vector of the world_from_camera3d FrameTransform
        
        This is comparable to the function get_camera_center() in DeepDRR.

        Returns:
            Point3D: the center of the camera in center.
        """
        
        world_from_camera3d = self.camera3d_from_world.inv
        return world_from_camera3d(point(0, 0, 0))

    def get_center_in_volume(self, volume: vol.Volume) -> Point3D:
        """Get the camera center in IJK-space.

        In original deepdrr, this is the `source_point` of `get_canonical_proj_matrix()`

        Args:
            volume (Volume): the volume to get the camera center in.

        Returns:
            Point3D: the camera center in the volume's IJK-space.
        """
        return volume.ijk_from_world @ self.center_in_world
 
    def get_ray_transform(self, volume: vol.Volume) -> Transform:
        """Get the ray transform for the camera, in IJK-space.

        ijk_from_index transformation that goes from Point2D to Vector3D, with the vector in the Point2D frame.

        The ray transform takes a Point2D and converts it to a Vector3D. 

        Analogous to get_canonical_projection_matrix. Gets "RT_Kinv" for CUDA kernel.

        """
        return volume.ijk_from_world @ self.world_from_index
