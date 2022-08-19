from __future__ import annotations

from typing import Union, Optional, Any, TYPE_CHECKING
import numpy as np

from .core import Transform, FrameTransform, point, Point3D, get_data, Plane
from .camera_intrinsic_transform import CameraIntrinsicTransform

if TYPE_CHECKING:
    from ..vol import Volume


# TODO: reorganize geo so you have primitives.py and transforms.py. Have separate classes for each type of transform?


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

        Args:
            intrinsic (CameraIntrinsicTransform): the camera intrinsic matrix, or a mapping to 2D image index coordinates
                from camera coordinates, i.e. index_from_camera2d.
            extrinsic (FrameTransform): the camera extrinsic matrix, or simply a FrameTransform to camera coordinates
                 from world coordinates, i.e. camera3d_from_world.

        """
        self.index_from_camera2d = (
            intrinsic
            if isinstance(intrinsic, CameraIntrinsicTransform)
            else CameraIntrinsicTransform(intrinsic)
        )
        self.camera3d_from_world = (
            extrinsic
            if isinstance(extrinsic, FrameTransform)
            else FrameTransform(extrinsic)
        )
        index_from_world = self.index_from_camera3d @ self.camera3d_from_world
        super().__init__(
            get_data(index_from_world), _inv=get_data(index_from_world.inv)
        )

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
