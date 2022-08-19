from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import pyvista as pv

from .. import geo


class Device(ABC):
    """A parent class representing X-ray device interfaces in DeepDRR.

    Attributes:
        sensor_height (int): the height of the sensor in pixels.
        sensor_width (int): the width of the sensor in pixels.
        pixel_size (float): the size of a pixel in mm.

    """

    sensor_height: int
    sensor_width: int
    pixel_size: float

    camera_intrinsics: geo.CameraIntrinsicTransform
    source_to_detector_distance: float
    world_from_device: geo.FrameTransform

    @property
    def device_from_world(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's local frame.

        Args:
            world_transform (FrameTransform): the "world_from_device" frame transformation for the device.

        Returns:
            FrameTransform: the "device_from_world" frame transformation for the device.
        """
        return self.world_from_device.inverse()

    @property
    @abstractmethod
    def device_from_camera3d(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_device frame (in the current pose).

        Args:
            camera3d_transform (FrameTransform): the "camera3d_from_device" frame transformation for the device.

        Returns:
            FrameTransform: the "device_from_camera3d" frame transformation for the device.
        """
        pass

    @property
    def camera3d_from_device(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_device frame (in the current pose).

        Returns:
            FrameTransform: the "camera3d_from_device" frame transformation for the device.
        """
        return self.device_from_camera3d.inverse()

    @property
    def world_from_camera3d(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_world frame (in the current pose).

        Returns:
            FrameTransform: the "world_from_camera3d" frame transformation for the device.
        """
        return self.world_from_device @ self.device_from_camera3d

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_world frame (in the current pose).

        Returns:
            FrameTransform: the "camera3d_from_world" frame transformation for the device.
        """
        return self.camera3d_from_device @ self.device_from_world

    def get_camera_projection(self) -> geo.CameraProjection:
        """Get the camera projection for the device in the current pose.

        Returns:
            CameraProjection: the "index_from_world" camera projection for the device.
        """
        return geo.CameraProjection(self.camera_intrinsics, self.camera3d_from_world)

    @property
    def index_from_world(self) -> geo.CameraProjection:
        """Get the camera projection for the device in the current pose.

        Returns:
            CameraProjection: the "index_from_world" camera projection for the device.
        """
        return self.get_camera_projection()

    @property
    @abstractmethod
    def principle_ray(self) -> geo.Vector3D:
        """Get the principle ray for the device in the current pose in the device frame.

        The principle ray is the direction of the ray that passes through the center of the
        image. It points from the source toward the detector.

        Returns:
            Vector3D: the principle ray for the device as a unit vector.

        """
        pass

    @property
    def principle_ray_in_world(self) -> geo.Vector3D:
        """Get the principle ray for the device in the current pose in the world frame.

        The principle ray is the direction of the ray that passes through the center of the
        image. It points from the source toward the detector.

        Returns:
            Vector3D: the principle ray for the device as a unit vector.
        """
        return (self.world_from_device @ self.principle_ray).normalized()

    @property
    def source_in_world(self) -> geo.Point3D:
        return self.world_from_camera3d @ geo.p(0, 0, 0)

    def get_mesh_in_world(self, full=False, use_cached=True):
        """Get a really simple camera mesh for the device in the current pose.

        Subclasses may want to override this with more detailed meshes (full=True).

        """

        # In camera frame
        s = geo.p(0, 0, 0)
        c = s + geo.v(0, 0, self.source_to_detector_distance)
        cx = self.pixel_size * self.sensor_height / 2.0
        cy = self.pixel_size * self.sensor_width / 2.0
        ul = geo.p(-cx, cy, self.source_to_detector_distance)
        ur = geo.p(cx, cy, self.source_to_detector_distance)
        bl = geo.p(-cx, -cy, self.source_to_detector_distance)
        br = geo.p(cx, -cy, self.source_to_detector_distance)

        mesh = (
            pv.Line(ur, ul)
            + pv.Line(br, bl)
            + pv.Line(ur, br)
            + pv.Line(ul, bl)
            + pv.Line(s, ul)
            + pv.Line(s, ur)
            + pv.Line(s, bl)
            + pv.Line(s, br)
            + pv.Line(c, s)
            + pv.Sphere(5, c)
        )
        mesh.transform(geo.get_data(self.world_from_camera3d), inplace=True)
        return mesh
