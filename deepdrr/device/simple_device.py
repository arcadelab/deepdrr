from typing import Optional, Tuple, Union, List
import numpy as np
import logging
from .. import geo
from .device import Device

log = logging.getLogger(__name__)


class SimpleDevice(Device):
    """A simple C-arm with a point, direction interface to set views.

    The "point" being positioned is always at the midpoint of the source and detector. The direction
    is the direction from the source to the detector. The up-vector of images can also be provided,
    not necessarily in the image plane (projected onto it).

    Any of the device's attributes can be set directly. The default values are not based on any
    particular device.

    This class may be useful for understanding basic concepts.

    Attributes:
        sensor_height (int): the height of the sensor in pixels.
        sensor_width (int): the width of the sensor in pixels.
        pixel_size (float): the size of a pixel in mm.
        source_to_detector_distance (float): the distance from the source to the detector in mm.
        world_from_device (FrameTransform): the "world_from_device" frame transformation for the device.

    """

    def __init__(
        self,
        sensor_height: int = 300,
        sensor_width: int = 300,
        pixel_size: float = 0.1,
        source_to_detector_distance: float = 1000.0,
        world_from_device: Optional[geo.FrameTransform] = None,
    ):
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        self.pixel_size = pixel_size
        self.source_to_detector_distance = source_to_detector_distance
        self.world_from_device = geo.frame_transform(world_from_device)

        # Default view centered on the origin. Sets the device_from_camera3d
        self.set_view([0, 0, 0], [0, 0, 1], [0, -1, 0])

    @property
    def camera_intrinsics(self) -> geo.CameraIntrinsicTransform:
        """Get the camera intrinsics for the device.

        Returns:
            CameraIntrinsicTransform: the camera intrinsics for the device.
        """
        f = self.source_to_detector_distance / self.pixel_size
        data = np.array(
            [
                [f, 0, self.sensor_width / 2],
                [0, f, self.sensor_height / 2],
                [0, 0, 1],
            ]
        )
        return geo.CameraIntrinsicTransform(
            data,
            sensor_height=self.sensor_height,
            sensor_width=self.sensor_width,
        )

    def set_view(
        self,
        point: geo.Point3D,
        direction: geo.Vector3D,
        up: Optional[geo.Vector3D] = None,
    ):
        """Set the view of the device.

        Args:
            point (Point3D): the point at the center of the source and detector, in world coordinates.
            direction (Vector3D): the direction from the source to the detector, in world coordinates.
            up (Vector3D): the up vector of the image, in world_coordinates. It's projection
                corresponds to the -Y axis in the camera3d frame. If None, the up vector is set to the -Y
                axis of the device frame.
        """
        point_in_device = self.device_from_world @ geo.point(point)
        direction_in_device = self.device_from_world @ geo.vector(direction)
        if up is not None:
            up_in_device = self.device_from_world @ geo.vector(up)
        else:
            up_in_device = geo.vector(0, -1, 0)

        # Get the "ray" frame, which has its origin at the isocenter and its z-axis along the ray.
        # The "ray" frame has an arbitrary rotation about the z-axis.
        z_axis = geo.vector(0, 0, 1)
        if (rotvec := z_axis.cross(direction_in_device)).norm() < 1e-6:
            # The direction is parallel to the z-axis. The ray frame is the device frame.
            rotvec = geo.vector(0, 0, 0)
        else:
            rotvec = rotvec.hat() * z_axis.angle(direction_in_device)
        rot = geo.Rotation.from_rotvec(rotvec)
        device_from_ray = geo.F.from_rt(rot, point_in_device)

        # Get the "ray-up" frame, which is rotated about Z to align the up vector with the -Y axis.
        neg_y_axis = geo.vector(0, -1, 0)
        up_vector_in_ray = device_from_ray @ up_in_device
        up_vector_in_image_plane = geo.vector(
            up_vector_in_ray[0], up_vector_in_ray[1], 0
        )

        if (rotvec := neg_y_axis.cross(up_vector_in_image_plane)).norm() < 1e-6:
            # The up vector is parallel to the -Y axis. The ray-up frame is the ray frame.
            rotvec = geo.vector(0, 0, 0)
        else:
            rotvec = rotvec.hat() * neg_y_axis.angle(up_vector_in_image_plane)
        rot = geo.Rotation.from_rotvec(rotvec)
        ray_from_ray_up = geo.F.from_rt(rot, geo.point(0, 0, 0))

        # Get the "camera3d" frame, which is translated in negative z.
        ray_up_from_camera3d = geo.F.from_rt(
            translation=geo.vector(0, 0, -self.source_to_detector_distance)
        )

        self._device_from_camera3d = (
            device_from_ray @ ray_from_ray_up @ ray_up_from_camera3d
        )

    @property
    def device_from_camera3d(self) -> geo.FrameTransform:
        return self._device_from_camera3d
