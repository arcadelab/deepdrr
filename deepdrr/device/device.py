from abc import ABC, abstractmethod

from .. import geo


class Device(ABC):
    """A parent class representing X-ray device interfaces in DeepDRR."""

    @property
    @abstractmethod
    def camera_intrinsics(self) -> geo.CameraIntrinsicTransform:
        """Get the camera intrinsics for the device in the current pose.

        Returns:
            CameraIntrinsicTransform: the camera intrinsics for the device.
        """
        pass

    @property
    @abstractmethod
    def world_from_device(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's local frame.

        Returns:
            FrameTransform: the "world_from_device" frame transformation for the device.
        """
        pass

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
    def camera3d_from_device(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_device frame (in the current pose).

        Returns:
            FrameTransform: the "camera3d_from_device" frame transformation for the device.
        """
        pass

    @property
    def device_from_camera3d(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_device frame (in the current pose).

        Args:
            camera3d_transform (FrameTransform): the "camera3d_from_device" frame transformation for the device.

        Returns:
            FrameTransform: the "device_from_camera3d" frame transformation for the device.
        """
        return self.camera3d_from_device.inverse()

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_world frame (in the current pose).

        Returns:
            FrameTransform: the "camera3d_from_world" frame transformation for the device.
        """
        return self.camera3d_from_device @ self.device_from_world

    @property
    def world_from_camera3d(self) -> geo.FrameTransform:
        """Get the FrameTransform for the device's camera3d_from_world frame (in the current pose).

        Returns:
            FrameTransform: the "world_from_camera3d" frame transformation for the device.
        """
        return self.camera3d_from_world.inverse()

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
    @abstractmethod
    def source_to_detector_distance(self) -> float:
        """Get the distance between the source and the detector in the current pose in mm.

        Returns:
            float: the distance between the source and the detector.
        """
        pass
