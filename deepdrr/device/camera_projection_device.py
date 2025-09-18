from typing import Optional, Tuple, Union, List
import numpy as np
import logging
from .. import geo
from .device import Device

log = logging.getLogger(__name__)


class CameraProjectionDevice(Device):

    def __init__(
        self,
        projection: geo.CameraProjection,
        pixel_size: float = 1.0,
    ):
        self.pixel_size = pixel_size
        self.projection = projection

    @property
    def projection(self) -> geo.CameraProjection:
        return self._projection

    @projection.setter
    def projection(self, projection: geo.CameraProjection):
        self._projection = projection

        self.sensor_height = projection.intrinsic.sensor_height
        self.sensor_width = projection.intrinsic.sensor_width

        self.camera_intrinsics = projection.intrinsic
        self.source_to_detector_distance = (
            projection.intrinsic.focal_length * self.pixel_size
        )
        self.world_from_device = projection.world_from_camera3d

    @property
    def device_from_camera3d(self) -> geo.FrameTransform:
        return geo.FrameTransform.identity()

    def __str__(self):
        return (
            f"CameraProjectionDevice("
            f"sensor_height={self.sensor_height}, "
            f"sensor_width={self.sensor_width}, "
            f"pixel_size={self.pixel_size}, "
            f"camera_intrinsics={self.camera_intrinsics}, "
            f"source_to_detector_distance={self.source_to_detector_distance}, "
            f"world_from_device={self.world_from_device})"
        )
