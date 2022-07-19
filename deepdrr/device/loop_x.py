import logging
import math
from typing import Dict, Optional
import numpy as np
from .device import Device
from .. import geo

log = logging.getLogger(__name__)


class LoopX(Device):
    """A class for representing the Loop-X device from BrainLab.

    For more info, see: https://www.brainlab.com/surgery-products/overview-platform-products/robotic-intraoperative-mobile-cbct/



    Attributes:
    """

    _source_angle_rad: float
    _detector_angle_rad: float
    _lateral_mm: float
    _longitudinal_mm: float
    _traction_yaw_rad: float
    _gantry_tilt_rad: float

    @property
    def source_angle(self) -> float:
        """Angular position of the source in degrees."""
        return math.degrees(self._source_angle_rad)

    @property
    def detector_angle(self) -> float:
        """Angular position of the detector in degrees."""
        return math.degrees(self._detector_angle_rad)

    @property
    def lateral(self) -> float:
        """The lateral position of the device in cm."""
        return self._lateral_mm / 10

    @property
    def longitudinal(self) -> float:
        """The longitudinal position of the device in cm."""
        return self._longitudinal_mm / 10

    @property
    def traction_yaw(self) -> float:
        """The traction yaw of the device in degrees."""
        return math.degrees(self._traction_yaw_rad)

    @property
    def gantry_tilt(self) -> float:
        """The gantry tilt of the device in degrees."""
        return math.degrees(self._gantry_tilt_rad)

    @source_angle.setter
    def source_angle(self, value: float):
        """Set the source angle of the device in degrees."""
        self._source_angle_rad = math.radians(value)

    @detector_angle.setter
    def detector_angle(self, value: float):
        """Set the detector angle of the device in degrees."""
        self._detector_angle_rad = math.radians(value)

    @lateral.setter
    def lateral(self, value: float):
        """Set the lateral position of the device in cm."""
        self._lateral_mm = value * 10

    @longitudinal.setter
    def longitudinal(self, value: float):
        """Set the longitudinal position of the device in cm."""
        self._longitudinal_mm = value * 10

    @traction_yaw.setter
    def traction_yaw(self, value: float):
        """Set the traction yaw of the device in degrees."""
        self._traction_yaw_rad = math.radians(value)

    @gantry_tilt.setter
    def gantry_tilt(self, value: float):
        """Set the gantry tilt of the device in degrees."""
        self._gantry_tilt_rad = math.radians(value)

    def set_pose(
        self,
        source_angle: Optional[float] = None,
        detector_angle: Optional[float] = None,
        lateral: Optional[float] = None,
        longitudinal: Optional[float] = None,
        traction_yaw: Optional[float] = None,
        gantry_tilt: Optional[float] = None,
    ) -> None:
        """Move the device to the given pose, as read directly off the device.

        This pose can be read/set by clicking on the loop icon in the top right of the screen.

        Args:
            source_angle: The source angle of the device in degrees.
            detector_angle: The detector angle of the device in degrees.
            lateral: The lateral position of the device in cm.
            longitudinal: The longitudinal position of the device in cm.
            traction_yaw: The traction yaw of the device in degrees.
            gantry_tilt: The gantry tilt of the device in degrees.

        """
        if source_angle is not None:
            self.source_angle = source_angle
        if detector_angle is not None:
            self.detector_angle = detector_angle
        if lateral is not None:
            self.lateral = lateral
        if longitudinal is not None:
            self.longitudinal = longitudinal
        if traction_yaw is not None:
            self.traction_yaw = traction_yaw
        if gantry_tilt is not None:
            self.gantry_tilt = gantry_tilt

    def get_pose(self) -> Dict[str, float]:
        return dict(
            source_angle=self.source_angle,
            detector_angle=self.detector_angle,
            lateral=self.lateral,
            longitudinal=self.longitudinal,
            traction_yaw=self.traction_yaw,
            gantry_tilt=self.gantry_tilt,
        )

    def set_home(self) -> None:
        """Set the home position of the device in world.

        The device frame is defined in terms of the home position.

        """
        raise NotImplementedError(
            "determine the most convenient way to set the home position, e.g. based on tracking of the home position"
        )

    