from typing import Optional

import numpy as np

from .geo import FrameTransform, Vector3D, Point3D
from . import utils


class CArm(object):
    def __init__(
        self,
        isocenter: Point3D,
        isocenter_distance: float,
    ) -> None:
        """Make a CArm device

        Args:
            isocenter (Point3D): isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            isocenter_distance (float): the distance from the isocenter to the camera center, that is the source point of the rays.
        """
        self.isocenter = isocenter
        self.isocenter_distance = isocenter_distance

    @property
    def isocenter_from_world(self) -> FrameTransform:
        # translate points to the frame at the center of the c-arm's rotation
        return FrameTransform.from_origin(self.isocenter)

    def at(
        self,
        phi: float,
        theta: float,
        rho: Optional[float] = 0,
        degrees: bool = True,
        offset: Optional[Vector3D] = None,
    ) -> FrameTransform:
        """Get the FrameTransform for the C-Arm device at the given pose.

        Args:
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (Optional[float], optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to True.
            offset (Optional[Vector3D], optional): world-space offset to add to the initial C-arm isocenter. Defaults to None.

        Returns:
            FrameTransform: the extrinsic matrix or "camera3d_from_world" frame transformation for the oriented C-Arm camera.
        """

        if degrees:
            phi = np.radians(phi)
            theta = np.radians(theta)
            rho = np.radians(rho)

        # get the rotation corresponding to the c-arm, then translate to the camera-center frame, along z-axis.
        R = utils.make_detector_rotation(phi, theta, rho)
        t = np.array([0, 0, self.isocenter_distance])
        camera3d_from_isocenter = FrameTransform.from_rt(R, t)

        if offset is None:
            offset = FrameTransform.identity()
        else:
            offset = FrameTransform.from_translation(offset)

        return camera3d_from_isocenter @ offset @ self.isocenter_from_world