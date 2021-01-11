from typing import Optional

import logging
import numpy as np

from . import geo
from . import utils


logger = logging.getLogger(__name__)


class CArm(object):
    """C-arm device for positioning a camera in space.

    TODO: maintain position as internal state.
    
    """
    def __init__(
        self,
        isocenter_distance: float,
        isocenter: Optional[geo.Point3D] = None,
        phi: float = 0,
        theta: float = 0,
        rho: float = 0,
        degrees: bool = True,
    ) -> None:
        """Make a CArm device.

        Args:
            isocenter_distance (float): the distance from the isocenter to the camera center, i.e. radius of c arm.
            isocenter (Point3D): isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (Optional[float], optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to True.
        """
        self.isocenter_distance = isocenter_distance
        self.isocenter = geo.point(0, 0, 0) if isocenter is None else isocenter
        self.phi, self.theta, self.rho = utils.radians(phi, theta, rho, degrees=degrees)

    @property
    def isocenter_from_world(self) -> geo.FrameTransform:
        # translate points to the frame at the center of the c-arm's rotation
        return geo.FrameTransform.from_origin(self.isocenter)

    def move_to(
        self, 
        isocenter: Optional[geo.Point3D] = None,
        phi: Optional[float] = None,
        theta: Optional[float] = None,
        rho: Optional[float] = None,
        degrees: bool = True,
    ) -> None:
        """Move the C-arm to the specified pose.

        Args:
            isocenter (Point3D): isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (float, optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to True.
        """
        if isocenter is not None:
            self.isocenter = geo.point(isocenter)
        if phi is not None:
            self.phi = utils.radians(phi, degrees=degrees)
        if theta is not None:
            self.theta = utils.radians(theta, degrees=degrees)
        if rho is not None:
            self.rho = utils.radians(rho, degrees=degrees)

    def move_by(
        self,
        offset: Optional[geo.Vector3D] = None,
        delta_phi: Optional[float] = None,
        delta_theta: Optional[float] = None,
        delta_rho: Optional[float] = None,
        degrees: bool = True,
    ) -> None:
        """Move the C-arm by the specified deltas.

        Args:
            offset (Vector3D): offset for the isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (float, optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to True.
        """
        if offset is not None:
            self.isocenter += geo.vector(offset)
        if delta_phi is not None:
            self.phi += utils.radians(delta_phi, degrees=degrees)
        if delta_theta is not None:
            self.theta += utils.radians(delta_theta, degrees=degrees)
        if delta_rho is not None:
            self.rho += utils.radians(delta_rho, degrees=degrees)

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        return self.at(self.phi, self.theta, self.rho, degrees=False)

    def at(
        self,
        phi: float,
        theta: float,
        rho: Optional[float] = 0,
        degrees: bool = True,
        offset: Optional[geo.Vector3D] = None,
    ) -> geo.FrameTransform:
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
        phi, theta, rho = utils.radians(phi, theta, rho, degrees=degrees)

        # get the rotation corresponding to the c-arm, then translate to the camera-center frame, along z-axis.
        R = utils.make_detector_rotation(phi, theta, rho)
        t = np.array([0, 0, self.isocenter_distance]) # TODO: divide by 2?
        camera3d_from_isocenter = geo.FrameTransform.from_rt(R, t)

        if offset is None:
            offset = geo.FrameTransform.identity()
        else:
            offset = geo.FrameTransform.from_translation(offset)

        return camera3d_from_isocenter @ offset @ self.isocenter_from_world
