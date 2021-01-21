from typing import Optional

import logging
import numpy as np

from . import geo
from . import utils


logger = logging.getLogger(__name__)


def make_detector_rotation(phi: float, theta: float, rho: float):
    """Make the rotation matrix for a CArm detector at the given angles.

    Args:
        phi (float): phi.
        theta (float): theta.
        rho (float): rho

    Returns:
        np.ndarray: Rotation matrix.
    """
    # rotation around phi and theta
    sin_p = np.sin(phi)
    neg_cos_p = -np.cos(phi)
    z = 0
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    omc = 1 - cos_t

    # Rotation by theta about vector [sin(phi), -cos(phi), z].
    R = np.array([
        [
            sin_p * sin_p * omc + cos_t,
            sin_p * neg_cos_p * omc - z * sin_t, 
            sin_p * z * omc + neg_cos_p * sin_t,
        ],
        [
            sin_p * neg_cos_p * omc + z * sin_t,
            neg_cos_p * neg_cos_p * omc + cos_t,
            neg_cos_p * z * omc - sin_p * sin_t,
        ],
        [
            sin_p * z * omc - neg_cos_p * sin_t,
            neg_cos_p * z * omc + sin_p * sin_t,
            z * z * omc + cos_t,
        ]])
    # rotation around detector priniciple axis
    rho = -phi + np.pi * 0.5 + rho
    R_principle = np.array([[np.cos(rho), -np.sin(rho), 0],
                            [np.sin(rho), np.cos(rho), 0],
                            [0, 0, 1]])
    R = np.matmul(R_principle, R)

    return R


class CArm(object):
    """C-arm device for positioning a camera in space.

    """
    def __init__(
        self,
        isocenter_distance: float,
        isocenter: Optional[geo.Point3D] = None,
        phi: float = 0,
        theta: float = 0,
        rho: float = 0,
        degrees: bool = False,
    ) -> None:
        """Make a CArm device.

        Args:
            isocenter_distance (float): the distance from the X-ray source to the isocenter of the CAarm. (The center of rotation).
            isocenter (Point3D): isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (Optional[float], optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
        """
        self.isocenter_distance = isocenter_distance
        self.isocenter = geo.point(0, 0, 0) if isocenter is None else isocenter
        self.phi, self.theta, self.rho = utils.radians(phi, theta, rho, degrees=degrees)

    def move_to(
        self, 
        isocenter: Optional[geo.Point3D] = None,
        phi: Optional[float] = None,
        theta: Optional[float] = None,
        rho: Optional[float] = None,
        degrees: bool = False,
    ) -> None:
        """Move the C-arm to the specified pose.

        Args:
            isocenter (Point3D): isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (float, optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
        """
        if isocenter is not None:
            self.isocenter = geo.point(isocenter)
        if phi is not None:
            self.phi = utils.radians(float(phi), degrees=degrees)
        if theta is not None:
            self.theta = utils.radians(float(theta), degrees=degrees)
        if rho is not None:
            self.rho = utils.radians(float(rho), degrees=degrees)

    def move_by(
        self,
        delta_isocenter: Optional[geo.Vector3D] = None,
        delta_phi: Optional[float] = None,
        delta_theta: Optional[float] = None,
        delta_rho: Optional[float] = None,
        degrees: bool = False,
        min_isocenter: Optional[geo.Point3D] = None,
        max_isocenter: Optional[geo.Point3D] = None,
        min_phi: Optional[float] = None,
        max_phi: Optional[float] = None,
        min_theta: Optional[float] = None,
        max_theta: Optional[float] = None,
    ) -> None:
        """Move the C-arm by the specified deltas.

        Clips the internal state by the provided values if not None.

        Args:
            delta_isocenter (Vector3D): offset for the isocenter of the C-arm in world-space. This is the center about which rotations are performed.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (float, optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
        """
        if delta_isocenter is not None:
            self.isocenter += geo.vector(delta_isocenter)
        if min_isocenter is not None or max_isocenter is not None:
            self.isocenter = geo.point(np.clip(min_isocenter, max_isocenter, self.isocenter))

        if delta_phi is not None:
            self.phi += utils.radians(float(delta_phi), degrees=degrees)
        if min_phi is not None or max_phi is not None:
            self.phi = np.clip(min_phi, max_phi, self.phi)

        if delta_theta is not None:
            self.theta += utils.radians(float(delta_theta), degrees=degrees)
        if min_theta is not None or max_theta is not None:
            self.theta = np.clip(min_theta, max_theta, self.theta)

        if delta_rho is not None:
            self.rho += utils.radians(float(delta_rho), degrees=degrees)

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        return self.get_camera3d_from_world(self.isocenter, self.phi, self.theta, self.rho, degrees=False)

    def get_camera3d_from_world(
        self,
        isocenter: geo.Point3D,
        phi: float,
        theta: float,
        rho: Optional[float] = 0,
        degrees: bool = False,
    ) -> geo.FrameTransform:
        """Get the FrameTransform for the C-Arm device at the given pose.

        This ignores the internal state except for the isocenter_distance.
        
        Args:
            isocenter (geo.Point3D): isocenter of the device.
            phi (float): CRAN/CAUD angle of the C-Arm (along the actual arc of the arm)
            theta (float): Lect/Right angulation of C-arm (rotation at the base)
            rho (Optional[float], optional): rotation about principle axis, after main rotation. Defaults to 0.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
            offset (Optional[Vector3D], optional): world-space offset to add to the initial C-arm isocenter. Defaults to None.

        Returns:
            FrameTransform: the extrinsic matrix or "camera3d_from_world" frame transformation for the oriented C-Arm camera.
        """
        #  TODO: A staticmethod function may be more appropriate.
        phi, theta, rho = utils.radians(phi, theta, rho, degrees=degrees)

        # get the rotation corresponding to the c-arm, then translate to the camera-center frame, along z-axis.
        R = make_detector_rotation(phi, theta, rho)
        t = np.array([0, 0, self.isocenter_distance])
        camera3d_from_isocenter = geo.FrameTransform.from_rt(R, t)
        isocenter_from_world = geo.FrameTransform.from_origin(isocenter)

        return camera3d_from_isocenter @ isocenter_from_world
