from typing import Optional, Tuple

import logging
import numpy as np
from scipy.spatial.transform import Rotation

from . import geo
from . import utils


logger = logging.getLogger(__name__)

PI = np.float32(np.pi)
DEFAULT_MIN_ALPHA = -2 * PI / 3 # -120
DEFAULT_MAX_ALPHA = 2 * PI / 3 # 120
DEFAULT_MIN_BETA = -PI / 4 # -45
DEFAULT_MAX_BETA = PI / 4 # 45



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
    
    rho = -phi + np.pi * 0.5 + rho
    R_principle = np.array([[np.cos(rho), -np.sin(rho), 0],
                            [np.sin(rho), np.cos(rho), 0],
                            [0, 0, 1]])
    R = np.matmul(R_principle, R)

    return R


def pose_vector_angles(pose: geo.Vector3D) -> Tuple[float, float]:
    """Get the C-arm angles alpha, beta corrsponding the the pose vector.

    TODO(killeen): make a part of the MobileCArm object, to convert from a world-space vector.

    Args:
        pose (geo.Vector3D): the vector pointing from the isocenter (or camera) to the detector.

    Returns:
        Tuple[float, float]: carm angulation (alpha, beta) in radians.
    """
    x, y, z = pose
    alpha = np.arctan2(y, np.sqrt(x * x + z * z))
    beta = -np.arctan2(x, z)
    return alpha, beta


class MobileCArm(object):
    def __init__(
        self,
        isocenter_distance: float = 800,
        isocenter: Optional[geo.Point3D] = None,
        alpha: float = 0,
        beta: float = 0,
        min_alpha: Optional[float] = None,
        max_alpha: Optional[float] = None,
        min_beta: Optional[float] = None,
        max_beta: Optional[float] = None,
        degrees: bool = False,
        world_from_carm: Optional[geo.FrameTransform] = None,
    ) -> None:
        """Make a CArm device.

        The geometry follows figure 2 in Kausch et al: https://pubmed.ncbi.nlm.nih.gov/32533315/

        TODO(killeen): limit the translation of the C-arm, as it would be in reality.

        Args:
            isocenter_distance (float): the distance from the X-ray source to the isocenter of the CAarm. (The center of rotation).
            isocenter (Point3D): isocenter of the C-arm in carm-space. This is the center about which rotations are performed.
            alpha (float): initial LAO/RAO angle of the C-Arm. alpha > 0 is in the RAO direction. This is the angle along arm of the C-arm.
            beta (float): initial CRA/CAU angulation of the C-arm. beta > 0 is in the CAU direction.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
            world_from_carm: (Optional[geo.FrameTransform], optional): Transform that defines the CArm space in world coordinates. None is the identity transform. Defaults to None.
        """
        self.isocenter_distance = isocenter_distance
        self.isocenter = geo.point(0, 0, 0) if isocenter is None else isocenter
        self.alpha = utils.radians(alpha, degrees=degrees)
        self.beta = utils.radians(beta, degrees=degrees)
        self.min_alpha = DEFAULT_MIN_ALPHA if min_alpha is None else utils.radians(min_alpha, degrees=degrees)
        self.max_alpha = DEFAULT_MAX_ALPHA if max_alpha is None else utils.radians(max_alpha, degrees=degrees)
        self.min_beta = DEFAULT_MIN_BETA if min_beta is None else utils.radians(min_beta, degrees=degrees)
        self.max_beta = DEFAULT_MAX_BETA if max_beta is None else utils.radians(max_beta, degrees=degrees)
        self.world_from_carm = geo.frame_transform(world_from_carm)

    def __str__(self):
        return f'MobileCArm(isocenter={np.array_str(np.array(self.isocenter))}, alpha={np.degrees(self.alpha)}, beta={np.degrees(self.beta)}, degrees=True)'

    @property
    def carm_from_world(self) -> geo.FrameTransform:
        return self.world_from_carm.inv

    @property
    def isocenter_in_world(self) -> geo.Point3D:
        return self.world_from_carm @ self.isocenter

    def move_by(
        self,
        delta_isocenter: Optional[geo.Vector3D] = None,
        delta_alpha: Optional[float] = None,
        delta_beta: Optional[float] = None,
        degrees: bool = False,
    ) -> None:
        """Move the C-arm to the specified pose.

        Args:
            delta_isocenter (Optional[geo.Vector3D], optional): isocenter (Point3D): isocenter of the C-arm in C-arm-space. This is the center about which rotations are performed. Defaults to None.
            delta_alpha (Optional[float], optional): change in alpha. Defaults to None.
            delta_beta (Optional[float], optional): change in beta. Defaults to None.
            degrees (bool, optional): whether the given angles are in degrees. Defaults to False.

        """
        if delta_isocenter is not None:
            self.isocenter += geo.vector(delta_isocenter)

        if delta_alpha is not None:
            assert np.isscalar(delta_alpha)
            self.alpha += utils.radians(float(delta_alpha), degrees=degrees)
            self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)

        if delta_beta is not None:
            assert np.isscalar(delta_beta)
            self.beta += utils.radians(float(delta_beta), degrees=degrees)
            self.beta = np.clip(self.beta, self.min_beta, self.max_beta)

    def move_to(
        self,
        isocenter: Optional[geo.Point3D] = None,
        alpha: Optional[geo.Point3D] = None,
        beta: Optional[geo.Point3D] = None,
        degrees: bool = False,
    ) -> None:
        """Move to the specified point.

        Args:
            isocenter (Optional[geo.Point3D], optional): the desired isocenter in carm coordinates. Defaults to None.
            alpha (Optional[geo.Point3D], optional): the desired alpha angulation. Defaults to None.
            beta (Optional[geo.Point3D], optional): the desired secondary angulation. Defaults to None.
            degrees (bool, optional): whether angles are in degrees or radians. Defaults to False.
        """
        if isocenter is not None:
            self.isocenter = geo.point(isocenter)
        if alpha is not None:
            self.alpha = utils.radians(float(alpha), degrees=degrees)
            self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)
        if beta is not None:
            self.beta = utils.radians(float(beta), degrees=degrees)
            self.beta = np.clip(self.beta, self.min_beta, self.max_beta)

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        return self.get_camera3d_from_world()

    def get_pose_vector(self) -> geo.Vector3D:
        """Get the unit vector pointing toward the detector of the C-arm from its isocenter, in the carm space."""
        rot = Rotation.from_euler('xy', [-self.alpha, -self.beta])
        x = rot.apply([0, 0, 1])
        return geo.vector(x)

    def get_pose_vector_in_world(self) -> geo.Vector3D:
        return (self.world_from_carm @ self.get_pose_vector()).hat()

    def get_camera3d_from_world(self) -> geo.FrameTransform:
        """Rigid transformation of the C-arm camera pose."""
        # get the rotation corresponding to the c-arm, then translate to the camera-center frame, along z-axis.
        # Note the difference between this rotation and the one to get the pose vector. This is going the opposite way.
        rot = Rotation.from_euler('xy', [self.alpha, self.beta]).as_matrix()
        t = np.array([0, 0, self.isocenter_distance])
        camera3d_from_isocenter = geo.FrameTransform.from_rt(rot, t)
        isocenter_from_carm = geo.FrameTransform.from_origin(self.isocenter)

        return camera3d_from_isocenter @ isocenter_from_carm @ self.carm_from_world

class CArm(object):
    """C-arm device for positioning a camera in space."""
    def __init__(
        self,
        isocenter_distance: float,
        isocenter: Optional[geo.Point3D] = None,
        phi: float = 0,
        theta: float = 0,
        rho: float = 0,
        degrees: bool = False,
    ) -> None:
        logger.warning('Previously, device.CArm used phi, theta as device angulation. This was incorrect. To accomplish something similar, use device.MobileCArm instead.')

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
            # TODO: check min_isocenter < max_isocenter
            self.isocenter = geo.point(np.clip(self.isocenter, min_isocenter, max_isocenter))

        if delta_phi is not None:
            assert np.isscalar(delta_phi)
            self.phi += utils.radians(float(delta_phi), degrees=degrees)
        if min_phi is not None or max_phi is not None:
            self.phi = np.clip(self.phi, min_phi, max_phi)

        if delta_theta is not None:
            assert np.isscalar(delta_theta)
            self.theta += utils.radians(float(delta_theta), degrees=degrees)
        if min_theta is not None or max_theta is not None:
            self.theta = np.clip(self.theta, min_theta, max_theta)

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
