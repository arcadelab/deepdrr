import math
from typing import Any, Dict, Optional, Tuple, Union, List

import logging
import numpy as np
from numpy.lib.utils import source
from scipy.spatial.transform import Rotation

from .device import Device
from .. import geo
from .. import utils

pv, pv_available = utils.try_import_pyvista()


log = logging.getLogger(__name__)

PI = np.float32(np.pi)


def pose_vector_angles(pose: geo.Vector3D) -> Tuple[float, float]:
    """Get the C-arm angles alpha, beta corrsponding the the pose vector.

    TODO(killeen): make a part of the MobileCArm object, to convert from a world-space vector. Get rid of the Gimble lock problem.

    Args:
        pose (geo.Vector3D): the vector pointing from the isocenter (or camera) to the detector. (Along the principle ray.)

    Returns:
        Tuple[float, float]: carm angulation (alpha, beta) in radians.
    """
    x, y, z = pose
    alpha = -math.atan2(y, np.sqrt(x * x + z * z))
    beta = math.atan2(x, z)
    return alpha, beta


class MobileCArm(Device):
    # basic parameters which can be safely set by user, but move_by() and reposition() are recommended.
    isocenter: geo.Point3D  # the isocenter point in the device frame
    alpha: float  # alpha angle in radians
    beta: float  # beta angle in radians
    world_from_device: geo.FrameTransform  # can be set by the user.

    def __init__(
        self,
        world_from_device: Optional[geo.FrameTransform] = None,
        isocenter: geo.Point3D = [0, 0, 0],
        alpha: float = 0,
        beta: float = 0,
        gamma: float = 0,  # rotation of detector about principle ray
        degrees: bool = True,
        horizontal_movement: float = 200,  # width of window in X and Y planes.
        vertical_travel: float = 430,  # width of window in Z plane.
        min_alpha: float = -40,
        max_alpha: float = 110,
        # note that this would collide with the patient. Suggested to limit to +/- 45
        min_beta: float = -225,
        max_beta: float = 225,
        source_to_detector_distance: float = 1020,
        # vertical component of the source point offset from the isocenter of rotation, in -Z. Previously called `isocenter_distance`
        source_to_isocenter_vertical_distance: float = 530,
        # horizontal offset of the principle ray from the isocenter of rotation, in +Y. Defaults to 9, but should be 200 in document.
        source_to_isocenter_horizontal_offset: float = 0,
        # horizontal distance from principle ray to inner C-arm circumference. Used for visualization
        immersion_depth: float = 730,
        # distance from central ray to edge of arm. Used for visualization
        free_space: float = 820,
        sensor_height: int = 1536,
        sensor_width: int = 1536,
        pixel_size: float = 0.194,
        rotate_camera_left: bool = True,  # make it so that down in the image corresponds to -x, so that patient images appear as expected (when gamma=0).
        enforce_isocenter_bounds: bool = False,  # Allow the isocenter to travel arbitrarily far from the device origin
    ) -> None:
        """A simulated C-arm imaging device with orbital movement (alpha), angulation (beta) and 3D translation.

        A MobileCArm has its own `device` frame, which moves independent of the `arm` frame and the
        `camera` frame, which actually defines the projection.

        The geometry and default values, are based on the Cios Fusion device, specifically this_
        document, with some exceptions. Rather than incorporating a swivel, this device allows for
        simple horizontal translations. Additionally, by default the principle ray is not offset from
        the isocenter.

        .. _this: https://www.lomisa.com/app/download/10978681598/ARCO_EN_C_SIEMENS_CIOS_FUSION.pdf?t=1490962065

        Additionally, the orbital movement and angulation are reversed from the convention used in `Kausch
        et al`_, Figure 2.

        .. _Kausch et al: https://pubmed.ncbi.nlm.nih.gov/32533315/

        All length units are in millimeters.

        Args:
            world_from_device: (Optional[geo.FrameTransform], optional): Transform that defines the device coordinate space in world coordinates. None is the identity transform. Defaults to None.
            isocenter (geo.Point3D): the initial isocenter of in the device frame. This is the point
                about which rotations are performed.
            isocenter_distance (float): the distance from the X-ray source to the isocenter of the CAarm. (The center of rotation).
            alpha (float): initial LAO/RAO angle of the C-Arm. alpha > 0 is in the RAO direction. This is the angle along arm of the C-arm.
            beta (float): initial CRA/CAU angulation of the C-arm. beta > 0 is in the CAU direction.
            degrees (bool, optional): Whether given angles are in degrees. Defaults to False.
            camera_intrinsics: (Optional[Union[geo.CameraIntrinsicTransform, dict]], optional): either a CameraIntrinsicTransform instance or kwargs for CameraIntrinsicTransform.from_sizes

        """
        self.world_from_device = geo.frame_transform(world_from_device)
        self.enforce_isocenter_bounds = enforce_isocenter_bounds
        self.isocenter = geo.point(isocenter)
        self.alpha = utils.radians(alpha, degrees=degrees)
        self.beta = utils.radians(beta, degrees=degrees)
        self.gamma = utils.radians(gamma, degrees=degrees)
        self.horizontal_movement = horizontal_movement
        self.vertical_travel = vertical_travel
        self.min_alpha = utils.radians(min_alpha, degrees=degrees)
        self.max_alpha = utils.radians(max_alpha, degrees=degrees)
        self.min_beta = utils.radians(min_beta, degrees=degrees)
        self.max_beta = utils.radians(max_beta, degrees=degrees)
        self.source_to_detector_distance = source_to_detector_distance
        self.source_to_isocenter_vertical_distance = (
            source_to_isocenter_vertical_distance
        )
        self.source_to_isocenter_horizontal_offset = (
            source_to_isocenter_horizontal_offset
        )
        self.immersion_depth = immersion_depth
        self.free_space = free_space
        self.pixel_size = pixel_size
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.camera_intrinsics = geo.CameraIntrinsicTransform.from_sizes(
            sensor_size=(sensor_width, sensor_height),
            pixel_size=pixel_size,
            source_to_detector_distance=self.source_to_detector_distance,
        )
        self.rotate_camera_left = rotate_camera_left

        # May upset some code that was erroneously using isocenter to position the Carm.
        if enforce_isocenter_bounds and (
            np.any(np.array(isocenter) < self.min_isocenter)
            or np.any(np.array(isocenter) > self.max_isocenter)
        ):
            raise ValueError(
                f"isocenter {self.isocenter} is out of bounds. Use world_from_device transform to position the carm in the world."
            )

        # points in the arm frame don't change.
        self.viewpoint_in_arm = geo.point(
            0, self.source_to_isocenter_horizontal_offset, 0
        )

        self._enforce_bounds()
        self._static_mesh = None

    def __str__(self):
        return (
            f"MobileCArm(isocenter={np.array_str(np.array(self.isocenter))}, "
            f"alpha={np.degrees(self.alpha)}, beta={np.degrees(self.beta)}, degrees=True)"
        )

    def to_config(self) -> Dict[str, Any]:
        """Get a json-safe dictionary that can be used to initialize the C-arm in its current pose."""
        return utils.jsonable(
            dict(
                world_from_device=self.world_from_device,
                isocenter=self.isocenter,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                degrees=False,
                horizontal_movement=self.horizontal_movement,
                vertical_travel=self.vertical_travel,
                min_alpha=self.min_alpha,
                max_alpha=self.max_alpha,
                min_beta=self.min_beta,
                max_beta=self.max_beta,
                source_to_detector_distance=self.source_to_detector_distance,
                source_to_isocenter_vertical_distance=self.source_to_isocenter_vertical_distance,
                source_to_isocenter_horizontal_offset=self.source_to_isocenter_horizontal_offset,
                immersion_depth=self.immersion_depth,
                free_space=self.free_space,
                sensor_height=self.sensor_height,
                sensor_width=self.sensor_width,
                pixel_size=self.pixel_size,
                rotate_camera_left=self.rotate_camera_left,
                enforce_isocenter_bounds=self.enforce_isocenter_bounds,
            )
        )

    @property
    def max_isocenter(self) -> np.ndarray:
        return (
            np.array(
                [
                    self.horizontal_movement,
                    self.horizontal_movement,
                    self.vertical_travel,
                ]
            )
            / 2
        )

    @property
    def min_isocenter(self) -> np.ndarray:
        return -self.max_isocenter

    @property
    def isocenter_in_world(self) -> geo.Point3D:
        return self.world_from_device @ self.isocenter

    @property
    def device_from_arm(self) -> geo.FrameTransform:
        # First, rotate points, then translate back by the isocenter.
        rot = Rotation.from_euler("xy", [self.alpha, self.beta]).as_matrix()
        return geo.FrameTransform.from_rt(rotation=rot, translation=self.isocenter)

    @property
    def arm_from_device(self) -> geo.FrameTransform:
        """Transformation from the device frame (which doesn't move) to the arm frame (which rotates and translates with the arm, origin at the isocenter)."""
        return self.device_from_arm.inv

    @property
    def camera3d_from_device(self) -> geo.FrameTransform:
        """Get the camera3d frame from device coordinates

        The Z axis points from the source to the detector."""
        camera3d_from_arm = geo.FrameTransform.from_rt(
            translation=geo.point(
                0,
                -self.source_to_isocenter_horizontal_offset,
                self.source_to_isocenter_vertical_distance,
            )
        )
        if self.rotate_camera_left:
            camera3d_from_arm = (
                geo.frame_transform(Rotation.from_euler("z", 90, degrees=True))
                @ camera3d_from_arm
            )

        # This is a little hacky, since the "device" frame is now kind of wrong, and modeling this
        # C-arm wouldn't show the rotation about the principle ray, but it will work to rotate the
        # images produced.
        gamma_rotation = geo.frame_transform(
            Rotation.from_euler("z", self.gamma, degrees=False)
        )

        return gamma_rotation @ camera3d_from_arm @ self.arm_from_device

    @property
    def device_from_camera3d(self) -> geo.FrameTransform:
        return self.camera3d_from_device.inv

    @property
    def camera3d_from_world(self) -> geo.FrameTransform:
        """Rigid transformation of the C-arm camera pose."""
        return self.get_camera3d_from_world()

    def get_camera3d_from_world(self) -> geo.FrameTransform:
        return self.camera3d_from_device @ self.device_from_world

    def get_camera_projection(self) -> geo.CameraProjection:
        return geo.CameraProjection(
            self.camera_intrinsics, self.get_camera3d_from_world()
        )

    @property
    def viewpoint(self) -> geo.Point3D:
        """Get the point along the principle ray, where objects of interest should ideally be placed.

        Returns:
            geo.Point3D: the viewpoint in the device frame.
        """
        return self.device_from_arm @ self.viewpoint_in_arm

    @property
    def viewpoint_in_world(self) -> geo.Point3D:
        return self.world_from_device @ self.viewpoint

    @property
    def principle_ray(self) -> geo.Vector3D:
        """Unit vector along principle ray."""
        return (self.device_from_arm @ geo.vector(0, 0, 1)).hat()

    def _enforce_bounds(self):
        """Enforce the CArm movement bounds."""
        if self.enforce_isocenter_bounds:
            self.isocenter = geo.point(
                np.clip(self.isocenter, self.min_isocenter, self.max_isocenter)
            )
        self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)
        self.beta = np.clip(self.beta, self.min_beta, self.max_beta)

    def move_by(
        self,
        delta_isocenter: Optional[geo.Vector3D] = None,
        delta_alpha: Optional[float] = None,
        delta_beta: Optional[float] = None,
        delta_gamma: Optional[float] = None,
        degrees: bool = True,
    ) -> None:
        """Move the C-arm to the specified pose.

        Args:
            delta_isocenter (Optional[geo.Vector3D], optional): change to the isocenter in DEVICE space
                (as a vector, this only matters if the scaling/rotation is different).
                This is the center about which rotations are performed. Defaults to None.
            delta_alpha (Optional[float], optional): change in alpha. Defaults to None.
            delta_beta (Optional[float], optional): change in beta. Defaults to None.
            degrees (bool, optional): whether the given angles are in degrees. Defaults to False.

        """
        # TODO: don't let out-of-bounds movement pass silently.

        if delta_isocenter is not None:
            self.isocenter += geo.vector(delta_isocenter)
        if delta_alpha is not None:
            assert np.isscalar(delta_alpha)
            self.alpha += utils.radians(float(delta_alpha), degrees=degrees)
        if delta_beta is not None:
            assert np.isscalar(delta_beta)
            self.beta += utils.radians(float(delta_beta), degrees=degrees)
        if delta_gamma is not None:
            assert np.isscalar(delta_gamma)
            self.gamma += utils.radians(float(delta_gamma), degrees=degrees)

        self._enforce_bounds()

    def move_to(
        self,
        isocenter: Optional[geo.Point3D] = None,
        isocenter_in_world: Optional[geo.Point3D] = None,
        alpha: float = None,
        beta: float = None,
        gamma: float = None,
        degrees: bool = True,
        interest_point_in_world: Optional[geo.Point3D] = None,
        principle_ray_in_world: Optional[geo.Vector3D] = None,
    ) -> None:
        """Move to the specified point.

        Args:
            isocenter_in_world (Optional[geo.Point3D], optional): the desired isocenter in world coordinates.
                Overrides `isocenter` if provided. Defaults to None.
            isocenter: Desired isocenter in device coordinates.
            alpha (Optional[float], optional): the desired alpha angulation. Defaults to None.
            beta (Optional[float], optional): the desired secondary angulation. Defaults to None.
            degrees (bool, optional): whether angles are in degrees or radians. Defaults to False.
            interest_point (Point3D, optional): If this world-space point is provided, add a translation such that the rotation
                maintains the camera-space position of this point. Overrides `isocenter`. Defaults to None.
            principle_ray_in_world (Optional[Vector3D], optional): If this world-space vector is provided, override alpha, beta so the C-arm points along this vector.

        """

        old_principle_ray = self.principle_ray

        if principle_ray_in_world is not None:
            principle_ray = self.device_from_world @ geo.vector(principle_ray_in_world)
            alpha, beta = pose_vector_angles(principle_ray)
            if gamma is not None:
                gamma = utils.radians(float(gamma), degrees=degrees)
            degrees = False

        if alpha is not None:
            self.alpha = utils.radians(float(alpha), degrees=degrees)
        if beta is not None:
            self.beta = utils.radians(float(beta), degrees=degrees)
        if gamma is not None:
            self.gamma = utils.radians(float(gamma), degrees=degrees)

        # Get the isocenter from convenience arguments, if supplied.
        if interest_point_in_world is not None:
            principle_ray = self.principle_ray
            interest_point = self.device_from_world @ interest_point_in_world

            # Get the length along the old ray to reach the interest point, then subtract along the new priniple ray
            # to move the isocenter to a comparable point.
            # NOTE: this is a reasonable approximation, but it's not perfect.
            isocenter = (
                interest_point
                - (interest_point - self.isocenter).dot(old_principle_ray.hat())
                * principle_ray.hat()
            )
        elif isocenter_in_world is not None:
            isocenter = self.device_from_world @ geo.point(isocenter_in_world)

        if isocenter is not None:
            self.isocenter = geo.point(isocenter)

        self._enforce_bounds()

    def reposition(
        self,
        viewpoint_in_world: Optional[geo.Point3D] = None,
        device_in_world: Optional[geo.Point3D] = None,
    ) -> None:
        """Reposition the C-arm by resetting its internal pose to the parameters and adjusting the world_from_device transform.

        TODO: currently, this eliminates any scaling/rotation of the device in world.

        May provide either the isocenter location (device_in_world) or viewpoint (viewpoint_in_world)

        Args:
            viewpoint_in_world (geo.Point3D): the initial viewpoint the device should have.
            device_in_world (): initial isocenter.

        """
        self.move_to(isocenter=[0, 0, 0], alpha=0, beta=0, degrees=False)
        if device_in_world is None:
            assert viewpoint_in_world is not None
            device_in_world = viewpoint_in_world - geo.vector(*self.viewpoint_in_arm)
        self.world_from_device = geo.FrameTransform.from_translation(device_in_world)

    # shape parameters
    source_height = 200
    source_radius = 100
    detector_height = 100
    arm_width = 100

    @property
    def source_in_arm(self) -> geo.Point3D:
        return geo.point(
            0,
            self.source_to_isocenter_horizontal_offset,
            -self.source_to_isocenter_vertical_distance,
        )

    @property
    def source_in_device(self) -> geo.Point3D:
        return self.device_from_arm @ self.source_in_arm

    def _make_mesh(self, full=True, include_labels: bool = False):
        """Make the mesh of the C-arm, centered and upright.

        This DOES NOT use the current isocenter, alpha, or beta.

        Returns:
        """
        assert (
            pv_available
        ), f"PyVista not available for obtaining MobileCArm surface model. Try: `pip install pyvista`"
        if include_labels:
            log.warning(f"C-arm mesh labels not supported yet")

        source_point = geo.point(
            0,
            self.source_to_isocenter_horizontal_offset,
            -self.source_to_isocenter_vertical_distance,
        )
        center_point = geo.point(0, self.source_to_isocenter_horizontal_offset, 0)

        mesh = (
            pv.Line(
                list(source_point),
                list(source_point + geo.vector(0, 0, self.source_to_detector_distance)),
            )
            + pv.Line(
                list(center_point + geo.vector(-100, 0, 0)),
                list(center_point + geo.vector(100, 0, 0)),
            )
            + pv.Line(
                list(center_point + geo.vector(0, -100, 0)),
                list(center_point + geo.vector(0, 100, 0)),
            )
        )

        if full:
            # Source
            mesh += pv.Cylinder(
                center=source_point,
                direction=[0, 0, 1],
                radius=self.source_radius,
                height=self.source_height,
            )

            # Sensor
            mesh += pv.Box(
                bounds=[
                    -self.pixel_size * self.sensor_width / 2,
                    self.pixel_size * self.sensor_width / 2,
                    -self.pixel_size * self.sensor_height / 2
                    + self.source_to_isocenter_horizontal_offset,
                    self.pixel_size * self.sensor_height / 2
                    + self.source_to_isocenter_horizontal_offset,
                    -self.source_to_isocenter_vertical_distance
                    + self.source_to_detector_distance,
                    -self.source_to_isocenter_vertical_distance
                    + self.source_to_detector_distance
                    + self.detector_height,
                ],
            )

            # Arm
            arm = pv.ParametricTorus(
                ringradius=self.source_to_isocenter_vertical_distance,
                crosssectionradius=self.arm_width / 2,
            )
            y = max(
                -self.pixel_size * self.sensor_height / 2
                + self.source_to_isocenter_horizontal_offset,
                source_point.y - self.source_radius,
            )
            arm.clip(normal="y", origin=[0, y, 0], inplace=True)
            arm.rotate_y(90)
            mesh += arm

        return mesh

    def get_mesh_in_world(self, full=False, use_cached=True):
        """Get the pyvista mesh for the C-arm, in its world-space orientation.

        Raises:
            RuntimeError: if pyvista is not available.

        """
        if self._static_mesh is None:
            self._static_mesh = self._make_mesh(full=full)

        mesh = self._static_mesh.copy()
        mesh.rotate_x(np.degrees(self.alpha))
        mesh.rotate_y(np.degrees(self.beta))
        mesh.translate(self.isocenter)
        mesh.transform(geo.get_data(self.world_from_device), inplace=True)

        # TODO: add cone window.

        return mesh

    def jitter(self):
        # semi-realistic jitter about the axis.
        raise NotImplementedError
