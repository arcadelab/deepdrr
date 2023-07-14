from typing import Dict, Optional
import logging
import numpy as np
from pathlib import Path

from . import Volume
from .. import geo
from ..utils import data_utils, radians

log = logging.getLogger(__name__)


class KWire(Volume):
    _mesh_material = "iron"

    diameter = 2.0  # mm
    tip_in_IJK: geo.Point3D
    base_in_IJK: geo.Point3D

    def __init__(
        self,
        *args,
        tip: Optional[geo.Point3D] = None,
        base: Optional[geo.Point3D] = None,
        **kwargs,
    ) -> None:
        """A special volume which can be positioned using the tip and base points.

        Use the `from_example()` class method to create a KWire from the example volume (which will be downloaded).

        Args:
            tip (geo.Point3D): The location of the tool tip in RAS.
            base (geo.Point3D): The location of the tool base in RAS.
        """

        super(KWire, self).__init__(*args, **kwargs)
        assert tip is not None and base is not None
        self.tip_in_IJK = self.IJK_from_anatomical @ geo.point(tip)
        self.base_in_IJK = self.IJK_from_anatomical @ geo.point(base)

    @classmethod
    def from_example(
        cls,
        diameter: float = 2,
        density: float = 7.5,
        world_from_anatomical: Optional[geo.F] = None,
        **kwargs,
    ):
        """Creates a KWire from the provided download link.

        Args:
            density (float, optional): Density of the K-wire metal.

        Returns:
            KWire: The example KWire built into DeepDRR.
        """
        url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/ERoEsDbaFj9InktoRKnrT-MBSF2oCOrZ9uyOeWViRx4-Qg?e=s5fofv&download=1"
        md5 = "83ba7b63ebc0912d34ed5880460f81bd"
        filename = "Kwire2.nii.gz"
        path = data_utils.download(url, filename, md5=md5)
        shape = (100, 100, 2000)
        tip = geo.point(-1, -1, 0)
        base = geo.point(-1, -1, 200)
        tool = cls.from_nifti(
            path,
            density_kwargs=dict(density=density),
            tip=tip,
            base=base,
            world_from_anatomical=world_from_anatomical,
            **kwargs,
        )

        # scale the tool to the desired radius
        tool.scale(diameter / tool.diameter)

        return tool

    def scale(self, factor: float) -> None:
        """Scales the volume by the given factor.

        Args:
            factor (float): The factor by which to scale the tool. 1 would be no scaling.
        """
        scaling = geo.F(
            np.diag([factor, factor, factor, 1.0]),
        )
        self.anatomical_from_IJK = scaling @ self.anatomical_from_IJK

    @staticmethod
    def _convert_hounsfield_to_density(hu_values: np.ndarray, density: float = 7.5):
        # TODO: coefficient should be 2?
        # Should be density of steel.
        return density * hu_values

    @staticmethod
    def _segment_materials(
        hu_values: np.ndarray, use_thresholding: bool = True
    ) -> Dict[str, np.ndarray]:
        if not use_thresholding:
            raise NotImplementedError

        return dict(iron=(hu_values > 0))

    @property
    def tip(self) -> geo.Point3D:
        """The tip of the tool in world space."""
        return self.anatomical_from_IJK @ self.tip_in_IJK

    @property
    def base(self) -> geo.Point3D:
        """The base of the tool in world space."""
        return self.anatomical_from_IJK @ self.base_in_IJK

    @property
    def tip_in_ijk(self) -> geo.Point3D:
        return self.tip_in_IJK

    @property
    def base_in_ijk(self) -> geo.Point3D:
        return self.base_in_IJK

    @property
    def tip_in_anatomical(self) -> geo.Point3D:
        """Get the location of the tool tip (the pointy end) in anatomical coordinates."""
        return self.tip

    @property
    def base_in_anatomical(self) -> geo.Point3D:
        """Get the location of the tool base in anatomical coordinates."""
        return self.base

    @property
    def tip_in_world(self) -> geo.Point3D:
        """Get the location of the tool tip (the pointy end) in world coordinates."""
        return self.world_from_IJK @ self.tip_in_IJK

    @property
    def base_in_world(self) -> geo.Point3D:
        """Get the location of the tool base in world coordinates."""
        return self.world_from_IJK @ self.base_in_IJK

    @property
    def length_in_world(self):
        return (self.tip_in_world - self.base_in_world).norm()

    def align(
        self,
        startpoint_in_world: geo.Point3D,
        endpoint_in_world: geo.Point3D,
        progress: float = 1.0,
        distance: Optional[float] = None,
    ) -> None:
        """Align the tool so that it lies between the two points, tip pointing toward the endpoint.

        Args:
            start_point_in_world (geo.Point3D): The first point, in world space.
            end_point_in_world (geo.Point3D): The second point, in world space. The tip of the tool points toward this point.
            progress (float, optional): Where to place the tip of the tool between the start and end point,
                on a scale from 0 to 1. 0 corresponds to the tip placed at the start point, 1 at the end point. Defaults to 1.0.
            distance (Optional[float], optional): The distance of the tip along the trajectory. 0 corresponds
                to the tip placed at the start point, |startpoint - endpoint| at the end point.
                Overrides progress if provided. Defaults to None.

        """
        # useful: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        startpoint_in_world = geo.point(startpoint_in_world)
        endpoint_in_world = geo.point(endpoint_in_world)

        if distance is None:
            distance = (endpoint_in_world - startpoint_in_world).norm() * progress

        # interpolate along the direction of the tool to get the desired points in world.
        direction = (endpoint_in_world - startpoint_in_world).hat()
        desired_tip_in_world = startpoint_in_world + distance * direction
        desired_base_in_world = desired_tip_in_world - direction * self.length_in_world

        self.world_from_anatomical = geo.FrameTransform.from_line_segments(
            desired_tip_in_world,
            desired_base_in_world,
            self.tip_in_anatomical,
            self.base_in_anatomical,
        )

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def trajectory_in_world(self) -> geo.Ray3D:
        return geo.Ray3D.from_pn(self.tip_in_world, self.base_in_world)

    @property
    def centerline_in_world(self) -> geo.Line3D:
        return geo.line(self.tip_in_world, self.base_in_world)

    def orient(
        self,
        startpoint: geo.Point3D,
        direction: geo.Vector3D,
        distance: float = 0,
    ):
        """Place the tip at startpoint and orient the tool to point toward the direction."""
        return self.align(
            startpoint,
            startpoint + direction.hat(),
            distance=distance,
        )

    def twist(self, angle: float, degrees: bool = True):
        """Rotate the tool clockwise (when looking down on it) by `angle`.

        Args:
            angle (float): The angle.
            degrees (bool, optional): Whether `angle` is in degrees. Defaults to True.
        """
        rotvec = (self.tip - self.base).hat()
        rotvec *= radians(angle, degrees=degrees)
        self.world_from_anatomical = self.world_from_anatomical @ geo.frame_transform(
            geo.Rotation.from_rotvec(rotvec)
        )

    def advance(self, distance: float):
        """Move the tool forward by the given distance.

        Args:
            distance (float): The distance to move the tool forward.
        """
        self.align(
            self.tip_in_world,
            self.tip_in_world + (self.tip_in_world - self.base_in_world),
            distance=distance,
        )
