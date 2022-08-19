from typing import Dict, Optional
import logging
import numpy as np
from pathlib import Path

from . import Volume
from .. import geo
from ..utils import data_utils

log = logging.getLogger(__name__)


class KWire(Volume):

    _mesh_material = "titanium"

    diameter = 2.0  # mm

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
        self.tip = geo.point(tip)
        self.base = geo.point(base)

    @classmethod
    def from_example(cls, density: float = 7.5, **kwargs):
        """Creates a KWire from the provided download link.

        Args:
            density (float, optional): Density of the K-wire metal.

        Returns:
            KWire: The example KWire built into DeepDRR.
        """
        url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/Ec2XGMXg_ItGtYWR_FfqHmUBwXJ1LmLBbbs4J_-3rJJQZg?e=fFWq6f&download=1"
        md5 = "83ba7b63ebc0912d34ed5880460f81bd"
        filename = "Kwire2.nii.gz"
        path = data_utils.download(url, filename, md5=md5)
        shape = (100, 100, 2000)
        tip = geo.point(-1, -1, 0)
        base = geo.point(-1, -1, 200)
        tool = cls.from_nifti(
            path, density_kwargs=dict(density=density), tip=tip, base=base, **kwargs
        )
        return tool

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

        return dict(titanium=(hu_values > 0))

    @property
    def tip_in_ijk(self) -> geo.Point3D:
        return self.ijk_from_anatomical @ self.tip

    @property
    def base_in_ijk(self) -> geo.Point3D:
        return self.ijk_from_anatomical @ self.base

    @property
    def tip_in_anatomical(self) -> geo.Point3D:
        """Get the location of the tool tip (the pointy end) in anatomical coordinates."""
        return self.tip

    @property
    def base_in_anatomical(self) -> geo.Point3D:
        """Get the location of the tool base in anatomical coordinates."""
        return self.anatomical_from_ijk @ self.base_in_ijk

    @property
    def tip_in_world(self) -> geo.Point3D:
        """Get the location of the tool tip (the pointy end) in world coordinates."""
        return self.world_from_ijk @ self.tip_in_ijk

    @property
    def base_in_world(self) -> geo.Point3D:
        """Get the location of the tool base in world coordinates."""
        return self.world_from_ijk @ self.base_in_ijk

    @property
    def length_in_world(self):
        return (self.tip_in_world - self.base_in_world).norm()

    def align(
        self,
        startpoint_in_world: geo.Point3D,
        endpoint_in_world: geo.Point3D,
        progress: float = 1.0,
    ) -> None:
        """Align the tool so that it lies between the two points, tip pointing toward the endpoint.

        Args:
            start_point_in_world (geo.Point3D): The first point, in world space.
            end_point_in_world (geo.Point3D): The second point, in world space. The tip of the tool points toward this point.
            progress (float, optional): Where to place the tip of the tool between the start and end point,
                on a scale from 0 to 1. 0 corresponds to the tip placed at the start point, 1 at the end point. Defaults to 1.0.
        """
        # useful: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        startpoint_in_world = geo.point(startpoint_in_world)
        endpoint_in_world = geo.point(endpoint_in_world)
        progress = float(progress)

        # interpolate along the direction of the tool to get the desired points in world.
        trajectory_vector = endpoint_in_world - startpoint_in_world

        desired_tip_in_world = endpoint_in_world - (1 - progress) * trajectory_vector
        desired_base_in_world = (
            desired_tip_in_world - trajectory_vector.hat() * self.length_in_world
        )

        self.world_from_anatomical = geo.FrameTransform.from_line_segments(
            desired_tip_in_world,
            desired_base_in_world,
            self.tip_in_anatomical,
            self.base_in_anatomical,
        )

    @property
    def radius(self) -> float:
        return self.diameter / 2
