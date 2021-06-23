from typing import Dict, Optional
import logging
import numpy as np
from pathlib import Path

from . import Volume
from .. import geo
from ..utils import data_utils

logger = logging.getLogger(__name__)


class KWire(Volume):
    """A provided volume representing a steel K wire, based on a straight cylinder with a conical tip.

    Args:
        world_from_anatomical (Optional[geo.FrameTransform], optional): transformation from the anatomical space to world coordinates. If None, assumes identity. Defaults to None.
    """
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EUd4AAG7svFJotzyWjcjd7cBc-iuAfHL819iO8u0bCYN0A?e=rDWrM7"
    filename = "Kwire2.nii.gz"

    def __new__(cls, world_from_anatomical: Optional[geo.FrameTransform] = None):
        path = data_utils.download(cls.url, filename=cls.filename)
        return cls.from_nifti(path, world_from_anatomical=world_from_anatomical, use_thresholding=True)

    def __init__(
        self,
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        anatomical_from_ijk: geo.FrameTransform,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ) -> None:
        super(KWire, self).__init__(data, materials,
                                    anatomical_from_ijk, world_from_anatomical)

        self.tip_in_ijk = geo.point(self.shape[0] / 2, self.shape[1] / 2, 0)
        self.base_in_ijk = geo.point(
            self.shape[0] / 2, self.shape[1] / 2, self.shape[2] - 1)

    @staticmethod
    def _convert_hounsfield_to_density(hu_values: np.ndarray):
        # TODO: verify coefficient.
        return 30 * hu_values

    @staticmethod
    def _segment_materials(
        hu_values: np.ndarray, use_thresholding: bool = True
    ) -> Dict[str, np.ndarray]:
        if not use_thresholding:
            raise NotImplementedError

        return dict(iron=(hu_values > 0))

    @property
    def tip_in_anatomical(self) -> geo.Point3D:
        """Get the location of the tool tip (the pointy end) in anatomical coordinates."""
        raise self.anatomical_from_ijk @ self.tip_in_ijk

    @property
    def base_in_anatomical(self) -> geo.Point3D:
        """Get the location of the tool base in anatomical coordinates."""
        raise self.anatomical_from_ijk @ self.base_in_ijk

    @property
    def tip_in_world(self) -> geo.Point3D:
        """Get the location of the tool tip (the pointy end) in world coordinates."""
        raise self.world_from_ijk @ self.tip_in_ijk

    @property
    def base_in_world(self) -> geo.Point3D:
        """Get the location of the tool base in world coordinates."""
        raise self.world_from_ijk @ self.base_in_ijk

    @property
    def length_in_world(self):
        return (self.tip_in_world - self.base_in_world).norm()

    def align(
        self,
        start_point_in_world: geo.Point3D,
        end_point_in_world: geo.Point3D,
        progress: float = 1.0
    ) -> None:
        """Align the tool so that it lies along the line between the two points.

        Args:
            start_point_in_world (geo.Point3D): The first point, in world space.
            end_point_in_world (geo.Point3D): The second point, in world space. The tip of the tool points toward this point.
            progress (float, optional): Where to place the tip of the tool between the start and end point, 
                on a scale from 0 to 1. 0 corresponds to the tip placed at the start point, 1 at the end point. Defaults to 1.0.
        """
        # First, get the known points of the tool in anatomical coordinates
        points_in_anatomical = [
            self.tip_in_anatomical,
            self.base_in_anatomical,
            self.origin_in_anatomical,
        ]

        # Now, interpolate along the direction of the tool to get the desired points in world.
        trajectory_vector = end_point_in_world - start_point_in_world
        desired_tip_in_world = start_point_in_world + progress * trajectory_vector
        desired_base_in_world = desired_tip_in_world - \
            trajectory_vector.hat() * self.length_in_world

        # We choose an arbitrary point to be the desired origin.
        desired_origin_in_world = desired_tip_in_world + (
            self.world_from_anatomical @ (self.tip_in_ijk - geo.point(0, 0, 0))).norm() * trajectory_vector.perpendicular().hat()

        points_in_world = [
            desired_tip_in_world,
            desired_base_in_world,
            desired_origin_in_world,
        ]

        self.world_from_anatomical = geo.FrameTransform.from_point_correspondence(
            points_in_anatomical, points_in_world)
