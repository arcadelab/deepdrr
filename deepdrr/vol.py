"""Volume class for CT volume.
"""

from typing import Union, Tuple

import numpy as np
from pathlib import Path

from .geo import Frame, Point3D, Vector3D, point, vector


# TODO: create a VolumeData class containing the volume, materials, and
# relationship with the world coordinates.
class Volume(object):
    def __init__(
        self, 
        volume: np.ndarray,
        segmentation: Union[dict[str, np.ndarray], np.ndarray],
        voxel_size: Tuple[float, float, float] = (1, 1, 1),
        origin: Point3D = [0, 0, 0], # world space point the zero image index (wrt LPS coordinates)
    ):
        self.volume = volume
        self.segmentation = segmentation
        self.voxel_size = np.array(voxel_size)
        self.origin = point(origin)

        self.lps_to_


    @classmethod
    def from_hu(
        cls,
        hu_volume: np.ndarray,
        **kwargs
    ):
        pass
    
    @classmethod
    def from_density(
        cls,
        volume: np.ndarray,
        **kwargs,
    ):
        pass

