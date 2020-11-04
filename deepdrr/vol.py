"""Volume class for CT volume.
"""

import numpy as np
from pathlib import Path

from .geo import Frame


# TODO: create a VolumeData class containing the volume, materials, and
# relationship with the world coordinates.
class Volume(object):
    def __init__(
        self, 
        volume: np.ndarray,
        materials: Union[Dict[str, np.ndarray], np.ndarray],
        voxel_size: np.ndarray,
        origin: np.ndarray,
    ):
        pass

