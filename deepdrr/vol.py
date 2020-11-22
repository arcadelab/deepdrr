"""Volume class for CT volume.
"""

from typing import Union, Tuple, Literal, List, Optional

import numpy as np
from pathlib import Path

from .load_dicom import conv_hu_to_density, conv_hu_to_materials, conv_hu_to_materials_thresholding
from .geo import FrameTransform, Point3D, Vector3D, point, vector


class Volume(object):
    def __init__(
        self, 
        data: np.ndarray,
        materials: dict[str, np.ndarray],
        origin: Optional[Point3D] = None,
        spacing: Optional[Vector3D] = (1, 1, 1),
        anatomical_coordinate_system: Literal['LPS', 'RAS'] = 'LPS',
        world_from_anatomical: Optional[FrameTransform] = None,
    ):
        """Create a volume object with a segmentation of the materials, with its own anatomical coordinate space.

        Note that the anatomical coordinate system is not the world coordinate system (which is cartesion). 
        
        Suggested anatomical coordinate space units is milimeters. 
        A helpful introduction to the geometry is can be found [here](https://www.slicer.org/wiki/Coordinate_systems).

        Args:
            volume (np.ndarray): the volume density data.
            materials (dict[str, np.ndarray]): mapping from material names to binary segmentation of that material.
            origin (Point3D, optional): Location of the volume's origin in the anatomical coordinate system.
            spacing (Tuple[float, float, float], optional): Spacing of the volume in the anatomical coordinate system. Defaults to (1, 1, 1).
            anatomical_coordinate_system (Literal['LPS', 'RAS']): anatomical coordinate system convention.
            world_from_anatomical (FrameTransform, optional): Optional transformation from anatomical to world coordinates. 
                If None, then identity is used. Defaults to None.
        """
        self.data = np.array(data, dtype=np.float32)
        self.materials = materials
        self.origin = point(origin)
        self.spacing = vector(spacing)
        self.anatomical_coordinate_system = anatomical_coordinate_system
        self.world_from_anatomical = FrameTransform.identity(3) if world_from_anatomical is None else world_from_anatomical

        assert self.spacing.shape == (3,)

        self.volume_shape = vector(self.data.shape)

        # define anatomical_from_indices FrameTransform
        if self.anatomical_coordinate_system == 'LPS':
            # IJKtoLPS = LPS_from_IJK
            rotation = [
                [self.spacing[0], 0, 0],
                [0, 0, self.spacing[2]],
                [0, -self.spacing[1], 0],
            ]
            self.anatomical_from_index = FrameTransform.from_matrices(R=rotation, t=self.origin)
        else:
            raise NotImplementedError("conversion from RAS (not hard, look at LPS example)")

        self.world_from_index = self.world_from_anatomical @ self.anatomical_from_index
        self.index_from_world = self.world_from_index.inv

    @classmethod
    def from_hu(
        cls,
        hu_values: np.ndarray,
        use_thresholding: bool = True,
        **kwargs,
    ) -> None:
        data = conv_hu_to_density(hu_values)

        if use_thresholding:
            materials = conv_hu_to_materials_thresholding(hu_values)
        else:
            materials = conv_hu_to_materials(hu_values)

        return cls(data, materials, **kwargs)
    