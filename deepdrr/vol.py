"""Volume class for CT volume.
"""

from __future__ import annotations
from typing import Union, Tuple, Literal, List, Optional, Dict

import numpy as np
from pathlib import Path

from .load_dicom import conv_hu_to_density, conv_hu_to_materials, conv_hu_to_materials_thresholding
from .geo import FrameTransform, Point3D, Vector3D, point, vector


class Volume(object):
    def __init__(
        self, 
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        origin: Optional[Point3D] = None,
        spacing: Optional[Vector3D] = (1, 1, 1),
        anatomical_coordinate_system: Literal['LPS', 'RAS', 'none'] = 'none',
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
            anatomical_coordinate_system (Literal['LPS', 'RAS', 'none']): anatomical coordinate system convention. Defaults to 'LPS'.
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
        if self.anatomical_coordinate_system == 'none':
            raise NotImplementedError('not sure what this would mean')
        elif self.anatomical_coordinate_system == 'LPS':
            # IJKtoLPS = LPS_from_IJK
            rotation = [
                [self.spacing[0], 0, 0],
                [0, 0, self.spacing[2]],
                [0, -self.spacing[1], 0],
            ]
            self.anatomical_from_voxel = FrameTransform.from_rt(R=rotation, t=self.origin)
        else:
            raise NotImplementedError("conversion from RAS (not hard, look at LPS example)")

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

    @property
    def world_from_voxel(self):
        return self.world_from_anatomical @ self.anatomical_from_voxel

    @property
    def voxel_from_world(self):
        return self.world_from_voxel.inv
    
    def itow(self, other: Union[Point3D, Vector3D]) -> Union[Point3D, Vector3D]:
        """voxel-to-world. Take an voxel-space representation and return the world-space representation of the point or vector.

        Args:
            other (Point3D): the point or vector representation in the volume's voxel space.

        Returns:
            Point3D: 
        """
        return self.world_from_voxel @ other

    def wtoi(self, other: Union[Point3D, Vector3D]) -> Union[Point3D, Vector3D]:
        """World-to-voxel. Take a world-space representation of a point or vector and return the voxel-space representation.

        Note the preferred format would be to just use self.voxel_from_world as a function, since it is a callable.

        Args:
            other (PointOrVector3D): the point or vector.

        Returns:
            PointOrVector3D: [description]
        """
        return self.voxel_from_world @ other

    @classmethod
    def from_dicom(
        cls,
        path: Union[str, Path],
    ) -> Volume:
        """Create the volume from a DICOM file."""
        raise NotImplementedError('load a volume from a dicom file')

    def to_dicom(self, path: Union[str, Path]):
        """Write the volume to a DICOM file.

        Args:
            path (str): the path to the file.
        """
        path = Path(path)

        raise NotImplementedError('save volume to dicom file')



