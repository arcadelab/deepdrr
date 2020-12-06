"""Volume class for CT volume.
"""

from __future__ import annotations
from typing import Union, Tuple, Literal, List, Optional, Dict

import numpy as np
from pathlib import Path

from .load_dicom import conv_hu_to_density, conv_hu_to_materials, conv_hu_to_materials_thresholding
from . import geo

#  FrameTransform, Point3D, Vector3D, point, vector


class Volume(object):



    def __init__(
        self, 
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        origin: geo.Point3D,
        spacing: Optional[geo.Vector3D] = [1, 1, 1],
        anatomical_coordinate_system: Optional[Literal['LPS', 'RAS', 'none']] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ):
        """Create a volume object with a segmentation of the materials, with its own anatomical coordinate space.

        Note that the anatomical coordinate system is not the world coordinate system (which is cartesion). 
        
        Suggested anatomical coordinate space units is milimeters. 
        A helpful introduction to the geometry is can be found [here](https://www.slicer.org/wiki/Coordinate_systems).

        Args:
            volume (np.ndarray): the volume density data.
            materials (dict[str, np.ndarray]): mapping from material names to binary segmentation of that material.
            origin (Point3D): Location of the volume's origin in the anatomical coordinate system.
            spacing (Tuple[float, float, float], optional): Spacing of the volume in the anatomical coordinate system. Defaults to (1, 1, 1).
            anatomical_coordinate_system (Literal['LPS', 'RAS', 'none']): anatomical coordinate system convention. Defaults to 'none'.
            world_from_anatomical (FrameTransform, optional): Optional transformation from anatomical to world coordinates. 
                If None, then identity is used. Defaults to None.
        """
        self.data = np.array(data, dtype=np.float32)
        self.materials = self._format_materials(materials)
        self.origin = geo.point(origin)
        self.spacing = geo.vector(spacing)
        self.anatomical_coordinate_system = anatomical_coordinate_system
        self.world_from_anatomical = geo.FrameTransform.identity(3) if world_from_anatomical is None else world_from_anatomical

        assert self.spacing.dim == 3

        # define anatomical_from_indices FrameTransform
        if self.anatomical_coordinate_system is None or self.anatomical_coordinate_system == 'none':
            self.anatomical_from_voxel = geo.FrameTransform.from_scaling(scaling=self.spacing, translation=self.origin)
        elif self.anatomical_coordinate_system == 'LPS':
            # IJKtoLPS = LPS_from_IJK
            rotation = [
                [self.spacing[0], 0, 0],
                [0, 0, self.spacing[2]],
                [0, -self.spacing[1], 0],
            ]
            self.anatomical_from_voxel = geo.FrameTransform.from_rt(rotation=rotation, translation=self.origin)
        else:
            raise NotImplementedError("conversion from RAS (not hard, look at LPS example)")

    def _format_materials(
        self, 
        materials: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Standardize the input segmentation to a one-hot array.

        Args:
            materials (Dict[str, np.ndarray]): Either a mapping of material name to segmentation, 
                a segmentation with the same shape as the volume, or a one-hot segmentation.

        Returns:
            np.ndarray: dict from material names to np.float32 segmentations.
        """
        for mat in materials:
            materials[mat] = np.array(materials[mat]).astype(np.float32)

        return materials

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def from_hu(
        cls,
        hu_values: np.ndarray,
        origin: geo.Point3D,
        use_thresholding: bool = True,
        spacing: Optional[geo.Vector3D] = (1, 1, 1),
        anatomical_coordinate_system: Optional[Literal['LPS', 'RAS', 'none']] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ) -> None:
        data = conv_hu_to_density(hu_values)

        if use_thresholding:
            materials = conv_hu_to_materials_thresholding(hu_values)
        else:
            materials = conv_hu_to_materials(hu_values)

        return cls(
            data,
            materials, 
            origin=origin, 
            spacing=spacing, 
            anatomical_coordinate_system=anatomical_coordinate_system, 
            world_from_anatomical=world_from_anatomical,
        )

    @property
    def world_from_voxel(self):
        return self.world_from_anatomical @ self.anatomical_from_voxel

    @property
    def voxel_from_world(self):
        return self.world_from_voxel.inv
    
    def itow(self, other: Union[geo.Point3D, geo.Vector3D]) -> Union[geo.Point3D, geo.Vector3D]:
        """voxel-to-world. Take an voxel-space representation and return the world-space representation of the point or vector.

        Args:
            other (Point3D): the point or vector representation in the volume's voxel space.

        Returns:
            Point3D: 
        """
        return self.world_from_voxel @ other

    def wtoi(self, other: Union[geo.Point3D, geo.Vector3D]) -> Union[geo.Point3D, geo.Vector3D]:
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



