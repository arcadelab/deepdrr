from __future__ import annotations
from typing import Any, Union, Tuple, List, Optional, Dict, Type

import logging
import numpy as np
from pathlib import Path
import nibabel as nib
from pydicom.filereader import dcmread
import nrrd
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv

from .. import load_dicom
from .. import geo
from .. import utils
from ..utils import data_utils
from ..utils import mesh_utils
from ..device import Device
from ..projector.material_coefficients import material_coefficients

vtk, nps, vtk_available = utils.try_import_vtk()


log = logging.getLogger(__name__)


class Mesh(object):
    anatomical_from_IJK: geo.FrameTransform
    world_from_anatomical: geo.FrameTransform
    mesh: pv.PolyData
    material: str
    density: float

    def __init__(
        self,
        material: str,
        density: float,
        mesh: pv.PolyData,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ) -> None:
        self.anatomical_from_IJK = geo.frame_transform(None)
        self.world_from_anatomical = (
            geo.FrameTransform.identity(3)
            if world_from_anatomical is None
            else geo.frame_transform(world_from_anatomical)
        )
        self.mesh = mesh
        self.material = material
        self.density = density


    @property
    def world_from_IJK(self) -> geo.FrameTransform:
        return self.world_from_anatomical @ self.anatomical_from_IJK

    @property
    def world_from_ijk(self) -> geo.FrameTransform:
        return self.world_from_IJK

    @property
    def IJK_from_world(self) -> geo.FrameTransform:
        return self.world_from_IJK.inverse()

    @property
    def ijk_from_world(self) -> geo.FrameTransform:
        return self.world_from_IJK.inv

    @property
    def anatomical_from_world(self):
        return self.world_from_anatomical.inv

    @property
    def ijk_from_anatomical(self):
        return self.anatomical_from_IJK.inv

    @property
    def IJK_from_anatomical(self):
        return self.anatomical_from_IJK.inv

    @property
    def origin(self) -> geo.Point3D:
        """The origin of the volume in anatomical space."""
        return geo.point(self.anatomical_from_ijk.t)

    origin_in_anatomical = origin

    @property
    def origin_in_world(self) -> geo.Point3D:
        """The origin of the volume in world space."""
        return geo.point(self.world_from_ijk.t)

    @property
    def center_in_world(self) -> geo.Point3D:
        """The center of the volume in world coorindates. Useful for debugging."""
        return self.world_from_ijk @ geo.point(np.array(self.shape) / 2)

    # def get_bounding_box_in_world(self) -> Tuple[geo.Point3D, geo.Point3D]:
    #     """Get the corners of a bounding box enclosing the volume in world coordinates.

    #     Assumes cell-centered sampling.

    #     Returns:
    #         geo.Point3D: The lower corner of the bounding box.
    #         geo.Point3D: The upper corner of the bounding box.
    #     """
    #     x, y, z = np.array(self.shape) - 0.5
    #     corners_ijk = [
    #         geo.point(-0.5, -0.5, -0.5),
    #         geo.point(-0.5, -0.5, z),
    #         geo.point(-0.5, y, -0.5),
    #         geo.point(-0.5, y, z),
    #         geo.point(x, -0.5, -0.5),
    #         geo.point(x, -0.5, z),
    #         geo.point(x, y, -0.5),
    #         geo.point(x, y, z),
    #     ]

    #     corners = np.array([np.array(self.world_from_ijk @ p) for p in corners_ijk])
    #     min_corner = geo.point(corners.min(0))
    #     max_corner = geo.point(corners.max(0))
    #     return min_corner, max_corner

