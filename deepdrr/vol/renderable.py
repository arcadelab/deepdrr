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
from abc import ABC, abstractmethod

from .. import load_dicom
from .. import geo
from .. import utils
from ..utils import data_utils
from ..utils import mesh_utils
from ..device import Device
from ..projector.material_coefficients import material_coefficients

vtk, nps, vtk_available = utils.try_import_vtk()


log = logging.getLogger(__name__)


class Renderable(ABC):
    anatomical_from_IJK: geo.FrameTransform
    world_from_anatomical: geo.FrameTransform

    def __init__(
        self,
        anatomical_from_IJK: Optional[geo.FrameTransform] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        anatomical_from_ijk: Optional[geo.FrameTransform] = None,
        enabled: bool = True,
    ) -> None:
        if anatomical_from_ijk is not None:
            # Deprecation warning
            anatomical_from_IJK = anatomical_from_ijk
        self.anatomical_from_IJK = geo.frame_transform(anatomical_from_IJK)
        self.world_from_anatomical = (
            geo.FrameTransform.identity(3)
            if world_from_anatomical is None
            else geo.frame_transform(world_from_anatomical)
        )
        self.set_enabled(enabled)

    @abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        pass

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

    @property
    def origin_in_anatomical(self) -> geo.Point3D:
        """The origin of the volume in anatomical space."""
        return self.origin

    @property
    def origin_in_world(self) -> geo.Point3D:
        """The origin of the volume in world space."""
        return geo.point(self.world_from_ijk.t)

    @abstractmethod
    def get_center(self) -> geo.Point3D:
        """The "center" of the renderable in anatomical space."""
        pass

    @property
    def center_in_world(self) -> geo.Point3D:
        """The center of the volume in world coorindates. Useful for debugging."""
        return self.world_from_ijk @ self.get_center()

    @abstractmethod
    def get_bounding_box_in_world(self) -> Tuple[geo.Point3D, geo.Point3D]:
        """Get the corners of a bounding box enclosing the volume in world coordinates.

        Assumes cell-centered sampling.

        Returns:
            geo.Point3D: The lower corner of the bounding box.
            geo.Point3D: The upper corner of the bounding box.
        """
        pass

    def place(
        self, point_in_anatomical: geo.Point3D, desired_point_in_world: geo.Point3D
    ) -> None:
        """Translate the volume so that x_in_anatomical corresponds to x_in_world."""
        p_A = np.array(point_in_anatomical)
        p_W = np.array(desired_point_in_world)
        r_WA = self.world_from_anatomical.R
        t_WA = p_W - r_WA @ p_A
        self.world_from_anatomical.t = t_WA  # fancy setter

    def place_center(self, x: geo.Point3D) -> None:
        """Translate the volume so that its center is located at world-space point x.

        Only changes the translation elements of the world_from_anatomical transform. Preserves the current rotation of the

        Args:
            x (geo.Point3D): the world-space point.

        """

        x = geo.point(x)
        center_anatomical = self.get_center()
        # center_world = self.world_from_anatomical @ center_anatomical
        self.place(center_anatomical, x)

    def apply_transform(self, transform: geo.FrameTransform) -> None:
        """Apply a transformation to the volume.

        Args:
            transform (geo.FrameTransform): The transformation to apply.
        """
        self.world_from_anatomical = transform @ self.world_from_anatomical

    def translate(self, t: geo.Vector3D) -> Renderable:
        """Translate the volume by `t`.

        Args:
            t (geo.Vector3D): The vector to translate by, in world space.
        """
        t = geo.vector(t)
        T = geo.FrameTransform.from_translation(t)
        self.world_from_anatomical = T @ self.world_from_anatomical
        return self

    def rotate(
        self,
        rotation: Union[geo.Vector3D, Rotation, geo.FrameTransform],
        center: Optional[geo.Point3D] = None,
    ) -> Renderable:
        """Rotate the volume by `rotation` about `center`.

        Args:
            rotation (Union[geo.Vector3D, Rotation]): the rotation in world-space. If it is a vector, `Rotation.from_rotvec(rotation)` is used.
            center (geo.Point3D, optional): the center of rotation in world space coordinates. If None, the center of the volume is used.
        """

        if isinstance(rotation, Rotation):
            R = geo.FrameTransform.from_rotation(rotation.as_matrix())
        elif isinstance(rotation, geo.FrameTransform):
            R = rotation
        else:
            r = geo.vector(rotation)
            R = geo.FrameTransform.from_rotation(Rotation.from_rotvec(r).as_matrix())

        if center is None:
            center = self.center_in_world

        T = geo.FrameTransform.from_translation(center)
        self.world_from_anatomical = T @ R @ T.inv @ self.world_from_anatomical
        return self
