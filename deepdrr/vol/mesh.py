from __future__ import annotations
from re import T
from typing import Any, Union, Tuple, List, Optional, Dict, Type, TypeVar
from functools import cached_property

import logging
from killeengeo import Point3D
import numpy as np
from pathlib import Path
import nibabel as nib
from pydicom.filereader import dcmread
import nrrd
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv
import killeengeo as kg

from ..pyrenderdrr import DRRMaterial
from .. import load_dicom
from .. import utils
from ..utils import data_utils
from ..utils import mesh_utils
from ..device import Device
from ..projector.material_coefficients import material_coefficients
from .renderable import Renderable
import pyrender
import trimesh
from trimesh.repair import fix_normals

# from deepdrr.utils.mesh_utils import polydata_to_trimesh


vtk, nps, vtk_available = utils.try_import_vtk()

log = logging.getLogger(__name__)

T = TypeVar("T", bound="Mesh")


class Mesh(Renderable):
    mesh: pyrender.Mesh

    def __init__(
        self,
        anatomical_from_IJK: Optional[kg.FrameTransform] = None,
        world_from_anatomical: Optional[kg.FrameTransform] = None,
        anatomical_from_ijk: Optional[kg.FrameTransform] = None,
        mesh: pyrender.Mesh = None,
        **kwargs,
    ) -> None:
        if mesh is None:
            raise ValueError("mesh must be specified")

        self.mesh = mesh

        Renderable.__init__(
            self,
            anatomical_from_IJK=anatomical_from_IJK,
            world_from_anatomical=world_from_anatomical,
            anatomical_from_ijk=anatomical_from_ijk,
            **kwargs,
        )

    def set_enabled(self, enabled: bool) -> None:
        self.mesh.is_visible = enabled

    @property
    def enabled(self) -> bool:
        return self.mesh.is_visible

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self.set_enabled(enabled)

    def get_center(self) -> kg.Point3D:
        return kg.point(self.mesh.centroid)

    def get_bounding_box_in_world(self) -> Tuple[Point3D, Point3D]:
        raise NotImplementedError("TODO")

    def get_bounding_box_volume(self) -> float:
        min_bounds, max_bounds = self.get_bounding_box_in_world()
        return abs(np.prod(max_bounds - min_bounds))

    @classmethod
    def from_stl(
        cls: Type[T],
        path: Union[Union[str, Path], List[Union[str, Path]]],
        material: Union[
            Union[str, DRRMaterial], List[Union[str, DRRMaterial]]
        ] = "iron",
        convert_to_RAS: bool = False,
        tag: Optional[str] = None,
        **kwargs,
    ) -> T:
        """Create a mesh for the given material with default density.

        Args:
            path (Union[str, Path]): Path to the STL file.
            material (str | DRRMaterial, optional): Material to use. Defaults to "iron".
            convert_to_RAS (bool, optional): Good practice is to store meshes in LPS coordinates. When loading a mesh
                from a CT file, such as a segmentation, that was saved in LPS coordinates, this should be set to True.
                Defaults to False.

        Returns:
            Mesh: The mesh.

        """
        if not isinstance(path, list):
            path = [path]

        path = [Path(p) for p in path]

        for p in path:
            if not p.exists():
                raise FileNotFoundError(f"Could not find file {p}")

        if not isinstance(material, list):
            material = [material]

        material = [
            DRRMaterial.from_name(m, tag=tag) if isinstance(m, str) else m
            for m in material
        ]

        prims = []
        for p, m in zip(path, material):
            mesh = mesh_utils.load_trimesh(p, convert_to_RAS=convert_to_RAS)
            fix_normals(mesh)
            prims.append(pyrender.Mesh.from_trimesh(mesh, material=m).primitives[0])

        mesh = pyrender.Mesh(primitives=prims)
        return cls(mesh=mesh, **kwargs)

    @cached_property
    def get_bounding_AABB(self):
        prims = self.mesh.primitives

        minpt = np.inf * np.ones(3)
        maxpt = -np.inf * np.ones(3)

        for prim in prims:
            positions = prim.positions
            minpt = np.minimum(minpt, np.min(positions, axis=0))
            maxpt = np.maximum(maxpt, np.max(positions, axis=0))

        return minpt, maxpt

    @cached_property
    def get_loose_bounding_sphere(self):
        # use the center of the AABB as the center of the sphere
        minpt, maxpt = self.get_bounding_AABB
        center = (minpt + maxpt) / 2

        # use the distance from the center to the furthest point as the radius
        prims = self.mesh.primitives
        radius = 0
        for prim in prims:
            positions = prim.positions
            radius = max(radius, np.max(np.linalg.norm(positions - center, axis=1)))

        return center, radius

    def get_points(self) -> np.ndarray:
        return np.array(self.mesh.primitives[0].positions)


class LinearToolMesh(Mesh):
    """A mesh with a base and a tip defined in anatomical coordinates."""

    def __init__(
        self,
        base: kg.Point3D,
        tip: kg.Point3D,
        **kwargs,
    ) -> None:
        self.base = kg.point(base)
        self.tip = kg.point(tip)
        super().__init__(**kwargs)

    def align(
        self,
        a_in_world: kg.Point3D,
        b_in_world: kg.Point3D,
        progress: float = 1,
        distance: Optional[float] = None,
    ):
        """Align the renderable so that base and tip are aligned with a_in_world and b_in_world."""
        super().align(
            a_in_world,
            b_in_world,
            progress=progress,
            distance=distance,
            a=self.base,
            b=self.tip,
        )
