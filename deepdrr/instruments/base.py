#!/usr/bin/env python3
import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pyvista as pv
from deepdrr import geo

from ..vol import Volume
from .. import utils
from ..utils import data_utils
from ..utils.mesh_utils import voxelize_multisurface


log = logging.getLogger(__name__)


class Instrument(Volume, ABC):
    """A class for representing instruments based on voxelized surface models.

    In the DeepDRR_DATA directory, place an STL file for each material you want to use, inside a
    directory determined by the class name. For example, if you have a
    class called `MyTool` with steel and plastic components, place the STL files `steel.stl` and
    `plastic.stl` in `DeepDRR_DATA/instruments/MyTool`.

    TODO: multiple components of the same material.

    DeepDRR_DATA
    └── instruments
        ├── ToolClassName
        │   ├── material_1.stl
        │   │── material_2.stl
        │   └── material_3.stl
        └── ToolClassName2
            ├── material_1.stl
            │── material_2.stl
            └── material_3.stl

    """

    # Every tool should define the tip in anatomical (modeling) coordinates, which is the center point begin inserted into the body along
    # the main axis of the tool, and the base, another point on that axis, so that they can be aligned.
    base: geo.Point3D
    tip: geo.Point3D
    radius: float

    # Available materials may be found at:
    # https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients

    _material_mapping = {
        "ABS Plastic": "polyethylene",
        "Ceramic": "concrete",
        "Stainless Steel": "iron",
        "stainless_steel": "iron",
        "steel": "iron",
        "cement": "concrete",
        "plastic": "polyethylene",
        "metal": "iron",
        "bone": "bone",
        "titanium": "titanium",
    }

    _default_densities = {
        "polyethylene": 1.05,  # polyethyelene is 0.97, but ABS plastic is 1.05
        "concrete": 1.5,
        "iron": 7.5,
        "titanium": 7,
        "bone": 1.5,
    }

    NUM_POINTS = 4000

    def __init__(
        self,
        density: float = 0.1,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        densities: Dict[str, float] = {},
    ):
        """Create the tool.

        Args:
            density: The spacing of the voxelization for each component of the tool.
            world_from_anatomical: Defines the pose of the tool in world.
            densities: Optional overrides to the material densities

        """
        self.density = density
        self._densities = self._default_densities.copy()
        self._densities.update(densities)

        self.instruments_dir = data_utils.deepdrr_data_dir() / "instruments"
        self.instruments_dir.mkdir(parents=True, exist_ok=True)

        self.surfaces = {}
        for material_dir, model_paths in self.get_model_paths():
            surface = pv.PolyData()
            for p in model_paths:
                s = pv.read(p)
                if len(s.points) > self.NUM_POINTS:
                    s = s.decimate(1 - self.NUM_POINTS / len(s.points))
                surface += s

            material_dirname = (
                material_dir.name if isinstance(material_dir, Path) else material_dir
            )
            self.surfaces[material_dirname] = surface

        # Convert from actual materials to DeepDRR compatible.
        materials = dict(
            (self._material_mapping[m], surf) for m, surf in self.surfaces.items()
        )

        volume_kwargs = voxelize_multisurface(
            voxel_size=density,
            surfaces=[(material, -1, surface) for material, surface in materials.items()],
            default_densities=self._densities,
        )

        super().__init__(
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system=None,
            **volume_kwargs,
        )

    def get_model_paths(self) -> List[Tuple[Path, List[Path]]]:
        """Get the model paths associated with this Tool.

        Returns:
            List[Tuple[Path, List[Path]]]: List of tuples containing the material dir and a list of paths with STL files for that material.
        """
        stl_dir = self.instruments_dir / self.__class__.__name__
        model_paths = [(p.stem, [p]) for p in stl_dir.glob("*.stl")]
        if not model_paths:
            raise FileNotFoundError(
                f"couldn't find materials for {self.__class__.__name__} in {stl_dir}"
            )
        return model_paths

    def get_cache_dir(self) -> Path:
        cache_dir = (
            data_utils.deepdrr_data_dir()
            / "cache"
            / self.__class__.__name__
            / "{}mm".format(str(self.density).replace(".", "-"))
        )
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    @property
    def base_in_world(self) -> geo.Point3D:
        return self.world_from_anatomical @ self.base

    @property
    def tip_in_world(self) -> geo.Point3D:
        return self.world_from_anatomical @ self.tip

    @property
    def length_in_world(self):
        return (self.tip_in_world - self.base_in_world).norm()

    def align(
        self,
        startpoint: geo.Point3D,
        endpoint: geo.Point3D,
        progress: float = 1,
        distance: Optional[float] = None,
    ):
        """Place the tool along the line between startpoint and endpoint.

        Args:
            startpoint (geo.Point3D): Startpoint in world.
            endpoint (geo.Point3D): Point in world toward which the tool points.
            progress (float): The fraction between startpoint and endpoint to place the tip of the tool. Defaults to 1.
            distance (Optional[float], optional): The distance of the tip along the trajectory. 0 corresponds
                to the tip placed at the start point, |startpoint - endpoint| at the end point.
                Overrides progress if provided. Defaults to None.


        """
        # useful: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        if distance is not None:
            progress = distance / self.length_in_world

        # interpolate along the direction of the tool to get the desired points in world.
        startpoint = geo.point(startpoint)
        endpoint = geo.point(endpoint)
        progress = float(progress)
        trajectory_vector = endpoint - startpoint

        desired_tip_in_world = startpoint.lerp(endpoint, progress)
        desired_base_in_world = (
            desired_tip_in_world - trajectory_vector.hat() * self.length_in_world
        )

        self.world_from_anatomical = geo.FrameTransform.from_line_segments(
            desired_tip_in_world,
            desired_base_in_world,
            self.tip,
            self.base,
        )

    def orient(
        self,
        startpoint: geo.Point3D,
        direction: geo.Vector3D,
        distance: float = 0,
    ):
        return self.align(
            startpoint,
            startpoint + direction.hat(),
            distance=distance,
        )

    def twist(self, angle: float, degrees: bool = True):
        """Rotate the tool clockwise (when looking down on it) by `angle`.

        Args:
            angle (float): The angle.
            degrees (bool, optional): Whether `angle` is in degrees. Defaults to True.
        """
        rotvec = (self.tip - self.base).hat()
        rotvec *= utils.radians(angle, degrees=degrees)
        self.world_from_anatomical = self.world_from_anatomical @ geo.frame_transform(
            geo.Rotation.from_rotvec(rotvec)
        )

    def get_mesh_in_world(self, full: bool = True, use_cached: bool = True):
        mesh = sum(self.surfaces.values(), pv.PolyData())
        mesh.transform(geo.get_data(self.world_from_anatomical), inplace=True)
        # meshh += pv.Sphere(
        #     center=list(self.world_from_ijk @ geo.point(0, 0, 0)), radius=5
        # )

        x, y, z = np.array(self.shape) - 1
        points = [
            [0, 0, 0],
            [0, 0, z],
            [0, y, 0],
            [0, y, z],
            [x, 0, 0],
            [x, 0, z],
            [x, y, 0],
            [x, y, z],
        ]

        points = [list(self.world_from_ijk @ geo.point(p)) for p in points]
        mesh += pv.Line(points[0], points[1])
        mesh += pv.Line(points[0], points[2])
        mesh += pv.Line(points[3], points[1])
        mesh += pv.Line(points[3], points[2])
        mesh += pv.Line(points[4], points[5])
        mesh += pv.Line(points[4], points[6])
        mesh += pv.Line(points[7], points[5])
        mesh += pv.Line(points[7], points[6])
        mesh += pv.Line(points[0], points[4])
        mesh += pv.Line(points[1], points[5])
        mesh += pv.Line(points[2], points[6])
        mesh += pv.Line(points[3], points[7])

        return mesh

    @property
    def center(self) -> geo.Point3D:
        return self.base.lerp(self.tip, 0.5)

    @property
    def center_in_world(self) -> geo.Point3D:
        return self.world_from_anatomical @ self.center

    @property
    def trajectory_in_world(self) -> geo.Ray3D:
        return geo.Ray3D.from_pn(
            self.tip_in_world, self.tip_in_world - self.base_in_world
        )

    @property
    def centerline_in_world(self) -> geo.Line3D:
        return geo.line(self.tip_in_world, self.base_in_world)

    def advance(self, distance: float):
        """Move the tool forward by the given distance.

        Args:
            distance (float): The distance to move the tool forward.
        """
        self.align(
            self.tip_in_world,
            self.tip_in_world + (self.tip_in_world - self.base_in_world),
            distance=distance,
        )
