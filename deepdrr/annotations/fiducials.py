from __future__ import annotations

import logging
from typing import List, Literal, Optional
from pathlib import Path
import numpy as np
import json
import pyvista as pv
import pandas as pd

from .. import geo, utils
from ..vol import Volume

log = logging.getLogger(__name__)


class FiducialList:
    # Can be treated like a list of Point3Ds
    def __init__(
        self,
        points: List[geo.Point3D],
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        anatomical_coordinate_system: Literal["RAS", "LPS"] = "RAS",
        names: Optional[List[str]] = None,
    ):
        self.points = points
        self.world_from_anatomical = world_from_anatomical
        self.anatomical_coordinate_system = anatomical_coordinate_system

        if names is None:
            num_digits = len(str(len(points)))
            self.names = [f"P{str(i).zfill(num_digits)}" for i in range(len(points))]
        else:
            assert len(names) == len(points), "Number of names must match number of points"
            self.names = names

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __repr__(self):
        return f"FiducialList({self.points})"

    def __str__(self):
        return str(self.points)

    def to_RAS(self) -> FiducialList:
        if self.anatomical_coordinate_system == "RAS":
            return self
        else:
            return FiducialList(
                [geo.RAS_from_LPS @ p for p in self.points],
                self.world_from_anatomical,
                "RAS",
            )

    def to_LPS(self) -> FiducialList:
        if self.anatomical_coordinate_system == "LPS":
            return self
        else:
            return FiducialList(
                [geo.LPS_from_RAS @ p for p in self.points],
                self.world_from_anatomical,
                "LPS",
            )

    @classmethod
    def from_fcsv(
        cls, path: Path, world_from_anatomical: Optional[geo.FrameTransform] = None
    ) -> FiducialList:
        """Load a FCSV file from Slicer3D

        Args:
            path (Path): Path to the FCSV file

        Returns:
            np.ndarray: Array of 3D points
        """
        with open(path, "r") as f:
            lines = f.readlines()
        points = []
        coordinate_system = None
        for line in lines:
            if line.startswith("# CoordinateSystem"):
                coordinate_system = line.split("=")[1].strip()
            elif line.startswith("#"):
                continue
            else:
                x, y, z = line.split(",")[1:4]
                points.append(geo.point(float(x), float(y), float(z)))

        if coordinate_system is None:
            log.warning("No coordinate system specified in FCSV file. Assuming LPS.")
            coordinate_system = "LPS"
        assert coordinate_system in ["RAS", "LPS"], "Unknown coordinate system"

        return cls(
            points,
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system=coordinate_system,
        )

    @classmethod
    def from_json(cls, path: Path, world_from_anatomical: Optional[geo.FrameTransform] = None):
        # TODO: add support for associated IDs of the fiducials. Should really be a list/dict.
        data = pd.read_json(path)
        control_points_table = pd.DataFrame.from_dict(data["markups"][0]["controlPoints"])
        coordinate_system = data["markups"][0]["coordinateSystem"]
        # TODO: not sure if this works.
        points = [geo.point(*row["position"]) for _, row in control_points_table.iterrows()]
        names = control_points_table["label"].values.tolist()

        return cls(
            points,
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system=coordinate_system,
            names=names,
        )

    def save(self, path: Path):
        raise NotImplementedError()


class Fiducial(geo.Point3D):
    @classmethod
    def from_fcsv(
        cls,
        path: Path,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ):
        fiducial_list = FiducialList.from_fcsv(path)
        assert len(fiducial_list) == 1, "Expected a single fiducial"
        return cls(
            fiducial_list[0].data,
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system=fiducial_list.anatomical_coordinate_system,
        )

    @classmethod
    def from_json(cls, path: Path, world_from_anatomical: Optional[geo.FrameTransform] = None):
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError
