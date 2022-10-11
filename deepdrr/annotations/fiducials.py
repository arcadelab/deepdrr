from __future__ import annotations

import logging
from typing import List, Literal, Optional
from pathlib import Path
import numpy as np
import json
import pyvista as pv

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
    ):
        self.points = points
        self.world_from_anatomical = world_from_anatomical
        self.anatomical_coordinate_system = anatomical_coordinate_system

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

    @classmethod
    def from_fcsv(cls, path: Path) -> FiducialList:
        """Load a FCSV file from Slicer3D

        Args:
            path (Path): Path to the FCSV file
        """
        # TODO: load the points from the fcsv


class Fiducial(geo.Point3D):
    @classmethod
    def from_fcsv(cls):
        pass

    @classmethod
    def from_json(cls):
        pass

    def save(self):
        pass
