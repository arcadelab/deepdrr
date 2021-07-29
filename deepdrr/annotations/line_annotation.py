from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path
import numpy as np
import json
import pyvista as pv

from .. import geo
from ..vol import Volume, AnyVolume

logger = logging.getLogger(__name__)


class LineAnnotation(object):
    def __init__(
        self, startpoint: geo.Point, endpoint: geo.Point, volume: AnyVolume
    ) -> None:
        # all points in anatomical coordinates, matching the provided volume.
        self.startpoint = geo.point(startpoint)
        self.endpoint = geo.point(endpoint)
        self.volume = volume

        assert (
            self.startpoint.dim == self.endpoint.dim
        ), "annotation points must have matching dim"

    def __str__(self):
        return f"LineAnnotation({self.startpoint}, {self.endpoint})"

    @classmethod
    def from_markup(cls, path: str, volume: AnyVolume) -> LineAnnotation:
        with open(path, "r") as file:
            ann = json.load(file)

        control_points = ann["markups"][0]["controlPoints"]
        points = [geo.point(cp["position"]) for cp in control_points]

        coordinate_system = ann["markups"][0]["coordinateSystem"]
        logger.debug(f"coordinate system: {coordinate_system}")

        if volume.anatomical_coordinate_system == "LPS":
            if coordinate_system == "LPS":
                pass
            elif coordinate_system == "RAS":
                logger.debug("converting to LPS")
                points = [geo.LPS_from_RAS @ p for p in points]
            else:
                raise ValueError
        elif volume.anatomical_coordinate_system == "RAS":
            if coordinate_system == "LPS":
                logger.debug("converting to RAS")
                points = [geo.RAS_from_LPS @ p for p in points]
            elif coordinate_system == "RAS":
                pass
            else:
                raise ValueError
        else:
            logger.warning(
                "annotation may not be in correct coordinate system. "
                "Unable to check against provided volume, probably "
                "because volume was created manually. Proceed with caution."
            )

        return cls(*points, volume)

    @property
    def startpoint_in_world(self) -> geo.Point:
        return self.volume.world_from_anatomical @ self.startpoint

    @property
    def endpoint_in_world(self) -> geo.Point:
        return self.volume.world_from_anatomical @ self.endpoint

    @property
    def midpoint_in_world(self) -> geo.Point:
        return self.volume.world_from_anatomical @ self.startpoint.lerp(
            self.endpoint, 0.5
        )

    def get_mesh_in_world(self, full: bool = True):
        u = self.startpoint_in_world
        v = self.endpoint_in_world

        mesh = pv.Line(u, v)
        mesh += pv.Sphere(2.5, u)
        mesh += pv.Sphere(2.5, v)
        return mesh
