from __future__ import annotations

import logging
from typing import List, Optional
from pathlib import Path
import numpy as np
import json
import pyvista as pv

from .. import geo, utils
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

    def save(self, path: str, color: List[float] = [0.5, 0.5, 0.5]):
        """Save the Line annotation to a mrk.json file, which can be opened by 3D Slicer.

        Args:
            path (str): Output path to the file.
            color (List[int], optional): The color of the saved annotation.
        """
        path = Path(path).expanduser()

        markup = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
            "markups": [
                {
                    "type": "Line",
                    "coordinateSystem": self.volume.anatomical_coordinate_system,
                    "locked": True,
                    "labelFormat": "%N-%d",
                    "controlPoints": [
                        {
                            "id": "1",
                            "label": "entry",
                            "description": "",
                            "associatedNodeID": "",
                            "position": utils.jsonable(self.startpoint),
                            "orientation": [
                                -1.0,
                                -0.0,
                                -0.0,
                                -0.0,
                                -1.0,
                                -0.0,
                                0.0,
                                0.0,
                                1.0,
                            ],
                            "selected": True,
                            "locked": False,
                            "visibility": True,
                            "positionStatus": "defined",
                        },
                        {
                            "id": "2",
                            "label": "exit",
                            "description": "",
                            "associatedNodeID": "",
                            "position": utils.jsonable(self.endpoint),
                            "orientation": [
                                -1.0,
                                -0.0,
                                -0.0,
                                -0.0,
                                -1.0,
                                -0.0,
                                0.0,
                                0.0,
                                1.0,
                            ],
                            "selected": True,
                            "locked": False,
                            "visibility": True,
                            "positionStatus": "defined",
                        },
                    ],
                    "measurements": [
                        {
                            "name": "length",
                            "enabled": True,
                            "value": 124.90054351814699,
                            "printFormat": "%-#4.4gmm",
                        }
                    ],
                    "display": {
                        "visibility": True,
                        "opacity": 1.0,
                        "color": color,
                        "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                        "activeColor": [0.4, 1.0, 0.0],
                        "propertiesLabelVisibility": True,
                        "pointLabelsVisibility": True,
                        "textScale": 3.0,
                        "glyphType": "Sphere3D",
                        "glyphScale": 5.800000000000001,
                        "glyphSize": 5.0,
                        "useGlyphScale": True,
                        "sliceProjection": False,
                        "sliceProjectionUseFiducialColor": True,
                        "sliceProjectionOutlinedBehindSlicePlane": False,
                        "sliceProjectionColor": [1.0, 1.0, 1.0],
                        "sliceProjectionOpacity": 0.6,
                        "lineThickness": 0.2,
                        "lineColorFadingStart": 1.0,
                        "lineColorFadingEnd": 10.0,
                        "lineColorFadingSaturation": 1.0,
                        "lineColorFadingHueOffset": 0.0,
                        "handlesInteractive": False,
                        "snapMode": "toVisibleSurface",
                    },
                }
            ],
        }

        with open(path, "w") as file:
            json.dump(markup, file)

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
