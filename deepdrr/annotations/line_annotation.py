from __future__ import annotations

import logging
from typing import List, Optional
from pathlib import Path
import numpy as np
import json
import pyvista as pv

from .. import geo, utils
from ..vol import Volume

log = logging.getLogger(__name__)


class LineAnnotation(object):
    def __init__(
        self,
        startpoint: geo.Point,
        endpoint: geo.Point,
        volume: Optional[Volume] = None,
        anatomical_from_world: Optional[geo.FrameTransform] = None,
        anatomical_coordinate_system: Optional[str] = None,
    ) -> None:
        """Create a Line Annotation.

        Must provide either the volume this relies on, or the minimal attributes (pose of volume in world and coordinate system).

        Args:
            startpoint (geo.Point): The startpoint in anatomical coordinates.
            endpoint (geo.Point): The endpoint in anatomical coordinates.
            volume (Optional[Volume], optional): The volume, with a given pose. If not provided, must provide `anatomical_from_world` and `anatomical_coordinate_system`. Defaults to None.
            anatomical_from_world (Optional[geo.FrameTransform], optional): [description]. Defaults to None.
            anatomical_coordinate_system (Optional[str], optional): [description]. Defaults to None.
        """
        # all points in anatomical coordinates, matching the provided volume.
        self.startpoint = geo.point(startpoint)
        self.endpoint = geo.point(endpoint)
        if volume is None:
            assert (
                anatomical_from_world is not None
                and anatomical_coordinate_system.upper() in ["RAS", "LPS"]
            )
            self.anatomical_coordinate_system = anatomical_coordinate_system.upper()
            self.anatomical_from_world = geo.frame_transform(anatomical_from_world)
        else:
            self.anatomical_from_world = volume.anatomical_from_world
            self.anatomical_coordinate_system = volume.anatomical_coordinate_system

        assert (
            self.startpoint.dim == self.endpoint.dim
        ), "annotation points must have matching dim"

    def __str__(self):
        return f"LineAnnotation({self.startpoint}, {self.endpoint})"

    @classmethod
    def from_markup(
        cls,
        path: str,
        volume: Optional[Volume] = None,
        anatomical_from_world: Optional[geo.FrameTransform] = None,
        anatomical_coordinate_system: Optional[str] = None,
    ) -> LineAnnotation:
        with open(path, "r") as file:
            ann = json.load(file)

        if volume is None:
            assert (
                anatomical_from_world is not None
                and anatomical_coordinate_system is not None
            )
        else:
            anatomical_coordinate_system = volume.anatomical_coordinate_system
            anatomical_from_world = volume.anatomical_from_world

        control_points = ann["markups"][0]["controlPoints"]
        points = [geo.point(cp["position"]) for cp in control_points]

        coordinate_system = ann["markups"][0]["coordinateSystem"]
        log.debug(f"loading markup with coordinate system: {coordinate_system}")

        if anatomical_coordinate_system == "LPS":
            if coordinate_system == "LPS":
                pass
            elif coordinate_system == "RAS":
                log.debug("converting to LPS")
                points = [geo.LPS_from_RAS @ p for p in points]
            else:
                raise ValueError
        elif anatomical_coordinate_system == "RAS":
            if coordinate_system == "LPS":
                log.debug("converting to RAS")
                points = [geo.RAS_from_LPS @ p for p in points]
            elif coordinate_system == "RAS":
                pass
            else:
                raise ValueError
        else:
            log.warning(
                "annotation may not be in correct coordinate system. "
                "Unable to check against provided volume, probably "
                "because volume was created manually. Proceed with caution."
            )

        return cls(
            *points,
            volume=volume,
            anatomical_from_world=anatomical_from_world,
            anatomical_coordinate_system=anatomical_coordinate_system,
        )

    def save(
        self,
        path: str,
        color: List[float] = [1.0, 0.5000076295109484, 0.5000076295109484],
    ):
        """Save the Line annotation to a mrk.json file, which can be opened by 3D Slicer.

        Args:
            path (str): Output path to the file.
            color (List[int], optional): The color of the saved annotation.
        """
        path = Path(path).expanduser()

        def to_lps(x):
            if self.anatomical_coordinate_system == "LPS":
                return list(x)
            elif self.anatomical_coordinate_system == "RAS":
                return list(geo.LPS_from_RAS @ x)
            else:
                raise ValueError

        # log.info(f"start, end: {self.startpoint, self.endpoint}")
        # log.info(
        #     f"start, end in world: {self.startpoint_in_world, self.endpoint_in_world}"
        # )

        markup = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
            "markups": [
                {
                    "type": "Line",
                    "coordinateSystem": self.anatomical_coordinate_system,
                    "locked": False,
                    "labelFormat": r"%N-%d",
                    "controlPoints": [
                        {
                            "id": "1",
                            "label": "startpoint",
                            "description": "",
                            "associatedNodeID": "",
                            "position": list(self.startpoint),
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
                            "label": "endpoint",
                            "description": "",
                            "associatedNodeID": "",
                            "position": list(self.endpoint),
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
                            "printFormat": r"%-#4.4gmm",
                        }
                    ],
                    "display": {
                        "visibility": True,
                        "opacity": 1.0,
                        "color": [0.5, 0.5, 0.5],
                        "selectedColor": color,
                        "activeColor": [0.4, 1.0, 0.0],
                        "propertiesLabelVisibility": False,
                        "pointLabelsVisibility": False,
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

        markup = utils.jsonable(markup)

        with open(path, "w") as file:
            json.dump(markup, file)

    @property
    def world_from_anatomical(self) -> geo.FrameTransform:
        return self.anatomical_from_world.inv

    @property
    def startpoint_in_world(self) -> geo.Point:
        return self.world_from_anatomical @ self.startpoint

    @property
    def endpoint_in_world(self) -> geo.Point:
        return self.world_from_anatomical @ self.endpoint

    @property
    def midpoint_in_world(self) -> geo.Point:
        return self.world_from_anatomical @ self.startpoint.lerp(self.endpoint, 0.5)

    def get_mesh_in_world(self, full: bool = True):
        u = self.startpoint_in_world
        v = self.endpoint_in_world

        mesh = pv.Line(u, v)
        mesh += pv.Sphere(2.5, u)
        mesh += pv.Sphere(2.5, v)
        return mesh
