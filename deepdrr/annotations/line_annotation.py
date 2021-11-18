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

    def save(self, path: str):
        # todo(sean): save markups with color options based on the following template
        # markup = {
        #     "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        #     "markups": [
        #         {
        #             "type": "Line",
        #             "coordinateSystem": "LPS",
        #             "locked": false,
        #             "labelFormat": "%N-%d",
        #             "controlPoints": [
        #                 {
        #                     "id": "1",
        #                     "label": "entry",
        #                     "description": "",
        #                     "associatedNodeID": "",
        #                     "position": [
        #                         3.980233907699585,
        #                         -40.96451187133789,
        #                         -1392.0523681640626,
        #                     ],
        #                     "orientation": [
        #                         -1.0,
        #                         -0.0,
        #                         -0.0,
        #                         -0.0,
        #                         -1.0,
        #                         -0.0,
        #                         0.0,
        #                         0.0,
        #                         1.0,
        #                     ],
        #                     "selected": true,
        #                     "locked": false,
        #                     "visibility": true,
        #                     "positionStatus": "defined",
        #                 },
        #                 {
        #                     "id": "2",
        #                     "label": "exit",
        #                     "description": "",
        #                     "associatedNodeID": "",
        #                     "position": [
        #                         100.19214630126953,
        #                         -2.4773364067077638,
        #                         -1322.3232421875,
        #                     ],
        #                     "orientation": [
        #                         -1.0,
        #                         -0.0,
        #                         -0.0,
        #                         -0.0,
        #                         -1.0,
        #                         -0.0,
        #                         0.0,
        #                         0.0,
        #                         1.0,
        #                     ],
        #                     "selected": true,
        #                     "locked": false,
        #                     "visibility": true,
        #                     "positionStatus": "defined",
        #                 },
        #             ],
        #             "measurements": [
        #                 {
        #                     "name": "length",
        #                     "enabled": true,
        #                     "value": 124.90054351814699,
        #                     "printFormat": "%-#4.4gmm",
        #                 }
        #             ],
        #             "display": {
        #                 "visibility": true,
        #                 "opacity": 1.0,
        #                 "color": [0.5, 0.5, 0.5],
        #                 "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
        #                 "activeColor": [0.4, 1.0, 0.0],
        #                 "propertiesLabelVisibility": true,
        #                 "pointLabelsVisibility": true,
        #                 "textScale": 3.0,
        #                 "glyphType": "Sphere3D",
        #                 "glyphScale": 5.800000000000001,
        #                 "glyphSize": 5.0,
        #                 "useGlyphScale": true,
        #                 "sliceProjection": false,
        #                 "sliceProjectionUseFiducialColor": true,
        #                 "sliceProjectionOutlinedBehindSlicePlane": false,
        #                 "sliceProjectionColor": [1.0, 1.0, 1.0],
        #                 "sliceProjectionOpacity": 0.6,
        #                 "lineThickness": 0.2,
        #                 "lineColorFadingStart": 1.0,
        #                 "lineColorFadingEnd": 10.0,
        #                 "lineColorFadingSaturation": 1.0,
        #                 "lineColorFadingHueOffset": 0.0,
        #                 "handlesInteractive": false,
        #                 "snapMode": "toVisibleSurface",
        #             },
        #         }
        #     ],
        # }

        raise NotImplementedError

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
