"""Visualization functions for DeepDRR.

DeepDRR uses pyvista to visualize the volumes and devices in 3D.
This is useful for debugging and verification purposes, but it
is not meant to replace a purpose built renderer. To view a scene
in detail, save the meshes to disk and open them in a suitable
viewer, such as 3D Slicer.

Note that these visualizations have the same limitations as PyVista.
They may not function properly in Jupyter notebooks.

Any object with the `get_mesh_in_world()` method can be visualized.

NOTE: often, PyVista will not render in an ssh window. To fix this, try some of the following:
```bash
#!/bin/bash
sudo apt-get install xvfb
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
export MESA_GL_VERSION_OVERRIDE=3.2
export MESA_GLSL_VERSION_OVERRIDE=150
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3
```

"""

import logging
from typing import Any, Union, List, Optional
import numpy as np
import os
import pyvista as pv

from . import utils
from . import geo


log = logging.getLogger(__name__)


def show(
    *item: Any,
    full: Union[bool, List[bool]] = False,
    colors: List[str] = ["tan", "cyan", "green", "red"],
    background: str = "white",
    use_cached: Union[bool, List[bool]] = True,
    offscreen: bool = False,
    mesh: Optional[pv.PolyData] = None,
    mesh_color: str = "black",
) -> Optional[np.ndarray]:
    """Show the given items in a pyvista window.

    Args:
        full (bool, optional): [description]. Defaults to True.
    """
    if offscreen:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
    os.environ["PYVISTA_USE_IPYVTK"] = "true"
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.2"
    os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "150"

    log.debug("display: {}".format(os.environ["DISPLAY"]))
    if offscreen and os.environ.get("DISPLAY") != ":99":
        os.environ["DISPLAY"] = ":99"
        os.system("Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &")
        os.system("sleep 3")

    plotter = pv.Plotter()
    plotter.show_axes()
    plotter.set_background(background)

    if mesh is not None:
        plotter.add_mesh(mesh, color=mesh_color)

    items = item
    fulls = utils.listify(full, len(items))
    use_cacheds = utils.listify(use_cached, len(items))
    for i, item in enumerate(items):
        color = colors[i % len(colors)]
        if hasattr(item, "get_mesh_in_world"):
            mesh = item.get_mesh_in_world(full=fulls[i], use_cached=use_cacheds[i])
        else:
            mesh = item
        plotter.add_mesh(mesh, color=color)

    plotter.reset_camera()
    plotter.show(auto_close=False)
    try:
        image = plotter.screenshot()
    except RuntimeError as e:
        log.warning(f"Failed to take screenshot: {e}")
        image = None
    plotter.close()
    return image


def get_frustum_mesh(
    camera_projection: geo.CameraProjection,
    pixel_size: float,
    image_path: Optional[str] = None,
    image_plane_distance: Optional[float] = None,
    full_frustum: bool = True,
) -> pv.PolyData:
    """Get a really simple camera mesh for the camera projections.

    Args:
        camera_projection (geo.CameraProjection): The camera projection.
        pixel_size (float): The pixel size in mm.
        image_path (str, optional): The path to the image. Defaults to None.
        image_plane_distance (float, optional): The distance from the camera to the image plane visualization. Defaults to None,
            which uses the distance from the camera to the image plane.
        full_frustum (bool, optional): Whether to show the full frustum, or just the principle ray. Defaults to True.

    Returns:
        pv.PolyData: Mesh representing the C-arm frustum.
    """

    focal_length_mm = camera_projection.intrinsic.focal_length * pixel_size
    sensor_height_mm = camera_projection.intrinsic.sensor_height * pixel_size
    sensor_width_mm = camera_projection.intrinsic.sensor_width * pixel_size

    # In camera frame
    s = geo.p(0, 0, 0)
    c = s + geo.v(0, 0, focal_length_mm)
    cx = pixel_size * camera_projection.intrinsic.cx
    cy = pixel_size * camera_projection.intrinsic.cy

    bl = geo.p(-cx, -cy, focal_length_mm)
    br = bl + geo.v(sensor_width_mm, 0, 0)
    ul = bl + geo.v(0, sensor_height_mm, 0)
    ur = bl + geo.v(sensor_width_mm, sensor_height_mm, 0)

    log.debug(
        f"sensor_width_mm: {sensor_width_mm}, sensor_height_mm: {sensor_height_mm}"
    )
    log.debug(f"cx: {cx}, cy: {cy}")
    log.debug(f"bl: {bl}, br: {br}, ul: {ul}, ur: {ur}")

    mesh = pv.Sphere(10, center=s)
    if full_frustum:
        mesh += (
            pv.Line(ur, ul)
            + pv.Line(br, bl)
            + pv.Line(ur, br)
            + pv.Line(ul, bl)
            + pv.Line(s, ul)
            + pv.Line(s, ur)
            + pv.Line(s, bl)
            + pv.Line(s, br)
        )
    else:
        mesh += pv.Line(s, c)

    if image_plane_distance is not None:
        pixel_size_at_plane = pixel_size / focal_length_mm * image_plane_distance
        cx_at_plane = pixel_size_at_plane * camera_projection.intrinsic.cx
        cy_at_plane = pixel_size_at_plane * camera_projection.intrinsic.cy
    else:
        image_plane_distance = focal_length_mm
        pixel_size_at_plane = pixel_size
        cx_at_plane = cx
        cy_at_plane = cy

    if image_path is not None:
        image = pv.read(image_path)
        # This is just a hack because some of rob's images are rotated by 180 degrees
        if camera_projection.intrinsic.fx > 0:
            image = image.transform(
                np.array(
                    [
                        [pixel_size_at_plane, 0, 0, -cx_at_plane],
                        [0, -pixel_size_at_plane, 0, cy_at_plane],
                        [0, 0, 1, image_plane_distance],
                        [0, 0, 0, 1],
                    ]
                ),
                inplace=False,
            )
        else:
            image = image.transform(
                np.array(
                    [
                        [-pixel_size_at_plane, 0, 0, cx_at_plane],
                        [0, pixel_size_at_plane, 0, -cy_at_plane],
                        [0, 0, 1, image_plane_distance],
                        [0, 0, 0, 1],
                    ]
                ),
                inplace=False,
            )
    else:
        image = pv.Plane(
            center=[0, 0, focal_length_mm],
            direction=[0, 0, 1],
            i_size=sensor_width_mm,
            j_size=sensor_height_mm,
        )

    image = image.transform(
        geo.get_data(camera_projection.world_from_camera3d), inplace=False
    )
    mesh.transform(geo.get_data(camera_projection.world_from_camera3d), inplace=True)
    return mesh, image
