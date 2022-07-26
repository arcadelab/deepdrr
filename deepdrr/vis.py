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

from . import utils

pv, pv_available = utils.try_import_pyvista()

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
