"""Visualization functions for DeepDRR.

DeepDRR uses pyvista to visualize the volumes and devices in 3D.
This is useful for debugging and verification purposes, but it
is not meant to replace a purpose built renderer. To view a scene
in detail, save the meshes to disk and open them in a suitable
viewer, such as 3D Slicer.

Note that these visualizations have the same limitations as PyVista.
They may not function properly in Jupyter notebooks.

Any object with the `get_mesh_in_world()` method can be visualized.

"""

import logging
from typing import Any, Union, List
import numpy as np

from . import utils

pv, pv_available = utils.try_import_pyvista()

logger = logging.getLogger(__name__)


def show(
    *item: Any,
    full: Union[bool, List[bool]] = False,
    colors: List[str] = ["tan", "cyan", "green", "red"],
    background: str = "#4d94b0",
) -> np.ndarray:
    """Show the given items in a pyvista window.

    Args:
        full (bool, optional): [description]. Defaults to True.
    """
    plotter = pv.Plotter()
    plotter.show_axes()
    plotter.set_background(background)

    items = item
    fulls = utils.listify(full, len(items))
    for i, (item, full) in enumerate(zip(items, fulls)):
        color = colors[i % len(colors)]
        plotter.add_mesh(item.get_mesh_in_world(full=full), color=color)

    plotter.reset_camera()
    plotter.show(auto_close=False)
    image = plotter.screenshot()
    plotter.close()
    return image
