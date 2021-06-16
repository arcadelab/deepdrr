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

from typing import Any, Union, List
from . import utils

pv, pv_available = utils.try_import_pyvista()


def show(*item: Any, full: Unionbool = False) -> None:
    """[summary]

    Args:
        full (bool, optional): [description]. Defaults to True.
    """
    renderer = pv.Plotter()
    renderer.show_axes()
    renderer.set_background("#4d94b0")
    renderer.set_position([500, 1500, 1200])
    renderer.set_viewup([0, 0, 1])

    items = item
    fulls = utils.listify(full, len(item))
    for item, full in zip(items, fulls):
        renderer.add_mesh(item.get_mesh_in_world(full=full))

    renderer.show()
