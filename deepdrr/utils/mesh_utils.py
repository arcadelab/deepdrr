from typing import Optional

import logging
import numpy as np
import vtk
from vtk.util import numpy_support as nps
import pyvista as pv

log = logging.getLogger(__name__)


def isosurface(
    data: np.ndarray,
    value: float = 0.5,
    label: Optional[int] = None,
    node_centered: bool = True,
    smooth: bool = True,
    smooth_iter: int = 30,
    relaxation_factor: float = 0.25,
) -> pv.PolyData:
    """Create an isosurface model using marching cubes.

    Args:
        data (np.ndarray): A 3-D array with scalar or integer data.
        value (float, optional): The value of the surface in `data`. Defaults to 0.5.
        label (Optional[int], optional): Get the isosurface of the `data == label` segmentation. Defaults to None.
        node_centered (bool, optional): Whether the values in the data are sampled in the node-centered style. Defaults to true.
        smooth (bool, optional): whether to apply smoothing. Defaults to True.
        smooth_iter (int, optional): number of smoothing iterations to run.
        relaxation_factor (float): passed to surface.smooth.

    Returns:
        pv.PolyData: a Pyvista mesh.
    """
    vol = vtk.vtkStructuredPoints()
    vol.SetDimensions(*data.shape[:3])
    if node_centered:
        vol.SetOrigin(0, 0, 0)
    else:
        vol.SetOrigin(0.5, 0.5, 0.5)
    vol.SetSpacing(1, 1, 1)

    if label is not None:
        data = (data == label).astype(np.uint8)
    else:
        data = (data > value).astype(np.uint8)

    scalars = nps.numpy_to_vtk(data.ravel(order="F"), deep=True)
    vol.GetPointData().SetScalars(scalars)

    log.debug("marching cubes...")
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(vol)
    dmc.GenerateValues(1, 1, 1)
    dmc.ComputeGradientsOff()
    dmc.ComputeNormalsOff()
    dmc.Update()

    surface = pv.wrap(dmc.GetOutput())
    if not surface.is_all_triangles():
        surface.triangulate(inplace=True)

    log.debug("postprocess...")
    surface.decimate_pro(
        0.01,
        feature_angle=60,
        splitting=False,
        preserve_topology=True,
        inplace=True,
    )

    if smooth:
        surface.smooth(
            n_iter=smooth_iter,
            relaxation_factor=relaxation_factor,
            feature_angle=70,
            boundary_smoothing=False,
            inplace=True,
        )

    surface.compute_normals(inplace=True)

    return surface
