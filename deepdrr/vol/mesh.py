from __future__ import annotations
from re import T
from typing import Any, Union, Tuple, List, Optional, Dict, Type

import logging
import numpy as np
from pathlib import Path
import nibabel as nib
from pydicom.filereader import dcmread
import nrrd
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv

from .. import load_dicom
from .. import geo
from .. import utils
from ..utils import data_utils
from ..utils import mesh_utils
from ..device import Device
from ..projector.material_coefficients import material_coefficients
from .renderable import Renderable
import pyrender
import trimesh
# from deepdrr.utils.mesh_utils import polydata_to_trimesh


vtk, nps, vtk_available = utils.try_import_vtk()

log = logging.getLogger(__name__)

class Mesh(Renderable):
    def __init__(
        self,
        anatomical_from_IJK: Optional[geo.FrameTransform] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        anatomical_from_ijk: Optional[geo.FrameTransform] = None,
        mesh: pyrender.Mesh = None,
        **kwargs
    ) -> None:
        Renderable.__init__(self, 
            anatomical_from_IJK=anatomical_from_IJK,
            world_from_anatomical=world_from_anatomical,
            anatomical_from_ijk=anatomical_from_ijk
        )
        if mesh is None:
            raise ValueError("mesh must be specified")
        
        self.mesh = mesh