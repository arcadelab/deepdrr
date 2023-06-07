from __future__ import annotations
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
from .primitive import Primitive

vtk, nps, vtk_available = utils.try_import_vtk()


log = logging.getLogger(__name__)


class Mesh(Renderable):
    primitives: List[Primitive]

    def __init__(
        self,
        primitives: Optional[List[Primitive]] = None,
        morph_weights: Optional[np.ndarray] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ) -> None:
        Renderable.__init__(self, None, world_from_anatomical)
        self.primitives = primitives if primitives is not None else []
        for primitive in self.primitives:
            primitive.set_parent_mesh(self)
        self.morph_weights = morph_weights if morph_weights is not None else np.zeros(len(self.morph_targets))
        if len(primitives) > 0:
            assert len(self.morph_weights) == len(self.primitives[0].morph_targets)
