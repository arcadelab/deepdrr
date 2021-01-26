import logging
import numpy as np
from pycuda import gpuarray, cumath
from pycuda.tools import DeviceMemoryPool

from .material_coefficients import material_coefficients


logger = logging.getLogger(__name__)


def get_absorbtion_coefs(x, material: str):
    """Returns the absorbtion coefficient for the specified material at the specified energy level (in keV)
    
    Args:
        x: energy level of photon/ray (keV)
        material (str): the material

    Returns:
        the absorbtion coefficient (in [cm^2 / g]), interpolated from the data in material_coefficients.py
    """
    # returns absorbtion coefficient at x in keV
    xMev = x.copy() / 1000
    return log_interp(xMev, material_coefficients[material][:, 0], material_coefficients[material][:, 1])


def log_interp(xInterp, x, y):
    # xInterp is the single energy value to interpolate an absorbtion coefficient for, 
    # interpolating from the data from "x" (energy value array from slicing material_coefficients)
    # and from "y" (absorbtion coefficient array from slicing material_coefficients)
    xInterp = np.log10(xInterp.copy())
    x = np.log10(x.copy())
    y = np.log10(y.copy())
    yInterp = np.power(10, np.interp(xInterp, x, y)) # np.interp is 1-D linear interpolation
    return yInterp
