import logging
import numpy as np
from pycuda import gpuarray, cumath
from pycuda.tools import DeviceMemoryPool

from .material_coefficients import material_coefficients


logger = logging.getLogger(__name__)


def get_absorbtion_coefs(x, material):
    # returns absorbtion coefficient at x in keV
    ###xMev = x.copy() / 1000
    ###return log_interp(xMev, material_coefficients[material][:, 0], material_coefficients[material][:, 1])
    xMev = x.copy() / 1000
    ret = log_interp(xMev, material_coefficients[material][:, 0], material_coefficients[material][:, 1])
    ###print(f"energy={xMev:}, mat={material}: coef={ret:1.6f}")
    return ret


def log_interp(xInterp, x, y):
    # xInterp is the single energy value to interpolate an absorbtion coefficient for, 
    # interpolating from the data from "x" (energy value array from slicing material_coefficients)
    # and from "y" (absorbtion coefficient array from slicing material_coefficients)
    xInterp = np.log10(xInterp.copy())
    x = np.log10(x.copy())
    y = np.log10(y.copy())
    yInterp = np.power(10, np.interp(xInterp, x, y)) # np.interp is 1-D linear interpolation
    return yInterp
