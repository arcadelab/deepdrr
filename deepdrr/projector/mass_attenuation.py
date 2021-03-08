import logging
import numpy as np

try:
    from pycuda import gpuarray, cumath
    from pycuda.tools import DeviceMemoryPool
except ImportError:
    logging.warning('pycuda unavailable')
    
from .material_coefficients import material_coefficients


logger = logging.getLogger(__name__)


def calculate_intensity_from_spectrum(projections, spectrum, blocksize=50):
    pool = DeviceMemoryPool()
    energies = spectrum[:, 0] / 1000
    pdf = spectrum[:, 1] / np.sum(spectrum[:, 1])
    projection_shape = projections[next(iter(projections))].shape
    num_blocks = np.ceil(projection_shape[0] / blocksize).astype(int)
    intensity = np.zeros(projection_shape, dtype=np.float32)
    photon_prob = np.zeros(projections[next(iter(projections))].shape, dtype=np.float32)
    
    logger.info('running mass attenuation...')
    for i in range(0, num_blocks):
        logger.debug(f"running block: {i + 1} / {num_blocks}")
        lower_i = i * blocksize
        upper_i = min([(i + 1) * blocksize, projection_shape[0]])
        intensity_gpu = gpuarray.zeros((upper_i - lower_i, projection_shape[1], projection_shape[2]), dtype=np.float32, allocator=pool.allocate)
        photon_prob_gpu = gpuarray.zeros((upper_i - lower_i, projection_shape[1], projection_shape[2]), dtype=np.float32, allocator=pool.allocate)

        projections_gpu = {}
        for mat in projections:
            projections_gpu[mat] = gpuarray.to_gpu(projections[mat][lower_i:upper_i, :, :], allocator=pool.allocate)

        for i, _ in enumerate(pdf):
            logger.debug(f"evaluating: {i + 1} / {len(pdf)} spectral bins")
            intensity_tmp = calculate_attenuation_gpu(projections_gpu, energies[i], pdf[i], pool)
            intensity_gpu = intensity_gpu.mul_add(1, intensity_tmp, 1)
            photon_prob_gpu = photon_prob_gpu.mul_add(1, intensity_tmp, 1 / energies[i])

        intensity[lower_i:upper_i, :, :] = intensity_gpu.get()
        photon_prob[lower_i:upper_i, :, :] = photon_prob_gpu.get()
    return intensity, photon_prob


def calculate_attenuation_gpu(projections_gpu, energy, p, pool):
    attenuation_gpu = gpuarray.zeros(projections_gpu[next(iter(projections_gpu))].shape, dtype=np.float32, allocator=pool.allocate)
    for mat in projections_gpu:
        # logger.debug(f'attenuating {mat}')
        attenuation_gpu = attenuation_gpu.mul_add(1.0, projections_gpu[mat], -get_absorption_coefs(energy, mat))
    attenuation_gpu = cumath.exp(attenuation_gpu) * energy * p
    return attenuation_gpu


def get_absorption_coefs(energy_keV, material):
    """Returns the absorption coefficient for the specified material at the specified energy level (in keV)
    
    Args:
        energy_keV: energy level of photon/ray (keV)
        material (str): the material

    Returns:
        the absorption coefficient (in [cm^2 / g]), interpolated from the data in material_coefficients.py
    """
    xMev = energy_keV / 1000
    return log_interp(xMev, material_coefficients[material][:, 0], material_coefficients[material][:, 1])


def log_interp(xInterp, x, y):
    # xInterp is the single energy value to interpolate an absorption coefficient for, 
    # interpolating from the data from "x" (energy value array from slicing material_coefficients)
    # and from "y" (absorption coefficient array from slicing material_coefficients)
    xInterp = np.log10(xInterp.copy())
    x = np.log10(x.copy())
    y = np.log10(y.copy())
    yInterp = np.power(10, np.interp(xInterp, x, y)) # np.interp is 1-D linear interpolation
    return yInterp
