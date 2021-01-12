import numpy as np
from pycuda import gpuarray, cumath
from pycuda.tools import DeviceMemoryPool

from .material_coefficients import material_coefficients


def calculate_intensity_from_spectrum(projections, spectrum, blocksize=50):
    pool = DeviceMemoryPool()
    energies = spectrum[:, 0] / 1000
    pdf = spectrum[:, 1] / np.sum(spectrum[:, 1])

    projection_shape = projections[next(iter(projections))].shape
    # next(iter(projections)) is the first key in the projections dictionary
    # Per the comments in projector.py, projection_shape is (num_projections, self.sensor_size[1], self.sensor_size[0])

    num_blocks = np.ceil(projection_shape[0] / blocksize).astype(int)
    intensity = np.zeros(projection_shape, dtype=np.float32)
    photon_prob = np.zeros(projection_shape, dtype=np.float32)

    for i in range(0, num_blocks):
        print("running block:", i + 1, "of", num_blocks)
        lower_i = i * blocksize
        upper_i = min([(i + 1) * blocksize, projection_shape[0]]) # logic to avoid overflowing bounds when in the GPU
        intensity_gpu = gpuarray.zeros((upper_i - lower_i, projection_shape[1], projection_shape[2]), dtype=np.float32, allocator=pool.allocate)
        photon_prob_gpu = gpuarray.zeros((upper_i - lower_i, projection_shape[1], projection_shape[2]), dtype=np.float32, allocator=pool.allocate)

        projections_gpu = {}
        for mat in projections:
            projections_gpu[mat] = gpuarray.to_gpu(projections[mat][lower_i:upper_i, :, :], allocator=pool.allocate)
            # copies the appropriate slive of projections[mat] to the GPU

        for i, _ in enumerate(pdf):
            print("evaluating:", i + 1, "/", pdf.__len__(), "spectral bins")
            intensity_tmp = calculate_attenuation_gpu(projections_gpu, energies[i], pdf[i], pool)
            # mul_add(self, selffac, other, otherfac) is equivlent to:
            #   return (self * selffac) + (other * otherfac) <-- SUPER easily parallelizable in a CUDA kernel function
            intensity_gpu = intensity_gpu.mul_add(1, intensity_tmp, 1)
            photon_prob_gpu = photon_prob_gpu.mul_add(1, intensity_tmp, 1 / energies[i])

        # The .get() calls copy from the GPU memory to the CPU memory
        intensity[lower_i:upper_i, :, :] = intensity_gpu.get()
        photon_prob[lower_i:upper_i, :, :] = photon_prob_gpu.get()
    return intensity, photon_prob


def calculate_attenuation_gpu(projections_gpu, energy, p, pool):
    attenuation_gpu = gpuarray.zeros(projections_gpu[next(iter(projections_gpu))].shape, dtype=np.float32, allocator=pool.allocate)
    # attenuation_gpu is the same shape as intensity_gpu and photon_prob_gpu from calculate_intensity_from_spectrum(...).
    # This makes sense--it HAS to be the same hape, else the calls to mul_add(...) wouldn't work

    # for each material:
    #   perform a mul_add(1, projections_gpu[mat], -get_absorbtion_coefs(energy, mat))
    #       - Since this is a mul_add(...) operation, it EASILY parallelizes pixel-by-pixel
    #
    # NOTE: for a single pixel (i.e., in the CUDA kernel function), this is effectively an additive 
    #   accumulation, almost (maybe exactly?) like a dot product between two (self.num_materials)-D 
    #   vectors that are indexed by mat.
    #       vector: projections_gpu[mat] -- the values for a single pixel
    #       vector: absorb_coeffs_E[mat] -- the absorbtion values for energy E
    #   attenuation_gpu(pixel) <- projections_gpu(pixel) <dot> {-1 * absorb_coeffs_E}
    # NOTE: absorb_coeffs_E does not depend on the pixel. Accordingly, it will be best if the get_absorbtion_coefs(...)
    #   function (or macro?) is a separate function than the projectKernel function
    for mat in projections_gpu:
        #print(get_absorbtion_coefs(energy,mat), mat)
        attenuation_gpu = attenuation_gpu.mul_add(1.0, projections_gpu[mat], -get_absorbtion_coefs(energy, mat))

    # Element-wise update of attenuation_gpu <-- EASILY parallelizes pixel-by-pixel
    attenuation_gpu = cumath.exp(attenuation_gpu) * energy * p
    return attenuation_gpu


def get_absorbtion_coefs(x, material):
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
    yInterp = np.power(10, np.interp(xInterp, x, y))
    return yInterp
