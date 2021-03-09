from typing import Tuple
import logging

from . import scatter
from .. import geo
from .projector import SingleProjector
from .cuda_scatter_structs import CudaPlaneSurfaceStruct
try:
    import pycuda.driver as cuda
    from pycuda.autoinit import context
except ImportError:
    logging.warning('pycuda unavailable')

import numpy as np

def simulate_scatter_gpu(
    proj: SingleProjector,
    camera_projection: geo.CameraProjection
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TODO: DOCUMENTATION
    """
    camera_center_in_volume = np.array(camera_projection.get_center_in_volume(proj.volume)).astype(np.float32)

    index_from_ijk = camera_projection.get_ray_transform(proj.volume).inv
    index_from_ijk = np.ascontiguousarray(np.array(index_from_ijk)[0:2, 0:3]).astype(np.float32)
    cuda.memcpy_htod(proj.index_from_ijk_gpu, index_from_ijk)

    ijk_from_index = np.array(camera_projection.get_ray_transform(proj.volume))
    detector_plane = scatter.get_detector_plane(
        ijk_from_index,
        camera_projection.index_from_camera2d,
        proj.source_to_detector_distance,
        geo.Point3D.from_any(camera_center_in_volume),
        proj.output_shape
    )
    detector_plane_struct = CudaPlaneSurfaceStruct(detector_plane, int(proj.detector_plane_gpu))

    E_abs_keV = 5 # E_abs == 5000 eV

    histories_per_thread = 16 # This is arbitrary

    scatter_args = [
        np.int32(camera_projection.sensor_width),           # detector_width
        np.int32(camera_projection.sensor_height),          # detector_height
        np.int32(histories_per_thread),                     # histories_for_thread
        proj.labeled_segmentation_gpu,                      # labeled_segmentation
        camera_center_in_volume[0],                        # sx
        camera_center_in_volume[1],                        # sy
        camera_center_in_volume[2],                        # sz
        np.float32(proj.source_to_detector_distance),       # sdd
        np.int32(proj.volume.shape[0]),                     # volume_shape_x
        np.int32(proj.volume.shape[1]),                     # volume_shape_y
        np.int32(proj.volume.shape[2]),                     # volume_shape_z
        np.float32(-0.5),                       # gVolumeEdgeMinPointX
        np.float32(-0.5),                       # gVolumeEdgeMinPointY
        np.float32(-0.5),                       # gVolumeEdgeMinPointZ
        np.float32(proj.volume.shape[0] - 0.5), # gVolumeEdgeMaxPointX
        np.float32(proj.volume.shape[1] - 0.5), # gVolumeEdgeMaxPointY
        np.float32(proj.volume.shape[2] - 0.5), # gVolumeEdgeMaxPointZ
        np.float32(proj.volume.spacing[0]),         # gVoxelElementSizeX
        np.float32(proj.volume.spacing[1]),         # gVoxelElementSizeY
        np.float32(proj.volume.spacing[2]),         # gVoxelElementSizeZ
        proj.index_from_ijk_gpu,                        # index_from_ijk
        proj.mat_mfp_structs_gpu,                       # mat_mfp_arr
        proj.woodcock_struct_gpu,                       # woodcock_mfp
        proj.compton_structs_gpu,                       # compton_arr
        proj.rita_structs_gpu,                          # rita_arr
        proj.detector_plane_gpu,                        # detector_plane
        np.int32(proj.spectrum.shape[0]),               # n_bins
        proj.energies_gpu,                              # spectrum_energies
        proj.cdf_gpu,                                   # spectrum_cdf
        np.float32(E_abs_keV),                          # E_abs
        np.int32(12345),                                # seed_input TODO
        proj.scatter_deposits_gpu,                      # deposited_energy
        proj.num_scattered_hits_gpu,                    # num_scattered_hits
        proj.num_unscattered_hits_gpu,                  # num_unscattered_hits
    ]

    # Calculate required blocks
    histories_per_block = (proj.threads * proj.threads) * histories_per_thread
    blocks_n = np.int(np.ceil(proj.scatter_num / histories_per_block))
    block = (proj.threads * proj.threads, 1, 1) # same number of threads per block as the ray-casting

    # Call the kernel
    if blocks_n <= proj.max_block_index:
        proj.simulate_scatter(*scatter_args, block=block, grid=(blocks_n, 1))
    else:
        for i in range(np.ceil(blocks_n / proj.max_block_index)):
            blocks_left_to_run = blocks_n - (i * proj.max_block_index)
            blocks_for_grid = min(blocks_left_to_run, proj.max_block_index)
            proj.simulate_scatter(*scatter_args, block=block, gird=(blocks_for_grid, 1))
            context.synchronize()

    # Copy results from the kernel
    scatter_img = np.empty(proj.output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(scatter_img, proj.scatter_deposits_gpu)
    scatter_img = np.swapaxes(scatter_img, 0, 1).copy()
    # Here, scatter_img is just the recorded deposited_energy.  Will need to adjust later

    num_scattered_hits = np.empty(proj.output_shape, dtype=np.int32)
    cuda.memcpy_dtoh(num_scattered_hits, proj.num_scattered_hits_gpu)
    num_scattered_hits = np.swapaxes(num_scattered_hits, 0, 1).copy()

    num_unscattered_hits = np.empty(proj.output_shape, dtype=np.int32)
    cuda.memcpy_dtoh(num_unscattered_hits, proj.num_unscattered_hits_gpu)
    num_unscattered_hits = np.swapaxes(num_unscattered_hits, 0, 1).copy()

    # Adjust scatter_img to reflect the "intensity per photon". We need to account for the
    # fact that the pixels are not uniform in term of solid angle.
    #   [scatter_intensity] = [ideal deposited_energy] / [ideal number of recorded photons],
    # where
    #   [ideal number of recorded photons] = [recorded photons] * (solid_angle[pixel] / average(solid_angle)) 
    # Since [ideal deposited_energy] would be transformed the same way, we simply calculate:
    #   [scatter_intensity] = [recorded deposited_energy] / [recorded number of photons]
    assert np.all(np.equal(0 == scatter_img, 0 == num_scattered_hits))
    # Since [deposited_energy] is zero whenever [num_scattered_hits] is zero, we can add 1 to 
    # every pixel that [num_scattered_hits] is zero to avoid a "divide by zero" error

    scatter_img = np.divide(scatter_img, 1 * (0 == num_scattered_hits) + num_scattered_hits * (0 != num_scattered_hits))
    # scatter_img is now the "intensity per photon"

    return scatter_img, num_scattered_hits, num_unscattered_hits