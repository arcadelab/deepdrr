from typing import Tuple
import logging

from . import scatter
from .. import geo
from .projector import SingleProjector
from .cuda_scatter_structs import CudaPlaneSurfaceStruct
try:
    import pycuda.driver as cuda
except ImportError:
    logging.warning('pycuda unavailable')

import numpy as np

def simulate_scatter_gpu(
    proj: SingleProjector,
    camera_projection: geo.CameraProjection
) -> Tuple[np.ndarray, int, int]:
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

    histories_for_thread = 0 # TODO

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
        np.int32(12345),                                 # seed_input TODO
        proj.scatter_deposits_gpu,                      # deposited_energy
        proj.num_scattered_hits_gpu,                    # num_scattered_hits
        proj.num_unscattered_hits_gpu,                  # num_unscattered_hits
    ]

    # Calculate required blocks
    # TODO

    # Call the kernel
    # TODO

    scatter_img = np.empty(proj.output_shape, dtype=np.float32)
    num_scattered_hits = np.int32(0)
    num_unscattered_hits = np.int32(0)
    cuda.memcpy_dtoh(scatter_img, proj.scatter_deposits_gpu) # copy results from GPU
    scatter_img = np.swapaxes(scatter_img, 0, 1).copy()
    cuda.memcpy_dtoh(num_scattered_hits, proj.num_scattered_hits_gpu)
    cuda.memcpy_dtoh(num_unscattered_hits, proj.num_unscattered_hits_gpu)

    return scatter_img, num_scattered_hits, num_unscattered_hits