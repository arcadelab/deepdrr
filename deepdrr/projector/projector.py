from typing import List, Union, Tuple, Optional, Dict, Any

import logging
import numpy as np
from pathlib import Path

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.autoinit import context
    from pycuda.compiler import SourceModule

    pycuda_available = True
except ImportError:
    pycuda_available = False
    SourceModule = Any
    logging.warning("pycuda unavailable")

from . import spectral_data
from . import mass_attenuation
from . import scatter
from . import analytic_generators
from .material_coefficients import material_coefficients
from .mcgpu_mfp_data import MFP_DATA
from .mcgpu_compton_data import COMPTON_DATA
from .mcgpu_rita_samplers import rita_samplers
from .. import geo
from .. import vol
from ..device import MobileCArm
from .. import utils
from .cuda_scatter_structs import CudaPlaneSurfaceStruct, CudaRitaStruct, CudaComptonStruct, CudaMatMfpStruct, CudaWoodcockStruct
import time


logger = logging.getLogger(__name__)


def _get_spectrum(spectrum: Union[np.ndarray, str]):
    """Get the data corresponding to the given spectrum name.

    Args:
        spectrum (Union[np.ndarray, str]): the spectrum array or the spectrum itself.

    Raises:
        TypeError: If the spectrum is not recognized.

    Returns:
        np.ndarray: The X-ray spectrum data.
    """
    if isinstance(spectrum, np.ndarray):
        return spectrum
    elif isinstance(spectrum, str):
        if spectrum not in spectral_data.spectrums:
            raise KeyError(f"unrecognized spectrum: {spectrum}")
        return spectral_data.spectrums[spectrum]
    else:
        raise TypeError(f"unrecognized spectrum type: {type(spectrum)}")


def _get_kernel_projector_module(num_volumes: int, num_materials: int) -> SourceModule:
    """Compile the cuda code for the kernel projector.

    Assumes `project_kernel.cu` and `cubic` interpolation library is in the same directory as THIS
    file.

    Args:
        num_materials (int): The number of materials to assume

    Returns:
        SourceModule: pycuda SourceModule object.

    """
    #path to files for cubic interpolation (folder cubic in DeepDRR)
    d = Path(__file__).resolve().parent
    bicubic_path = str(d / 'cubic')
    source_path = None
    if 1 == num_volumes:
        source_path = str(d / 'project_kernel_single.cu')
    else:
        source_path = str(d / 'project_kernel_multi.cu')

    with open(source_path, "r") as file:
        source = file.read()

    if 1 == num_volumes:
        logger.debug(f'compiling {source_path} with NUM_MATERIALS={num_materials}')
        return SourceModule(source, include_dirs=[bicubic_path], no_extern_c=True, options=['-D', f'NUM_MATERIALS={num_materials}'])
    else:
        logger.debug(f'compiling {source_path} with NUM_VOLUMES={num_volumes}, NUM_MATERIALS={num_materials}')
        return SourceModule(source, include_dirs=[bicubic_path, str(d)], no_extern_c=True, options=['-D', f'NUM_VOLUMES={num_volumes}', '-D', f'NUM_MATERIALS={num_materials}'])


def _get_kernel_scatter_module(num_materials) -> SourceModule:
    """Compile the cuda code for the scatter simulation.

    Assumes 'scatter_kernel.cu' and 'scatter_header.cu' are in the same directory as THIS file.

    Returns:
        SourceModule: pycuda SourceModule object.
    """
    d = Path(__file__).resolve().parent
    source_path = str(d / 'scatter_kernel.cu')

    with open(source_path, 'r') as file:
        source = file.read()

    logger.debug(f"compiling {source_path} with NUM_MATERIALS={num_materials}")
    return SourceModule(source, include_dirs=[str(d)], no_extern_c=True, options=['-D', f'NUM_MATERIALS={num_materials}'])


<<<<<<< HEAD

class SingleProjector(object):
    initialized: bool = False

    def __init__(
        self,
        volume: vol.Volume,
        camera_intrinsics: geo.CameraIntrinsicTransform,
        source_to_detector_distance: float, 
        step: float,
        photon_count: int,
        mode: Literal['linear'] = 'linear',
        spectrum: Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43']] = '90KV_AL40',
        threads: int = 8,
        max_block_index: int = 1024,
        attenuation: bool = True,
        collected_energy: bool = False,
        add_scatter: bool = False,
        scatter_num: int = 0
    ) -> None:
        """Create the projector, which has info for simulating the DRR, for a single projection angle.

        Args:
            volume (Volume): a volume object with materials segmented.
            camera_intrinsics (CameraIntrinsicTransform): intrinsics of the projector's camera. (used for sensor size).
            source_to_detector_distance (float): distance from source to detector in millimeters.
            step (float, optional): size of the step along projection ray in voxels.
            photon_count (int, optional): the average number of photons that hit each pixel. (The expected number of photons that hit each pixel is not uniform over each pixel because the detector is a flat panel.)
            mode (Literal['linear']): [description].
            spectrum (Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43'], optional): spectrum array or name of spectrum to use for projection. Defaults to '90KV_AL40'.
            threads (int, optional): number of threads per "side" in a 2-D GPU block. Defaults to 8.
            max_block_index (int, optional): maximum GPU block. Defaults to 1024.
            attenuation (bool, optional): whether the mass-attenuation calculation is performed in the CUDA kernel. Defaults to True.
            collected_energy (bool, optional): Whether to return data of "intensity" (energy deposited per photon, [keV]) or "collected energy" (energy deposited on pixel, [keV / mm^2]). Defaults to False ("intensity").
            add_scatter (bool, optional): whether to add scatter noise from artifacts. Defaults to False. DEPRECATED: use scatter_num instead.  If add_scatter is True, and scatter_num is unspecified, uses 10^6
            scatter_num (int, optional): the number of photons to sue in the scatter simulation.  If zero, scatter is not simulated.
        """
        logger.warning('Previously, projector.SingleProjector used add_scatter as the switch to control scatter. Now, use the scatter_num switch. add_scatter=True is currently equivalent to scatter_num=(10**6)')

        # set variables
        self.volume = volume # spacing units defaults to mm
        self.camera_intrinsics = camera_intrinsics
        self.source_to_detector_distance = source_to_detector_distance
        self.step = step
        self.photon_count = photon_count
        self.mode = mode
        self.spectrum = _get_spectrum(spectrum)
        self.threads = threads
        self.max_block_index = max_block_index
        self.attenuation = attenuation
        self.collected_energy = collected_energy

        #print(f"SPECTRUM ARGUMENT: {spectrum}")
        #print(f"SPECTRUM ARRAY: {self.spectrum}")

        if (scatter_num == 0) and add_scatter:
            self.scatter_num = 1000000 # 10^6
        else:
            self.scatter_num = max(scatter_num, 0) # in case scatter_num < 0

        self.num_materials = len(self.volume.materials)

        # compile the module
        # TODO: fix attenuation vs no-attenuation ugliness.
        self.mod = _get_kernel_projector_module(self.num_materials, attenuation=self.attenuation)
        self.project_kernel = self.mod.get_function("projectKernel")

        self.scatter_mod = _get_kernel_scatter_module(self.num_materials)
        self.simulate_scatter = self.scatter_mod.get_function("simulate_scatter")

        # assertions
        for mat in self.volume.materials:
            assert mat in material_coefficients, f'unrecognized material: {mat}'

    def project(
        self,
        camera_projection: geo.CameraProjection,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Perform the projection over just one image.

        Args:
            camera_projection (geo.CameraProjection): a camera projection transform.

        Raises:
            RuntimeError: if the projector has not been initialized.

        Returns:
            np.ndarray: the intensity image
            np.ndarray: the photon probability field
        """
        if not self.initialized:
            raise RuntimeError("Projector has not been initialized.")

        assert isinstance(self.spectrum, np.ndarray)

        # initialize projection-specific arguments
        camera_center_in_volume = np.array(camera_projection.get_center_in_volume(self.volume)).astype(np.float32)
        logger.debug(f'camera_center_ijk (source point): {camera_center_in_volume}')

        ijk_from_index = camera_projection.get_ray_transform(self.volume)
        logger.debug('center ray: {}'.format(ijk_from_index @ geo.point(self.output_shape[0] / 2, self.output_shape[1] / 2)))

        ijk_from_index = np.array(ijk_from_index).astype(np.float32)

        # spacing
        spacing = self.volume.spacing # units: [mm]

        # copy the projection matrix to CUDA (output array initialized to zero by the kernel)
        cuda.memcpy_htod(self.rt_kinv_gpu, ijk_from_index)

        # Make the arguments to the CUDA "projectKernel".
        if self.attenuation:
            args = [
                np.int32(camera_projection.sensor_width),          # out_width
                np.int32(camera_projection.sensor_height),          # out_height
                np.float32(self.step),                  # step
                np.float32(-0.5),                       # gVolumeEdgeMinPointX
                np.float32(-0.5),                       # gVolumeEdgeMinPointY
                np.float32(-0.5),                       # gVolumeEdgeMinPointZ
                np.float32(self.volume.shape[0] - 0.5), # gVolumeEdgeMaxPointX
                np.float32(self.volume.shape[1] - 0.5), # gVolumeEdgeMaxPointY
                np.float32(self.volume.shape[2] - 0.5), # gVolumeEdgeMaxPointZ
                np.float32(spacing[0]),         # gVoxelElementSizeX
                np.float32(spacing[1]),         # gVoxelElementSizeY
                np.float32(spacing[2]),         # gVoxelElementSizeZ
                camera_center_in_volume[0],                        # sx
                camera_center_in_volume[1],                        # sy
                camera_center_in_volume[2],                        # sz
                self.rt_kinv_gpu,                       # RT_Kinv
                np.int32(self.spectrum.shape[0]),       # n_bins
                self.energies_gpu,                      # energies
                self.pdf_gpu,                           # pdf
                self.absorption_coef_table_gpu,          # absorb_coef_table
                self.intensity_gpu,              # intensity
                self.photon_prob_gpu,                   # photon_prob
                self.solid_angle_gpu,                   # solid_angle
            ]
        else:
            args = [
                np.int32(camera_projection.sensor_width),          # out_width
                np.int32(camera_projection.sensor_height),          # out_height
                np.float32(self.step),                  # step
                np.float32(-0.5),                       # gVolumeEdgeMinPointX
                np.float32(-0.5),                       # gVolumeEdgeMinPointY
                np.float32(-0.5),                       # gVolumeEdgeMinPointZ
                np.float32(self.volume.shape[0] - 0.5), # gVolumeEdgeMaxPointX
                np.float32(self.volume.shape[1] - 0.5), # gVolumeEdgeMaxPointY
                np.float32(self.volume.shape[2] - 0.5), # gVolumeEdgeMaxPointZ
                np.float32(spacing[0]),         # gVoxelElementSizeX
                np.float32(spacing[1]),         # gVoxelElementSizeY
                np.float32(spacing[2]),         # gVoxelElementSizeZ
                camera_center_in_volume[0],                        # sx
                camera_center_in_volume[1],                        # sy
                camera_center_in_volume[2],                        # sz
                self.rt_kinv_gpu,                       # RT_Kinv
                self.output_gpu,                        # output
            ]

        # Calculate required blocks
        blocks_w = np.int(np.ceil(self.output_shape[0] / self.threads))
        blocks_h = np.int(np.ceil(self.output_shape[1] / self.threads))
        block = (self.threads, self.threads, 1)
        # lfkj("running:", blocks_w, "x", blocks_h, "blocks with ", self.threads, "x", self.threads, "threads")

        if blocks_w <= self.max_block_index and blocks_h <= self.max_block_index:
            offset_w = np.int32(0)
            offset_h = np.int32(0)
            self.project_kernel(*args, offset_w, offset_h, block=block, grid=(blocks_w, blocks_h))
        else:
            # lfkj("running kernel patchwise")
            for w in range((blocks_w - 1) // (self.max_block_index + 1)):
                for h in range((blocks_h - 1) // (self.max_block_index + 1)):
                    offset_w = np.int32(w * self.max_block_index)
                    offset_h = np.int32(h * self.max_block_index)
                    self.project_kernel(*args, offset_w, offset_h, block=block, grid=(self.max_block_index, self.max_block_index))
                    context.synchronize()

        if self.attenuation:
            intensity = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(intensity, self.intensity_gpu)
            # transpose the axes, which previously have width on the slow dimension
            intensity = np.swapaxes(intensity, 0, 1).copy()

            photon_prob = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(photon_prob, self.photon_prob_gpu)
            photon_prob = np.swapaxes(photon_prob, 0, 1).copy()

            if (self.scatter_num > 0):
                logger.info(f"Starting scatter simulation, scatter_num={self.scatter_num}. Time: {time.asctime()}")
                index_from_ijk = camera_projection.get_ray_transform(self.volume).inv
                index_from_ijk = np.ascontiguousarray(np.array(index_from_ijk)[0:2, 0:3]).astype(np.float32)
                cuda.memcpy_htod(self.index_from_ijk_gpu, index_from_ijk)

                detector_plane = scatter.get_detector_plane(
                    ijk_from_index,
                    camera_projection.index_from_camera2d,
                    self.source_to_detector_distance,
                    geo.Point3D.from_any(camera_center_in_volume),
                    self.output_shape
                )
                detector_plane_struct = CudaPlaneSurfaceStruct(detector_plane, int(self.detector_plane_gpu))

                E_abs_keV = 5 # E_abs == 5000 eV
                histories_per_thread = int(np.ceil(self.scatter_num / (4 * self.threads * self.threads)))
                print(f"histories_per_thread: {histories_per_thread}")

                scatter_args = [
                    np.int32(camera_projection.sensor_width),           # detector_width
                    np.int32(camera_projection.sensor_height),          # detector_height
                    np.int32(histories_per_thread),                     # histories_for_thread
                    self.labeled_segmentation_gpu,                      # labeled_segmentation
                    camera_center_in_volume[0],                        # sx
                    camera_center_in_volume[1],                        # sy
                    camera_center_in_volume[2],                        # sz
                    np.float32(self.source_to_detector_distance),       # sdd
                    np.int32(self.volume.shape[0]),                     # volume_shape_x
                    np.int32(self.volume.shape[1]),                     # volume_shape_y
                    np.int32(self.volume.shape[2]),                     # volume_shape_z
                    np.float32(-0.5),                       # gVolumeEdgeMinPointX
                    np.float32(-0.5),                       # gVolumeEdgeMinPointY
                    np.float32(-0.5),                       # gVolumeEdgeMinPointZ
                    np.float32(self.volume.shape[0] - 0.5), # gVolumeEdgeMaxPointX
                    np.float32(self.volume.shape[1] - 0.5), # gVolumeEdgeMaxPointY
                    np.float32(self.volume.shape[2] - 0.5), # gVolumeEdgeMaxPointZ
                    np.float32(self.volume.spacing[0]),         # gVoxelElementSizeX
                    np.float32(self.volume.spacing[1]),         # gVoxelElementSizeY
                    np.float32(self.volume.spacing[2]),         # gVoxelElementSizeZ
                    self.index_from_ijk_gpu,                        # index_from_ijk
                    self.mat_mfp_structs_gpu,                       # mat_mfp_arr
                    self.woodcock_struct_gpu,                       # woodcock_mfp
                    self.compton_structs_gpu,                       # compton_arr
                    self.rita_structs_gpu,                          # rita_arr
                    self.detector_plane_gpu,                        # detector_plane
                    np.int32(self.spectrum.shape[0]),               # n_bins
                    self.energies_gpu,                              # spectrum_energies
                    self.cdf_gpu,                                   # spectrum_cdf
                    np.float32(E_abs_keV),                          # E_abs
                    np.int32(12345),                                # seed_input TODO
                    self.scatter_deposits_gpu,                      # deposited_energy
                    self.num_scattered_hits_gpu,                    # num_scattered_hits
                    self.num_unscattered_hits_gpu,                  # num_unscattered_hits
                ]

                seed_input_index = 30 # so we can change the seed_input for each simulation block--TODO
                assert 12345 == scatter_args[seed_input_index]

                # Calculate required blocks
                histories_per_block = (4 * self.threads * self.threads) * histories_per_thread
                blocks_n = np.int(np.ceil(self.scatter_num / histories_per_block))
                block = (4 * self.threads * self.threads, 1, 1) # same number of threads per block as the ray-casting
                print(f"scatter_num: {self.scatter_num}. histories_per_block: {histories_per_block}. blocks_n: {blocks_n}")

                # Call the kernel
                if blocks_n <= self.max_block_index:
                    self.simulate_scatter(*scatter_args, block=block, grid=(blocks_n, 1))
                else:
                    for i in range(int(np.ceil(blocks_n / self.max_block_index))):
                        blocks_left_to_run = blocks_n - (i * self.max_block_index)
                        blocks_for_grid = min(blocks_left_to_run, self.max_block_index)
                        self.simulate_scatter(*scatter_args, block=block, grid=(blocks_for_grid, 1))
                        context.synchronize()

                # Copy results from the GPU
                scatter_intensity = np.empty(self.output_shape, dtype=np.float32)
                cuda.memcpy_dtoh(scatter_intensity, self.scatter_deposits_gpu)
                scatter_intensity = np.swapaxes(scatter_intensity, 0, 1).copy()
                # Here, scatter_intensity is just the recorded deposited_energy. Will need to adjust later

                n_sc = np.empty(self.output_shape, dtype=np.int32)
                cuda.memcpy_dtoh(n_sc, self.num_scattered_hits_gpu)
                n_sc = np.swapaxes(n_sc, 0, 1).copy()

                n_pri = np.empty(self.output_shape, dtype=np.int32)
                cuda.memcpy_dtoh(n_pri, self.num_unscattered_hits_gpu)
                n_pri = np.swapaxes(n_pri, 0, 1).copy()

                # Adjust scatter_img to reflect the "intensity per photon". We need to account for the
                # fact that the pixels are not uniform in term of solid angle.
                #   [scatter_intensity] = [ideal deposited_energy] / [ideal number of recorded photons],
                # where
                #   [ideal number of recorded photons] = [recorded photons] * (solid_angle[pixel] / average(solid_angle)) 
                # Since [ideal deposited_energy] would be transformed the same way, we simply calculate:
                #   [scatter_intensity] = [recorded deposited_energy] / [recorded number of photons]
                assert np.all(np.equal(0 == scatter_intensity, 0 == n_sc))
                # Since [deposited_energy] is zero whenever [num_scattered_hits] is zero, we can add 1 to 
                # every pixel that [num_scattered_hits] is zero to avoid a "divide by zero" error

                scatter_intensity = np.divide(scatter_intensity, 1 * (0 == n_sc) + n_sc * (0 != n_sc))
                # scatter_intensity now actually reflects "intensity per photon"
                logger.info(f"Finished scatter simulation, scatter_num={self.scatter_num}. Time: {time.asctime()}")

                hits_sc = np.sum(n_sc) # total number of recorded scatter hits
                hits_pri = np.sum(n_pri) # total number of recorded primary hits

                print(f"hits_sc: {hits_sc}, hits_pri: {hits_pri}")

                f_sc = hits_sc / (hits_pri + hits_sc)
                f_pri = hits_pri / (hits_pri + hits_sc)

                ### Reasoning: prob_tot = (f_pri * prob_pri) + (f_sc * prob_sc)
                ### such that: prob_tot / prob_pri = f_pri + f_sc * (prob_sc / prob_pri)
                #photon_prob *= (f_pri + f_sc * (n_sc / n_pri))

                # total intensity = (f_pri * intensity_pri) * (f_sc * intensity_sc)
                intensity = ((f_pri * intensity) + (f_sc * scatter_intensity)) ###/ f_pri

            if self.collected_energy:
                assert np.int32(0) != self.solid_angle_gpu
                solid_angle = np.empty(self.output_shape, dtype=np.float32)
                cuda.memcpy_dtoh(solid_angle, self.solid_angle_gpu)
                solid_angle = np.swapaxes(solid_angle, 0, 1).copy()

                pixel_size_x = self.source_to_detector_distance / camera_projection.index_from_camera2d.fx
                pixel_size_y = self.source_to_detector_distance / camera_projection.index_from_camera2d.fy

                # get energy deposited by multiplying [intensity] with [number of photons to hit each pixel]
                deposited_energy = np.multiply(intensity, solid_angle) * self.photon_count / np.average(solid_angle)
                # convert to keV / mm^2
                deposited_energy /= (pixel_size_x * pixel_size_y)
                return deposited_energy, photon_prob
            
            return intensity, photon_prob
        else:
            # copy the output to CPU
            output = np.empty(self.output_shape, np.float32)
            cuda.memcpy_dtoh(output, self.output_gpu)

            # transpose the axes, which previously have width on the slow dimension
            output = np.swapaxes(output, 0, 1).copy()

            # normalize to centimeters
            output /= 10

            return output

    def initialize(self):
        """Allocate GPU memory and transfer the volume, segmentations to GPU."""
        if self.initialized:
            raise RuntimeError("Close projector before initializing again.")

        # allocate and transfer volume texture to GPU
        # TODO: this axis-swap is messy and actually may be messing things up. Maybe use a FrameTransform in the Volume class instead?
        volume = np.array(self.volume)
        volume = np.moveaxis(volume, [0, 1, 2], [2, 1, 0]).copy() # TODO: is this axis swap necessary?
        self.volume_gpu = cuda.np_to_array(volume, order='C')
        self.volume_texref = self.mod.get_texref("volume")
        cuda.bind_array_to_texref(self.volume_gpu, self.volume_texref)
        
        # set the (interpolation?) mode
        if self.mode == 'linear':
            self.volume_texref.set_filter_mode(cuda.filter_mode.LINEAR)
        else:
            raise RuntimeError

        # allocate and transfer segmentation texture to GPU
        # TODO: remove axis swap?
        # self.segmentations_gpu = [cuda.np_to_array(seg, order='C') for mat, seg in self.volume.materials.items()]
        self.segmentations_gpu = [cuda.np_to_array(np.moveaxis(seg, [0, 1, 2], [2, 1, 0]).copy(), order='C') for mat, seg in self.volume.materials.items()]
        self.segmentations_texref = [self.mod.get_texref(f"seg_{m}") for m, _ in enumerate(self.volume.materials)]
        for seg, texref in zip(self.segmentations_gpu, self.segmentations_texref):
            cuda.bind_array_to_texref(seg, texref)
            if self.mode == 'linear':
                texref.set_filter_mode(cuda.filter_mode.LINEAR)
            else:
                raise RuntimeError

        # allocate ijk_from_index matrix array on GPU (3x3 array x 4 bytes per float32)
        self.rt_kinv_gpu = cuda.mem_alloc(3 * 3 * 4)

        if self.attenuation:
            # allocate deposited_energy array on GPU (4 bytes to a float32)
            self.intensity_gpu = cuda.mem_alloc(self.output_size * 4)
            logger.debug(f"bytes alloc'd for self.intensity_gpu: {self.output_size * 4}")

            # allocate photon_prob array on GPU (4 bytes to a float32)
            self.photon_prob_gpu = cuda.mem_alloc(self.output_size * 4)
            logger.debug(f"bytes alloc'd for self.photon_prob_gpu: {self.output_size * 4}")

            # allocate solid_angle array on GPU as needed (4 bytes to a float32)
            if self.collected_energy:
                self.solid_angle_gpu = cuda.mem_alloc(self.output_size * 4)
                logger.debug(f"bytes alloc'd for self.solid_angle_gpu: {self.output_size * 4}")
            else:
                self.solid_angle_gpu = np.int32(0) # NULL. Don't need to do solid angle calculation

            # allocate and transfer spectrum energies (4 bytes to a float32)
            assert isinstance(self.spectrum, np.ndarray)
            noncont_energies = self.spectrum[:,0].copy() / 1000 # [keV]
            contiguous_energies = np.ascontiguousarray(noncont_energies, dtype=np.float32) # [keV]
            n_bins = contiguous_energies.shape[0]
            self.energies_gpu = cuda.mem_alloc(n_bins * 4)
            cuda.memcpy_htod(self.energies_gpu, contiguous_energies)
            logger.debug(f"bytes alloc'd for self.energies_gpu: {n_bins * 4}")

            # allocate and transfer spectrum pdf (4 bytes to a float32)
            noncont_pdf = self.spectrum[:, 1]  / np.sum(self.spectrum[:, 1])
            contiguous_pdf = np.ascontiguousarray(noncont_pdf.copy(), dtype=np.float32)
            assert contiguous_pdf.shape == contiguous_energies.shape
            assert contiguous_pdf.shape[0] == n_bins
            self.pdf_gpu = cuda.mem_alloc(n_bins * 4)
            cuda.memcpy_htod(self.pdf_gpu, contiguous_pdf)
            logger.debug(f"bytes alloc'd for self.pdf_gpu {n_bins * 4}")

            # precompute, allocate, and transfer the get_absorption_coef(energy, material) table (4 bytes to a float32)
            absorption_coef_table = np.empty(n_bins * self.num_materials).astype(np.float32)
            for bin in range(n_bins): #, energy in enumerate(energies):
                for m, mat_name in enumerate(self.volume.materials):
                    absorption_coef_table[bin * self.num_materials + m] = mass_attenuation.get_absorption_coefs(contiguous_energies[bin], mat_name)
            self.absorption_coef_table_gpu = cuda.mem_alloc(n_bins * self.num_materials * 4)
            cuda.memcpy_htod(self.absorption_coef_table_gpu, absorption_coef_table)
            logger.debug(f"size alloc'd for self.absorption_coef_table_gpu: {n_bins * self.num_materials * 4}")
        else:
            # allocate output image array on GPU (4 bytes to a float32)
            self.output_gpu = cuda.mem_alloc(self.output_size * 4)
            logger.debug(f"bytes alloc'd for self.output_gpu {self.output_size * 4}")
        
        if self.scatter_num > 0:
            my_materials = list(self.volume.materials.keys())
            print(f"my_materials: {my_materials}")

            # Material MFP structs
            self.mat_mfp_struct_dict = dict()
            self.mat_mfp_structs_gpu = cuda.mem_alloc(self.num_materials * CudaMatMfpStruct.MEMSIZE)
            for i, mat in enumerate(my_materials):
                struct_gpu_ptr = int(self.mat_mfp_structs_gpu) + (i * CudaMatMfpStruct.MEMSIZE)
                self.mat_mfp_struct_dict[mat] = CudaMatMfpStruct(MFP_DATA[mat], struct_gpu_ptr)

            # Woodcock MFP struct
            wc_np_arr = scatter.make_woodcock_mfp(my_materials)
            self.woodcock_struct_gpu = cuda.mem_alloc(CudaWoodcockStruct.MEMSIZE)
            self.woodcock_struct = CudaWoodcockStruct(wc_np_arr, int(self.woodcock_struct_gpu))

            # Material Compton structs
            self.compton_struct_dict = dict()
            self.compton_structs_gpu = cuda.mem_alloc(self.num_materials * CudaComptonStruct.MEMSIZE)
            for i, mat in enumerate(my_materials):
                struct_gpu_ptr = int(self.compton_structs_gpu) + (i * CudaComptonStruct.MEMSIZE)
                self.compton_struct_dict[mat] = CudaComptonStruct(COMPTON_DATA[mat], struct_gpu_ptr)
            
            # Material RITA structs
            self.rita_struct_dict = dict()
            self.rita_structs_gpu = cuda.mem_alloc(self.num_materials * CudaRitaStruct.MEMSIZE)
            for i, mat in enumerate(my_materials):
                struct_gpu_ptr = int(self.rita_structs_gpu) + (i * CudaRitaStruct.MEMSIZE)
                self.rita_struct_dict[mat] = CudaRitaStruct(rita_samplers[mat], struct_gpu_ptr)
                #print(f"for material [{mat}], RITA structure at location {struct_gpu_ptr}")
                #for g in range(self.rita_struct_dict[mat].n_gridpts):
                #    print(f"[{self.rita_struct_dict[mat].x[g]}, {self.rita_struct_dict[mat].y[g]}, {self.rita_struct_dict[mat].a[g]}, {self.rita_struct_dict[mat].b[g]}]")
            
            # Labeled segmentation
            num_voxels = self.volume.shape[0] * self.volume.shape[1] * self.volume.shape[2]
            labeled_seg = np.zeros(self.volume.shape).astype(np.int8)
            for i, mat in enumerate(my_materials):
                labeled_seg = np.add(labeled_seg, i * self.volume.materials[mat]).astype(np.int8)
            labeled_seg = np.moveaxis(labeled_seg, [0, 1, 2], [2, 1, 0]).copy() # TODO: is this axis swap necessary?
            self.labeled_segmentation_gpu = cuda.mem_alloc(num_voxels)
            cuda.memcpy_htod(self.labeled_segmentation_gpu, labeled_seg)

            # Detector plane
            self.detector_plane_gpu = cuda.mem_alloc(CudaPlaneSurfaceStruct.MEMSIZE)

            # index_from_ijk
            self.index_from_ijk_gpu = cuda.mem_alloc(2 * 3 * 4) # (2, 3) array of floats

            # spectrum cdf
            n_bins = self.spectrum.shape[0]
            #spectrum_cdf = np.array([np.sum(self.spectrum[0:i+1, 1]) for i in range(n_bins)])
            #spectrum_cdf = (spectrum_cdf / np.sum(self.spectrum[:, 1])).astype(np.float32)
            spectrum_cdf = np.array([np.sum(contiguous_pdf[0:i+1]) for i in range(n_bins)])
            #print(f"spectrum CDF:\n{spectrum_cdf}")
            self.cdf_gpu = cuda.mem_alloc(n_bins * 4)
            cuda.memcpy_htod(self.cdf_gpu, spectrum_cdf)

            # output
            self.scatter_deposits_gpu = cuda.mem_alloc(self.output_size * 4)
            self.num_scattered_hits_gpu = cuda.mem_alloc(self.output_size * 4)
            self.num_unscattered_hits_gpu = cuda.mem_alloc(self.output_size * 4)

        # Mark self as initialized.
        self.initialized = True

    def free(self):
        if self.initialized:
            self.volume_gpu.free()
            for seg in self.segmentations_gpu:
                seg.free()
            self.rt_kinv_gpu.free()

            if self.attenuation:
                self.intensity_gpu.free()
                self.photon_prob_gpu.free()
                if self.collected_energy:
                    self.solid_angle_gpu.free()
                else:
                    assert np.int32(0) == self.solid_angle_gpu
                self.energies_gpu.free()
                self.pdf_gpu.free()
                self.absorption_coef_table_gpu.free()
            else:
                self.output_gpu.free()
            
            if self.scatter_num > 0:
                self.mat_mfp_structs_gpu.free()
                self.woodcock_struct_gpu.free()
                self.compton_structs_gpu.free()
                self.rita_structs_gpu.free()
                self.labeled_segmentation_gpu.free()
                self.detector_plane_gpu.free()
                self.index_from_ijk_gpu.free()
                self.cdf_gpu.free()
                self.scatter_deposits_gpu.free()
                self.num_scattered_hits_gpu.free()
                self.num_unscattered_hits_gpu.free()

        self.initialized = False
=======
>>>>>>> dev


class Projector(object):
    def __init__(
        self,
        volume: Union[vol.Volume, List[vol.Volume]],
        priorities: Optional[List[int]] = None,
        camera_intrinsics: Optional[geo.CameraIntrinsicTransform] = None,
<<<<<<< HEAD
        source_to_detector_distance: float = 1200, 
        carm: Optional[CArm] = None,
=======
        carm: Optional[MobileCArm] = None,
>>>>>>> dev
        step: float = 0.1,
        mode: str = "linear",
        spectrum: Union[np.ndarray, str] = "90KV_AL40",
        add_scatter: bool = False,
        scatter_num: int = 0,
        add_noise: bool = False,
        photon_count: int = 10000,
        threads: int = 8,
        max_block_index: int = 1024,
<<<<<<< HEAD
        collected_energy: bool = False,
=======
        collected_energy: bool = False,  # convert to keV / cm^2 or keV / mm^2
>>>>>>> dev
        neglog: bool = True,
        intensity_upper_bound: Optional[float] = None,
    ) -> None:
        """Create the projector, which has info for simulating the DRR.

        Usage:
        ```
        with Projector(volume, materials, ...) as projector:
            for projection in projections:
                yield projector(projection)
        ```

        Args:
<<<<<<< HEAD
            volume (Union[Volume, List[Volume]]): a volume object with materials segmented. If multiple volumes are provided, they should have mutually exclusive materials (not checked).
            camera_intrinsics (CameraIntrinsicTransform): intrinsics of the projector's camera. (used for sensor size). None is NotImplemented. Defaults to None.
            source_to_detector_distance (float): distance from source to detector in millimeters.
            carm (Optional[CArm], optional): Optional C-arm device, for convenience which can be used to get projections from C-Arm pose. If not provided, camera pose must be defined by user. Defaults to None.
            step (float, optional): size of the step along projection ray in voxels. Defaults to 0.1.
            mode (Literal['linear']): [description].
            spectrum (Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43'], optional): spectrum array or name of spectrum to use for projection. Defaults to '90KV_AL40'.
            add_scatter (bool, optional): whether to add scatter noise from artifacts. Defaults to False. DEPRECATED: use scatter_num instead.  If add_scatter is True, and scatter_num is unspecified, uses 10^6
            scatter_num (int, optional): the number of photons to sue in the scatter simulation.  If zero, scatter is not simulated.
            add_noise (bool, optional): whether to add Poisson noise. Defaults to False.
            photon_count (int, optional): the average number of photons that hit each pixel. (The expected number of photons that hit each pixel is not uniform over each pixel because the detector is a flat panel.) Defaults to 10^4.
            threads (int, optional): number of threads per "side" in a 2-D GPU block. Defaults to 8.
            max_block_index (int, optional): maximum GPU block. Defaults to 1024. TODO: determine from compute capability.
            collected_energy (bool, optional): Whether to return data of "intensity" (energy deposited per photon, [keV]) or "collected energy" (energy deposited on pixel, [keV / mm^2]). Defaults to False ("intensity").
=======
            volume (Union[Volume, List[Volume]]): a volume object with materials segmented, or a list of volume objects.
            priorities (Optional[List[int]], optional): Denotes the 'priority level' of the volumes in projection. At each position, if volumes with lower priority-integers are sampled from as long as they have a non-null 
                                segmentation at that location. valid priority levels are in the range [0, NUM_VOLUMES), with priority 0 being prioritized over other priority levels. Note that multiple volumes can share a 
                                priority level.  If a list of priorities is provided, the priorities are associated in-order to the provided volumes.  If no list is provided (the default), the volumes are assumed to have
                                distinct priority levels, and each volume is prioritized over the preceding volumes. (This behavior is equivalent to passing in the list: [NUM_VOLUMES - 1, ..., 1, 0].)
            camera_intrinsics (CameraIntrinsicTransform): intrinsics of the projector's camera. (used for sensor size). If None, the CArm object must be provided and have a camera_intrinsics attribute. Defaults to None.
            carm (Optional[MobileCArm], optional): Optional C-arm device, for convenience which can be used to get projections from C-Arm pose. If not provided, camera pose must be defined by user. Defaults to None.
            step (float, optional): size of the step along projection ray in voxels. Defaults to 0.1.
            mode (str): Interpolation mode for the kernel. Defaults to "linear".
            spectrum (Union[np.ndarray, str], optional): Spectrum array or name of spectrum to use for projection. Defaults to '90KV_AL40'.
            add_scatter (bool, optional): Whether to add scatter noise from artifacts. Defaults to False.
            add_noise: (bool, optional): Whether to add Poisson noise. Defaults to False.
            threads (int, optional): Number of threads to use. Defaults to 8.
            max_block_index (int, optional): Maximum GPU block. Defaults to 1024. TODO: determine from compute capability.
>>>>>>> dev
            neglog (bool, optional): whether to apply negative log transform to intensity images. If True, outputs are in range [0, 1]. Recommended for easy viewing. Defaults to False.
            intensity_upper_bound (Optional[float], optional): Maximum intensity, clipped before neglog, after noise and scatter. Defaults to 40 keV / sr.
        """
<<<<<<< HEAD
        logger.warning('Previously, projector.Projector used add_scatter as the switch to control scatter. Now, use the scatter_num switch. add_scatter=True is currently equivalent to scatter_num=(10**6)')
                    
=======

>>>>>>> dev
        # set variables
        volume = utils.listify(volume)
        self.volumes = []
        self.priorities = []
        for _vol in volume:
            assert isinstance(_vol, vol.Volume)
            self.volumes.append(_vol)
        if priorities is None:
            self.priorities = [len(self.volumes) - 1 - i for i in range(len(self.volumes))]
        else:
            for prio in priorities:
                assert isinstance(prio, int), "missing priority, or priority is not an integer"
                assert (0 <= prio) and (prio < len(volume)), "invalid priority outside range [0, NUM_VOLUMES)"
                self.priorities.append(prio)
        assert len(self.volumes) == len(self.priorities)

        self.camera_intrinsics = camera_intrinsics
        self.source_to_detector_distance = source_to_detector_distance
        self.carm = carm
        self.step = step
        self.mode = mode
        self.spectrum = _get_spectrum(spectrum)

        if (scatter_num == 0) and add_scatter:
            self.scatter_num = 1000000 # 10^6
        else:
            self.scatter_num = max(scatter_num, 0) # in case scatter_num < 0

        self.add_noise = add_noise
        self.photon_count = photon_count
        self.threads = threads
        self.max_block_index = max_block_index
        self.collected_energy = collected_energy
        self.neglog = neglog
        self.intensity_upper_bound = intensity_upper_bound
        # TODO: handle intensity_upper_bound when [collected_energy is True] -- I think this should be handled in the SingleProjector.project(...) method right after the solid-angle calculation?

        assert len(self.volumes) > 0

<<<<<<< HEAD
        self.projectors = [
            SingleProjector(
                volume,
                self.camera_intrinsics,
                self.source_to_detector_distance,
                step=step,
                photon_count=photon_count,
                mode=mode,
                spectrum=spectrum,
                threads=threads,
                max_block_index=max_block_index,
                attenuation=(1 == len(self.volumes)),
                collected_energy=self.collected_energy,
                add_scatter=add_scatter,
                scatter_num=self.scatter_num
            ) for volume in self.volumes
        ]
=======
        all_mats = []
        for _vol in self.volumes:
            all_mats.extend(list(_vol.materials.keys()))
        
        self.all_materials = list(set(all_mats))
        self.all_materials.sort()
        logger.info(f"ALL MATERIALS: {self.all_materials}")

        # compile the module
        self.mod = _get_kernel_projector_module(len(self.volumes), len(self.all_materials))
        self.project_kernel = self.mod.get_function("projectKernel")

        # assertions
        for mat in self.all_materials:
            assert mat in material_coefficients, f'unrecognized material: {mat}'

        if self.camera_intrinsics is None:
            assert self.carm is not None and hasattr(self.carm, "camera_intrinsics")
            self.camera_intrinsics = self.carm.camera_intrinsics
        
        self.is_initialized = False
>>>>>>> dev

    @property
    def initialized(self):
        # Has the cuda memory been allocated?
        return self.is_initialized

    @property
    def volume(self):
        logger.warning(f'volume is deprecated. Each projector can contain multiple volumes.')
        if len(self.volumes) != 1:
            raise AttributeError
        return self.volumes[0]

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self.camera_intrinsics.sensor_size
    
    @property
    def output_size(self) -> int:
        return int(np.prod(self.output_shape))

    def project(self, *camera_projections: geo.CameraProjection,) -> np.ndarray:
        """Perform the projection.

        Args:
            camera_projection: any number of camera projections. If none are provided, the Projector uses the CArm device to obtain a camera projection.

        Raises:
            ValueError: if no projections are provided and self.carm is None.

        Returns:
            np.ndarray: array of DRRs, after mass attenuation, etc.
        """
        if not self.initialized:
            raise RuntimeError("Projector has not been initialized.")

        if not camera_projections and self.carm is None:
            raise ValueError(
                "must provide a camera projection object to the projector, unless imaging device (e.g. CArm) is provided"
            )
        elif not camera_projections and self.carm is not None:
            camera_projections = [self.carm.get_camera_projection()]
            logger.debug(f'projecting with source at {camera_projections[0].center_in_world}, pointing toward isocenter at {self.carm.isocenter}...')

        assert isinstance(self.spectrum, np.ndarray)
        
        logger.info("Initiating projection and attenuation...")

<<<<<<< HEAD
        # TODO: handle multiple volumes more elegantly, i.e. in the kernel. (!)
        if len(self.projectors) == 1:
            projector = self.projectors[0]
            images = []
            photon_probs = []
            for i, proj in enumerate(camera_projections):
                logger.info(f"Projecting and attenuating camera position {i+1} / {len(camera_projections)}")
                image, photon_prob = projector.project(proj)
                images.append(image)
                photon_probs.append(photon_prob)

            images = np.stack(images)
            photon_prob = np.stack(photon_probs)
            logger.info("Completed projection and attenuation")
        else:
            # Separate the projection and mass attenuation
            forward_projections = dict()
            for pidx, projector in enumerate(self.projectors):
                outputs = []
                for proj in camera_projections:
                    outputs.append(projector.project(proj))

                outputs = np.stack(outputs)

                # convert forward_projections to dict over materials
                _forward_projections = dict((mat, outputs[:, :, :, m]) for m, mat in enumerate(projector.volume.materials))
                # if len(set(_forward_projections.keys()).intersection(set(forward_projections.keys()))) > 0:
                #     logger.error(f'{_forward_projections.keys()}')
                #     raise NotImplementedError(f'non mutually exclusive materials in multiple volumes.')

                # TODO: this is actively terrible.
                if isinstance(projector.volume, vol.MetalVolume):
                    for mat in ['air', 'soft tissue', 'bone']:
                        if mat not in forward_projections:
                            logger.warning(f'existing projections does not contain material: {mat}')
                            continue
                        elif mat not in _forward_projections:
                            logger.warning(f'new projections does not contain material: {mat}')
                            continue
                        forward_projections[mat] -= _forward_projections[mat]

                    if 'titanium' in forward_projections:
                        forward_projections['titanium'] += _forward_projections['titanium']
                    else:
                        forward_projections['titanium'] = _forward_projections['titanium']
                else:
                    forward_projections.update(_forward_projections)
=======
        intensities = []
        photon_probs = []
        for i, proj in enumerate(camera_projections):
            logger.info(f"Projecting and attenuating camera position {i+1} / {len(camera_projections)}")
            # initialize projection-specific arguments
            if 1 == len(self.volumes):
                _vol = self.volumes[0]
                source_ijk = np.array(proj.get_center_in_volume(_vol)).astype(np.float32)
                logger.debug(f'source point for volume: {source_ijk}')

                ijk_from_index = proj.get_ray_transform(_vol)
                logger.debug(f'center ray: {ijk_from_index @ geo.point(self.output_shape[0] / 2, self.output_shape[1] / 2)}')
                ijk_from_index = np.array(ijk_from_index).astype(np.float32)
                logger.debug(f'ijk_from_index (rt_kinv in kernel):\n{ijk_from_index}')
                cuda.memcpy_htod(int(self.rt_kinv_gpu), ijk_from_index)

                args = [
                    np.int32(proj.sensor_width),        # out_width
                    np.int32(proj.sensor_height),       # out_height
                    np.float32(self.step),              # step
                    np.float32(-0.5),                   # gVolumeEdgeMinPointX
                    np.float32(-0.5),                   # gVolumeEdgeMinPointY
                    np.float32(-0.5),                   # gVolumeEdgeMinPointZ
                    np.float32(_vol.shape[0] - 0.5),    # gVolumeEdgeMaxPointX
                    np.float32(_vol.shape[1] - 0.5),    # gVolumeEdgeMaxPointY
                    np.float32(_vol.shape[2] - 0.5),    # gVolumeEdgeMaxPointZ
                    np.float32(_vol.spacing[0]),        # gVoxelElementSizeX
                    np.float32(_vol.spacing[1]),        # gVoxelElementSizeY
                    np.float32(_vol.spacing[2]),        # gVoxelElementSizeZ
                    np.float32(source_ijk[0]),       # sx
                    np.float32(source_ijk[1]),       # sy
                    np.float32(source_ijk[2]),       # sz
                    self.rt_kinv_gpu,           # RT_Kinv
                    np.int32(self.spectrum.shape[0]),   # n_bins
                    self.energies_gpu,                  # energies
                    self.pdf_gpu,                       # pdf
                    self.absorption_coef_table_gpu,     # absorb_coef_table
                    self.intensity_gpu,         # intensity
                    self.photon_prob_gpu,       # photon_prob
                ]
            else:
                for vol_id, _vol in enumerate(self.volumes):
                    source_ijk = np.array(proj.get_center_in_volume(_vol)).astype(np.float32)
                    logger.debug(f'source point for volume #{vol_id}: {source_ijk}')
                    cuda.memcpy_htod(int(self.sourceX_gpu) + int(4 * vol_id), np.array([source_ijk[0]]))
                    cuda.memcpy_htod(int(self.sourceY_gpu) + int(4 * vol_id), np.array([source_ijk[1]]))
                    cuda.memcpy_htod(int(self.sourceZ_gpu) + int(4 * vol_id), np.array([source_ijk[2]]))

                    ijk_from_index = proj.get_ray_transform(_vol)
                    logger.debug(f'center ray: {ijk_from_index @ geo.point(self.output_shape[0] / 2, self.output_shape[1] / 2)}')
                    ijk_from_index = np.array(ijk_from_index).astype(np.float32)
                    logger.debug(f'ijk_from_index (rt_kinv in kernel):\n{ijk_from_index}')
                    cuda.memcpy_htod(int(self.rt_kinv_gpu) + (9 * 4) * vol_id, ijk_from_index)

                args = [
                    np.int32(proj.sensor_width),        # out_width
                    np.int32(proj.sensor_height),       # out_height
                    np.float32(self.step),              # step
                    self.priorities_gpu,          # priority
                    self.minPointX_gpu,         # gVolumeEdgeMinPointX
                    self.minPointY_gpu,         # gVolumeEdgeMinPointY
                    self.minPointZ_gpu,         # gVolumeEdgeMinPointZ
                    self.maxPointX_gpu,         # gVolumeEdgeMaxPointX
                    self.maxPointY_gpu,         # gVolumeEdgeMaxPointY
                    self.maxPointZ_gpu,         # gVolumeEdgeMaxPointZ
                    self.voxelSizeX_gpu,        # gVoxelElementSizeX
                    self.voxelSizeY_gpu,        # gVoxelElementSizeY
                    self.voxelSizeZ_gpu,        # gVoxelElementSizeZ
                    self.sourceX_gpu,       # sx
                    self.sourceY_gpu,       # sy
                    self.sourceZ_gpu,       # sz
                    self.rt_kinv_gpu,           # RT_Kinv
                    np.int32(self.spectrum.shape[0]),   # n_bins
                    self.energies_gpu,                  # energies
                    self.pdf_gpu,                       # pdf
                    self.absorption_coef_table_gpu,     # absorb_coef_table
                    self.intensity_gpu,         # intensity
                    self.photon_prob_gpu,       # photon_prob
                ]
            # 'endif' for num_volumes > 1 

            # Calculate required blocks
            blocks_w = np.int(np.ceil(self.output_shape[0] / self.threads))
            blocks_h = np.int(np.ceil(self.output_shape[1] / self.threads))
            block = (self.threads, self.threads, 1)
            logger.debug(f"Running: {blocks_w}x{blocks_h} blocks with {self.threads}x{self.threads} threads each")

            if blocks_w <= self.max_block_index and blocks_h <= self.max_block_index:
                offset_w = np.int32(0)
                offset_h = np.int32(0)
                self.project_kernel(*args, offset_w, offset_h, block=block, grid=(blocks_w, blocks_h))
            else:
                logger.debug("Running kernel patchwise")
                for w in range((blocks_w - 1) // (self.max_block_index + 1)):
                    for h in range((blocks_h - 1) // (self.max_block_index + 1)):
                        offset_w = np.int32(w * self.max_block_index)
                        offset_h = np.int32(h * self.max_block_index)
                        self.project_kernel(*args, offset_w, offset_h, block=block, grid=(self.max_block_index, self.max_block_index))
                        context.synchronize() 
>>>>>>> dev

            intensity = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(intensity, self.intensity_gpu)
            # transpose the axes, which previously have width on the slow dimension
            intensity = np.swapaxes(intensity, 0, 1).copy()

            photon_prob = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(photon_prob, self.photon_prob_gpu)
            photon_prob = np.swapaxes(photon_prob, 0, 1).copy()

            intensities.append(intensity)
            photon_probs.append(photon_prob)

        images = np.stack(intensities)
        photon_prob = np.stack(photon_probs)
        logger.info("Completed projection and attenuation")

<<<<<<< HEAD
=======
        if self.add_scatter:
            # lfkj('adding scatter (may cause Nan errors)')
            noise = self.scatter_net.add_scatter(images, self.camera)
            photon_prob *= 1 + noise / images
            images += noise

        # transform to collected energy in keV per cm^2 (or keV per mm^2)
        if self.collected_energy:
            logger.info("converting image to collected energy")
            images = images * (
                self.photon_count
                / (self.camera.pixel_size[0] * self.camera.pixel_size[1])
            )

>>>>>>> dev
        if self.add_noise:
            logger.info("adding Poisson noise")
            images = analytic_generators.add_noise(
                images, photon_prob, self.photon_count
            )

        if self.intensity_upper_bound is not None:
            images = np.clip(images, None, self.intensity_upper_bound)

        if self.neglog:
            logger.info("applying negative log transform")
            images = utils.neglog(images)

        if images.shape[0] == 1:
            return images[0]
        else:
            return images

    def project_over_carm_range(
        self,
        phi_range: Tuple[float, float, float],
        theta_range: Tuple[float, float, float],
        degrees: bool = True,
    ) -> np.ndarray:
        """Project over a range of angles using the included CArm.

        Ignores the CArm's internal pose, except for its isocenter.

        """
        if self.carm is None:
            raise RuntimeError("must provide carm device to projector")

        camera_projections = []
        phis, thetas = utils.generate_uniform_angles(phi_range, theta_range)
        for phi, theta in zip(phis, thetas):
            extrinsic = self.carm.get_camera3d_from_world(
                self.carm.isocenter, phi=phi, theta=theta, degrees=degrees,
            )

            camera_projections.append(
                geo.CameraProjection(self.camera_intrinsics, extrinsic)
            )

        return self.project(*camera_projections)

    def initialize(self):
        """Allocate GPU memory and transfer the volume, segmentations to GPU."""
        if self.initialized:
            raise RuntimeError("Close projector before initializing again.")

        # allocate and transfer the volume texture to GPU
        self.volumes_gpu = []
        self.volumes_texref = []
        for vol_id, volume in enumerate(self.volumes):
            # TODO: this axis-swap is messy and actually may be messing things up. Maybe use a FrameTransform in the Volume class instead?
            volume = np.array(volume)
            volume = np.moveaxis(volume, [0, 1, 2], [2, 1, 0]).copy() # TODO: is this axis swap necessary?
            vol_gpu = cuda.np_to_array(volume, order='C')
            vol_texref = self.mod.get_texref(f"volume_{vol_id}")
            cuda.bind_array_to_texref(vol_gpu, vol_texref)
            self.volumes_gpu.append(vol_gpu)
            self.volumes_texref.append(vol_texref)
        
        # set the (interpolation?) mode
        if self.mode == 'linear':
            for texref in self.volumes_texref:
                texref.set_filter_mode(cuda.filter_mode.LINEAR)
        else:
            raise RuntimeError
        
        self.segmentations_gpu = [] # List[List[segmentations]], indexing by (vol_id, material_id)
        self.segmentations_texref = [] # List[List[texrefs]], indexing by (vol_id, material_id)
        for vol_id, _vol in enumerate(self.volumes):
            seg_for_vol = []
            texref_for_vol = []
            for mat_id, mat in enumerate(self.all_materials):
                seg = None
                if mat in _vol.materials:
                    seg = _vol.materials[mat]
                else:
                    seg = np.zeros(_vol.shape).astype(np.float32)
                seg_for_vol.append(cuda.np_to_array(np.moveaxis(seg, [0, 1, 2], [2, 1, 0]).copy(), order='C'))
                texref = self.mod.get_texref(f'seg_{vol_id}_{mat_id}')
                texref_for_vol.append(texref)

            for seg, texref in zip(seg_for_vol, texref_for_vol):
                cuda.bind_array_to_texref(seg, texref)
                if self.mode == 'linear':
                    texref.set_filter_mode(cuda.filter_mode.LINEAR)
                else:
                    raise RuntimeError("Invalid texref filter mode")
            
            self.segmentations_gpu.append(seg_for_vol)
            self.segmentations_texref.append(texref)

        if len(self.volumes) > 1:
            # allocate volumes' priority level on the GPU
            self.priorities_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            for vol_id, prio in enumerate(self.priorities):
                cuda.memcpy_htod(int(self.priorities_gpu) + (4 * vol_id), np.int32(prio))

            # allocate gVolumeEdge{Min,Max}Point{X,Y,Z} and gVoxelElementSize{X,Y,Z} on the GPU
            self.minPointX_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.minPointY_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.minPointZ_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            
            self.maxPointX_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.maxPointY_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.maxPointZ_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            
            self.voxelSizeX_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.voxelSizeY_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.voxelSizeZ_gpu = cuda.mem_alloc(len(self.volumes) * 4)

            for i, _vol in enumerate(self.volumes):
                gpu_ptr_offset = (4 * i)
                cuda.memcpy_htod(int(self.minPointX_gpu) + gpu_ptr_offset, np.float32(-0.5))
                cuda.memcpy_htod(int(self.minPointY_gpu) + gpu_ptr_offset, np.float32(-0.5))
                cuda.memcpy_htod(int(self.minPointZ_gpu) + gpu_ptr_offset, np.float32(-0.5))

                cuda.memcpy_htod(int(self.maxPointX_gpu) + gpu_ptr_offset, np.float32(_vol.shape[0] - 0.5))
                cuda.memcpy_htod(int(self.maxPointY_gpu) + gpu_ptr_offset, np.float32(_vol.shape[1] - 0.5))
                cuda.memcpy_htod(int(self.maxPointZ_gpu) + gpu_ptr_offset, np.float32(_vol.shape[2] - 0.5))

                cuda.memcpy_htod(int(self.voxelSizeX_gpu) + gpu_ptr_offset, np.float32(_vol.spacing[0]))
                cuda.memcpy_htod(int(self.voxelSizeY_gpu) + gpu_ptr_offset, np.float32(_vol.spacing[1]))
                cuda.memcpy_htod(int(self.voxelSizeZ_gpu) + gpu_ptr_offset, np.float32(_vol.spacing[2]))
            logger.debug(f"gVolume information allocated and copied to GPU")

            # allocate source coord.s on GPU (4 bytes for each of {x,y,z} for each volume)
            self.sourceX_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.sourceY_gpu = cuda.mem_alloc(len(self.volumes) * 4)
            self.sourceZ_gpu = cuda.mem_alloc(len(self.volumes) * 4)
        # 'endif' for multi-volume allocation

        # allocate ijk_from_index matrix array on GPU (3x3 array x 4 bytes per float32)
        self.rt_kinv_gpu = cuda.mem_alloc(len(self.volumes) * 3 * 3 * 4)

        # allocate intensity array on GPU (4 bytes to a float32)
        self.intensity_gpu = cuda.mem_alloc(self.output_size * 4)
        logger.debug(f"bytes alloc'd for self.intensity_gpu: {self.output_size * 4}")

        # allocate photon_prob array on GPU (4 bytes to a float32)
        self.photon_prob_gpu = cuda.mem_alloc(self.output_size * 4)
        logger.debug(f"bytes alloc'd for self.photon_prob_gpu: {self.output_size * 4}")

        # allocate and transfer spectrum energies (4 bytes to a float32)
        assert isinstance(self.spectrum, np.ndarray)
        noncont_energies = self.spectrum[:,0].copy() / 1000
        contiguous_energies = np.ascontiguousarray(noncont_energies, dtype=np.float32)
        n_bins = contiguous_energies.shape[0]
        self.energies_gpu = cuda.mem_alloc(n_bins * 4)
        cuda.memcpy_htod(self.energies_gpu, contiguous_energies)
        logger.debug(f"bytes alloc'd for self.energies_gpu: {n_bins * 4}")

        # allocate and transfer spectrum pdf (4 bytes to a float32)
        noncont_pdf = self.spectrum[:, 1]  / np.sum(self.spectrum[:, 1])
        contiguous_pdf = np.ascontiguousarray(noncont_pdf.copy(), dtype=np.float32)
        assert contiguous_pdf.shape == contiguous_energies.shape
        assert contiguous_pdf.shape[0] == n_bins
        self.pdf_gpu = cuda.mem_alloc(n_bins * 4)
        cuda.memcpy_htod(self.pdf_gpu, contiguous_pdf)
        logger.debug(f"bytes alloc'd for self.pdf_gpu {n_bins * 4}")

        # precompute, allocate, and transfer the get_absorption_coef(energy, material) table (4 bytes to a float32)
        absorption_coef_table = np.empty(n_bins * len(self.all_materials)).astype(np.float32)
        for bin in range(n_bins): #, energy in enumerate(energies):
            for m, mat_name in enumerate(self.all_materials):
                absorption_coef_table[bin * len(self.all_materials) + m] = mass_attenuation.get_absorption_coefs(contiguous_energies[bin], mat_name)
        self.absorption_coef_table_gpu = cuda.mem_alloc(n_bins * len(self.all_materials) * 4)
        cuda.memcpy_htod(self.absorption_coef_table_gpu, absorption_coef_table)
        logger.debug(f"size alloc'd for self.absorption_coef_table_gpu: {n_bins * len(self.all_materials) * 4}")

        # Mark self as initialized.
        self.is_initialized = True

    def free(self):
        """Free the allocated GPU memory."""
        if self.initialized:
            for vol_id, vol_gpu in enumerate(self.volumes_gpu):
                vol_gpu.free()
                for seg in self.segmentations_gpu[vol_id]:
                    seg.free()
            
            if len(self.volumes) > 1:
                self.priorities_gpu.free()
                self.minPointX_gpu.free() # frees all of the gVolume, gVoxel data
                self.sourceX_gpu.free() # also frees source{Y,Z}_gpu

            self.rt_kinv_gpu.free()

            self.intensity_gpu.free()
            self.photon_prob_gpu.free()
            self.energies_gpu.free()
            self.pdf_gpu.free()
            self.absorption_coef_table_gpu.free()

        self.is_initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, tb):
        self.free()

    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)
