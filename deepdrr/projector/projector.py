from typing import Literal, List, Union, Tuple, Optional, Dict

import logging
import numpy as np
from pathlib import Path

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.autoinit import context
    from pycuda.compiler import SourceModule
except ImportError:
    SourceModule = "SourceModule"
    logging.warning('pycuda unavailable')

from . import spectral_data
from . import mass_attenuation
from . import scatter
from . import analytic_generators
from .material_coefficients import material_coefficients
from .. import geo
from .. import vol
from ..device import MobileCArm
from .. import utils


logger = logging.getLogger(__name__)


def _get_spectrum(spectrum):
    if isinstance(spectrum, np.ndarray):
        return spectrum
    elif isinstance(spectrum, str):
        assert spectrum in spectral_data.spectrums, f'unrecognized spectrum: {spectrum}'
        return spectral_data.spectrums[spectrum]
    else:
        raise TypeError(f'unrecognized spectrum: {type(spectrum)}')


def _get_kernel_projector_module(num_volumes, num_materials) -> SourceModule:
    """Compile the cuda code for the kernel projector.

    Assumes `project_kernel.cu` and `cubic` interpolation library is in the same directory as THIS file.

    Returns:
        SourceModule: pycuda SourceModule object.
    """
    #path to files for cubic interpolation (folder cubic in DeepDRR)
    d = Path(__file__).resolve().parent
    bicubic_path = str(d / 'cubic')
    source_path = str(d / 'project_kernel.cu')

    with open(source_path, 'r') as file:
        source = file.read()

    logger.debug(f'compiling {source_path} with NUM_VOLUMES={num_volumes}, NUM_MATERIALS={num_materials}')
    # TODO: replace the NUM_MATERIALS junk with some elegant meta-programming.
    return SourceModule(source, include_dirs=[bicubic_path, str(d)], no_extern_c=True, options=['-D', f'NUM_VOLUMES={num_volumes}', '-D', f'NUM_MATERIALS={num_materials}'])

class Projector(object):
    def __init__(
        self,
        volume: Union[vol.Volume, List[vol.Volume]],
        camera_intrinsics: Optional[geo.CameraIntrinsicTransform] = None,
        carm: Optional[MobileCArm] = None,
        step: float = 0.1,
        mode: Literal['linear'] = 'linear',
        spectrum: Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43']] = '90KV_AL40',
        add_scatter: bool = False,
        add_noise: bool = False,
        photon_count: int = 100000,
        threads: int = 8,
        max_block_index: int = 1024,
        collected_energy: bool = False, # convert to keV / cm^2 or keV / mm^2
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
            volume (Union[Volume, List[Volume]]): a volume object with materials segmented. If multiple volumes are provided, they should have mutually exclusive materials (not checked).
            camera_intrinsics (CameraIntrinsicTransform): intrinsics of the projector's camera. (used for sensor size). If None, the CArm object must be provided and have a camera_intrinsics attribute. Defaults to None.
            carm (Optional[MobileCArm], optional): Optional C-arm device, for convenience which can be used to get projections from C-Arm pose. If not provided, camera pose must be defined by user. Defaults to None.
            step (float, optional): size of the step along projection ray in voxels. Defaults to 0.1.
            mode (Literal['linear']): [description].
            spectrum (Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43'], optional): spectrum array or name of spectrum to use for projection. Defaults to '90KV_AL40'.
            add_scatter (bool, optional): whether to add scatter noise from artifacts. Defaults to False.
            add_noise: (bool, optional): whether to add Poisson noise. Defaults to False.
            threads (int, optional): number of threads to use. Defaults to 8.
            max_block_index (int, optional): maximum GPU block. Defaults to 1024. TODO: determine from compute capability.
            neglog (bool, optional): whether to apply negative log transform to intensity images. If True, outputs are in range [0, 1]. Recommended for easy viewing. Defaults to False.
            intensity_upper_bound (Optional[float], optional): Maximum intensity, clipped before neglog, after noise and scatter. Defaults to 40.
        """
                    
        # set variables
        self.volumes = utils.listify(volume)
        self.camera_intrinsics = camera_intrinsics
        self.carm = carm
        self.step = step
        self.mode = mode
        self.spectrum = _get_spectrum(spectrum)
        self.add_scatter = add_scatter
        self.add_noise = add_noise
        self.photon_count = photon_count
        self.threads = threads
        self.max_block_index = max_block_index
        self.collected_energy = collected_energy
        self.neglog = neglog
        self.intensity_upper_bound = intensity_upper_bound

        assert len(self.volumes) > 0

        all_mats = []
        for vol in self.volumes:
            all_mats.extend(list(vol.materials.keys()))
        
        self.all_materials = list(set(all_mats))

        # compile the module
        self.mod = _get_kernel_projector_module(len(self.volumes), len(self.all_materials))
        self.project_kernel = self.mod.get_function("projectKernel")

        # assertions
        for mat in self.all_materials:
            assert mat in material_coefficients, f'unrecognized material: {mat}'

        if self.camera_intrinsics is None:
            assert self.carm is not None and hasattr(self.carm, 'camera_intrinsics')
            self.camera_intrinsics = self.carm.camera_intrinsics
        
        self.is_initialized = False

    @property
    def initialized(self):
        # Has the cuda memory been allocated?
        return self.is_initialized

    @property
    def volume(self):
        if len(self.volumes) != 1: # TODO: what to do here? What do I return?
            raise DeprecationWarning(f'volume is deprecated. Each projector contains multiple "SingleProjectors", which contain their own volumes.')
        return self.volumes[0]

    @property
    def output_shape(self) -> Tuple[int, int]:
        return self.camera_intrinsics.sensor_size
    
    @property
    def output_size(self) -> int:
        return int(np.prod(self.output_shape))

    def project(
        self,
        *camera_projections: geo.CameraProjection,
    ) -> np.ndarray:
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
            raise ValueError('must provide a camera projection object to the projector, unless imaging device (e.g. CArm) is provided')
        elif not camera_projections and self.carm is not None:
            camera_projections = [self.carm.get_camera_projection()]
            logger.debug(f'projecting with source at {camera_projections[0].center_in_world}, pointing toward isocenter at {self.carm.isocenter}...')

        assert isinstance(self.spectrum, np.ndarray)
        
        logger.info("Initiating projection and attenuation...")

        intensities = []
        photon_probs = []
        for i, proj in enumerate(camera_projections):
            logger.info(f"Projecting and attenuating camera position {i+1} / {len(camera_projections)}")
            # initialize projection-specific arguments
            for vol_id, vol in enumerate(self.volumes):
                source_ijk = np.array(proj.get_center_in_volume(vol)).astype(np.float32)
                logger.debug(f'source point for volume #{vol_id}: {source_ijk}')
                cuda.memcpy_htod(int(self.sourceX_gpu) + int(4 * vol_id), np.array([source_ijk[0]]))
                cuda.memcpy_htod(int(self.sourceY_gpu) + int(4 * vol_id), np.array([source_ijk[1]]))
                cuda.memcpy_htod(int(self.sourceZ_gpu) + int(4 * vol_id), np.array([source_ijk[2]]))

                ijk_from_index = proj.get_ray_transform(vol)
                logger.debug(f'center ray: {ijk_from_index @ geo.point(self.output_shape[0] / 2, self.output_shape[1] / 2)}')
                ijk_from_index = np.array(ijk_from_index).astype(np.float32)
                logger.debug(f'ijk_from_index (rt_kinv in kernel):\n{ijk_from_index}')
                cuda.memcpy_htod(int(self.rt_kinv_gpu) + (9 * 4) * vol_id, ijk_from_index)

            args = [
                np.int32(proj.sensor_width),        # out_width
                np.int32(proj.sensor_height),       # out_height
                np.float32(self.step),              # step
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

        if self.add_scatter:
            # lfkj('adding scatter (may cause Nan errors)')
            noise = self.scatter_net.add_scatter(images, self.camera)
            photon_prob *= 1 + noise / images
            images += noise

        # transform to collected energy in keV per cm^2 (or keV per mm^2)
        if self.collected_energy:
            logger.info("converting image to collected energy")
            images = images * (self.photon_count / (self.camera.pixel_size[0] * self.camera.pixel_size[1]))

        if self.add_noise:
            logger.info("adding Poisson noise")
            images = analytic_generators.add_noise(images, photon_prob, self.photon_count)

        if self.intensity_upper_bound is not None:
            images = np.clip(images, None, self.intensity_upper_bound)

        if self.neglog:
            logger.info("applying negative log transform")
            images = utils.neglog(images)

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
                self.carm.isocenter,
                phi=phi,
                theta=theta,
                degrees=degrees,
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
        for vol_id, vol in enumerate(self.volumes):
            seg_for_vol = []
            for mat in self.all_materials:
                seg = None
                if mat in vol.materials:
                    seg = vol.materials[mat]
                else:
                    seg = np.zeros(vol.shape).astype(np.float32)
                seg_for_vol.append(cuda.np_to_array(np.moveaxis(seg, [0, 1, 2], [2, 1, 0]).copy(), order='C'))
            texref_for_vol = [self.mod.get_texref(f'seg_{vol_id}_{mat_id}') for mat_id, _ in enumerate(self.all_materials)]

            for seg, texref in zip(seg_for_vol, texref_for_vol):
                cuda.bind_array_to_texref(seg, texref)
                if self.mode == 'linear':
                    texref.set_filter_mode(cuda.filter_mode.LINEAR)
                else:
                    raise RuntimeError("Invalid texref filter mode")
            
            self.segmentations_gpu.append(seg_for_vol)
            self.segmentations_texref.append(texref)


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
        # gpu_ptr = cuda.mem_alloc(len(self.volumes) * 9 * 4)
        # print(f"gpu_ptr: {gpu_ptr}")
        # self.minPointX_gpu = int(gpu_ptr) + 0 * (4 * len(self.volumes))
        # self.minPointY_gpu = int(gpu_ptr) + 1 * (4 * len(self.volumes))
        # self.minPointZ_gpu = int(gpu_ptr) + 2 * (4 * len(self.volumes))
        
        # self.maxPointX_gpu = int(gpu_ptr) + 3 * (4 * len(self.volumes))
        # self.maxPointY_gpu = int(gpu_ptr) + 4 * (4 * len(self.volumes))
        # self.maxPointZ_gpu = int(gpu_ptr) + 5 * (4 * len(self.volumes))

        # self.voxelSizeX_gpu = int(gpu_ptr) + 6 * (4 * len(self.volumes))
        # self.voxelSizeY_gpu = int(gpu_ptr) + 7 * (4 * len(self.volumes))
        # self.voxelSizeZ_gpu = int(gpu_ptr) + 8 * (4 * len(self.volumes))

        for i, vol in enumerate(self.volumes):
            gpu_ptr_offset = (4 * i)
            cuda.memcpy_htod(int(self.minPointX_gpu) + gpu_ptr_offset, np.float32(-0.5))
            cuda.memcpy_htod(int(self.minPointY_gpu) + gpu_ptr_offset, np.float32(-0.5))
            cuda.memcpy_htod(int(self.minPointZ_gpu) + gpu_ptr_offset, np.float32(-0.5))

            cuda.memcpy_htod(int(self.maxPointX_gpu) + gpu_ptr_offset, np.float32(vol.shape[0] - 0.5))
            cuda.memcpy_htod(int(self.maxPointY_gpu) + gpu_ptr_offset, np.float32(vol.shape[1] - 0.5))
            cuda.memcpy_htod(int(self.maxPointZ_gpu) + gpu_ptr_offset, np.float32(vol.shape[2] - 0.5))

            cuda.memcpy_htod(int(self.voxelSizeX_gpu) + gpu_ptr_offset, np.float32(vol.spacing[0]))
            cuda.memcpy_htod(int(self.voxelSizeY_gpu) + gpu_ptr_offset, np.float32(vol.spacing[1]))
            cuda.memcpy_htod(int(self.voxelSizeZ_gpu) + gpu_ptr_offset, np.float32(vol.spacing[2]))
            # arr = np.array([
            #     -0.5, -0.5, -0.5,
            # ])
            # tmp = cuda.mem_alloc(48)
            # cuda.memcpy_htod(tmp, arr)
        logger.debug(f"gVolume information allocated and copied to GPU")

        # allocate source coord.s on GPU (4 bytes for each of {x,y,z} for each volume)
        self.sourceX_gpu = cuda.mem_alloc(len(self.volumes) * 4)
        self.sourceY_gpu = cuda.mem_alloc(len(self.volumes) * 4)
        self.sourceZ_gpu = cuda.mem_alloc(len(self.volumes) * 4)
        # source_gpu = cuda.mem_alloc(len(self.volumes) * 3 * 4)
        # self.sourceX_gpu = int(source_gpu + 0 * (4 * len(self.volumes)))
        # self.sourceY_gpu = int(source_gpu + 1 * (4 * len(self.volumes)))
        # self.sourceZ_gpu = int(source_gpu + 2 * (4 * len(self.volumes)))

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
