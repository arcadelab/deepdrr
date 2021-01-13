from typing import Literal, List, Union, Tuple, Optional, Dict

import logging
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
from multiprocessing import cpu_count
import numpy as np
import os
from pathlib import Path 

from . import spectral_data
from . import mass_attenuation
from . import scatter
from . import analytic_generators
from .material_coefficients import material_coefficients
from .. import geo
from ..vol import Volume
from ..device import CArm
from .. import utils


logger = logging.getLogger(__name__)


def _get_kernel_projector_module(num_materials) -> SourceModule:
    """Compile the cuda code for the kernel projector.

    Assumes `kernel_projector.cu` and `cubic` interpolation library is in the same directory as THIS file.

    Returns:
        SourceModule: pycuda SourceModule object.
    """
    #path to files for cubic interpolation (folder cubic in DeepDRR)
    d = Path(__file__).resolve().parent
    bicubic_path = str(d / 'cubic')
    source_path = str(d / 'project_kernel.cu')        

    with open(source_path, 'r') as file:
        source = file.read()

    # TODO: replace the NUM_MATERIALS junk with some elegant meta-programming.
    return SourceModule(source, include_dirs=[bicubic_path], no_extern_c=True, options=['-D', f'NUM_MATERIALS={num_materials}'])


class Projector(object):
    def __init__(
        self,
        volume: Volume,
        camera_intrinsics: geo.CameraIntrinsicTransform,
        carm: Optional[CArm] = None,
        step: float = 0.1,
        mode: Literal['linear'] = 'linear',
        spectrum: Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43']] = '90KV_AL40',
        add_scatter: bool = False, # add scatter noise
        add_noise: bool = False, # add poisson noise
        photon_count: int = 100000,
        threads: int = 8,
        max_block_index: int = 1024,
        collected_energy: bool = False, # convert to keV / cm^2 or keV / mm^2
        neglog: bool = False,
    ) -> None:
        """Create the projector, which has info for simulating the DRR.

        Usage:
        ```
        with Projector(volume, materials, ...) as projector:
            for projection in projections:
                yield projector(projection)
        ```

        Args:
            volume (Volume): a volume object with materials segmented.
            camera_intrinsics (CameraIntrinsicTransform): intrinsics of the projector's camera. (used for sensor size).
            carm (Optional[CArm], optional): Optional C-arm device, for convenience which can be used to get projections from C-Arm pose. If not provided, camera pose must be defined by user. Defaults to None.
            step (float, optional): size of the step along projection ray in voxels. Defaults to 0.1.
            mode (Literal['linear']): [description].
            spectrum (Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43'], optional): spectrum array or name of spectrum to use for projection. Defaults to '90KV_AL40'.
            add_scatter (bool, optional): whether to add scatter noise. Defaults to False.
            threads (int, optional): number of threads to use. Defaults to 8.
            max_block_index (int, optional): maximum GPU block. Defaults to 1024.
            neglog (bool, optional): whether to apply negative log transform to output images. Recommended for easy viewing. Defaults to False.
        """
                    
        # set variables
        self.volume = volume
        self.camera_intrinsics = camera_intrinsics
        self.carm = carm
        self.step = step
        self.mode = mode
        self.spectrum = self._get_spectrum(spectrum)
        self.add_scatter = add_scatter
        self.add_noise = add_noise
        self.photon_count = photon_count
        self.threads = threads
        self.max_block_index = max_block_index
        self.collected_energy = collected_energy
        self.neglog = neglog

        # other parameters
        self.sensor_size = self.camera_intrinsics.sensor_size
        self.num_materials = len(self.volume.materials)
        self.scatter_net = scatter.ScatterNet() if self.add_scatter else None

        # compile the module
        self.mod = _get_kernel_projector_module(self.num_materials) # TODO: make this not a compile-time option.
        self.project_kernel = self.mod.get_function("projectKernel")
        
        # assertions
        for mat in self.volume.materials:
            assert mat in material_coefficients, f'unrecognized material: {mat}'

        # Has the cuda memory been allocated?
        self.initialized = False

    def _project(
        self,
        camera_projection: geo.CameraProjection,
    ) -> np.ndarray:
        """Perform the projection over just one image.

        Args:
            camera_projection (geo.CameraProjection): a camera projection transform.

        Raises:
            RuntimeError: if the projector has not been initialized.

        Returns:
            np.ndarray: the output projection for each material.
        """
        if not self.initialized:
            raise RuntimeError("Projector has not been initialized.")

        # initialize projection-specific arguments
        camera_center_in_volume = np.array(camera_projection.get_center_in_volume(self.volume)).astype(np.float32)
        logger.debug(f'camera_center_ijk (source point): {camera_center_in_volume}')

        ijk_from_index = camera_projection.get_ray_transform(self.volume)
        ijk_from_index = np.array(ijk_from_index).astype(np.float32)

        # spacing
        spacing = self.volume.spacing

        # copy the projection matrix to CUDA (output array initialized to zero by the kernel)
        cuda.memcpy_htod(self.rt_kinv_gpu, ijk_from_index)

        # Make the arguments to the CUDA "projectKernel".
        args = [
            np.int32(self.camera_intrinsics.sensor_width),          # out_width
            np.int32(self.camera_intrinsics.sensor_height),          # out_height
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
        blocks_w = np.int(np.ceil(self.sensor_size[0] / self.threads))
        blocks_h = np.int(np.ceil(self.sensor_size[1] / self.threads))
        block = (8, 8, 1)
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
                
        # copy the output to CPU
        output = np.empty(self.output_shape, np.float32)
        cuda.memcpy_dtoh(output, self.output_gpu)

        # transpose the axes, which previously have width on the slow dimension
        output = np.swapaxes(output, 0, 1).copy()

        # normalize to centimeters
        output /= 10

        return output

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
        logger.debug(f'carm isocenter: {self.carm.isocenter}')
        if not camera_projections and self.carm is None:
            raise ValueError('must provide a camera projection object to the projector, unless imaging device (e.g. CArm) is provided')
        elif not camera_projections and self.carm is not None:
            camera_projections = [geo.CameraProjection(self.camera_intrinsics, self.carm.camera3d_from_world)]
        
        outputs = []

        for proj in camera_projections:
            outputs.append(self._project(proj))

        forward_projections = np.stack(outputs)

        # convert forward_projections to dict over materials
        # TODO: fix mass_attenuation so it doesn't require this conversion
        forward_projections = dict((mat, forward_projections[:, :, :, m]) for m, mat in enumerate(self.volume.materials))

        logger.info(f'performing mass attenuation...')
        images, photon_prob = mass_attenuation.calculate_intensity_from_spectrum(forward_projections, self.spectrum)
        logger.info('done.')

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
                self.carm.isocenter,
                phi=phi,
                theta=theta,
                degrees=degrees,
            )

            camera_projections.append(
                geo.CameraProjection(self.camera_intrinsics, extrinsic)
            )

        return self.project(*camera_projections)

    @property
    def output_shape(self):
        return (self.sensor_size[0], self.sensor_size[1], self.num_materials)
    
    @property
    def output_size(self):
        return self.sensor_size[0] * self.sensor_size[1] * self.num_materials

    def _get_spectrum(self, spectrum):
        if isinstance(spectrum, np.ndarray):
            return spectrum
        elif isinstance(spectrum, str):
            assert spectrum in spectral_data.spectrums, f'unrecognized spectrum: {spectrum}'
            return spectral_data.spectrums[spectrum]
        else:
            raise TypeError(f'unrecognized spectrum: {type(spectrum)}')

    def initialize(self):
        """Allocate GPU memory and transfer the volume, segmentations to GPU."""
        if self.initialized:
            raise RuntimeError("Close projector before initializing again.")

        # allocate and transfer volume texture to GPU
        # TODO: this axis-swap is messy and actually may be messing things up. Maybe use a FrameTransform in the Volume class instead?
        volume = self.volume.data
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

        # allocate output image array on GPU (4 bytes to a float32)
        self.output_gpu = cuda.mem_alloc(self.output_size * 4)

        # allocate ijk_from_index matrix array on GPU (3x3 array x 4 bytes per float32)
        self.rt_kinv_gpu = cuda.mem_alloc(3 * 3 * 4)
        
        # Mark self as initialized.
        self.initialized = True

    def free(self):
        """Free the allocated GPU memory."""
        if self.initialized:
            self.volume_gpu.free()
            for seg in self.segmentations_gpu:
                seg.free()

            self.output_gpu.free()
            self.rt_kinv_gpu.free()
        self.initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, tb):
        self.free()
        
    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)
