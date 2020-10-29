from typing import Literal, List, Union, Tuple, Optional, Dict

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np
import os
from pathlib import Path 

from . import spectral_data
from . import mass_attenuation
from . import scatter
from . import analytic_generators
from ..geometry.camera import Camera
from ..geometry.projection import Projection
from .. import utils


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
    """Forward projector object.

    The goal is to get to a point where reloads are done only when a new volume is needed.

    Usage:
    ```
    with Projector(volume, materials, ...) as projector:
        for projection in projections:
            yield projector(projection)
    ```

    """

    material_names = [
        "bone",
        "soft tissue",
        "air",
        "iron",
        "lung",
        "titanium",
        "teflon",
        "bone external",
        "soft tissue external",
        "air external",
        "iron external",
        "lung external",
        "titanium external",
        "teflon external",
    ]

    def __init__(
        self,
        volume: np.ndarray, # TODO: make volume class containing voxel_size, origin, materials, and other params, so that projector can just take in a Volume and a Camera.
        segmentation: Union[Dict[str, np.ndarray], np.ndarray],
        voxel_size: np.ndarray,
        camera: Camera,
        materials: List[str] = ['air', 'soft tissue', 'bone'], # list of materials in order corresponding to zero-indexed integer labels
        origin: np.ndarray = [0, 0, 0], # origin of the volume?
        step: float = 0.1, # step size along ray
        mode: Literal['linear'] = 'linear',
        spectrum: Union[np.ndarray, Literal['60KV_AL35', '90KV_AL40', '120KV_AL43']] = '90KV_AL40',
        add_scatter: bool = True, # add scatter noise
        photon_count: int = 100000,
        threads: int = 8,
        max_block_index: int = 1024,
        centimeters: bool = True,       # convert to centimeters
        collected_energy: bool = False, # convert to keV / cm^2 or keV / mm^2
    ):
                    
        # set variables
        self.volume = volume
        self.segmentation = self._format_segmentation(segmentation, num_materials=len(materials))
        self.materials = materials
        self.voxel_size = np.array(voxel_size)
        self.camera = camera
        self.sensor_size = camera.sensor_size
        self.origin = np.array(origin)
        self.step = step
        self.mode = mode
        self.spectrum = self._get_spectrum(spectrum)
        self.add_scatter = add_scatter
        self.photon_count = photon_count
        self.threads = threads
        self.max_block_index = max_block_index
        self.centimeters = centimeters
        self.collected_energy = collected_energy

        # other parameters
        self.num_materials = len(materials)
        self.scatter_net = scatter.ScatterNet() if self.add_scatter else None

        # compile the module
        self.mod = _get_kernel_projector_module(self.num_materials) # TODO: make this not a compile-time option.
        self.project_kernel = self.mod.get_function("projectKernel")

        # assertions
        for mat in self.materials:
            assert mat in self.material_names, f'unrecognized material: {mat}'
        assert self.segmentation.shape[0] == self.num_materials
        assert self.segmentation.shape[1:] == self.volume.shape, f'bad materials segmentation shape: {self.segmentation.shape}, volume: {self.volume.shape}'

        # Has the cuda memory been allocated.
        self.initialized = False

    def _project(
        self,
        projection: Projection,
    ) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Projector has not been initialized.")

        # initialize projection-specific arguments
        inv_proj, source_point = projection.get_ray_transform(
            voxel_size=self.voxel_size,
            volume_size=self.volume.shape,
            origin=self.origin,
            dtype=np.float32,
        )

        # copy the projection matrix to CUDA (output array initialized to zero by the kernel)
        cuda.memcpy_htod(self.inv_proj_gpu, inv_proj)

        # Make the arguments to the CUDA "projectKernel".
        args = [
            np.int32(self.sensor_size[0]),          # out_width
            np.int32(self.sensor_size[1]),          # out_height
            np.float32(self.step),                  # step
            np.float32(-0.5),                       # gVolumeEdgeMinPointX
            np.float32(-0.5),                       # gVolumeEdgeMinPointY
            np.float32(-0.5),                       # gVolumeEdgeMinPointZ
            np.float32(self.volume.shape[0] - 0.5), # gVolumeEdgeMaxPointX
            np.float32(self.volume.shape[1] - 0.5), # gVolumeEdgeMaxPointY
            np.float32(self.volume.shape[2] - 0.5), # gVolumeEdgeMaxPointZ
            np.float32(self.voxel_size[0]),         # gVoxelElementSizeX
            np.float32(self.voxel_size[1]),         # gVoxelElementSizeY
            np.float32(self.voxel_size[2]),         # gVoxelElementSizeZ
            source_point[0],                        # sx
            source_point[1],                        # sy
            source_point[2],                        # sz
            self.inv_proj_gpu,                      # gInvARmatrix
            self.output_gpu,                        # output
        ]

        # Calculate required blocks
        blocks_w = np.int(np.ceil(self.sensor_size[0] / self.threads))
        blocks_h = np.int(np.ceil(self.sensor_size[1] / self.threads))
        block = (8, 8, 1)
        print("running:", blocks_w, "x", blocks_h, "blocks with ", self.threads, "x", self.threads, "threads")

        if blocks_w <= self.max_block_index and blocks_h <= self.max_block_index:
            offset_w = np.int32(0)
            offset_h = np.int32(0)
            self.project_kernel(*args, offset_w, offset_h, block=block, grid=(blocks_w, blocks_h))
        else:
            print("running kernel patchwise")
            for w in range((blocks_w - 1) // (self.max_block_index + 1)):
                for h in range((blocks_h - 1) // (self.max_block_index + 1)):
                    offset_w = np.int32(w * self.max_block_index)
                    offset_h = np.int32(h * self.max_block_index)
                    self.project_kernel(*args, offset_w, offset_h, block=block, grid=(self.max_block_index, self.max_block_index))
                    context.synchronize()
                
        # copy the output to CPU
        output = np.empty(self.output_shape, np.float32)
        cuda.memcpy_dtoh(output, self.output_gpu)

        # transpose the axes
        output = np.swapaxes(output, 0, 1).copy()

        # convert to centimeters
        if self.centimeters:
            output /= 10
            
        return output

    def project(
        self,
        *projections: Projection,
    ) -> np.ndarray:
        outputs = []

        for proj in projections:
            outputs.append(self._project(proj))

        forward_projections = np.stack(outputs)

        # convert forward_projections to dict over materials
        # (TODO: fix mass_attenuation so it doesn't require this conversion)
        forward_projections = dict((k, forward_projections[:, :, :, i]) for i, k in enumerate(self.materials))
        
        # calculate intensity at detector (images: mean energy one photon emitted from the source
        # deposits at the detector element, photon_prob: probability of a photon emitted from the
        # source to arrive at the detector)
        images, photon_prob = mass_attenuation.calculate_intensity_from_spectrum(forward_projections, self.spectrum)

        if self.add_scatter:
            noise = self.scatter_net.add_scatter(images, self.camera)
            photon_prob *= 1 + noise / images
            images += noise

        # transform to collected energy in keV per cm^2 (or keV per mm^2)
        if self.collected_energy:
            images = images * (self.photon_count / (self.camera.pixel_size[0] * self.camera.pixel_size[1]))

        images = analytic_generators.add_noise(images, photon_prob, self.photon_count)

        return images

    def from_view(
            self,
            phi: float,
            theta: float,
            rho: float = 0,
            offset: List[float] = [0., 0., 0.],
    ):
        projection = self.camera.make_projections([phi], [theta], [rho], [offset])[0]
        return self.project(projection)
        
    def over_range(
        self,
        phi_range: Tuple[float, float, float],
        theta_range: Tuple[float, float, float],        
    ) -> np.ndarray:
        projections = self.camera.make_projections_from_range(phi_range=phi_range, theta_range=theta_range)
        return self.project(*projections)

    @property
    def output_shape(self):
        return (self.sensor_size[0], self.sensor_size[1], self.num_materials)
    
    @property
    def output_size(self):
        return self.sensor_size[0] * self.sensor_size[1] * self.num_materials

    def _format_segmentation(
        self, 
        segmentation: Union[Dict[str, np.ndarray], np.ndarray],
        num_materials: int,
    ) -> np.ndarray:
        """Standardize the input segmentation to a one-hot array.

        Args:
            segmentation (Union[Dict[str, np.ndarray], np.ndarray]): Either a mapping of material name to segmentation, 
                a segmentation with the same shape as the volume, or a one-hot segmentation.
            num_materials (int): number of materials in the segmentation

        Returns:
            np.ndarray: 4D one-hot segmentation of the materials with labels along the 0th axis.
        """
        if isinstance(segmentation, dict):
            assert len(segmentation) == num_materials
            segmentation = np.stack([seg == i for i, seg in enumerate(segmentation.values())], axis=0)
        elif segmentation.ndim == 3:
            assert np.all(segmentation < num_materials) # TODO: more flexibility?
            segmentation = utils.one_hot(segmentation, num_materials, axis=0)
        elif segmentation.ndim == 4:
            pass
        else:
            raise TypeError

        return segmentation.astype(np.float32)

    def _get_spectrum(self, spectrum):
        if isinstance(spectrum, np.ndarray):
            return spectrum
        elif isinstance(spectrum, str):
            assert spectrum in spectral_data.spectrums, f'unrecognized spectrum: {spectrum}'
            return spectral_data.spectrums[spectrum]

    def initialize(self):
        """Allocate GPU memory and transfer the volume, segmentations to GPU."""
        if self.initialized:
            raise RuntimeError("Close projector before initializing again.")

        # allocate and transfer volume texture to GPU
        volume = np.moveaxis(self.volume, [0, 1, 2], [2, 1, 0]).copy()
        self.volume_gpu = cuda.np_to_array(volume, order='C')
        self.volume_texref = self.mod.get_texref("volume")
        cuda.bind_array_to_texref(self.volume_gpu, self.volume_texref)
        if self.mode == 'linear':
            self.volume_texref.set_filter_mode(cuda.filter_mode.LINEAR)

        # allocate and transfer segmentation texture to GPU
        segmentation = np.moveaxis(self.segmentation, [1, 2, 3], [3, 2, 1]).copy()
        self.segmentation_gpu = [cuda.np_to_array(segmentation[m], order='C') for m in range(self.num_materials)]
        self.segmentation_texref = [self.mod.get_texref(f"seg_{m}") for m in range(self.num_materials)]
        for seg, tex in zip(self.segmentation_gpu, self.segmentation_texref):
            cuda.bind_array_to_texref(seg, tex)
            if self.mode == 'linear':
                tex.set_filter_mode(cuda.filter_mode.LINEAR)

        # allocate output array on GPU
        self.output_gpu = cuda.mem_alloc(self.output_size * 4)

        # allocate inverse projection matrix array on GPU (3x3 array x 4 bytes)
        self.inv_proj_gpu = cuda.mem_alloc(3 * 3 * 4)
        
        # Mark self as initialized.
        self.initialized = True

    def free(self):
        """Free the allocated GPU memory."""
        if self.initialized:
            self.volume_gpu.free()
            for seg in self.segmentation_gpu:
                seg.free()

            self.output_gpu.free()
            self.inv_proj_gpu.free()
        self.initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, tb):
        self.free()
        
    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)

