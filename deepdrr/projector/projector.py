from typing import Literal, List, Union, Tuple, Optional, Dict

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np
import os
from pathlib import Path 

from ..geometry.camera import Camera
from ..geometry.projection import Projection
from .. import utils


def _get_kernel_projector_module() -> SourceModule:
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

    return SourceModule(source, include_dirs=[bicubic_path], no_extern_c=True)


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

    NUM_MATERIALS = 3 # unfortunately this is hard-coded in the kernel.
    mod = _get_kernel_projector_module()
    project_kernel = mod.get_function("projectKernel")

    def __init__(
        self,
        volume: np.ndarray, # TODO: make volume class containing voxel_size, origin, materials, and other params, so that projector can just take in a Volume and a Camera.
        materials: Union[Dict[str, np.ndarray], np.ndarray],
        voxel_size: np.ndarray,
        camera: Camera,
        origin: np.ndarray = [0, 0, 0], # origin of the volume?
        step: float = 0.1, # step size along ray
        mode: Literal['linear'] = 'linear',
        threads: int = 8,
        max_block_index: int = 1024,
        centimeters: bool = True,
    ):
                    
        # set variables
        self.volume = volume
        self.materials = self._format_materials(materials)
        self.voxel_size = np.array(voxel_size)
        self.camera = camera
        self.sensor_size = camera.sensor_size
        self.origin = np.array(origin)
        self.step = step
        self.mode = mode
        self.threads = threads
        self.max_block_index = max_block_index
        self.centimeters = centimeters

        # Has the cuda memory been allocated.
        self.initialized = False

        # assertions
        assert self.materials.shape[1:] == self.volume.shape, f'materials segmentation shape does not match the volume: {self.materials.shape}, {self.volume.shape}'

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

        return np.stack(outputs)

    def over_range(
        self,
        phi_range: Tuple[float, float, float],
        theta_range: Tuple[float, float, float],        
    ) -> np.ndarray:
        projections = self.camera.make_projections_from_range(phi_range=phi_range, theta_range=theta_range)
        return self.project(*projections)

    @property
    def output_shape(self):
        return (self.sensor_size[0], self.sensor_size[1], self.NUM_MATERIALS)
    
    @property
    def output_size(self):
        return self.sensor_size[0] * self.sensor_size[1] * self.NUM_MATERIALS

    def _format_materials(
        self, 
        materials: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Standardize the input materials to a one-hot array.

        Args:
            materials (Union[Dict[str, np.ndarray], np.ndarray]): Either a mapping of material name to segmentation, 
                a segmentation with the same shape as the volume, or a one-hot segmentation.

        Returns:
            np.ndarray: 4D one-hot segmentation of the materials with labels along the 0th axis.
        """
        if isinstance(materials, dict):
            assert len(materials) == self.NUM_MATERIALS
            materials = np.stack([mat == i for i, mat in enumerate(materials.values())], axis=0)
        elif materials.ndim == 3:
            assert np.all(materials < self.NUM_MATERIALS)
            materials = utils.one_hot(materials, self.NUM_MATERIALS, axis=0)
        elif materials.ndim == 4:
            pass
        else:
            raise TypeError

        assert materials.shape[0] == self.NUM_MATERIALS
        materials = materials.astype(np.float32)
        return materials

    def initialize(self):
        """Allocate GPU memory and transfer the volume, materials to GPU."""
        if self.initialized:
            raise RuntimeError("Close projector before initializing again.")

        # allocate and transfer volume texture to GPU
        volume = np.moveaxis(self.volume, [0, 1, 2], [2, 1, 0]).copy()
        self.volume_gpu = cuda.np_to_array(volume, order='C')
        self.volume_texref = self.mod.get_texref("volume")
        cuda.bind_array_to_texref(self.volume_gpu, self.volume_texref)
        if self.mode == 'linear':
            self.volume_texref.set_filter_mode(cuda.filter_mode.LINEAR)

        # allocate and transfer materials texture to GPU
        materials = np.moveaxis(self.materials, [1, 2, 3], [3, 2, 1]).copy()
        self.materials_gpu = [cuda.np_to_array(materials[m], order='C') for m in range(self.NUM_MATERIALS)]
        self.materials_texref = [self.mod.get_texref(f"materials_{m}") for m in range(self.NUM_MATERIALS)]
        for mat, tex in zip(self.materials_gpu, self.materials_texref):
            cuda.bind_array_to_texref(mat, tex)
            if self.mode == 'linear':
                tex.set_filter_mode(cuda.filter_mode.LINEAR)

        # allocate output array on GPU
        self.output_gpu = cuda.mem_alloc(self.output_size * 4)

        # allocate inverse projection matrix array on GPU (3x3 array x 4 bytes)
        self.inv_proj_gpu = cuda.mem_alloc(3 * 3 * 4)
        
        # Mark self as initialized.
        self.initialized = True

    def close(self):
        """Free the allocated GPU memory."""
        if self.initialized:
            self.volume_gpu.free()
            for mat in self.materials_gpu:
                mat.free()

            self.output_gpu.free()
            self.inv_proj_gpu.free()
        self.initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, tb):
        self.close()
        
    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)

"""
Begin deprecated code.
"""

class ForwardProjector():
    def __init__(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray,
        voxelsize: np.ndarray,
        origin: List[float] = [0.0,0.0,0.0],
        stepsize: float = 0.1,
        mode: Literal["linear"] = "linear",
    ) -> None:
        """Initialize the forward projector for one material type.

        Args:
            volume (np.ndarray): the density volume from the CT.
            segmentation (np.ndarray): binary segmentation of the material being projected over.
            voxelsize (np.ndarray): size of the voxel in [x, y, z].
            origin (List[float], optional): 3D coordinate of the origin. Defaults to [0.0,0.0,0.0].
            stepsize (float, optional): size of the step along the projection ray. Defaults to 0.1.
            mode (Literal[, optional): filter mode. Defaults to "linear".
        """
        #generate kernels
        self.mod = self.generateKernelModuleProjector()
        self.projKernel = self.mod.get_function("projectKernel")
        self.volumesize = volume.shape
        self.volume = np.moveaxis(volume, [0, 1, 2], [2, 1, 0]).copy()
        self.segmentation = np.moveaxis(segmentation.astype(np.float32), [0, 1, 2], [2, 1, 0]).copy()
        # print("done swap")
        self.volume_gpu = cuda.np_to_array(self.volume, order='C')
        self.texref_volume = self.mod.get_texref("tex_density")
        cuda.bind_array_to_texref(self.volume_gpu, self.texref_volume)
        self.segmentation_gpu = cuda.np_to_array(self.segmentation, order='C')
        self.texref_segmentation = self.mod.get_texref("tex_segmentation")
        cuda.bind_array_to_texref(self.segmentation_gpu, self.texref_segmentation)
        if mode =="linear":
            self.texref_volume.set_filter_mode(cuda.filter_mode.LINEAR)
            self.texref_segmentation.set_filter_mode(cuda.filter_mode.LINEAR)
        self.voxelsize = voxelsize
        self.stepsize = np.float32(stepsize)
        self.origin = origin
        self.initialized = False
        print("initialized projector")

    def initialize_sensor(self, proj_width, proj_height):
        self.proj_width = np.int32(proj_width)
        self.proj_height = np.int32(proj_height)
        self.initialized = True

    def setOrigin(self, origin):
        self.origin = origin

    def generateKernelModuleProjector(self):
        #path to files for cubic interpolation (folder cubic in DeepDRR)
        d = Path(__file__).resolve().parent
        bicubic_path = str(d / 'cubic')
        source_path = str(d / 'project_kernel.cu')        

        with open(source_path, 'r') as file:
            source = file.read()

        return SourceModule(source, include_dirs=[bicubic_path], no_extern_c=True, build_options=['-D', f'NUM_MATERIALS={NUM_MATERIALS}'])

    def project(
        self, 
        proj_mat: Projection, 
        threads: int = 8,
        max_blockind: int = 1024,
    ) -> np.ndarray:
        if not self.initialized:
            print("Projector is not initialized")
            return

        inv_ar_mat, source_point = proj_mat.get_canonical_matrix(
            voxel_size=self.voxelsize,
            volume_size=self.volumesize, 
            origin_shift=self.origin
        )

        can_proj_matrix = inv_ar_mat.astype(np.float32)
        pixel_array = np.zeros((self.proj_width, self.proj_height)).astype(np.float32)
        sourcex = source_point[0]
        sourcey = source_point[1]
        sourcez = source_point[2]
        g_volume_edge_min_point_x = np.float32(-0.5)
        g_volume_edge_min_point_y = np.float32(-0.5)
        g_volume_edge_min_point_z = np.float32(-0.5)
        g_volume_edge_max_point_x = np.float32(self.volumesize[0] - 0.5)
        g_volume_edge_max_point_y = np.float32(self.volumesize[1] - 0.5)
        g_volume_edge_max_point_z = np.float32(self.volumesize[2] - 0.5)
        g_voxel_element_size_x = self.voxelsize[0]
        g_voxel_element_size_y = self.voxelsize[1]
        g_voxel_element_size_z = self.voxelsize[2]

        #copy to gpu
        proj_matrix_gpu = cuda.mem_alloc(can_proj_matrix.nbytes)
        cuda.memcpy_htod(proj_matrix_gpu, can_proj_matrix)
        pixel_array_gpu = cuda.mem_alloc(pixel_array.nbytes)
        cuda.memcpy_htod(pixel_array_gpu, pixel_array)

        #calculate required blocks
        #threads = 8
        blocks_w = np.int(np.ceil(self.proj_width / threads))
        blocks_h = np.int(np.ceil(self.proj_height / threads))
        print("running:", blocks_w, "x", blocks_h, "blocks with ", threads, "x", threads, "threads")

        if blocks_w <= max_blockind and blocks_h <= max_blockind:
            #run kernel
            offset_w = np.int32(0)
            offset_h = np.int32(0)
            self.projKernel(
                self.proj_width, 
                self.proj_height, 
                self.stepsize, 
                g_volume_edge_min_point_x, 
                g_volume_edge_min_point_y, 
                g_volume_edge_min_point_z,
                g_volume_edge_max_point_x, 
                g_volume_edge_max_point_y, 
                g_volume_edge_max_point_z, 
                g_voxel_element_size_x, 
                g_voxel_element_size_y, 
                g_voxel_element_size_z, 
                sourcex, 
                sourcey, 
                sourcez,
                proj_matrix_gpu, 
                pixel_array_gpu, 
                offset_w, 
                offset_h, 
                block=(8, 8, 1), 
                grid=(blocks_w, blocks_h)
            )
        else:
            print("running kernel patchwise")
            for w in range(0, (blocks_w-1) // max_blockind+1):
                for h in range(0, (blocks_h-1) // max_blockind+1):
                    offset_w = np.int32(w * max_blockind)
                    offset_h = np.int32(h * max_blockind)
                    # print(offset_w, offset_h)
                    self.projKernel(
                        self.proj_width, 
                        self.proj_height, 
                        self.stepsize,
                        g_volume_edge_min_point_x,
                        g_volume_edge_min_point_y, 
                        g_volume_edge_min_point_z,
                        g_volume_edge_max_point_x, 
                        g_volume_edge_max_point_y, 
                        g_volume_edge_max_point_z,
                        g_voxel_element_size_x, 
                        g_voxel_element_size_y, 
                        g_voxel_element_size_z, 
                        sourcex, 
                        sourcey,
                        sourcez,
                        proj_matrix_gpu, 
                        pixel_array_gpu, 
                        offset_w, 
                        offset_h, 
                        block=(8, 8, 1),
                        grid=(max_blockind, max_blockind)
                    )
                    context.synchronize()

        #context.synchronize()
        cuda.memcpy_dtoh(pixel_array, pixel_array_gpu)

        pixel_array = np.swapaxes(pixel_array, 0, 1)
        #normalize to cm
        return pixel_array/10


def generate_projections(projection_matrices, density, materials, origin, voxel_size, sensor_width, sensor_height, mode="linear", max_blockind = 1024, threads = 8):
    # projections = np.zeros((projection_matrices.__len__(), sensor_width, sensor_height, materials.__len__()),dtype=np.float32)
    projections = {}

    for mat in materials:
        print("projecting:", mat)
        curr_projections = np.zeros((projection_matrices.__len__(), sensor_height, sensor_width), dtype=np.float32)
        projector = ForwardProjector(density, materials[mat], voxel_size, origin=origin, mode = mode)
        projector.initialize_sensor(sensor_width, sensor_height)

        for i, proj_mat in enumerate(projection_matrices):
            curr_projections[i, :, :] = projector.project(proj_mat, max_blockind=max_blockind,threads=threads)
        projections[mat] = curr_projections
        #clean projector to free Memory on GPU
        projector = None

    return projections