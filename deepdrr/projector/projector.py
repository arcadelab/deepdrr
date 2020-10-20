from typing import Literal, List, Union, Tuple, Optional, Dict

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
import numpy as np
import os
from pathlib import Path 


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

    """

    mod = _get_kernel_projector_module()
    project_kernel = mod.get_function("projectKernel")

    def __init__(
        self,
        volume: np.ndarray,
        materials: Union[Dict[str, np.ndarray], np.ndarray],
        voxel_size: np.ndarray,
        origin: np.ndarray = [0, 0, 0],
        step: float = 0.1,
        mode: Literal['linear'] = 'linear',
    ) -> None:
        self.volume = volume
        






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

        return SourceModule(source, include_dirs=[bicubic_path], no_extern_c=True)


    def project(self, proj_mat, threads = 8, max_blockind = 1024):
        if not self.initialized:
            print("Projector is not initialized")
            return

        inv_ar_mat, source_point = proj_mat.get_conanical_proj_matrix(voxel_size=self.voxelsize, volume_size=self.volumesize, origin_shift=self.origin)

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
            self.projKernel(self.proj_width, self.proj_height, self.stepsize, g_volume_edge_min_point_x, g_volume_edge_min_point_y, g_volume_edge_min_point_z,
                            g_volume_edge_max_point_x, g_volume_edge_max_point_y, g_volume_edge_max_point_z, g_voxel_element_size_x, g_voxel_element_size_y, g_voxel_element_size_z, sourcex, sourcey, sourcez,
                            proj_matrix_gpu, pixel_array_gpu, offset_w, offset_h, block=(8, 8, 1), grid=(blocks_w, blocks_h))
        else:
            print("running kernel patchwise")
            for w in range(0, (blocks_w-1)//max_blockind+1):
                for h in range(0, (blocks_h-1) // max_blockind+1):
                    offset_w = np.int32(w * max_blockind)
                    offset_h = np.int32(h * max_blockind)
                    # print(offset_w, offset_h)
                    self.projKernel(self.proj_width, self.proj_height, self.stepsize, g_volume_edge_min_point_x,
                                    g_volume_edge_min_point_y, g_volume_edge_min_point_z,
                                    g_volume_edge_max_point_x, g_volume_edge_max_point_y, g_volume_edge_max_point_z,
                                    g_voxel_element_size_x, g_voxel_element_size_y, g_voxel_element_size_z, sourcex, sourcey,
                                    sourcez,
                                    proj_matrix_gpu, pixel_array_gpu, offset_w, offset_h, block=(8, 8, 1),
                                    grid=(max_blockind, max_blockind))
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
            curr_projections[i, :, :] = projector.project(proj_mat, max_blockind= max_blockind, threads=threads)
        projections[mat] = curr_projections
        #clean projector to free Memory on GPU
        projector = None

    return projections