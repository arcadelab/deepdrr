from __future__ import annotations


import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import warnings

import math
from pyvista import DeprecationError
import torch
import numpy as np
from pyparsing import alphas
from collections import defaultdict

from OpenGL.GL import GL_TEXTURE_RECTANGLE
from OpenGL.GL import *

from pyrender import (
    IntrinsicsCamera,
    Primitive,
    Mesh,
    Node,
    Scene,
    RenderFlags,
    MetallicRoughnessMaterial,
)
from ..pyrenderdrr.renderer import DRRMode
from ..pyrenderdrr.material import DRRMaterial

from .. import geo, utils, vol
from ..device import Device, MobileCArm
from . import analytic_generators, spectral_data
from ..material import Material
from .mcgpu_compton_data import COMPTON_DATA
from .mcgpu_mfp_data import MFP_DATA
from .mcgpu_rita_samplers import rita_samplers

from pyrender.platforms import egl
from ..pyrenderdrr.renderer import Renderer

from functools import lru_cache

import cupy as cp
import cupy

import numpy
from cuda import cudart

from ..utils.cuda_utils import check_cudart_err, format_cudart_err

# from cupyx.profiler import time_range
# import cupyx.profiler

from matplotlib import pyplot as plt

from ..utils.output_logger import OutputLogger
import contextlib

log = logging.getLogger(__name__)


NUMBYTES_INT8 = 1
NUMBYTES_INT32 = 4
NUMBYTES_FLOAT32 = 4


# @cupyx.profiler.time_range()
def gl_tex_to_gpu(reg_img, dst_ptr, width, height, num_channels):
    check_cudart_err(cudart.cudaGraphicsMapResources(1, reg_img, None))

    cuda_array = check_cudart_err(
        cudart.cudaGraphicsSubResourceGetMappedArray(reg_img, 0, 0)
    )

    check_cudart_err(
        cudart.cudaMemcpy2DFromArray(
            dst=dst_ptr,
            dpitch=int(width * num_channels * NUMBYTES_FLOAT32),
            src=cuda_array,
            wOffset=0,
            hOffset=0,
            width=int(width * num_channels * NUMBYTES_FLOAT32),
            height=int(height),
            kind=cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        )
    )
    check_cudart_err(cudart.cudaGraphicsUnmapResources(1, reg_img, None))


# @time_range()
def gl_gpu_to_tex(reg_img, src_ptr, width, height, num_channels):
    check_cudart_err(cudart.cudaGraphicsMapResources(1, reg_img, None))

    cuda_array = check_cudart_err(
        cudart.cudaGraphicsSubResourceGetMappedArray(reg_img, 0, 0)
    )

    check_cudart_err(
        cudart.cudaMemcpy2DToArray(
            dst=cuda_array,
            wOffset=0,
            hOffset=0,
            src=src_ptr,
            spitch=int(width * num_channels * NUMBYTES_FLOAT32),
            width=int(width * num_channels * NUMBYTES_FLOAT32),
            height=int(height),
            kind=cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
        )
    )
    check_cudart_err(cudart.cudaGraphicsUnmapResources(1, reg_img, None))


def create_cuda_texture(
    array,
    texture_shape: Tuple[int, ...] = None,
    num_channels: int = 1,
    normalised_values: bool = False,
    normalised_coords: bool = False,
    sampling_mode: str = "linear",
    address_mode: str = "clamp",
    dtype=None,
):
    """Creates a Cuda texture and takes care of a lot of the needed 'know-how'

    Parameters
    ----------
    array
    texture_shape
    num_channels
    normalised_values
    normalised_coords
    sampling_mode
    address_mode
    dtype

    Returns
    -------

    """
    import cupy

    if texture_shape is None:
        if num_channels > 1:
            texture_shape = array.shape[0:-1]
        else:
            texture_shape = array.shape

    if array.dtype == numpy.float16 or dtype == numpy.float16:
        raise ValueError("float16 types not yet supported!")

    if dtype is None:
        dtype = array.dtype

    if not (1 <= len(texture_shape) <= 3):
        raise ValueError(
            f"Invalid number of dimensions ({len(texture_shape)}), must be 1, 2 or 3 (shape={texture_shape}) "
        )

    if not (num_channels == 1 or num_channels == 2 or num_channels == 4):
        raise ValueError(
            f"Invalid number of channels ({num_channels}), must be 1, 2., 3 or 4"
        )

    if array.size != numpy.prod(texture_shape) * num_channels:
        raise ValueError(
            f"Texture shape {texture_shape}, num of channels ({num_channels}), "
            + f"and array size ({array.size}) are mismatched!"
        )

    dtype = numpy.dtype(dtype)

    if array.dtype != dtype:
        array = array.astype(dtype, copy=False)

    nbits = 8 * dtype.itemsize
    channels = (nbits,) * num_channels + (0,) * (4 - num_channels)
    if "f" in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindFloat
    elif "i" in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindSigned
    elif "u" in dtype.kind:
        channel_type = cupy.cuda.runtime.cudaChannelFormatKindUnsigned
    else:
        raise ValueError(f"dtype '{address_mode}' is not supported")

    format_descriptor = cupy.cuda.texture.ChannelFormatDescriptor(
        *channels, channel_type
    )
    
    cuda_array = cupy.cuda.texture.CUDAarray(format_descriptor, *(texture_shape[::-1]))
    ressource_descriptor = cupy.cuda.texture.ResourceDescriptor(
        cupy.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_array
    )

    if address_mode == "clamp":
        address_mode = cupy.cuda.runtime.cudaAddressModeClamp
    elif address_mode == "border":
        address_mode = cupy.cuda.runtime.cudaAddressModeBorder
    elif address_mode == "wrap":
        address_mode = cupy.cuda.runtime.cudaAddressModeWrap
    elif address_mode == "mirror":
        address_mode = cupy.cuda.runtime.cudaAddressModeMirror
    else:
        raise ValueError(f"Address mode '{address_mode}' is not supported")

    address_mode = (address_mode,) * len(texture_shape)

    if sampling_mode == "nearest":
        filter_mode = cupy.cuda.runtime.cudaFilterModePoint
    elif sampling_mode == "linear":
        filter_mode = cupy.cuda.runtime.cudaFilterModeLinear
    else:
        raise ValueError(f"Sampling mode '{sampling_mode}' is not supported")

    if normalised_values:
        read_mode = cupy.cuda.runtime.cudaReadModeNormalizedFloat
    else:
        read_mode = cupy.cuda.runtime.cudaReadModeElementType

    texture_descriptor = cupy.cuda.texture.TextureDescriptor(
        addressModes=address_mode,
        filterMode=filter_mode,
        readMode=read_mode,
        sRGB=None,
        borderColors=None,
        normalizedCoords=normalised_coords,
        maxAnisotropy=None,
    )

    texture_object = cupy.cuda.texture.TextureObject(
        ressource_descriptor, texture_descriptor
    )

    # 'copy_from' from CUDAArray requires that the num of channels be multiplied
    # to the last axis of the array (see cupy docs!)
    if num_channels > 1:
        array_shape_for_copy = texture_shape[:-1] + (texture_shape[-1] * num_channels,)
    else:
        array_shape_for_copy = texture_shape
    axp = cupy.get_array_module(array)
    array = axp.reshape(array, newshape=array_shape_for_copy)

    if not array.flags.owndata or not not array.flags.c_contiguous:
        # the array must be contiguous, we check if this is a derived array,
        # if yes we must unfortunately copy the data...
        array = array.copy()

    # We need to synchronise otherwise some weird stuff happens! see warp 3d demo does not work withoutv this!
    # Backend.current().synchronise()
    cuda_array.copy_from(array)

    del format_descriptor, texture_descriptor, ressource_descriptor

    return texture_object, cuda_array


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


# @lru_cache(maxsize=1)
# def max_block_dim(): # TODO (liam): use this maybe?
#     ret = np.inf
#     for devicenum in range(cuda.Device.count()):
#         attrs = cuda.Device(devicenum).get_attributes()
#         ret = min(attrs[cuda.device_attribute.MAX_BLOCK_DIM_X], ret)
#     return ret

# @lru_cache(maxsize=1)
# def max_grid_dim(): # TODO (liam): use this maybe?
#     ret = np.inf
#     for devicenum in range(cuda.Device.count()):
#         attrs = cuda.Device(devicenum).get_attributes()
#         ret = min(attrs[cuda.device_attribute.MAX_GRID_DIM_X], ret)
#     return ret


def _get_kernel_peel_postprocess_module(
    num_intersections: int,
) -> cp.RawModule:
    d = Path(__file__).resolve().parent

    source_path = str(d / "peel_postprocess_kernel.cu")

    with open(source_path, "r") as file:
        source = file.read()

    options = []
    if os.name == "nt":
        log.warning("running on windows is not thoroughly tested")

    options += [
        "-D",
        f"NUM_INTERSECTIONS={num_intersections}",
    ]
    assert num_intersections % 4 == 0, "num_intersections must be a multiple of 4"

    sm = cp.RawModule(code=source, options=tuple(options), backend="nvcc")
    return sm


def _get_kernel_projector_module(
    num_volumes: int,
    num_materials: int,
    mesh_additive_enabled: bool,
    # mesh_additive_and_subtractive_enabled: bool,
    max_mesh_hits: int,
    mesh_layers: int,
    air_index: int,
    attenuate_outside_volume: bool = False,
) -> cp.RawModule:
    """Compile the cuda code for the kernel projector.

    Assumes `project_kernel.cu`, `kernel_vol_seg_data.cu`, and `cubic` interpolation library is in the same directory as THIS
    file.

    Args:
        num_volumes (int): The number of volumes to assume
        num_materials (int): The number of materials to assume

    Returns:
        RawModule: The compiled cuda module

    """
    # path to files for cubic interpolation (folder cubic in DeepDRR)
    d = Path(__file__).resolve().parent
    bicubic_path = str(d / "cubic")
    source_path = str(d / "project_kernel.cu")

    with open(source_path, "r") as file:
        source = file.read()

    options = []
    if os.name == "nt":
        log.warning("running on windows is not thoroughly tested")
        #     options.append("--compiler-options")
        #     options.append('"-D _WIN64"')

        # options.append("-ccbin")
        # options.append(
        #     '"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.32.31326\\bin\\Hostx64\\x64"'
        # )

    options += [
        "-D",
        f"NUM_VOLUMES={num_volumes}",
        "-D",
        f"MESH_ADDITIVE_ENABLED={int(mesh_additive_enabled)}",
        # "-D",
        # f"MESH_ADDITIVE_AND_SUBTRACTIVE_ENABLED={int(mesh_additive_and_subtractive_enabled)}",
        "-D",
        f"NUM_MATERIALS={num_materials}",
        "-D",
        f"MAX_MESH_HITS={max_mesh_hits}",
        "-D",
        f"MESH_LAYERS={mesh_layers}",
        "-D",
        f"ATTENUATE_OUTSIDE_VOLUME={int(attenuate_outside_volume)}",
        "-D",
        f"AIR_INDEX={air_index}",
        "-I",
        bicubic_path,
        "-I",
        str(d),
    ]
    log.debug(
        f"compiling {source_path} with NUM_VOLUMES={num_volumes}, NUM_MATERIALS={num_materials}"
    )

    sm = cp.RawModule(code=source, options=tuple(options), backend="nvcc")
    return sm


class Projector(object):
    volumes: List[vol.Volume]

    def __init__(
        self,
        volume: vol.Renderable | list[vol.Renderable],
        priorities: Optional[List[int]] = None,
        camera_intrinsics: Optional[geo.CameraIntrinsicTransform] = None,
        device: Optional[Device] = None,
        step: float = 0.1,
        mode: str = "linear",
        spectrum: Union[np.ndarray, str] = "90KV_AL40",
        add_scatter: Optional[bool] = None,
        scatter_num: int = 0,
        add_noise: bool = False,
        photon_count: int = 10000,
        threads: int = 8,
        max_block_index: int = 65535,
        collected_energy: bool = False,  # TODO: add unit test for this
        neglog: bool = True,
        intensity_upper_bound: Optional[float] = None,
        attenuate_outside_volume: bool = False,  # TODO: add unit tests for this, doesn't work with meshes?
        source_to_detector_distance: float = -1,
        carm: Optional[Device] = None,
        max_mesh_hits=32,
        mesh_layers=2,
        cuda_device_id=None,
    ) -> None:
        """Create the projector, which has info for simulating the DRR.

        Usage:
        ```
        with Projector(volume, materials, ...) as projector:
            for projection in projections:
                yield projector(projection)
        ```

        Args:
            volume (Union[Volume, List[Volume]]): a volume object with materials segmented, or a list of volume objects.
            priorities (List[int], optional): Denotes the 'priority level' of the volumes in projection by assigning an integer rank to each volume. At each position, volumes with lower rankings are sampled from as long
                                as they have a non-null segmentation at that location. Valid ranks are in the range [0, NUM_VOLUMES), with rank 0 having precedence over other ranks. Note that multiple volumes can share a
                                rank. If a list of ranks is provided, the ranks are associated in-order to the provided volumes.  If no list is provided (the default), the volumes are assumed to have distinct ranks, and
                                each volume has precedence over the preceding volumes. (This behavior is equivalent to passing in the list: [NUM_VOLUMES - 1, ..., 1, 0].)
            camera_intrinsics (CameraIntrinsicTransform): intrinsics of the projector's camera. (used for sensor size). If None, the CArm object must be provided and have a camera_intrinsics attribute. Defaults to None.
            device (Device, optional): Optional X-ray device object to use, which can provide a mapping from real C-arms to camera poses. If not provided, camera pose must be defined by user. Defaults to None.
            step (float, optional): size of the step along projection ray in voxels. Defaults to 0.1.
            mode (str): Interpolation mode for the kernel. Defaults to "linear".
            spectrum (Union[np.ndarray, str], optional): Spectrum array or name of spectrum to use for projection. Options are `'60KV_AL35'`, `'90KV_AL40'`, and `'120KV_AL43'`. Defaults to '90KV_AL40'.
            add_scatter (bool, optional): Whether to add scatter noise from artifacts. This is deprecated in favor of `scatter_num`. Defaults to None.
            scatter_num (int, optional): the number of photons to sue in the scatter simulation.  If zero, scatter is not simulated.
            add_noise: (bool, optional): Whether to add Poisson noise. Defaults to False.
            photon_count (int, optional): the average number of photons that hit each pixel. (The expected number of photons that hit each pixel is not uniform over each pixel because the detector is a flat panel.) Defaults to 10^4.
            threads (int, optional): Number of threads to use. Defaults to 8.
            max_block_index (int, optional): Maximum GPU block. Defaults to 1024. TODO: determine from compute capability.
            collected_energy (bool, optional): Whether to return data of "intensity" (energy deposited per photon, [keV]) or "collected energy" (energy deposited on pixel, [keV / mm^2]). Defaults to False ("intensity").
            neglog (bool, optional): whether to apply negative log transform to intensity images. If True, outputs are in range [0, 1]. Recommended for easy viewing. Defaults to False.
            intensity_upper_bound (float, optional): Maximum intensity, clipped before neglog, after noise and scatter. A good value is 40 keV / photon. Defaults to None.
            source_to_detector_distance (float, optional): If `device` is not provided, this is the distance from the source to the detector. This limits the lenght rays are traced for. Defaults to -1 (no limit).
            carm (MobileCArm, optional): Deprecated alias for `device`. See `device`.
        """

        self._egl_platform = None

        self.cuda_device_id = cuda_device_id

        self.mesh_layers = mesh_layers

        # set variables
        volume = utils.listify(volume)
        self.volumes = []
        self.priorities = []
        self.primitives = []
        self.meshes = []
        for _vol in volume:
            if isinstance(_vol, vol.Volume):
                self.volumes.append(_vol)
            elif isinstance(_vol, vol.Mesh):
                self.meshes.append(_vol)
                for prim in _vol.mesh.primitives:
                    if isinstance(prim.material, DRRMaterial):
                        self.primitives.append(prim)
                    else:
                        raise ValueError(f"unrecognized material type: {type(_vol)}.")
            else:
                raise ValueError(f"unrecognized Renderable type: {type(_vol)}.")
        log.info("volumes")
        log.info(self.volumes)
        self.mesh_additive_enabled = len(self.meshes) > 0
        # self.mesh_subtractive_enabled = False

        # for prim in self.primitives:
        #     if prim.material.subtractive:
        #         self.mesh_subtractive_enabled = True

        if priorities is None:
            self.priorities = [
                len(self.volumes) - 1 - i for i in range(len(self.volumes))
            ]
        else:
            for prio in priorities:
                assert isinstance(
                    prio, int
                ), "missing priority, or priority is not an integer"
                assert (0 <= prio) and (
                    prio < len(volume)
                ), "invalid priority outside range [0, NUM_VOLUMES)"
                self.priorities.append(prio)
        assert len(self.volumes) == len(self.priorities)

        if carm is not None:
            warnings.warn("carm is deprecated, use device instead", DeprecationWarning)
            self.device = carm
        else:
            self.device = device

        self._camera_intrinsics = camera_intrinsics

        self.step = float(step)
        self.mode = mode
        self.spectrum_arr = _get_spectrum(spectrum)
        self._source_to_detector_distance = source_to_detector_distance

        if add_scatter is not None:
            log.warning("add_scatter is deprecated. Set scatter_num instead.")
            if scatter_num != 0:
                raise ValueError("Only set scatter_num.")
            self.scatter_num = 1e7 if add_scatter else 0
        elif scatter_num < 0:
            raise ValueError(f"scatter_num must be non-negative.")
        else:
            self.scatter_num = scatter_num

        if self.scatter_num > 0 and self.device is None:
            raise ValueError("Must provide device to simulate scatter.")

        if self.scatter_num > 0:
            raise DeprecationError("Scatter is deprecated.")

        self.add_noise = add_noise
        self.photon_count = photon_count
        self.threads = threads
        self.max_block_index = max_block_index
        self.collected_energy = collected_energy
        self.neglog = neglog
        self.intensity_upper_bound = intensity_upper_bound
        # TODO (mjudish): handle intensity_upper_bound when [collected_energy is True]
        # Might want to disallow using intensity_upper_bound, due to nonsensicalness

        self.max_mesh_hits = max_mesh_hits
        if self.max_mesh_hits < 4 or self.max_mesh_hits % 4 != 0:
            raise ValueError("max_mesh_depth must be a multiple of 4 and >= 4")

        all_mats = []
        for _vol in self.volumes:
            all_mats.extend(list(_vol.materials[0].keys()))

        for _vol in self.primitives:
            all_mats.append(_vol.material.drrMatName)

        if attenuate_outside_volume:
            all_mats.append("air")

        self.all_materials = list(set(all_mats))
        self.all_materials.sort()
        log.debug(f"MATERIALS: {self.all_materials}")

        if attenuate_outside_volume:
            assert "air" in self.all_materials
            air_index = self.all_materials.index("air")
        else:
            air_index = 0

        self.air_index = air_index
        self.attenuate_outside_volume = attenuate_outside_volume

        # assertions
        for mat in self.all_materials:
            try:
                Material.from_string(mat)
            except AttributeError:
                raise ValueError(
                    f"Material {mat} not found in material database. Please check the material name."
                )

        # initialized when arrays are allocated.
        self.output_shape = None

        self.initialized = False

    @property
    def source_to_detector_distance(self) -> float:
        if self.device is not None:
            return self.device.source_to_detector_distance
        else:
            return self._source_to_detector_distance

    @property
    def camera_intrinsics(self) -> geo.CameraIntrinsicTransform:
        if self.device is not None:
            return self.device.camera_intrinsics
        elif self._camera_intrinsics is not None:
            return self._camera_intrinsics
        else:
            raise RuntimeError(
                "No device provided. Set the device attribute by passing `device=<device>` to the constructor."
            )

    @camera_intrinsics.setter
    def camera_intrinsics(self, value: geo.CameraIntrinsicTransform):
        if self.device is not None:
            raise RuntimeError(
                "Cannot set camera intrinsics when a device is provided. Use the device's camera_intrinsics instead."
            )
        elif isinstance(value, geo.CameraIntrinsicTransform):
            self._camera_intrinsics = value
        else:
            raise TypeError(
                f"Expected geo.CameraIntrinsicTransform, got {type(value)} instead."
            )

    @property
    def volume(self):
        if len(self.volumes) != 1:
            raise AttributeError(
                f"projector contains multiple volumes. Access them with `projector.volumes[i]`"
            )
        return self.volumes[0]

    @property
    def output_size(self) -> int:
        return int(np.prod(self.output_shape))

    # @time_range()
    def _prepare_project(self, camera_projections: geo.CameraProjection):
        if not self.initialized:
            raise RuntimeError("Projector has not been initialized.")

        if not camera_projections and self.device is None:
            raise ValueError(
                "must provide a camera projection object to the projector, unless imaging device (e.g. CArm) is provided"
            )
        elif not camera_projections and self.device is not None:
            camera_projections = [self.device.get_camera_projection()]
            # log.debug(
            # f"projecting with source at {camera_projections[0].center_in_world}, pointing in {self.device.principle_ray_in_world}..."
            # )
            self.max_ray_length = (
                math.sqrt(  # TODO: can these really change after construction?
                    self.device.source_to_detector_distance**2
                    + self.device.detector_height**2
                    + self.device.detector_width**2
                )
            )
        else:
            # self.max_ray_length = -1
            self.max_ray_length = self.source_to_detector_distance * 4

        return camera_projections

    # @time_range()
    def project(
        self,
        *camera_projections: geo.CameraProjection,
    ) -> np.ndarray:
        """Perform the projection.

        Args:
            camera_projection: any number of camera projections. If none are provided, the Projector uses the CArm device to obtain a camera projection.

        Raises:
            ValueError: if no projections are provided and self.device is None.

        Returns:
            np.ndarray: array of DRRs, after mass attenuation, etc.
        """

        camera_projections = self._prepare_project(camera_projections)

        intensities = []
        photon_probs = []
        for i, proj in enumerate(camera_projections):
            log.debug(
                f"Projecting and attenuating camera position {i+1} / {len(camera_projections)}"
            )
            intensity, photon_prob = self._render_single(proj)
            intensities.append(intensity)
            photon_probs.append(photon_prob)

        images = np.stack(intensities)
        photon_prob = np.stack(photon_probs)
        log.debug("Completed projection and attenuation")

        if self.add_noise:  # TODO: add tests
            log.info("adding Poisson noise")
            images = analytic_generators.add_noise(
                images, photon_prob, self.photon_count
            )

        if self.intensity_upper_bound is not None:
            images = np.clip(images, None, self.intensity_upper_bound)

        if self.neglog:
            log.debug("applying negative log transform")
            images = utils.neglog(images)

        if images.shape[0] == 1:
            return images[0]
        else:
            return images

    def _render_single(self, proj: geo.CameraProjection) -> np.ndarray:
        # Only re-allocates if the output shape has changed.
        self.initialize_output_arrays(proj.intrinsic.sensor_size)

        self._update_object_locations(proj)

        if self.mesh_additive_enabled:
            self._render_mesh(proj)

        args = [
            np.uint64(self.volumes_texobs_gpu.data.ptr),
            np.uint64(self.seg_texobs_gpu.data.ptr),
            np.int32(proj.sensor_width),  # out_width
            np.int32(proj.sensor_height),  # out_height
            np.float32(self.step),  # step
            np.uint64(self.priorities_gpu.data.ptr),  # priority
            np.uint64(self.minPointX_gpu.data.ptr),  # gVolumeEdgeMinPointX
            np.uint64(self.minPointY_gpu.data.ptr),  # gVolumeEdgeMinPointY
            np.uint64(self.minPointZ_gpu.data.ptr),  # gVolumeEdgeMinPointZ
            np.uint64(self.maxPointX_gpu.data.ptr),  # gVolumeEdgeMaxPointX
            np.uint64(self.maxPointY_gpu.data.ptr),  # gVolumeEdgeMaxPointY
            np.uint64(self.maxPointZ_gpu.data.ptr),  # gVolumeEdgeMaxPointZ
            np.uint64(self.voxelSizeX_gpu.data.ptr),  # gVoxelElementSizeX
            np.uint64(self.voxelSizeY_gpu.data.ptr),  # gVoxelElementSizeY
            np.uint64(self.voxelSizeZ_gpu.data.ptr),  # gVoxelElementSizeZ
            np.uint64(self.sourceX_gpu.data.ptr),  # sx_ijk
            np.uint64(self.sourceY_gpu.data.ptr),  # sy_ijk
            np.uint64(self.sourceZ_gpu.data.ptr),  # sz_ijk
            np.float32(self.max_ray_length),  # max_ray_length
            np.uint64(self.world_from_index_gpu.data.ptr),  # world_from_index
            np.uint64(self.ijk_from_world_gpu.data.ptr),  # ijk_from_world
            np.int32(self.spectrum_arr.shape[0]),  # n_bins
            np.uint64(self.energies_gpu.data.ptr),  # energies
            np.uint64(self.pdf_gpu.data.ptr),  # pdf
            np.uint64(self.absorption_coef_table_gpu.data.ptr),  # absorb_coef_table
            np.uint64(self.intensity_gpu.data.ptr),  # intensity
            np.uint64(self.photon_prob_gpu.data.ptr),  # photon_prob
            self.solid_angle_gpu,  # solid_angle
            np.uint64(self.mesh_hit_alphas_gpu.data.ptr),
            np.uint64(self.mesh_hit_facing_gpu.data.ptr),
            np.uint64(self.mesh_sub_layer_valid.data.ptr),
            np.uint64(self.additive_densities_gpu.data.ptr),
            np.uint64(self.prim_unique_materials_gpu.data.ptr),
            np.int32(len(self.prim_unique_materials)),
            # np.int32(self.mesh_layers),
            # np.int32(self.max_mesh_hits),
        ]

        # Calculate required blocks
        blocks_w = int(np.ceil(self.output_shape[0] / self.threads))
        blocks_h = int(np.ceil(self.output_shape[1] / self.threads))
        block = (self.threads, self.threads, 1)
        log.debug(
            f"Running: {blocks_w}x{blocks_h} blocks with {self.threads}x{self.threads} threads each"
        )

        log.debug("args: {}".format("\n".join(map(str, args))))
        if blocks_w <= self.max_block_index and blocks_h <= self.max_block_index:
            offset_w = np.int32(0)
            offset_h = np.int32(0)
            self.project_kernel(
                block=block,
                grid=(blocks_w, blocks_h),
                args=(*args, offset_w, offset_h),
            )
        else:
            raise DeprecationWarning(
                "Patchwise projection is deprecated, try increasing max_block_index and/or threads. Please raise an issue if you need this feature."
            )

            # def fast_host_to_device(d_a, a):
            #     d_a.data.copy_from(a.ctypes.data, a.nbytes)

            # def fast_device_to_host(a, d_a):
            #     a.ctypes.data.copy_from(d_a.data, a.nbytes)

        intensity = cp.asnumpy(self.intensity_gpu)
        intensity = intensity.reshape(self.output_shape)
        intensity = np.swapaxes(intensity, 0, 1).copy()

        photon_prob = cp.asnumpy(self.photon_prob_gpu)
        photon_prob = photon_prob.reshape(self.output_shape)
        photon_prob = np.swapaxes(photon_prob, 0, 1).copy()

        collected_energy_data = intensity
        if self.collected_energy:
            collected_energy_data = self._calculate_collected_energy_per_pixel(
                proj, intensity
            )

        return collected_energy_data, photon_prob

    def _update_object_locations(self, proj: geo.CameraProjection) -> None:
        world_from_index = np.array(proj.world_from_index[:-1, :]).astype(np.float32)
        self.world_from_index_gpu = cp.asarray(world_from_index)

        sourceX = np.zeros(len(self.volumes), dtype=np.float32)
        sourceY = np.zeros(len(self.volumes), dtype=np.float32)
        sourceZ = np.zeros(len(self.volumes), dtype=np.float32)

        ijk_from_world_cpu = np.zeros(len(self.volumes) * 3 * 4, dtype=np.float32)

        for vol_id, _vol in enumerate(self.volumes):
            source_ijk = np.array(
                _vol.IJK_from_world
                @ proj.center_in_world  # TODO (liam): Remove unused center arguments
            ).astype(np.float32)

            sourceX[vol_id] = source_ijk[0]
            sourceY[vol_id] = source_ijk[1]
            sourceZ[vol_id] = source_ijk[2]

            # TODO: prefer toarray() to get transform throughout
            IJK_from_world = _vol.IJK_from_world.toarray()
            ijk_from_world_cpu[
                IJK_from_world.size * vol_id : IJK_from_world.size * (vol_id + 1)
            ] = IJK_from_world.flatten()
        self.ijk_from_world_gpu = cp.asarray(ijk_from_world_cpu)

        self.sourceX_gpu = cp.asarray(sourceX)
        self.sourceY_gpu = cp.asarray(sourceY)
        self.sourceZ_gpu = cp.asarray(sourceZ)

    def _calculate_collected_energy_per_pixel(
        self, proj: geo.CameraProjection, intensity: np.ndarray
    ) -> np.ndarray:
        # transform to collected energy in keV per cm^2 (or keV per mm^2)
        assert np.array_equal(self.solid_angle_gpu, np.uint64(0)) == False
        solid_angle = cp.asnumpy(self.solid_angle_gpu).reshape(self.output_shape)
        solid_angle = np.swapaxes(solid_angle, 0, 1).copy()

        # TODO (mjudish): is this calculation valid? SDD is in [mm], what does f{x,y} measure?
        pixel_size_x = self.source_to_detector_distance / proj.index_from_camera2d.fx
        pixel_size_y = self.source_to_detector_distance / proj.index_from_camera2d.fy

        # get energy deposited by multiplying [intensity] with [number of photons to hit each pixel]
        deposited_energy = (
            np.multiply(intensity, solid_angle)
            * self.photon_count
            / np.average(solid_angle)
        )
        # convert to keV / mm^2
        deposited_energy /= pixel_size_x * pixel_size_y
        return deposited_energy

    def _setup_pyrender_scene(self, proj: geo.CameraProjection):
        for mesh_id, mesh in enumerate(self.meshes):
            self.mesh_nodes[mesh_id]._matrix = mesh.world_from_ijk

        self.cam.fx = proj.intrinsic.fx
        self.cam.fy = proj.intrinsic.fy
        self.cam.cx = proj.intrinsic.cx
        self.cam.cy = proj.intrinsic.sensor_height - proj.intrinsic.cy

        self.cam.znear = 1  # self.device.source_to_detector_distance / 1000
        self.cam.zfar = self.source_to_detector_distance * 4

        deepdrr_to_opengl_cam = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )

        self.cam_node._matrix = np.array(proj.extrinsic.inv) @ deepdrr_to_opengl_cam

        zfar = self.source_to_detector_distance * 4  # TODO (liam)

        return zfar

    def meshes_bounding_sphere_in_frustum(
        self, meshes, index_from_world: Optional[geo.CameraProjection] = None
    ) -> bool:
        camera_projections = self._prepare_project(index_from_world)
        proj = camera_projections

        self._setup_pyrender_scene(proj)

        width = proj.intrinsic.sensor_width
        height = proj.intrinsic.sensor_height

        gl_proj_matrix = self.cam.get_projection_matrix(width=width, height=height)

        ndc_top = np.array([0, 1, 1, 1])
        ndc_bottom = np.array([0, -1, 1, 1])
        ndc_left = np.array([-1, 0, 1, 1])
        ndc_right = np.array([1, 0, 1, 1])

        cam_top = np.dot(np.linalg.inv(gl_proj_matrix), ndc_top)
        cam_bottom = np.dot(np.linalg.inv(gl_proj_matrix), ndc_bottom)
        cam_left = np.dot(np.linalg.inv(gl_proj_matrix), ndc_left)
        cam_right = np.dot(np.linalg.inv(gl_proj_matrix), ndc_right)

        cam_v = cam_top - cam_bottom
        cam_h = cam_right - cam_left

        top_face_normal = np.cross(cam_h[:3], cam_top[:3])
        bottom_face_normal = np.cross(cam_bottom[:3], cam_h[:3])
        left_face_normal = np.cross(cam_v[:3], cam_left[:3])
        right_face_normal = np.cross(cam_right[:3], cam_v[:3])

        top_face_normal /= np.linalg.norm(top_face_normal)
        bottom_face_normal /= np.linalg.norm(bottom_face_normal)
        left_face_normal /= np.linalg.norm(left_face_normal)
        right_face_normal /= np.linalg.norm(right_face_normal)

        res = []
        for mesh in meshes:
            center, radius = mesh.get_loose_bounding_sphere

            # transform to world space
            center = np.array([center[0], center[1], center[2], 1])
            center = np.dot(mesh.world_from_ijk, center)

            # transform to camera space
            camspace_center = np.dot(proj.extrinsic, center)

            camspace_center = camspace_center[:3]

            camspace_center[2] = -camspace_center[2]  # flip z axis

            # if sphere is at all in frustum
            is_in_frustum = (
                np.dot(camspace_center, top_face_normal) < radius
                and np.dot(camspace_center, bottom_face_normal) < radius
                and np.dot(camspace_center, left_face_normal) < radius
                and np.dot(camspace_center, right_face_normal) < radius
            )

            res.append(is_in_frustum)

        return res

    def project_seg(
        self,
        *camera_projections: geo.CameraProjection,
        tags: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Project segmentation masks for mesh objects.

        Args:
            camera_projections: TODO
            tags: [C] list of tags to project.
            seg_node_map: TODO

        Returns:
            np.ndarray: [C, H, W] array of segmentation masks
        """
        if len(camera_projections) > 1:
            raise NotImplementedError("multiple projections")

        camera_projections = self._prepare_project(camera_projections)
        log.info(type(camera_projections))
        log.info(camera_projections)
        return self._render_seg(camera_projections[0], tags=tags)

    def _render_seg(
        self, proj: geo.CameraProjection, tags: Optional[List[str]] = None
    ) -> np.ndarray:
        zfar = self._setup_pyrender_scene(proj)
        log.info(zfar)
        log.info(proj)
        res = self._render_mesh_seg(proj, zfar, tags=tags)
        return res

    def project_hits(
        self,
        *camera_projections: geo.CameraProjection,
        tags: Optional[List[Optional[List[str]]]] = None,
    ) -> cupy.array:
        """
        For each mesh layer, compute a list of entry and exit alpha values for each pixel.
        Each pixel list will have an even number of elements and is padded by [inf] values.
        The list is sorted by closest to farthest intersection.
        For example: [entry0, exit0, entry1, exit1, inf, inf, ...].

        Args:
            camera_projections: TODO
            seg_node_map: TODO

        Returns:
            np.array: Numpy array of Float32s of shape (mesh_layers, height, width, max_mesh_hits)
        """
        if len(camera_projections) > 1:
            raise NotImplementedError("multiple projections")
        camera_projections = self._prepare_project(camera_projections)
        return self._render_hits(camera_projections[0], tags=tags)

    def _render_hits(
        self,
        proj: geo.CameraProjection,
        tags: Optional[List[Optional[List[str]]]] = None,
    ) -> cupy.array:
        zfar = self._setup_pyrender_scene(proj)
        res = []
        if tags is not None:
            for tag in tags:
                self._render_mesh_subtractive_single(
                    proj, zfar, layer_idx=0, hits_mode=True, tags=tag
                )
                res.append(
                    self.mesh_hit_alphas_gpu[0]
                    .get()
                    .reshape(
                        self.output_shape[1], self.output_shape[0], self.max_mesh_hits
                    )
                )
        return res

    def project_travel(
        self,
        *camera_projections: geo.CameraProjection,
        tags: Optional[List[Optional[List[str]]]] = None,
    ) -> cupy.array:
        if len(camera_projections) > 1:
            raise NotImplementedError("multiple projections")
        camera_projections = self._prepare_project(camera_projections)
        return self._render_distance(camera_projections[0], tags=tags)

    def _render_distance(
        self,
        proj: geo.CameraProjection,
        tags: Optional[List[Optional[List[str]]]] = None,
    ) -> cupy.array:
        zfar = self._setup_pyrender_scene(proj)
        res = []
        if tags is not None:
            for tag in tags:
                layer_idx = 0
                mat_idx = 0
                rend_out = self.gl_renderer.render(
                    self.scene,
                    drr_mode=DRRMode.DENSITY,
                    flags=RenderFlags.RGBA,
                    zfar=zfar,
                    mat=None,
                    mat_idx=0,
                    tags=tag,
                    layer_idx=layer_idx,
                    density_override=1,
                )
                out_im = rend_out[:, :, 0]
                out_im[np.abs(rend_out[:, :, 1]) > 0.01] = 0
                out_im[out_im < 0] = 0
                res.append(out_im)
        return res

    def _render_mesh(self, proj: geo.CameraProjection) -> None:
        zfar = self._setup_pyrender_scene(proj)

        self._render_mesh_additive(proj, zfar)
        self._render_mesh_subtractive(proj, zfar)
        self._subtract_from_additive(proj, zfar)
        self._transfer_additive_to_cuda(proj, zfar)

    def _render_mesh_additive(
        self, proj: geo.CameraProjection, zfar: float, density_override=None
    ) -> None:
        """
        For each mesh layer and material combination, get the ray density.

        Args:
            proj (geo.CameraProjection): The camera projection.
            zfar (float): The far clipping plane distance.
        """
        width = proj.intrinsic.sensor_width
        height = proj.intrinsic.sensor_height
        total_pixels = width * height

        for layer_idx in range(self.mesh_layers):
            for mat_idx, mat in enumerate(self.prim_unique_materials):
                self.gl_renderer.render(
                    self.scene,
                    drr_mode=DRRMode.DENSITY,
                    flags=RenderFlags.RGBA,
                    zfar=zfar,
                    mat=mat,
                    mat_idx=mat_idx,
                    layer_idx=layer_idx,
                    density_override=density_override,
                )

    def _render_mesh_seg(
        self, proj: geo.CameraProjection, zfar: float, tags: Optional[List[str]] = None
    ) -> None:
        width = proj.intrinsic.sensor_width
        height = proj.intrinsic.sensor_height
        # log.info(f"sensor size: {width}x{height}")
        total_pixels = width * height

        #     res = self.gl_renderer.render(
        #         self.scene,
        #         drr_mode=DRRMode.SEG,
        #         flags=RenderFlags.RGBA,
        #         zfar=zfar,
        #         tags=tags,
        #     )
        #     res = np.flip(res, axis=0)
        batch_size = 4
        batched = []
        for i in range(0, len(tags), batch_size):
            batched.append(tags[i : i + batch_size])
        res = []
        for batch in batched:
            res_batch = self.gl_renderer.render(
                self.scene,
                drr_mode=DRRMode.SEG,
                flags=RenderFlags.RGBA,
                zfar=zfar,
                tags=batch,
            )
            for i in range(len(batch)):
                res.append(res_batch[:, :, i])

        return res

    def _render_mesh_subtractive(
        self, proj: geo.CameraProjection, zfar: float, hits_mode: bool = False
    ) -> None:
        for layer_idx in range(self.mesh_layers):
            layer_enabled = False
            for mesh in self.meshes:
                if mesh.mesh.is_visible:
                    for prim in mesh.mesh.primitives:
                        if (
                            prim.material.layer == layer_idx
                            and prim.material.subtractive
                        ):
                            layer_enabled = True
                            break

            self.mesh_sub_layer_valid[layer_idx] = 1 if layer_enabled else 0

            if layer_enabled:
                self._render_mesh_subtractive_single(proj, zfar, layer_idx, hits_mode)

    def _render_mesh_subtractive_single(
        self,
        proj: geo.CameraProjection,
        zfar: float,
        layer_idx: int,
        hits_mode: bool,
        tags: Optional[List[str]] = None,
    ) -> None:
        width = proj.intrinsic.sensor_width
        height = proj.intrinsic.sensor_height
        total_pixels = width * height

        self.gl_renderer.render(
            self.scene,
            drr_mode=DRRMode.DIST,
            flags=RenderFlags.RGBA,
            zfar=zfar,
            layer_idx=None if hits_mode else layer_idx,
            force_all_subtract=hits_mode,
            tags=tags,
        )

        for tex_idx in range(self.gl_renderer.num_peel_passes):
            pointer_into_hit_alphas = int(
                int(self.mesh_hit_alphas_tex_gpu.data.ptr)
                + layer_idx
                * self.gl_renderer.num_peel_passes
                * total_pixels
                * 4
                * NUMBYTES_FLOAT32
                + tex_idx * total_pixels * 4 * NUMBYTES_FLOAT32
            )
            gl_tex_to_gpu(
                self.gl_renderer.subtractive_reg_ims[tex_idx],
                pointer_into_hit_alphas,
                width,
                height,
                4,
            )

        self.kernel_reorder(
            args=(
                np.uint64(
                    self.mesh_hit_alphas_tex_gpu.data.ptr
                    + layer_idx
                    * self.gl_renderer.num_peel_passes
                    * total_pixels
                    * 4
                    * NUMBYTES_FLOAT32
                ),
                np.uint64(
                    self.mesh_hit_alphas_gpu.data.ptr
                    + layer_idx
                    * self.gl_renderer.num_peel_passes
                    * total_pixels
                    * 4
                    * NUMBYTES_FLOAT32
                ),
                np.int32(total_pixels),
            ),
            block=(256, 1, 1),  # TODO (liam)
            grid=(128, 1),  # TODO (liam)
        )

        self.kernel_tide(
            args=(
                np.uint64(
                    self.mesh_hit_alphas_gpu.data.ptr
                    + layer_idx
                    * self.gl_renderer.num_peel_passes
                    * total_pixels
                    * 4
                    * NUMBYTES_FLOAT32
                ),
                np.uint64(
                    self.mesh_hit_facing_gpu.data.ptr
                    + layer_idx
                    * self.gl_renderer.num_peel_passes
                    * total_pixels
                    * 4
                    * NUMBYTES_INT8
                ),
                np.int32(total_pixels),
                np.float32(self.source_to_detector_distance * 2),
            ),
            block=(32, 1, 1),  # TODO (liam)
            grid=(2048, 1),  # TODO (liam)
        )

        if not hits_mode:
            self.kernel_reorder2(
                args=(
                    np.uint64(
                        self.mesh_hit_alphas_gpu.data.ptr
                        + layer_idx
                        * self.gl_renderer.num_peel_passes
                        * total_pixels
                        * 4
                        * NUMBYTES_FLOAT32
                    ),
                    np.uint64(
                        self.mesh_hit_alphas_tex_gpu.data.ptr
                        + layer_idx
                        * self.gl_renderer.num_peel_passes
                        * total_pixels
                        * 4
                        * NUMBYTES_FLOAT32
                    ),
                    np.int32(total_pixels),
                ),
                block=(256, 1, 1),  # TODO (liam)
                grid=(128, 1),  # TODO (liam)
            )

    def _subtract_from_additive(self, proj: geo.CameraProjection, zfar: float) -> None:
        width = proj.intrinsic.sensor_width
        height = proj.intrinsic.sensor_height
        total_pixels = width * height

        # For each layer
        for subtrahend_layer_idx in range(self.mesh_layers - 1, -1, -1):
            if not self.mesh_sub_layer_valid[subtrahend_layer_idx]:
                continue

            # For each subtractive layer pair
            num_mesh_hit_pairs = self.max_mesh_hits // 2
            for tex_idx in range(num_mesh_hit_pairs):  # TODO: only mesh peel nonzero
                # transfer self.mesh_hit_alphas_gpu to gl textures
                pointer_into_mesh_hit_alphas_tex_gpu = int(
                    int(self.mesh_hit_alphas_tex_gpu.data.ptr)
                    + subtrahend_layer_idx
                    * num_mesh_hit_pairs
                    * total_pixels
                    * 2
                    * NUMBYTES_FLOAT32
                    + tex_idx * total_pixels * 2 * NUMBYTES_FLOAT32
                )
                gl_gpu_to_tex(
                    self.gl_renderer.mesh_sub_reg_ims[tex_idx],
                    pointer_into_mesh_hit_alphas_tex_gpu,
                    width,
                    height,
                    2,
                )

                for minuend_layer_idx in range(subtrahend_layer_idx):
                    for mat_idx, mat in enumerate(self.prim_unique_materials):
                        self.gl_renderer.render(
                            self.scene,
                            drr_mode=DRRMode.MESH_SUB,
                            flags=RenderFlags.RGBA,
                            zfar=zfar,
                            mat=mat,
                            mat_idx=mat_idx,
                            layer_idx=minuend_layer_idx,
                            tex_idx=tex_idx,
                        )

    def _transfer_additive_to_cuda(
        self, proj: geo.CameraProjection, zfar: float
    ) -> None:
        width = proj.intrinsic.sensor_width
        height = proj.intrinsic.sensor_height
        total_pixels = width * height

        # transfer all the additive
        for layer_idx in range(self.mesh_layers):
            for mat_idx, mat in enumerate(self.prim_unique_materials):
                pointer_into_additive_densities = (
                    int(self.additive_densities_gpu.data.ptr)
                    + layer_idx
                    * len(self.prim_unique_materials)
                    * total_pixels
                    * 2
                    * NUMBYTES_FLOAT32
                    + mat_idx * total_pixels * 2 * NUMBYTES_FLOAT32
                )

                gl_tex_to_gpu(
                    self.gl_renderer.additive_reg_ims[
                        layer_idx * len(self.prim_unique_materials) + mat_idx
                    ],
                    pointer_into_additive_densities,
                    width,
                    height,
                    2,
                )

    def project_over_carm_range(
        self,
        phi_range: Tuple[float, float, float],
        theta_range: Tuple[float, float, float],
        degrees: bool = True,
    ) -> np.ndarray:
        """Project over a range of angles using the included CArm.

        Ignores the CArm's internal pose, except for its isocenter.

        """
        raise DeprecationError(
            "project_over_carm_range is deprecated. See README for alternatives."
        )

    def initialize_output_arrays(self, sensor_size: Tuple[int, int]) -> None:
        """Allocate arrays dependent on the output size. Frees previously allocated arrays.

        This may have to be called multiple times if the output size changes.

        """
        # TODO: only allocate if the size grows. Otherwise, reuse the existing arrays.

        if self.initialized and self.output_shape == sensor_size:
            return

        if self.initialized:
            if len(self.primitives) > 0:
                # TODO (liam)
                raise RuntimeError(
                    "Changing sensor size while using meshes is not yet supported."
                )

            self.intensity_gpu = None
            self.photon_prob_gpu = None
            if self.collected_energy:
                self.solid_angle_gpu = None

        # Changes the output size as well
        self.output_shape = sensor_size

        # allocate intensity array on GPU (4 bytes to a float32)
        self.intensity_gpu = cp.zeros(self.output_size, dtype=np.float32)
        log.debug(
            f"bytes alloc'd for {self.output_shape} self.intensity_gpu: {self.output_size * NUMBYTES_FLOAT32}"
        )

        # allocate photon_prob array on GPU (4 bytes to a float32)
        self.photon_prob_gpu = cp.zeros(self.output_size, dtype=np.float32)
        log.debug(
            f"bytes alloc'd for {self.output_shape} self.photon_prob_gpu: {self.output_size * NUMBYTES_FLOAT32}"
        )

        # allocate solid_angle array on GPU as needed (4 bytes to a float32)
        if self.collected_energy:
            self.solid_angle_gpu = cp.zeros(self.output_size, dtype=np.float32)
            log.debug(
                f"bytes alloc'd for {self.output_shape} self.solid_angle_gpu: {self.output_size * NUMBYTES_FLOAT32}"
            )
        else:
            # NULL. Don't need to do solid angle calculation
            self.solid_angle_gpu = np.uint64(0)

    def initialize(self):
        """Allocate GPU memory and transfer the volume, segmentations to GPU."""
        if self.initialized:
            raise RuntimeError("Close projector before initializing again.")

        # TODO: in this function, there are several instances of axis swaps.
        # We may want to investigate if the axis swaps are necessary.

        log.debug(f"beginning call to Projector.initialize")
        init_tick = time.perf_counter()

        width = self.camera_intrinsics.sensor_width
        height = self.camera_intrinsics.sensor_height
        total_pixels = width * height

        device_id = int(os.environ.get("EGL_DEVICE_ID", "0"))
        egl_device = egl.get_device_by_index(device_id)
        self._egl_platform = egl.EGLPlatform(
            viewport_width=width, viewport_height=height, device=egl_device
        )
        self._egl_platform.init_context()
        self._egl_platform.make_current()

        self.cupy_device = cupy.cuda.Device(self.cuda_device_id)
        self.cupy_device.__enter__()

        cp.cuda.memory._set_thread_local_allocator(None)  # TODO: what does this do?

        # self.enable_unified_memory = False
        # self.mempool = cp.cuda.MemoryPool(cp.cuda.memory.malloc_managed if self.enable_unified_memory else None)
        # self._previous_allocator = cp.cuda.memory._get_thread_local_allocator()
        # cp.cuda.memory._set_thread_local_allocator(self.mempool.malloc)

        # cupy.cuda.set_allocator(cupy.cuda.MemoryAsyncPool().malloc)

        # compile the module, moved to to initialize because it needs to happen after the context is created
        self.mod = _get_kernel_projector_module(
            len(self.volumes),
            len(self.all_materials),
            self.mesh_additive_enabled,
            # self.mesh_subtractive_enabled,
            self.max_mesh_hits,
            self.mesh_layers,
            air_index=self.air_index,
            attenuate_outside_volume=self.attenuate_outside_volume,
        )
        self.project_kernel = self.mod.get_function("projectKernel")

        self.peel_postprocess_mod = _get_kernel_peel_postprocess_module(
            num_intersections=self.max_mesh_hits
        )
        self.kernel_tide = self.peel_postprocess_mod.get_function("kernelTide")
        self.kernel_reorder = self.peel_postprocess_mod.get_function("kernelReorder")
        self.kernel_reorder2 = self.peel_postprocess_mod.get_function("kernelReorder2")

        self.volumes_texobs = []
        self.volumes_texarrs = []
        for vol_id, volume in enumerate(self.volumes):
            volume_gpu = cp.asarray(volume)  # Move volume to GPU
            volume_gpu = cp.moveaxis(volume_gpu, [0, 1, 2], [2, 1, 0])  # Adjust axes on GPU
            volume = cp.asnumpy(volume_gpu)  # Move volume back to CPU for texture creation
            volume_gpu = None  # Free GPU memory
            vol_texobj, vol_texarr = create_cuda_texture(volume)  # Create texture
            self.volumes_texarrs.append(vol_texarr)
            self.volumes_texobs.append(vol_texobj)

        self.seg_texobs = []
        self.seg_texarrs = []
        for vol_id, _vol in enumerate(self.volumes):
            # Remap segmentation indices using cupy
            label_dict_index_remapping = cp.array(
                [_vol.materials[0][k] for k in self.all_materials], dtype=cp.uint16
            ).argsort()
            # Perform remapping and axis adjustment on GPU
            segmentation_gpu = cp.asarray(_vol.materials[1])
            segmentation_gpu = label_dict_index_remapping[segmentation_gpu]
            segmentation_gpu = cp.moveaxis(segmentation_gpu.astype(cp.uint16), [0, 1, 2], [2, 1, 0])

            segmentation = cp.asnumpy(segmentation_gpu)  # Move segmentation to CPU for texture creation
            segmentation_gpu = None # Free GPU memory

            # Create CUDA texture
            combined_texobj, combined_texarr = create_cuda_texture(
                segmentation, sampling_mode="nearest", dtype=np.uint16
            )
            self.seg_texobs.append(combined_texobj)
            self.seg_texarrs.append(combined_texarr)
        
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        self.volumes_texobs_gpu = cp.array(
            [x.ptr for x in self.volumes_texobs], dtype=np.uint64
        )
        self.seg_texobs_gpu = cp.array(
            [x.ptr for x in self.seg_texobs], dtype=np.uint64
        )

        self.prim_unique_materials = list(
            set([mesh.material.drrMatName for mesh in self.primitives])
        )
        self.prim_unique_materials.sort()  # for deterministic results
        self.prim_unique_materials_indices = [
            self.all_materials.index(mat) for mat in self.prim_unique_materials
        ]
        self.prim_unique_materials_gpu = cp.array(
            self.prim_unique_materials_indices, dtype=np.int32
        )

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing segmentations: {init_tock - init_tick}"
        )

        self.scene = Scene(bg_color=[0.0, 0.0, 0.0])

        self.mesh_nodes = []
        log.info("adding meshes to self.scene")
        # log.info(self.meshes)
        for drrmesh in self.meshes:
            node = Node()
            drrmesh.mesh.originmesh = drrmesh
            self.scene.add_node(node)
            self.mesh_nodes.append(node)
            self.scene.add(drrmesh.mesh, parent_node=node)

        cam_intr = self.camera_intrinsics

        self.cam = IntrinsicsCamera(
            fx=cam_intr.fx,
            fy=cam_intr.fy,
            cx=cam_intr.cx,
            cy=cam_intr.cy,
            znear=self.source_to_detector_distance
            / 1000,  # TODO (liam) near clipping plane parameter
            zfar=self.source_to_detector_distance
            * 4,  # TODO (killeen) far clipping plane parameter
        )

        self.cam_node = self.scene.add(self.cam)

        self._renderer = Renderer(
            viewport_width=width,
            viewport_height=height,
            num_peel_passes=self.max_mesh_hits // 4,
            mesh_layers=self.max_mesh_hits,
            prim_unqiue_materials=len(self.prim_unique_materials),
        )
        self.gl_renderer = self._renderer

        self.additive_densities_gpu = cp.zeros(
            self.mesh_layers * len(self.prim_unique_materials) * total_pixels * 2,
            dtype=np.float32,
        )

        # allocate volumes' priority level on the GPU
        self.priorities_gpu = cp.zeros(len(self.volumes), dtype=np.int32)
        for vol_id, prio in enumerate(self.priorities):
            self.priorities_gpu[vol_id] = prio

        # allocate gVolumeEdge{Min,Max}Point{X,Y,Z} and gVoxelElementSize{X,Y,Z} on the GPU
        self.minPointX_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.minPointY_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.minPointZ_gpu = cp.zeros(len(self.volumes), dtype=np.float32)

        self.maxPointX_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.maxPointY_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.maxPointZ_gpu = cp.zeros(len(self.volumes), dtype=np.float32)

        self.voxelSizeX_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.voxelSizeY_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.voxelSizeZ_gpu = cp.zeros(len(self.volumes), dtype=np.float32)

        for i, _vol in enumerate(self.volumes):
            gpu_ptr_offset = NUMBYTES_FLOAT32 * i
            self.minPointX_gpu[i] = np.float32(-0.5)
            self.minPointY_gpu[i] = np.float32(-0.5)
            self.minPointZ_gpu[i] = np.float32(-0.5)

            self.maxPointX_gpu[i] = np.float32(_vol.shape[0] - 0.5)
            self.maxPointY_gpu[i] = np.float32(_vol.shape[1] - 0.5)
            self.maxPointZ_gpu[i] = np.float32(_vol.shape[2] - 0.5)
            self.voxelSizeX_gpu[i] = np.float32(_vol.spacing[0])
            self.voxelSizeY_gpu[i] = np.float32(_vol.spacing[1])
            self.voxelSizeZ_gpu[i] = np.float32(_vol.spacing[2])
        log.debug(f"gVolume information allocated and copied to GPU")

        # allocate source coord.s on GPU (4 bytes for each of {x,y,z} for each volume)
        self.sourceX_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.sourceY_gpu = cp.zeros(len(self.volumes), dtype=np.float32)
        self.sourceZ_gpu = cp.zeros(len(self.volumes), dtype=np.float32)

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing multivolume stuff: {init_tock - init_tick}"
        )

        # allocate world_from_index matrix array on GPU (3x3 array x 4 bytes per float32)
        self.world_from_index_gpu = cp.zeros(3 * 3, dtype=np.float32)

        # allocate ijk_from_world for each volume.
        self.ijk_from_world_gpu = cp.zeros(len(self.volumes) * 3 * 4, dtype=np.float32)

        # Initializes the output_shape as well.
        self.initialize_output_arrays(self.camera_intrinsics.sensor_size)

        # allocate and transfer spectrum energies (4 bytes to a float32)
        assert isinstance(self.spectrum_arr, np.ndarray)
        noncont_energies = self.spectrum_arr[:, 0].copy() / 1000
        contiguous_energies = np.ascontiguousarray(noncont_energies, dtype=np.float32)
        n_bins = contiguous_energies.shape[0]
        self.energies_gpu = cp.asarray(contiguous_energies)
        log.debug(f"bytes alloc'd for self.energies_gpu: {n_bins * NUMBYTES_FLOAT32}")

        # allocate and transfer spectrum pdf (4 bytes to a float32)
        noncont_pdf = self.spectrum_arr[:, 1] / np.sum(self.spectrum_arr[:, 1])
        contiguous_pdf = np.ascontiguousarray(noncont_pdf.copy(), dtype=np.float32)
        assert contiguous_pdf.shape == contiguous_energies.shape
        assert contiguous_pdf.shape[0] == n_bins
        self.pdf_gpu = cp.asarray(contiguous_pdf)
        log.debug(f"bytes alloc'd for self.pdf_gpu {n_bins * NUMBYTES_FLOAT32}")

        # precompute, allocate, and transfer the get_absorption_coef(energy, material) table (4 bytes to a float32)
        absorption_coef_table = np.zeros(n_bins * len(self.all_materials)).astype(
            np.float32
        )
        for bin in range(n_bins):  # , energy in enumerate(energies):
            for m, mat_name in enumerate(self.all_materials):
                absorption_coef_table[bin * len(self.all_materials) + m] = (
                    Material.from_string(mat_name)
                    .get_coefficients(contiguous_energies[bin])
                    .mu_over_rho
                )
        self.absorption_coef_table_gpu = cp.asarray(absorption_coef_table)
        log.debug(
            f"size alloc'd for self.absorption_coef_table_gpu: {n_bins * len(self.all_materials) * NUMBYTES_FLOAT32}"
        )

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing rest of primary-signal stuff: {init_tock - init_tick}"
        )

        self.mesh_sub_layer_valid = cp.zeros((self.mesh_layers), dtype=np.int8)
        self.mesh_hit_alphas_gpu = cp.zeros(
            (self.mesh_layers, total_pixels, self.max_mesh_hits), dtype=np.float32
        )
        self.mesh_hit_alphas_tex_gpu = cp.zeros(
            (self.mesh_layers, total_pixels, self.max_mesh_hits), dtype=np.float32
        )
        self.mesh_hit_facing_gpu = cp.zeros(
            (self.mesh_layers, total_pixels, self.max_mesh_hits), dtype=np.int8
        )

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing rest of stuff: {init_tock - init_tick}"
        )
        
        # Free unused GPU memory blocks by cupy
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        # Mark self as initialized.
        self.initialized = True

    def free(self):
        """Free the allocated GPU memory."""
        if self.initialized:
            self.gl_renderer.delete()

            self.volumes_texobs = None
            self.volumes_texarrs = None
            self.seg_texobs = None
            self.seg_texarrs = None

            self.mesh_sub_layer_valid = None
            self.mesh_hit_alphas_gpu = None
            self.mesh_hit_alphas_tex_gpu = None
            self.mesh_hit_facing_gpu = None
            self.additive_densities_gpu = None
            self.prim_unique_materials_gpu = None

            self.priorities_gpu = None
            self.minPointX_gpu = None
            self.minPointY_gpu = None
            self.minPointZ_gpu = None
            self.maxPointX_gpu = None
            self.maxPointY_gpu = None
            self.maxPointZ_gpu = None
            self.voxelSizeX_gpu = None
            self.voxelSizeY_gpu = None
            self.voxelSizeZ_gpu = None
            self.sourceX_gpu = None
            self.sourceY_gpu = None
            self.sourceZ_gpu = None
            self.world_from_index_gpu = None
            self.ijk_from_world_gpu = None
            self.intensity_gpu = None
            self.photon_prob_gpu = None

            if self.collected_energy:
                self.solid_angle_gpu = None

            self.energies_gpu = None
            self.pdf_gpu = None
            self.absorption_coef_table_gpu = None

            self.cupy_device.__exit__()

        self.initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, tb):
        self.free()

    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)
