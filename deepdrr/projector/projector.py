import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .. import geo, utils, vol
from ..device import MobileCArm
from . import analytic_generators, mass_attenuation, scatter, spectral_data
from .cuda_scatter_structs import (
    CudaComptonStruct,
    CudaMatMfpStruct,
    CudaPlaneSurfaceStruct,
    CudaRayleighStruct,
    CudaWoodcockStruct,
)
from .material_coefficients import material_coefficients
from .mcgpu_compton_data import COMPTON_DATA
from .mcgpu_mfp_data import MFP_DATA
from .mcgpu_rita_samplers import rita_samplers

log = logging.getLogger(__name__)

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.autoinit import context
    from pycuda.compiler import SourceModule

    pycuda_available = True
except ImportError:
    pycuda_available = False
    SourceModule = Any
    log.warning("pycuda unavailable")


NUMBYTES_INT8 = 1
NUMBYTES_INT32 = 4
NUMBYTES_FLOAT32 = 4


log.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

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

    Assumes `project_kernel.cu`, `kernel_vol_seg_data.cu`, and `cubic` interpolation library is in the same directory as THIS
    file.

    Args:
        num_volumes (int): The number of volumes to assume
        num_materials (int): The number of materials to assume

    Returns:
        SourceModule: pycuda SourceModule object.

    """
    # path to files for cubic interpolation (folder cubic in DeepDRR)
    d = Path(__file__).resolve().parent
    bicubic_path = str(d / "cubic")
    source_path = str(d / "project_kernel.cu")

    with open(source_path, "r") as file:
        source = file.read()

    log.debug(
        f"compiling {source_path} with NUM_VOLUMES={num_volumes}, NUM_MATERIALS={num_materials}"
    )
    return SourceModule(
        source,
        include_dirs=[bicubic_path, str(d)],
        no_extern_c=True,
        options=[
            "-D",
            f"NUM_VOLUMES={num_volumes}",
            "-D",
            f"NUM_MATERIALS={num_materials}",
        ],
    )


def _get_kernel_scatter_module(num_materials) -> SourceModule:
    """Compile the cuda code for the scatter simulation.

    Assumes `scatter_kernel.cu` and `scatter_header.cu` are in the same directory as THIS file.

    Returns:
        SourceModule: pycuda SourceModule object.
    """
    d = Path(__file__).resolve().parent
    source_path = str(d / "scatter_kernel.cu")

    with open(source_path, "r") as file:
        source = file.read()

    log.debug(f"compiling {source_path} with NUM_MATERIALS={num_materials}")
    return SourceModule(
        source,
        include_dirs=[str(d)],
        no_extern_c=True,
        options=["-D", f"NUM_MATERIALS={num_materials}"],
    )


class Projector(object):
    def __init__(
        self,
        volume: Union[vol.Volume, List[vol.Volume]],
        priorities: Optional[List[int]] = None,
        camera_intrinsics: Optional[geo.CameraIntrinsicTransform] = None,
        carm: Optional[MobileCArm] = None,
        step: float = 0.1,
        mode: str = "linear",
        spectrum: Union[np.ndarray, str] = "90KV_AL40",
        add_scatter: Optional[bool] = None,
        scatter_num: int = 0,
        add_noise: bool = False,
        photon_count: int = 10000,
        threads: int = 8,
        max_block_index: int = 1024,
        collected_energy: bool = False,
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
            volume (Union[Volume, List[Volume]]): a volume object with materials segmented, or a list of volume objects.
            priorities (List[int], optional): Denotes the 'priority level' of the volumes in projection by assigning an integer rank to each volume. At each position, volumes with lower rankings are sampled from as long
                                as they have a non-null segmentation at that location. Valid ranks are in the range [0, NUM_VOLUMES), with rank 0 having precedence over other ranks. Note that multiple volumes can share a
                                rank. If a list of ranks is provided, the ranks are associated in-order to the provided volumes.  If no list is provided (the default), the volumes are assumed to have distinct ranks, and
                                each volume has precedence over the preceding volumes. (This behavior is equivalent to passing in the list: [NUM_VOLUMES - 1, ..., 1, 0].)
            camera_intrinsics (CameraIntrinsicTransform): intrinsics of the projector's camera. (used for sensor size). If None, the CArm object must be provided and have a camera_intrinsics attribute. Defaults to None.
            carm (MobileCArm, optional): Optional C-arm device, for convenience which can be used to get projections from C-Arm pose. If not provided, camera pose must be defined by user. Defaults to None.
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
        """
        # set variables
        volume = utils.listify(volume)
        self.volumes = []
        self.priorities = []
        for _vol in volume:
            assert isinstance(_vol, vol.Volume)
            self.volumes.append(_vol)

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

        self.camera_intrinsics = camera_intrinsics
        if carm is not None:
            self.source_to_detector_distance = carm.source_to_detector_distance
        else:
            log.warning(
                "No way to specify source-to-detector distance without a MobileCArm parameter"
            )
            raise ValueError("No source_to_detector_distance")
        self.carm = carm
        self.step = step
        self.mode = mode
        self.spectrum = _get_spectrum(spectrum)

        if add_scatter is not None:
            log.warning("add_scatter is deprecated. Set scatter_num instead.")
            if scatter_num != 0:
                raise ValueError("Only set scatter_num.")
            self.scatter_num = 1e7 if add_scatter else 0
        elif scatter_num < 0:
            raise ValueError(f"scatter_num must be non-negative.")
        else:
            self.scatter_num = scatter_num

        self.add_noise = add_noise
        self.photon_count = photon_count
        self.threads = threads
        self.max_block_index = max_block_index
        self.collected_energy = collected_energy
        self.neglog = neglog
        self.intensity_upper_bound = intensity_upper_bound
        # TODO (mjudish): handle intensity_upper_bound when [collected_energy is True]
        # Might want to disallow using intensity_upper_bound, due to nonsensicalness

        assert len(self.volumes) > 0

        all_mats = []
        for _vol in self.volumes:
            all_mats.extend(list(_vol.materials.keys()))

        self.all_materials = list(set(all_mats))
        self.all_materials.sort()
        log.debug(f"MATERIALS: {self.all_materials}")

        # compile the module
        self.mod = _get_kernel_projector_module(
            len(self.volumes), len(self.all_materials)
        )
        self.project_kernel = self.mod.get_function("projectKernel")

        if self.scatter_num > 0:
            self.scatter_mod = _get_kernel_scatter_module(len(self.all_materials))
            self.simulate_scatter = self.scatter_mod.get_function("simulate_scatter")

            # Calculate CUDA block parameters. Number of blocks is constant, each with
            # (self.threads * self.threads) threads, so that each block has same number
            # of threads as the projection kernel.
            self.num_scatter_blocks = min(32768, self.max_block_index)
            # TODO (mjudish): discuss with killeen max_block_index and what makes sense
            # for the scatter block structure
            
            total_threads = self.num_scatter_blocks * self.threads * self.threads
            log.info(f"total threads: {total_threads}")
            self.histories_per_thread = int(np.ceil(self.scatter_num / total_threads))

            self.scatter_num = self.histories_per_thread * total_threads
            log.info(f"input scatter_num: {scatter_num}, rounded up to {self.scatter_num}\nhistories per thread: {self.histories_per_thread}")

            if len(self.volumes) > 1:
                self.resample_megavolume = self.mod.get_function("resample_megavolume")

        # assertions
        for mat in self.all_materials:
            assert mat in material_coefficients, f"unrecognized material: {mat}"

        if self.camera_intrinsics is None:
            assert self.carm is not None and hasattr(self.carm, "camera_intrinsics")
            self.camera_intrinsics = self.carm.camera_intrinsics

        self.initialized = False

    @property
    def volume(self):
        if len(self.volumes) != 1:
            raise AttributeError(
                f"projector contains multiple volumes. Access them with `projector.volumes[i]`"
            )
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
            raise ValueError(
                "must provide a camera projection object to the projector, unless imaging device (e.g. CArm) is provided"
            )
        elif not camera_projections and self.carm is not None:
            camera_projections = [self.carm.get_camera_projection()]
            log.debug(
                f"projecting with source at {camera_projections[0].center_in_world}, pointing toward isocenter at {self.carm.isocenter}..."
            )

        assert isinstance(self.spectrum, np.ndarray)

        log.info("Initiating projection and attenuation...")

        project_tick = time.perf_counter()

        intensities = []
        photon_probs = []
        for i, proj in enumerate(camera_projections):
            log.debug(
                f"Projecting and attenuating camera position {i+1} / {len(camera_projections)}"
            )

            # Get the volume min/max points in world coordinates.
            sx, sy, sz = proj.get_center_in_world()
            world_from_index = np.array(proj.world_from_index).astype(np.float32)
            cuda.memcpy_htod(self.world_from_index_gpu, world_from_index)

            for vol_id, _vol in enumerate(self.volumes):
                source_ijk = np.array(proj.get_center_in_volume(_vol)).astype(
                    np.float32
                )
                log.debug(f"source point for volume #{vol_id}: {source_ijk}")
                cuda.memcpy_htod(
                    int(self.sourceX_gpu) + int(NUMBYTES_INT32 * vol_id),
                    np.array([source_ijk[0]]),
                )
                cuda.memcpy_htod(
                    int(self.sourceY_gpu) + int(NUMBYTES_INT32 * vol_id),
                    np.array([source_ijk[1]]),
                )
                cuda.memcpy_htod(
                    int(self.sourceZ_gpu) + int(NUMBYTES_INT32 * vol_id),
                    np.array([source_ijk[2]]),
                )

                ijk_from_world = np.array(_vol.ijk_from_world).astype(np.float32)
                cuda.memcpy_htod(
                    int(self.ijk_from_world_gpu)
                    + (ijk_from_world.size * NUMBYTES_FLOAT32) * vol_id,
                    ijk_from_world,
                )

            args = [
                np.int32(proj.sensor_width),  # out_width
                np.int32(proj.sensor_height),  # out_height
                np.float32(self.step),  # step
                self.priorities_gpu,  # priority
                self.minPointX_gpu,  # gVolumeEdgeMinPointX
                self.minPointY_gpu,  # gVolumeEdgeMinPointY
                self.minPointZ_gpu,  # gVolumeEdgeMinPointZ
                self.maxPointX_gpu,  # gVolumeEdgeMaxPointX
                self.maxPointY_gpu,  # gVolumeEdgeMaxPointY
                self.maxPointZ_gpu,  # gVolumeEdgeMaxPointZ
                self.voxelSizeX_gpu,  # gVoxelElementSizeX
                self.voxelSizeY_gpu,  # gVoxelElementSizeY
                self.voxelSizeZ_gpu,  # gVoxelElementSizeZ
                np.float32(sx),  # sx
                np.float32(sy),  # sy
                np.float32(sz),  # sz
                self.sourceX_gpu,  # sx_ijk
                self.sourceY_gpu,  # sy_ijk
                self.sourceZ_gpu,  # sz_ijk
                self.world_from_index_gpu,  # world_from_index
                self.ijk_from_world_gpu,  # ijk_from_world
                np.int32(self.spectrum.shape[0]),  # n_bins
                self.energies_gpu,  # energies
                self.pdf_gpu,  # pdf
                self.absorption_coef_table_gpu,  # absorb_coef_table
                self.intensity_gpu,  # intensity
                self.photon_prob_gpu,  # photon_prob
                self.solid_angle_gpu,  # solid_angle
            ]

            # Calculate required blocks
            blocks_w = int(np.ceil(self.output_shape[0] / self.threads))
            blocks_h = int(np.ceil(self.output_shape[1] / self.threads))
            block = (self.threads, self.threads, 1)
            log.debug(
                f"Running: {blocks_w}x{blocks_h} blocks with {self.threads}x{self.threads} threads each"
            )

            if blocks_w <= self.max_block_index and blocks_h <= self.max_block_index:
                offset_w = np.int32(0)
                offset_h = np.int32(0)
                self.project_kernel(
                    *args, offset_w, offset_h, block=block, grid=(blocks_w, blocks_h)
                )
            else:
                log.debug("Running kernel patchwise")
                for w in range((blocks_w - 1) // (self.max_block_index + 1)):
                    for h in range((blocks_h - 1) // (self.max_block_index + 1)):
                        offset_w = np.int32(w * self.max_block_index)
                        offset_h = np.int32(h * self.max_block_index)
                        self.project_kernel(
                            *args,
                            offset_w,
                            offset_h,
                            block=block,
                            grid=(self.max_block_index, self.max_block_index),
                        )
                        context.synchronize()

            project_tock = time.perf_counter()
            log.debug(
                f"projection #{i}: time elapsed after call to project_kernel: {project_tock - project_tick}"
            )

            intensity = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(intensity, self.intensity_gpu)
            # transpose the axes, which previously have width on the slow dimension
            log.debug("copied intensity from gpu")
            intensity = np.swapaxes(intensity, 0, 1).copy()
            log.debug("swapped intensity")

            photon_prob = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(photon_prob, self.photon_prob_gpu)
            log.debug("copied photon_prob")
            photon_prob = np.swapaxes(photon_prob, 0, 1).copy()
            log.debug("swapped photon_prob")

            project_tock = time.perf_counter()
            log.debug(
                f"projection #{i}: time elpased after copy from kernel: {project_tock - project_tick}"
            )

            if self.scatter_num > 0:
                print("starting scatter")
                # TODO (mjudish): the resampled density never gets used in the scatter kernel
                log.info(
                    f"Starting scatter simulation, scatter_num={self.scatter_num}. Time: {time.asctime()}"
                )

                #index_from_ijk = (
                #    self.megavol_ijk_from_world @ proj.world_from_index
                #).inv
                #index_from_ijk = np.array(index_from_ijk).astype(np.float32) # 2x4 matrix
                #print(f"index_from_ijk on GPU:\n{index_from_ijk}")
                #cuda.memcpy_htod(self.index_from_ijk_gpu, index_from_ijk)
                print(f"index_from_world on GPU:\n{np.array(proj.index_from_world)}")
                cuda.memcpy_htod(self.index_from_world_gpu, np.array(proj.index_from_world))

                scatter_source_ijk = np.array(
                    self.megavol_ijk_from_world @ proj.center_in_world
                ).astype(np.float32)

                print(
                    f"np.array(self.megavol_ijk_from_world) dims:{np.array(self.megavol_ijk_from_world).shape}\n{np.array(self.megavol_ijk_from_world)}"
                )
                print(
                    f"world_from_index:\n{world_from_index}"
                )

                scatter_source_world = np.array(proj.center_in_world).astype(np.float32)

                detector_plane = scatter.get_detector_plane(
                    #np.array(self.megavol_ijk_from_world @ proj.world_from_index),
                    np.array(proj.world_from_index),
                    proj.index_from_camera2d,
                    self.source_to_detector_distance,
                    geo.Point3D.from_any(scatter_source_world),
                    self.output_shape,
                )
                detector_plane_struct = CudaPlaneSurfaceStruct(
                    detector_plane, int(self.detector_plane_gpu)
                )

                # print the detector's corners in IJK
                _tmp_corners_idx = [
                    np.array([0, 0, 1]),
                    np.array([self.output_shape[0], 0, 1]),
                    np.array([self.output_shape[0], self.output_shape[1], 1]),
                    np.array([0, self.output_shape[1], 1])
                ]
                _tmp_corner_rays_world = [proj.world_from_index @ corner for corner in _tmp_corners_idx]

                print(f"Detector corner rays in world: (0,0), (W,0), (W,H), (0, H):")
                for _corner_ray in _tmp_corner_rays_world:
                    print(f"\t{_corner_ray}")
                # end print corners

                print(f"source in world:\n\t{proj.center_in_world}")
                detector_ctr_in_world = detector_plane.surface_origin + (detector_plane.basis_1 * self.output_shape[0] * 0.5) + (detector_plane.basis_2 * self.output_shape[1] * 0.5)
                print(f"detector center in world:\n\t{detector_ctr_in_world}")
                print(f"Detector corners in world, FROM RAYS:")
                for _corner_ray in _tmp_corner_rays_world:
                    print(f"\t{proj.center_in_world + self.source_to_detector_distance * _corner_ray}")
                print(f"Detector corners in world, FROM PLANE_SURFACE:")
                for indices in _tmp_corners_idx:
                    corner = detector_plane.surface_origin + (detector_plane.basis_1 * indices[0]) + (detector_plane.basis_2 * indices[1])
                    print(f"\t{corner}")

                world_from_ijk_arr = np.array(self.megavol_ijk_from_world.inv)
                cuda.memcpy_htod(self.world_from_ijk_gpu, world_from_ijk_arr)
                #print(f"world_from_ijk_arr:\n{world_from_ijk_arr}")

                ijk_from_world_arr = np.array(self.megavol_ijk_from_world)
                cuda.memcpy_htod(self.ijk_from_world_gpu, ijk_from_world_arr)
                #print(f"ijk_from_world_arr:\n{ijk_from_world_arr}")

                E_abs_keV = 5  # E_abs == 5000 eV

                scatter_args = [
                    np.int32(proj.sensor_width),  # detector_width
                    np.int32(proj.sensor_height),  # detector_height
                    np.int32(self.histories_per_thread),  # histories_for_thread
                    self.megavol_labeled_seg_gpu,  # labeled_segmentation
                    scatter_source_ijk[0],  # sx
                    scatter_source_ijk[1],  # sy
                    scatter_source_ijk[2],  # sz
                    np.float32(self.source_to_detector_distance),  # sdd
                    np.int32(self.megavol_shape[0]),  # volume_shape_x
                    np.int32(self.megavol_shape[1]),  # volume_shape_y
                    np.int32(self.megavol_shape[2]),  # volume_shape_z
                    np.float32(-0.5),  # gVolumeEdgeMinPointX
                    np.float32(-0.5),  # gVolumeEdgeMinPointY
                    np.float32(-0.5),  # gVolumeEdgeMinPointZ
                    np.float32(self.megavol_shape[0] - 0.5),  # gVolumeEdgeMaxPointX
                    np.float32(self.megavol_shape[1] - 0.5),  # gVolumeEdgeMaxPointY
                    np.float32(self.megavol_shape[2] - 0.5),  # gVolumeEdgeMaxPointZ
                    np.float32(self.megavol_spacing[0]),  # gVoxelElementSizeX
                    np.float32(self.megavol_spacing[1]),  # gVoxelElementSizeY
                    np.float32(self.megavol_spacing[2]),  # gVoxelElementSizeZ
                    self.index_from_world_gpu,  # index_from_world
                    self.mat_mfp_structs_gpu,  # mat_mfp_arr
                    self.woodcock_struct_gpu,  # woodcock_mfp
                    self.compton_structs_gpu,  # compton_arr
                    self.rayleigh_structs_gpu,  # rayleigh_arr
                    self.detector_plane_gpu,  # detector_plane
                    self.world_from_ijk_gpu,  # world_from_ijk
                    self.ijk_from_world_gpu,  # ijk_from_world
                    np.int32(self.spectrum.shape[0]),  # n_bins
                    self.energies_gpu,  # spectrum_energies
                    self.cdf_gpu,  # spectrum_cdf
                    np.float32(E_abs_keV),  # E_abs
                    np.int32(12345),  # seed_input
                    self.scatter_deposits_gpu,  # deposited_energy
                    self.num_scattered_hits_gpu,  # num_scattered_hits
                    self.num_unscattered_hits_gpu,  # num_unscattered_hits
                ]

                # same number of threads per block as the ray-casting
                block = (self.threads * self.threads, 1, 1)

                log.info("Starting scatter simulation")
                # Call the kernel
                if self.num_scatter_blocks <= self.max_block_index:
                    print("running single call to scatter kernel")
                    self.simulate_scatter(
                        *scatter_args, block=block, grid=(self.num_scatter_blocks, 1)
                    )
                else:
                    print("running scatter kernel patchwise")
                    for i in range(int(np.ceil(self.num_scatter_blocks / self.max_block_index))):
                        blocks_left_to_run = self.num_scatter_blocks - (i * self.max_block_index)
                        blocks_for_grid = min(blocks_left_to_run, self.max_block_index)
                        self.simulate_scatter(
                            *scatter_args, block=block, grid=(blocks_for_grid, 1)
                        )
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

                # TODO TEMP -- save the scatter outputs to .npy files
                np.save("scatter_intensity", scatter_intensity)
                np.save("hits_scatter", n_sc)
                np.save("hits_primary", n_pri)
                #

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

                scatter_intensity = np.divide(
                    scatter_intensity, 1 * (0 == n_sc) + n_sc * (0 != n_sc)
                )
                # scatter_intensity now actually reflects "intensity per photon"
                log.info(
                    f"Finished scatter simulation, scatter_num={self.scatter_num}. Time: {time.asctime()}"
                )

                hits_sc = np.sum(n_sc)  # total number of recorded scatter hits
                # total number of recorded primary hits
                hits_pri = np.sum(n_pri)

                log.debug(f"hits_sc: {hits_sc}, hits_pri: {hits_pri}")
                print(f"hits_sc: {hits_sc}, hits_pri: {hits_pri}")

                f_sc = hits_sc / (hits_pri + hits_sc)
                f_pri = hits_pri / (hits_pri + hits_sc)

                ### Reasoning: prob_tot = (f_pri * prob_pri) + (f_sc * prob_sc)
                # such that: prob_tot / prob_pri = f_pri + f_sc * (prob_sc / prob_pri)
                # photon_prob *= (f_pri + f_sc * (n_sc / n_pri))

                # total intensity = (f_pri * intensity_pri) * (f_sc * intensity_sc)
                intensity = (f_pri * intensity) + (f_sc * scatter_intensity)  # / f_pri
            # end scatter calculation

            # transform to collected energy in keV per cm^2 (or keV per mm^2)
            if self.collected_energy:
                assert np.int32(0) != self.solid_angle_gpu
                solid_angle = np.empty(self.output_shape, dtype=np.float32)
                cuda.memcpy_dtoh(solid_angle, self.solid_angle_gpu)
                solid_angle = np.swapaxes(solid_angle, 0, 1).copy()

                # TODO (mjudish): is this calculation valid? SDD is in [mm], what does f{x,y} measure?
                pixel_size_x = (
                    self.source_to_detector_distance / proj.index_from_camera2d.fx
                )
                pixel_size_y = (
                    self.source_to_detector_distance / proj.index_from_camera2d.fy
                )

                # get energy deposited by multiplying [intensity] with [number of photons to hit each pixel]
                deposited_energy = (
                    np.multiply(intensity, solid_angle)
                    * self.photon_count
                    / np.average(solid_angle)
                )
                # convert to keV / mm^2
                deposited_energy /= pixel_size_x * pixel_size_y
                intensities.append(deposited_energy)
            else:
                intensities.append(intensity)
            
            photon_probs.append(photon_prob)
        # end for-loop over the projections

        images = np.stack(intensities)
        photon_prob = np.stack(photon_probs)
        log.info("Completed projection and attenuation")

        if self.add_noise:
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

        # TODO: in this function, there are several instances of axis swaps. 
        # We may want to investigate if the axis swaps are necessary.

        log.debug(f"beginning call to Projector.initialize")
        init_tick = time.perf_counter()

        # allocate and transfer the volume texture to GPU
        self.volumes_gpu = []
        self.volumes_texref = []
        for vol_id, volume in enumerate(self.volumes):
            volume = np.array(volume)
            volume = np.moveaxis(volume, [0, 1, 2], [2, 1, 0]).copy()
            vol_gpu = cuda.np_to_array(volume, order="C")
            vol_texref = self.mod.get_texref(f"volume_{vol_id}")
            cuda.bind_array_to_texref(vol_gpu, vol_texref)
            self.volumes_gpu.append(vol_gpu)
            self.volumes_texref.append(vol_texref)

        init_tock = time.perf_counter()
        log.debug(f"time elapsed after intializing volumes: {init_tock - init_tick}")

        # set the interpolation mode
        if self.mode == "linear":
            for texref in self.volumes_texref:
                texref.set_filter_mode(cuda.filter_mode.LINEAR)
        else:
            raise RuntimeError

        # List[List[segmentations]], indexing by (vol_id, material_id)
        self.segmentations_gpu = []
        # List[List[texrefs]], indexing by (vol_id, material_id)
        self.segmentations_texref = []
        for vol_id, _vol in enumerate(self.volumes):
            seg_for_vol = []
            texref_for_vol = []
            for mat_id, mat in enumerate(self.all_materials):
                seg = None
                if mat in _vol.materials:
                    seg = _vol.materials[mat]
                else:
                    seg = np.zeros(_vol.shape).astype(np.float32)
                seg_for_vol.append(
                    cuda.np_to_array(
                        np.moveaxis(seg, [0, 1, 2], [2, 1, 0]).copy(), order="C"
                    )
                )
                texref = self.mod.get_texref(f"seg_{vol_id}_{mat_id}")
                texref_for_vol.append(texref)

            for seg, texref in zip(seg_for_vol, texref_for_vol):
                cuda.bind_array_to_texref(seg, texref)
                if self.mode == "linear":
                    texref.set_filter_mode(cuda.filter_mode.LINEAR)
                else:
                    raise RuntimeError("Invalid texref filter mode")

            self.segmentations_gpu.append(seg_for_vol)
            self.segmentations_texref.append(texref)

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing segmentations: {init_tock - init_tick}"
        )

        # allocate volumes' priority level on the GPU
        self.priorities_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_INT32)
        for vol_id, prio in enumerate(self.priorities):
            cuda.memcpy_htod(
                int(self.priorities_gpu) + (NUMBYTES_INT32 * vol_id), np.int32(prio)
            )

        # allocate gVolumeEdge{Min,Max}Point{X,Y,Z} and gVoxelElementSize{X,Y,Z} on the GPU
        self.minPointX_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.minPointY_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.minPointZ_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)

        self.maxPointX_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.maxPointY_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.maxPointZ_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)

        self.voxelSizeX_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.voxelSizeY_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.voxelSizeZ_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)

        for i, _vol in enumerate(self.volumes):
            gpu_ptr_offset = NUMBYTES_FLOAT32 * i
            cuda.memcpy_htod(int(self.minPointX_gpu) + gpu_ptr_offset, np.float32(-0.5))
            cuda.memcpy_htod(int(self.minPointY_gpu) + gpu_ptr_offset, np.float32(-0.5))
            cuda.memcpy_htod(int(self.minPointZ_gpu) + gpu_ptr_offset, np.float32(-0.5))

            cuda.memcpy_htod(
                int(self.maxPointX_gpu) + gpu_ptr_offset,
                np.float32(_vol.shape[0] - 0.5),
            )
            cuda.memcpy_htod(
                int(self.maxPointY_gpu) + gpu_ptr_offset,
                np.float32(_vol.shape[1] - 0.5),
            )
            cuda.memcpy_htod(
                int(self.maxPointZ_gpu) + gpu_ptr_offset,
                np.float32(_vol.shape[2] - 0.5),
            )
            cuda.memcpy_htod(
                int(self.voxelSizeX_gpu) + gpu_ptr_offset,
                np.float32(_vol.spacing[0]),
            )
            cuda.memcpy_htod(
                int(self.voxelSizeY_gpu) + gpu_ptr_offset,
                np.float32(_vol.spacing[1]),
            )
            cuda.memcpy_htod(
                int(self.voxelSizeZ_gpu) + gpu_ptr_offset,
                np.float32(_vol.spacing[2]),
            )
        log.debug(f"gVolume information allocated and copied to GPU")

        # allocate source coord.s on GPU (4 bytes for each of {x,y,z} for each volume)
        self.sourceX_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.sourceY_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)
        self.sourceZ_gpu = cuda.mem_alloc(len(self.volumes) * NUMBYTES_FLOAT32)

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing multivolume stuff: {init_tock - init_tick}"
        )

        # allocate ijk_from_index matrix array on GPU (3x3 array x 4 bytes per float32)
        # TODO: represent the factor of "3 x 3" in a more abstracted way
        self.world_from_index_gpu = cuda.mem_alloc(3 * 3 * NUMBYTES_FLOAT32)

        # allocate ijk_from_world for each volume.
        self.ijk_from_world_gpu = cuda.mem_alloc(
            len(self.volumes) * 3 * 4 * NUMBYTES_FLOAT32
        )

        # allocate intensity array on GPU (4 bytes to a float32)
        self.intensity_gpu = cuda.mem_alloc(self.output_size * NUMBYTES_FLOAT32)
        log.debug(
            f"bytes alloc'd for self.intensity_gpu: {self.output_size * NUMBYTES_FLOAT32}"
        )

        # allocate photon_prob array on GPU (4 bytes to a float32)
        self.photon_prob_gpu = cuda.mem_alloc(self.output_size * NUMBYTES_FLOAT32)
        log.debug(
            f"bytes alloc'd for self.photon_prob_gpu: {self.output_size * NUMBYTES_FLOAT32}"
        )

        # allocate solid_angle array on GPU as needed (4 bytes to a float32)
        if self.collected_energy:
            self.solid_angle_gpu = cuda.mem_alloc(self.output_size * NUMBYTES_FLOAT32)
            log.debug(
                f"bytes alloc'd for self.solid_angle_gpu: {self.output_size * NUMBYTES_FLOAT32}"
            )
        else:
            # NULL. Don't need to do solid angle calculation
            self.solid_angle_gpu = np.int32(0)

        # allocate and transfer spectrum energies (4 bytes to a float32)
        assert isinstance(self.spectrum, np.ndarray)
        noncont_energies = self.spectrum[:, 0].copy() / 1000
        contiguous_energies = np.ascontiguousarray(noncont_energies, dtype=np.float32)
        n_bins = contiguous_energies.shape[0]
        self.energies_gpu = cuda.mem_alloc(n_bins * NUMBYTES_FLOAT32)
        cuda.memcpy_htod(self.energies_gpu, contiguous_energies)
        log.debug(f"bytes alloc'd for self.energies_gpu: {n_bins * NUMBYTES_FLOAT32}")

        # allocate and transfer spectrum pdf (4 bytes to a float32)
        noncont_pdf = self.spectrum[:, 1] / np.sum(self.spectrum[:, 1])
        contiguous_pdf = np.ascontiguousarray(noncont_pdf.copy(), dtype=np.float32)
        assert contiguous_pdf.shape == contiguous_energies.shape
        assert contiguous_pdf.shape[0] == n_bins
        self.pdf_gpu = cuda.mem_alloc(n_bins * NUMBYTES_FLOAT32)
        cuda.memcpy_htod(self.pdf_gpu, contiguous_pdf)
        log.debug(f"bytes alloc'd for self.pdf_gpu {n_bins * NUMBYTES_FLOAT32}")

        # precompute, allocate, and transfer the get_absorption_coef(energy, material) table (4 bytes to a float32)
        absorption_coef_table = np.empty(n_bins * len(self.all_materials)).astype(
            np.float32
        )
        for bin in range(n_bins):  # , energy in enumerate(energies):
            for m, mat_name in enumerate(self.all_materials):
                absorption_coef_table[
                    bin * len(self.all_materials) + m
                ] = mass_attenuation.get_absorption_coefs(
                    contiguous_energies[bin], mat_name
                )
        self.absorption_coef_table_gpu = cuda.mem_alloc(
            n_bins * len(self.all_materials) * NUMBYTES_FLOAT32
        )
        cuda.memcpy_htod(self.absorption_coef_table_gpu, absorption_coef_table)
        log.debug(
            f"size alloc'd for self.absorption_coef_table_gpu: {n_bins * len(self.all_materials) * NUMBYTES_FLOAT32}"
        )

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing rest of primary-signal stuff: {init_tock - init_tick}"
        )

        # Scatter-specific initializations

        if self.scatter_num > 0:
            if len(self.volumes) > 1:
                log.debug(f"beginning scatter resampling")
                # Combine the multiple volumes into one single volume
                x_points_world = []
                y_points_world = []
                z_points_world = []

                for _vol in self.volumes:
                    corners_ijk = [
                        geo.point(-0.5, -0.5, -0.5),
                        geo.point(-0.5, -0.5, _vol.shape[2] - 0.5),
                        geo.point(-0.5, _vol.shape[1] - 0.5, -0.5),
                        geo.point(-0.5, _vol.shape[1] - 0.5, _vol.shape[2] - 0.5),
                        geo.point(_vol.shape[0] - 0.5, -0.5, -0.5),
                        geo.point(_vol.shape[0] - 0.5, -0.5, _vol.shape[2] - 0.5),
                        geo.point(_vol.shape[0] - 0.5, _vol.shape[1] - 0.5, -0.5),
                        geo.point(
                            _vol.shape[0] - 0.5,
                            _vol.shape[1] - 0.5,
                            _vol.shape[2] - 0.5,
                        ),
                    ]

                    for ijk in corners_ijk:
                        corner = _vol.world_from_ijk @ ijk
                        x_points_world.append(corner[0])
                        y_points_world.append(corner[1])
                        z_points_world.append(corner[2])

                # The points that define the bounding box of the combined volume
                min_world_point = geo.point(
                    min(x_points_world), min(y_points_world), min(z_points_world)
                )
                max_world_point = geo.point(
                    max(x_points_world), max(y_points_world), max(z_points_world)
                )

                # TODO: make this calculation more numpy-style
                largest_spacing_x = max([_vol.spacing[0] for _vol in self.volumes])
                largest_spacing_y = max([_vol.spacing[1] for _vol in self.volumes])
                largest_spacing_z = max([_vol.spacing[2] for _vol in self.volumes])

                self.megavol_spacing = geo.vector(
                    largest_spacing_x, largest_spacing_y, largest_spacing_z
                )

                # readjust the bounding box so that the voxels fit evenly
                for axis in range(3):
                    remainder = (
                        max_world_point[axis] - min_world_point[axis]
                    ) % self.megavol_spacing[axis]
                    if remainder > 0:
                        max_world_point[axis] = (
                            max_world_point[axis]
                            + self.megavol_spacing[axis]
                            - remainder
                        )

                log.info(f"megavol spacing: {self.megavol_spacing}")

                mega_x_len = int(
                    0.01
                    + (
                        (max_world_point[0] - min_world_point[0])
                        / self.megavol_spacing[0]
                    )
                )
                mega_y_len = int(
                    0.01
                    + (
                        (max_world_point[1] - min_world_point[1])
                        / self.megavol_spacing[1]
                    )
                )
                mega_z_len = int(
                    0.01
                    + (
                        (max_world_point[2] - min_world_point[2])
                        / self.megavol_spacing[2]
                    )
                )

                self.megavol_shape = (mega_x_len, mega_y_len, mega_z_len)

                # megavol.world_from_ijk == megavol.world_from_anatomical @ megavol.anatomical_from_ijk
                # We assume that megavol.world_from_anatomical is the identity transform
                # We assume that the origin for the maegvol is voxel (0,0,0)
                # Reference the Volume class for calculation of anatomical_from_ijk
                megavol_world_from_ijk = geo.FrameTransform.from_scaling(
                    scaling=self.megavol_spacing
                )
                self.megavol_ijk_from_world = megavol_world_from_ijk.inv

                log.info(f"max_world_point: {max_world_point}")
                log.info(f"min_world_point: {min_world_point}")
                log.info(
                    f"mega_[x,y,z]_len: ({mega_x_len}, {mega_y_len}, {mega_z_len})"
                )

                # allocate megavolume data and labeled (i.e., not binary) segmentation
                self.megavol_density_gpu = cuda.mem_alloc(
                    NUMBYTES_FLOAT32 * mega_x_len * mega_y_len * mega_z_len
                )
                self.megavol_labeled_seg_gpu = cuda.mem_alloc(
                    NUMBYTES_INT8 * mega_x_len * mega_y_len * mega_z_len
                )

                # TODO: discuss whether it is stylistically fine that these are allocated
                # and freed entirely within the Projector.initialized function
                inp_priority_gpu = cuda.mem_alloc(NUMBYTES_INT32 * len(self.volumes))
                inp_voxelBoundX_gpu = cuda.mem_alloc(NUMBYTES_INT32 * len(self.volumes))
                inp_voxelBoundY_gpu = cuda.mem_alloc(NUMBYTES_INT32 * len(self.volumes))
                inp_voxelBoundZ_gpu = cuda.mem_alloc(NUMBYTES_INT32 * len(self.volumes))
                inp_ijk_from_world_gpu = cuda.mem_alloc(
                    NUMBYTES_INT32
                    * np.array(self.volumes[0].ijk_from_world).size
                    * len(self.volumes)
                )

                for vol_id, _vol in enumerate(self.volumes):
                    int_offset = NUMBYTES_INT32 * vol_id
                    arr_offset = (
                        NUMBYTES_INT32 * np.array(_vol.ijk_from_world).size * vol_id
                    )
                    cuda.memcpy_htod(
                        int(inp_priority_gpu) + int_offset,
                        np.int32(self.priorities[vol_id]),
                    )
                    cuda.memcpy_htod(
                        int(inp_voxelBoundX_gpu) + int_offset, np.int32(_vol.shape[0])
                    )
                    cuda.memcpy_htod(
                        int(inp_voxelBoundY_gpu) + int_offset, np.int32(_vol.shape[1])
                    )
                    cuda.memcpy_htod(
                        int(inp_voxelBoundZ_gpu) + int_offset, np.int32(_vol.shape[2])
                    )
                    inp_ijk_from_world = np.ascontiguousarray(
                        np.array(_vol.ijk_from_world).astype(np.float32)
                    )
                    log.debug(inp_ijk_from_world)
                    # cuda.memcpy_htod(int(inp_ijk_from_world_gpu) + arr_offset, inp_ijk_from_world)
                    cuda.memcpy_htod(
                        int(inp_ijk_from_world_gpu) + arr_offset, np.int32(12345)
                    )

                # call the resampling kernel
                # TODO: null segmentation should be assigned AIR material
                # will need to figure out how to handle the case when AIR
                # is not in self.all_materials
                resampling_args = [
                    inp_priority_gpu,
                    inp_voxelBoundX_gpu,
                    inp_voxelBoundY_gpu,
                    inp_voxelBoundZ_gpu,
                    inp_ijk_from_world_gpu,
                    np.float32(min_world_point[0]),  # mega{Min,Max}{X,Y,Z}
                    np.float32(min_world_point[1]),
                    np.float32(min_world_point[2]),
                    np.float32(max_world_point[0]),
                    np.float32(max_world_point[1]),
                    np.float32(max_world_point[2]),
                    np.float32(self.megavol_spacing[0]),  # megaVoxelSize{X,Y,Z}
                    np.float32(self.megavol_spacing[1]),
                    np.float32(self.megavol_spacing[2]),
                    np.int32(mega_x_len),
                    np.int32(mega_y_len),
                    np.int32(mega_z_len),
                    self.megavol_density_gpu,
                    self.megavol_labeled_seg_gpu,
                ]

                init_tock = time.perf_counter()
                log.debug(
                    f"resampling kernel args set. time elapsed: {init_tock - init_tick}"
                )

                # Calculate block and grid sizes: each block is a 4x4x4 cube of voxels
                block = (1, 1, 1)
                blocks_x = np.int(np.ceil(mega_x_len / block[0]))
                blocks_y = np.int(np.ceil(mega_y_len / block[1]))
                blocks_z = np.int(np.ceil(mega_z_len / block[2]))
                log.info(
                    f"Resampling: {blocks_x}x{blocks_y}x{blocks_z} blocks with {block[0]}x{block[1]}x{block[2]} threads each"
                )

                if (
                    blocks_x <= self.max_block_index
                    and blocks_y <= self.max_block_index
                    and blocks_z <= self.max_block_index
                ):
                    offset_x = np.int32(0)
                    offset_y = np.int32(0)
                    offset_z = np.int32(0)
                    self.resample_megavolume(
                        *resampling_args,
                        offset_x,
                        offset_y,
                        offset_z,
                        block=block,
                        grid=(blocks_x, blocks_y, blocks_z),
                    )
                else:
                    log.debug("Running resampling kernel patchwise")
                    for x in range((blocks_x - 1) // (self.max_block_index + 1)):
                        for y in range((blocks_y - 1) // (self.max_block_index + 1)):
                            for z in range(
                                (blocks_z - 1) // (self.max_block_index + 1)
                            ):
                                offset_x = np.int32(x * self.max_block_index)
                                offset_y = np.int32(y * self.max_block_index)
                                offset_z = np.int32(z * self.max_block_index)
                                self.resample_megavolume(
                                    *resampling_args,
                                    offset_x,
                                    offset_y,
                                    offset_z,
                                    block=block,
                                    grid=(
                                        self.max_block_index,
                                        self.max_block_index,
                                        self.max_block_index,
                                    ),
                                )
                                context.synchronize()

                inp_priority_gpu.free()
                inp_voxelBoundX_gpu.free()
                inp_voxelBoundY_gpu.free()
                inp_voxelBoundZ_gpu.free()
                inp_ijk_from_world_gpu.free()

                init_tock = time.perf_counter()
                log.debug(
                    f"time elapsed after call to resampling kernel: {init_tock - init_tick}"
                )

            else:
                self.megavol_ijk_from_world = self.volumes[0].ijk_from_world
                print(
                    f"self.volumes[0].ijk_from_world dim:{self.volumes[0].ijk_from_world.dim}\n{self.volumes[0].ijk_from_world}"
                )
                self.megavol_spacing = self.volumes[0].spacing

                mega_x_len = self.volumes[0].shape[0]
                mega_y_len = self.volumes[0].shape[1]
                mega_z_len = self.volumes[0].shape[2]
                num_voxels = mega_x_len * mega_y_len * mega_z_len

                self.megavol_shape = (mega_x_len, mega_y_len, mega_z_len)

                self.megavol_density_gpu = cuda.mem_alloc(NUMBYTES_FLOAT32 * num_voxels)
                self.megavol_labeled_seg_gpu = cuda.mem_alloc(
                    NUMBYTES_INT8 * num_voxels
                )

                # TODO: null_seg should be assigned to AIR material.
                # will need to figure out how to handle the case where
                # AIR material was not originally in self.all_materials

                # copy over from self.volumes[0] to the gpu
                labeled_seg = np.zeros(self.volumes[0].shape).astype(np.int8)
                null_seg = np.ones(self.volumes[0].shape).astype(np.int8)
                for i, mat in enumerate(self.all_materials):
                    labeled_seg = np.add(
                        labeled_seg, i * self.volumes[0].materials[mat]
                    ).astype(np.int8)
                    null_seg = np.logical_and(
                        null_seg, 
                        np.logical_not(self.volumes[0].materials[mat])
                    ).astype(np.int8)
                # a labeled_seg value of NUM_MATERIALS indicates a null segmentation
                labeled_seg = np.add(
                    labeled_seg,
                    len(self.all_materials) * null_seg
                ).astype(np.int8)
                # NOTE: axis swap not necessary because using raw array, not texture
                cuda.memcpy_htod(self.megavol_labeled_seg_gpu, labeled_seg)

                # Copy volume density info to self.megavol_density_gpu
                # NOTE: axis swap not necessary because using raw array, not texture
                cuda.memcpy_htod(self.megavol_density_gpu, self.volumes[0].data)
                
                init_tock = time.perf_counter()
                log.debug(
                    f"time elapsed after copying megavolume to GPU: {init_tock - init_tick}"
                )
            # end initialization of megavolume

            # Material MFP structs
            self.mat_mfp_struct_dict = dict()
            self.mat_mfp_structs_gpu = cuda.mem_alloc(
                len(self.all_materials) * CudaMatMfpStruct.MEMSIZE
            )
            for i, mat in enumerate(self.all_materials):
                struct_gpu_ptr = int(self.mat_mfp_structs_gpu) + (
                    i * CudaMatMfpStruct.MEMSIZE
                )
                self.mat_mfp_struct_dict[mat] = CudaMatMfpStruct(
                    MFP_DATA[mat], struct_gpu_ptr
                )

            init_tock = time.perf_counter()
            log.debug(
                f"time elapsed after intializing MFP structs: {init_tock - init_tick}"
            )

            # Woodcock MFP struct
            wc_np_arr = scatter.make_woodcock_mfp(self.all_materials)
            self.woodcock_struct_gpu = cuda.mem_alloc(CudaWoodcockStruct.MEMSIZE)
            self.woodcock_struct = CudaWoodcockStruct(
                wc_np_arr, int(self.woodcock_struct_gpu)
            )

            init_tock = time.perf_counter()
            log.debug(
                f"time elapsed after intializing Woodcock struct: {init_tock - init_tick}"
            )

            # Material Compton structs
            self.compton_struct_dict = dict()
            self.compton_structs_gpu = cuda.mem_alloc(
                len(self.all_materials) * CudaComptonStruct.MEMSIZE
            )
            for i, mat in enumerate(self.all_materials):
                struct_gpu_ptr = int(self.compton_structs_gpu) + (
                    i * CudaComptonStruct.MEMSIZE
                )
                self.compton_struct_dict[mat] = CudaComptonStruct(
                    COMPTON_DATA[mat], struct_gpu_ptr
                )

            init_tock = time.perf_counter()
            log.debug(
                f"time elapsed after intializing Compton structs: {init_tock - init_tick}"
            )

            # Material Rayleigh structs
            self.rayleigh_struct_dict = dict()
            self.rayleigh_structs_gpu = cuda.mem_alloc(
                len(self.all_materials) * CudaRayleighStruct.MEMSIZE
            )
            for i, mat in enumerate(self.all_materials):
                struct_gpu_ptr = int(self.rayleigh_structs_gpu) + (
                    i * CudaRayleighStruct.MEMSIZE
                )
                self.rayleigh_struct_dict[mat] = CudaRayleighStruct(
                    rita_samplers[mat], mat, struct_gpu_ptr
                )

            init_tock = time.perf_counter()
            log.debug(
                f"time elapsed after intializing RITA structs: {init_tock - init_tick}"
            )

            # Detector plane
            self.detector_plane_gpu = cuda.mem_alloc(CudaPlaneSurfaceStruct.MEMSIZE)

            # world_from_ijk
            self.world_from_ijk_gpu = cuda.mem_alloc(3 * 4 * NUMBYTES_FLOAT32)

            # index_from_world
            # TODO: get the factor of "2 x 4" from a more abstract source
            self.index_from_world_gpu = cuda.mem_alloc(
                2 * 4 * NUMBYTES_FLOAT32
            )  # (2, 4) array of floats

            # spectrum cdf
            n_bins = self.spectrum.shape[0]
            # spectrum_cdf = np.array([np.sum(self.spectrum[0:i+1, 1]) for i in range(n_bins)])
            # spectrum_cdf = (spectrum_cdf / np.sum(self.spectrum[:, 1])).astype(np.float32)
            spectrum_cdf = np.array(
                [np.sum(contiguous_pdf[0 : i + 1]) for i in range(n_bins)]
            )
            # log.debug(f"spectrum CDF:\n{spectrum_cdf}")
            self.cdf_gpu = cuda.mem_alloc(n_bins * NUMBYTES_FLOAT32)
            cuda.memcpy_htod(self.cdf_gpu, spectrum_cdf)

            # output
            self.scatter_deposits_gpu = cuda.mem_alloc(
                self.output_size * NUMBYTES_FLOAT32
            )
            self.num_scattered_hits_gpu = cuda.mem_alloc(
                self.output_size * NUMBYTES_INT32
            )
            self.num_unscattered_hits_gpu = cuda.mem_alloc(
                self.output_size * NUMBYTES_INT32
            )

        init_tock = time.perf_counter()
        log.debug(
            f"time elapsed after intializing rest of stuff: {init_tock - init_tick}"
        )

        # Mark self as initialized.
        self.initialized = True

    def free(self):
        """Free the allocated GPU memory."""
        if self.initialized:
            for vol_id, vol_gpu in enumerate(self.volumes_gpu):
                vol_gpu.free()
                for seg in self.segmentations_gpu[vol_id]:
                    seg.free()

            self.priorities_gpu.free()

            self.minPointX_gpu.free()
            self.minPointY_gpu.free()
            self.minPointZ_gpu.free()

            self.maxPointX_gpu.free()
            self.maxPointY_gpu.free()
            self.maxPointZ_gpu.free()

            self.voxelSizeX_gpu.free()
            self.voxelSizeY_gpu.free()
            self.voxelSizeZ_gpu.free()

            self.sourceX_gpu.free()
            self.sourceY_gpu.free()
            self.sourceZ_gpu.free()

            self.world_from_index_gpu.free()
            self.ijk_from_world_gpu.free()
            self.intensity_gpu.free()
            self.photon_prob_gpu.free()

            if self.collected_energy:
                self.solid_angle_gpu.free()

            self.energies_gpu.free()
            self.pdf_gpu.free()
            self.absorption_coef_table_gpu.free()

            if self.scatter_num > 0:
                self.megavol_density_gpu.free()
                self.megavol_labeled_seg_gpu.free()
                self.mat_mfp_structs_gpu.free()
                self.woodcock_struct_gpu.free()
                self.compton_structs_gpu.free()
                self.rayleigh_structs_gpu.free()
                self.detector_plane_gpu.free()
                self.index_from_world_gpu.free()
                self.cdf_gpu.free()
                self.scatter_deposits_gpu.free()
                self.num_scattered_hits_gpu.free()
                self.num_unscattered_hits_gpu.free()

        self.initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, tb):
        self.free()

    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)
