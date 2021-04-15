from typing import Literal, List, Union, Tuple, Optional, Dict

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
    logging.warning("pycuda unavailable")

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
        assert spectrum in spectral_data.spectrums, f"unrecognized spectrum: {spectrum}"
        return spectral_data.spectrums[spectrum]
    else:
        raise TypeError(f"unrecognized spectrum: {type(spectrum)}")


def _get_kernel_projector_module(num_materials, attenuation=True) -> SourceModule:
    """Compile the cuda code for the kernel projector.

    Assumes `project_kernel.cu` and `cubic` interpolation library is in the same directory as THIS file.

    Returns:
        SourceModule: pycuda SourceModule object.
    """
    assert pycuda_available, f"pycuda must be available to run the projector kernel"

    # path to files for cubic interpolation (folder cubic in DeepDRR)
    d = Path(__file__).resolve().parent
    bicubic_path = str(d / "cubic")
    source_path = (
        str(d / "project_kernel.cu")
        if attenuation
        else str(d / "project_kernel_no-attenuation.cu")
    )

    with open(source_path, "r") as file:
        source = file.read()

    logger.debug(f"compiling {source_path} with NUM_MATERIALS={num_materials}")
    return SourceModule(
        source,
        include_dirs=[bicubic_path],
        no_extern_c=True,
        options=["-D", f"NUM_MATERIALS={num_materials}"],
    )


class SingleProjector(object):
    initialized: bool = False

    def __init__(
        self,
        volume: vol.Volume,
        camera_intrinsics: geo.CameraIntrinsicTransform,
        step: float = 0.1,
        mode: Literal["linear"] = "linear",
        spectrum: Union[
            np.ndarray, Literal["60KV_AL35", "90KV_AL40", "120KV_AL43"]
        ] = "90KV_AL40",
        threads: int = 8,
        max_block_index: int = 1024,
        attenuation: bool = True,
    ) -> None:
        self.volume = volume
        self.camera_intrinsics = camera_intrinsics
        self.step = step
        self.mode = mode
        self.spectrum = _get_spectrum(spectrum)
        self.threads = threads
        self.max_block_index = max_block_index
        self.attenuation = attenuation

        self.num_materials = len(self.volume.materials)

        # compile the module
        # TODO: fix attenuation vs no-attenuation ugliness.
        self.mod = _get_kernel_projector_module(
            self.num_materials, attenuation=self.attenuation
        )
        self.project_kernel = self.mod.get_function("projectKernel")

        # assertions
        for mat in self.volume.materials:
            assert mat in material_coefficients, f"unrecognized material: {mat}"

    @property
    def output_shape(self) -> Tuple[int, int]:
        if self.attenuation:
            return self.camera_intrinsics.sensor_size
        else:
            return (
                self.camera_intrinsics.sensor_width,
                self.camera_intrinsics.sensor_height,
                self.num_materials,
            )

    @property
    def output_size(self) -> int:
        return int(np.prod(self.output_shape))

    def project(
        self, camera_projection: geo.CameraProjection,
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
        camera_center_in_volume = np.array(
            camera_projection.get_center_in_volume(self.volume)
        ).astype(np.float32)
        logger.debug(f"camera_center_ijk (source point): {camera_center_in_volume}")

        ijk_from_index = camera_projection.get_ray_transform(self.volume)
        logger.debug(
            "center ray: {}".format(
                ijk_from_index
                @ geo.point(self.output_shape[0] / 2, self.output_shape[1] / 2)
            )
        )

        ijk_from_index = np.array(ijk_from_index).astype(np.float32)

        # spacing
        spacing = self.volume.spacing

        # copy the projection matrix to CUDA (output array initialized to zero by the kernel)
        cuda.memcpy_htod(self.rt_kinv_gpu, ijk_from_index)

        # Make the arguments to the CUDA "projectKernel".
        if self.attenuation:
            args = [
                np.int32(camera_projection.sensor_width),  # out_width
                np.int32(camera_projection.sensor_height),  # out_height
                np.float32(self.step),  # step
                np.float32(-0.5),  # gVolumeEdgeMinPointX
                np.float32(-0.5),  # gVolumeEdgeMinPointY
                np.float32(-0.5),  # gVolumeEdgeMinPointZ
                np.float32(self.volume.shape[0] - 0.5),  # gVolumeEdgeMaxPointX
                np.float32(self.volume.shape[1] - 0.5),  # gVolumeEdgeMaxPointY
                np.float32(self.volume.shape[2] - 0.5),  # gVolumeEdgeMaxPointZ
                np.float32(spacing[0]),  # gVoxelElementSizeX
                np.float32(spacing[1]),  # gVoxelElementSizeY
                np.float32(spacing[2]),  # gVoxelElementSizeZ
                camera_center_in_volume[0],  # sx
                camera_center_in_volume[1],  # sy
                camera_center_in_volume[2],  # sz
                self.rt_kinv_gpu,  # RT_Kinv
                self.intensity_gpu,  # intensity
                self.photon_prob_gpu,  # photon_prob (or NULL)
                np.int32(self.spectrum.shape[0]),  # n_bins
                self.energies_gpu,  # energies
                self.pdf_gpu,  # pdf
                self.absorption_coef_table_gpu,  # absorb_coef_table
            ]
        else:
            args = [
                np.int32(camera_projection.sensor_width),  # out_width
                np.int32(camera_projection.sensor_height),  # out_height
                np.float32(self.step),  # step
                np.float32(-0.5),  # gVolumeEdgeMinPointX
                np.float32(-0.5),  # gVolumeEdgeMinPointY
                np.float32(-0.5),  # gVolumeEdgeMinPointZ
                np.float32(self.volume.shape[0] - 0.5),  # gVolumeEdgeMaxPointX
                np.float32(self.volume.shape[1] - 0.5),  # gVolumeEdgeMaxPointY
                np.float32(self.volume.shape[2] - 0.5),  # gVolumeEdgeMaxPointZ
                np.float32(spacing[0]),  # gVoxelElementSizeX
                np.float32(spacing[1]),  # gVoxelElementSizeY
                np.float32(spacing[2]),  # gVoxelElementSizeZ
                camera_center_in_volume[0],  # sx
                camera_center_in_volume[1],  # sy
                camera_center_in_volume[2],  # sz
                self.rt_kinv_gpu,  # RT_Kinv
                self.output_gpu,  # output
            ]

        # Calculate required blocks
        blocks_w = np.int(np.ceil(self.output_shape[0] / self.threads))
        blocks_h = np.int(np.ceil(self.output_shape[1] / self.threads))
        block = (self.threads, self.threads, 1)
        # lfkj("running:", blocks_w, "x", blocks_h, "blocks with ", self.threads, "x", self.threads, "threads")

        if blocks_w <= self.max_block_index and blocks_h <= self.max_block_index:
            offset_w = np.int32(0)
            offset_h = np.int32(0)
            self.project_kernel(
                *args, offset_w, offset_h, block=block, grid=(blocks_w, blocks_h)
            )
        else:
            # lfkj("running kernel patchwise")
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

        if self.attenuation:
            intensity = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(intensity, self.intensity_gpu)
            # transpose the axes, which previously have width on the slow dimension
            intensity = np.swapaxes(intensity, 0, 1).copy()

            photon_prob = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(photon_prob, self.photon_prob_gpu)
            photon_prob = np.swapaxes(photon_prob, 0, 1).copy()

            #
            # TODO: ask about this np.swapaxes(...) call.  It's not clear to me why it's necessary or desirable, given that
            #   we were working off of self.output_shape, which basically goes off of self.sensor_shape
            #

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
        volume = np.moveaxis(
            volume, [0, 1, 2], [2, 1, 0]
        ).copy()  # TODO: is this axis swap necessary?
        self.volume_gpu = cuda.np_to_array(volume, order="C")
        self.volume_texref = self.mod.get_texref("volume")
        cuda.bind_array_to_texref(self.volume_gpu, self.volume_texref)

        # set the (interpolation?) mode
        if self.mode == "linear":
            self.volume_texref.set_filter_mode(cuda.filter_mode.LINEAR)
        else:
            raise RuntimeError

        # allocate and transfer segmentation texture to GPU
        # TODO: remove axis swap?
        # self.segmentations_gpu = [cuda.np_to_array(seg, order='C') for mat, seg in self.volume.materials.items()]
        self.segmentations_gpu = [
            cuda.np_to_array(np.moveaxis(seg, [0, 1, 2], [2, 1, 0]).copy(), order="C")
            for mat, seg in self.volume.materials.items()
        ]
        self.segmentations_texref = [
            self.mod.get_texref(f"seg_{m}") for m, _ in enumerate(self.volume.materials)
        ]
        for seg, texref in zip(self.segmentations_gpu, self.segmentations_texref):
            cuda.bind_array_to_texref(seg, texref)
            if self.mode == "linear":
                texref.set_filter_mode(cuda.filter_mode.LINEAR)
            else:
                raise RuntimeError

        # allocate ijk_from_index matrix array on GPU (3x3 array x 4 bytes per float32)
        self.rt_kinv_gpu = cuda.mem_alloc(3 * 3 * 4)

        if self.attenuation:
            # allocate intensity array on GPU (4 bytes to a float32)
            self.intensity_gpu = cuda.mem_alloc(self.output_size * 4)
            logger.debug(
                f"bytes alloc'd for self.intensity_gpu: {self.output_size * 4}"
            )

            # allocate photon_prob array on GPU (4 bytes to a float32)
            self.photon_prob_gpu = cuda.mem_alloc(self.output_size * 4)
            logger.debug(
                f"bytes alloc'd for self.photon_prob_gpu: {self.output_size * 4}"
            )

            # allocate and transfer spectrum energies (4 bytes to a float32)
            assert isinstance(self.spectrum, np.ndarray)
            noncont_energies = self.spectrum[:, 0].copy() / 1000
            contiguous_energies = np.ascontiguousarray(
                noncont_energies, dtype=np.float32
            )
            n_bins = contiguous_energies.shape[0]
            self.energies_gpu = cuda.mem_alloc(n_bins * 4)
            cuda.memcpy_htod(self.energies_gpu, contiguous_energies)
            logger.debug(f"bytes alloc'd for self.energies_gpu: {n_bins * 4}")

            # allocate and transfer spectrum pdf (4 bytes to a float32)
            noncont_pdf = self.spectrum[:, 1] / np.sum(self.spectrum[:, 1])
            contiguous_pdf = np.ascontiguousarray(noncont_pdf.copy(), dtype=np.float32)
            assert contiguous_pdf.shape == contiguous_energies.shape
            assert contiguous_pdf.shape[0] == n_bins
            self.pdf_gpu = cuda.mem_alloc(n_bins * 4)
            cuda.memcpy_htod(self.pdf_gpu, contiguous_pdf)
            logger.debug(f"bytes alloc'd for self.pdf_gpu {n_bins * 4}")

            # precompute, allocate, and transfer the get_absorption_coef(energy, material) table (4 bytes to a float32)
            absorption_coef_table = np.empty(n_bins * self.num_materials).astype(
                np.float32
            )
            for bin in range(n_bins):  # , energy in enumerate(energies):
                for m, mat_name in enumerate(self.volume.materials):
                    absorption_coef_table[
                        bin * self.num_materials + m
                    ] = mass_attenuation.get_absorption_coefs(
                        contiguous_energies[bin], mat_name
                    )
            self.absorption_coef_table_gpu = cuda.mem_alloc(
                n_bins * self.num_materials * 4
            )
            cuda.memcpy_htod(self.absorption_coef_table_gpu, absorption_coef_table)
            logger.debug(
                f"size alloc'd for self.absorption_coef_table_gpu: {n_bins * self.num_materials * 4}"
            )
        else:
            # allocate output image array on GPU (4 bytes to a float32)
            self.output_gpu = cuda.mem_alloc(self.output_size * 4)
            logger.debug(f"bytes alloc'd for self.output_gpu {self.output_size * 4}")

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
                self.energies_gpu.free()
                self.pdf_gpu.free()
                self.absorption_coef_table_gpu.free()
            else:
                self.output_gpu.free()

        self.initialized = False


class Projector(object):
    def __init__(
        self,
        volume: Union[vol.Volume, List[vol.Volume]],
        camera_intrinsics: Optional[geo.CameraIntrinsicTransform] = None,
        carm: Optional[MobileCArm] = None,
        step: float = 0.1,
        mode: Literal["linear"] = "linear",
        spectrum: Union[
            np.ndarray, Literal["60KV_AL35", "90KV_AL40", "120KV_AL43"]
        ] = "90KV_AL40",
        add_scatter: bool = False,
        add_noise: bool = False,
        photon_count: int = 100000,
        threads: int = 8,
        max_block_index: int = 1024,
        collected_energy: bool = False,  # convert to keV / cm^2 or keV / mm^2
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
        self.spectrum = _get_spectrum(spectrum)
        self.add_scatter = add_scatter
        self.add_noise = add_noise
        self.photon_count = photon_count
        self.collected_energy = collected_energy
        self.neglog = neglog
        self.intensity_upper_bound = intensity_upper_bound

        assert len(self.volumes) > 0

        if self.camera_intrinsics is None:
            assert self.carm is not None and hasattr(self.carm, "camera_intrinsics")
            self.camera_intrinsics = self.carm.camera_intrinsics

        self.projectors = [
            SingleProjector(
                volume,
                self.camera_intrinsics,
                step=step,
                mode=mode,
                spectrum=spectrum,
                threads=threads,
                max_block_index=max_block_index,
                attenuation=len(self.volumes) == 1,
            )
            for volume in self.volumes
        ]

        # other parameters
        self.scatter_net = scatter.ScatterNet() if self.add_scatter else None

    @property
    def initialized(self):
        # Has the cuda memory been allocated?
        return np.all([p.initialized for p in self.projectors])

    @property
    def volume(self):
        if len(self.projectors) != 1:
            raise DeprecationWarning(
                f'volume is deprecated. Each projector contains multiple "SingleProjectors", which contain their own volumes.'
            )
        return self.projectors[0].volume

    def project(self, *camera_projections: geo.CameraProjection,) -> np.ndarray:
        """Perform the projection.

        Args:
            camera_projection: any number of camera projections. If none are provided, the Projector uses the CArm device to obtain a camera projection.

        Raises:
            ValueError: if no projections are provided and self.carm is None.

        Returns:
            np.ndarray: array of DRRs, after mass attenuation, etc.
        """
        if not camera_projections and self.carm is None:
            raise ValueError(
                "must provide a camera projection object to the projector, unless imaging device (e.g. CArm) is provided"
            )
        elif not camera_projections and self.carm is not None:
            camera_projections = [self.carm.get_camera_projection()]
            logger.debug(
                f"projecting with source at {camera_projections[0].center_in_world}, pointing toward isocenter at {self.carm.isocenter}..."
            )

        logger.info("Initiating projection and attenuation...")

        # TODO: handle multiple volumes more elegantly, i.e. in the kernel. (!)
        if len(self.projectors) == 1:
            projector = self.projectors[0]
            intensities = []
            photon_probs = []
            for i, proj in enumerate(camera_projections):
                logger.info(
                    f"Projecting and attenuating camera position {i+1} / {len(camera_projections)}"
                )
                intensity, photon_prob = projector.project(proj)
                intensities.append(intensity)
                photon_probs.append(photon_prob)

            images = np.stack(intensities)
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
                _forward_projections = dict(
                    (mat, outputs[:, :, :, m])
                    for m, mat in enumerate(projector.volume.materials)
                )
                # if len(set(_forward_projections.keys()).intersection(set(forward_projections.keys()))) > 0:
                #     logger.error(f'{_forward_projections.keys()}')
                #     raise NotImplementedError(f'non mutually exclusive materials in multiple volumes.')

                # TODO: this is actively terrible.
                if isinstance(projector.volume, vol.MetalVolume):
                    for mat in ["air", "soft tissue", "bone"]:
                        if mat not in forward_projections:
                            logger.warning(
                                f"existing projections does not contain material: {mat}"
                            )
                            continue
                        elif mat not in _forward_projections:
                            logger.warning(
                                f"new projections does not contain material: {mat}"
                            )
                            continue
                        forward_projections[mat] -= _forward_projections[mat]

                    if "titanium" in forward_projections:
                        forward_projections["titanium"] += _forward_projections[
                            "titanium"
                        ]
                    else:
                        forward_projections["titanium"] = _forward_projections[
                            "titanium"
                        ]
                else:
                    forward_projections.update(_forward_projections)

            logger.info(f"performing mass attenuation...")
            images, photon_prob = mass_attenuation.calculate_intensity_from_spectrum(
                forward_projections, self.spectrum
            )
            logger.info("done.")

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

        for p in self.projectors:
            p.initialize()

    def free(self):
        """Free the allocated GPU memory."""
        for p in self.projectors:
            p.free()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, type, value, tb):
        self.free()

    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)
