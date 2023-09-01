"""Volume class for CT volume.

"""

from __future__ import annotations
from typing import Any, Union, Tuple, List, Optional, Dict, Type

import logging
import numpy as np
from pathlib import Path
import nibabel as nib
from pydicom.filereader import dcmread
import nrrd
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import pyvista as pv

from .. import load_dicom
from .. import geo
from .. import utils
from ..utils import data_utils
from ..utils import mesh_utils
from ..device import Device
from ..projector.material_coefficients import material_coefficients
from .renderable import Renderable

vtk, nps, vtk_available = utils.try_import_vtk()


log = logging.getLogger(__name__)


class Volume(Renderable):
    data: np.ndarray
    materials: Dict[str, np.ndarray]
    anatomical_coordinate_system: Optional[str]

    cache_dir: Optional[Path] = None

    # TODO: The current Volume class is really a scanned volume. We should have a BaseVolume or
    # GenericVolume, which might be subclassed by tools or other types of volumes not constructed
    # from array data, e.g. for which materials is perfectly known.
    def __init__(
        self,
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        anatomical_from_IJK: Optional[geo.FrameTransform] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        anatomical_coordinate_system: Optional[str] = None,
        cache_dir: Optional[str] = None,
        config: Dict[str, Any] = dict(),
    ) -> None:
        """A deepdrr Volume object with materials segmentation and orientation in world-space.

        The recommended way to create a Volume is to load from a file using the classmethods
        `from_nifti()` or `from_nrrd()`.

        Args:
            data (np.ndarray): The density data (a 3D array).
            materials (Dict[str, np.ndarray]): material segmentation of the volume, mapping material name to binary segmentation.
            anatomical_from_IJK (geo.FrameTransform): transformation from IJK space to anatomical (RAS or LPS).
            world_from_anatomical (Optional[geo.FrameTransform], optional): transformation from the anatomical space to world coordinates. If None, assumes identity. Defaults to None.
            anatomical_coordinate_system (str, optional): String denoting the coordinate system. Either "LPS", "RAS", or None.
                This may be useful for ensuring compatibility with other data, but it is not checked or used internally (yet). Defaults to None.
            cache_dir ()
        """
        Renderable.__init__(self, anatomical_from_IJK, world_from_anatomical)
        self.data = np.array(data).astype(np.float32)
        self.materials = self._format_materials(materials)
        self.anatomical_coordinate_system = anatomical_coordinate_system
        assert self.anatomical_coordinate_system in ["LPS", "RAS", None]
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.config = config

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration of the volume. Does not include volumetric data.

        Includes any info passed into `config`.

        Returns:
            Dict[str, Any]: The configuration of the volume.
        """
        config = self.config.copy()
        config.update(
            anatomical_from_IJK=self.anatomical_from_IJK,
            world_from_anatomical=self.world_from_anatomical,
            anatomical_coordinate_system=self.anatomical_coordinate_system,
        )
        return config

    @classmethod
    def from_parameters(
        cls,
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        origin: geo.Point3D,
        spacing: Optional[geo.Vector3D] = [1, 1, 1],
        anatomical_coordinate_system: Optional[str] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        **kwargs,
    ):
        """Create a volume object with a segmentation of the materials, from parameters.

        Note that the anatomical coordinate system is not the world coordinate system (which is cartesian).

        Suggested anatomical coordinate space units is millimeters.
        A helpful introduction to the geometry is can be found [here](https://www.slicer.org/wiki/Coordinate_systems).

        Args:
            volume (np.ndarray): the volume density data.
            materials (Dict[str, np.ndarray]): mapping from material names to binary segmentation of that material.
            origin (Point3D): Location of the volume's origin in the anatomical coordinate system.
            spacing (Tuple[float, float, float], optional): Spacing of the volume in the anatomical coordinate system. Defaults to (1, 1, 1).
            anatomical_coordinate_system (Optional[str]): anatomical coordinate system convention, either "RAS" or "LPS". Defaults to None.
            world_from_anatomical (FrameTransform, optional): Optional transformation from anatomical to world coordinates.
                If None, then identity is used. Defaults to None.

        """
        origin = geo.point(origin)
        spacing = geo.vector(spacing)
        assert spacing.dim == 3

        # define anatomical_from_ijk FrameTransform
        if (
            anatomical_coordinate_system is None
            or anatomical_coordinate_system == "none"
        ):
            raise NotImplementedError
            anatomical_from_IJK = geo.FrameTransform.from_scaling(
                scaling=spacing, translation=origin
            )
        elif anatomical_coordinate_system == "LPS":
            # IJKtoLPS = LPS_from_IJK
            rotation = [
                [spacing[0], 0, 0],
                [0, 0, spacing[2]],
                [0, -spacing[1], 0],
            ]
            anatomical_from_IJK = geo.FrameTransform.from_rt(
                rotation=rotation, translation=origin
            )
        elif anatomical_coordinate_system == "RAS":
            raise NotImplementedError(
                "conversion from RAS (not hard, look at LPS example)"
            )
        else:
            raise ValueError()

        return cls(
            data=data,
            materials=materials,
            anatomical_from_ijk=anatomical_from_IJK,
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system=anatomical_coordinate_system,
            **kwargs,
        )

    @classmethod
    def from_hu(
        cls,
        hu_values: np.ndarray,
        origin: geo.Point3D,
        use_thresholding: bool = True,
        spacing: Optional[geo.Vector3D] = (1, 1, 1),
        anatomical_coordinate_system: Optional[str] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        **kwargs,
    ) -> None:
        data = cls._convert_hounsfield_to_density(hu_values)
        materials = cls._segment_materials(hu_values, use_thresholding=use_thresholding)

        return cls.from_parameters(
            data,
            materials,
            origin=origin,
            spacing=spacing,
            anatomical_coordinate_system=anatomical_coordinate_system,
            world_from_anatomical=world_from_anatomical,
            **kwargs,
        )

    @classmethod
    def _get_cache_dir(cls, cache_dir: Optional[Path] = None) -> Optional[Path]:
        """Get the cache dir used for surfaces or other things.

        Args:
            cache_dir (Optional[Path], optional): If provided separately by the user. Saved if current cache_dir is None. Defaults to None.

        Returns:
            Optional[Path]: The cache_dir, if it can be found. Created.
        """
        if cache_dir is None:
            return None

        cache_dir = Path(cache_dir).expanduser()
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        return cache_dir

    @classmethod
    def _get_cache_path_root(
        cls,
        cache_name: str,
        cache_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Get the cache path."""
        if cache_dir is None:
            return None

        cache_dir = Path(cache_dir).expanduser()
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        return cache_dir / cache_name

    @staticmethod
    def _convert_hounsfield_to_density(hu_values: np.ndarray, smooth_air: bool = False):
        # Use two linear interpolations from data: (HU,g/cm^3)
        # use for lower HU: density = 0.001029*HU + 1.03
        # use for upper HU: density = 0.0005886*HU + 1.03

        # set air densities
        if smooth_air:
            hu_values[hu_values <= -900] = -1000
        # hu_values[hu_values > 600] = 5000;
        densities = np.maximum(
            np.minimum(0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0
        )
        return densities

    @staticmethod
    def _segment_materials(
        hu_values: np.ndarray,
        use_thresholding: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Segment the materials.

        Meant for internal use, particularly to be overriden by volumes with different materials.

        Args:
            hu_values (np.ndarray): volume data in Hounsfield Units.
            use_thretholding (bool, optional): whether to segment with thresholding (true) or a DNN. Defaults to True.

        Returns:
            Dict[str, np.ndarray]: materials segmentation.
        """
        if use_thresholding:
            return load_dicom.conv_hu_to_materials_thresholding(hu_values)
        else:
            return load_dicom.conv_hu_to_materials(hu_values)

    @classmethod
    def segment_materials(
        cls,
        hu_values: np.ndarray,
        anatomical_from_ijk: geo.FrameTransform,
        use_thresholding: bool = True,
        use_cached: bool = True,
        save_cache: bool = False,
        cache_dir: Optional[Path] = None,
        cache_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Segment the materials in a volume, potentially caching.

        If cache_dir is None, then

        Args:
            hu_values (np.ndarray): volume data in Hounsfield Units.
            use_thretholding (bool, optional): whether to segment with thresholding (true) or a DNN. Defaults to True.
            use_cached (bool, optional): use the cached segmentation, if it exists. Defaults to True.
            save_cache (bool, optional): save the segmentation to cache_dir. Defaults to True.
            cache_dir (Optional[Path], optional): where to look for the segmentation cache. If None, no caching performed. Defaults to None.
            cache_name (str, optional): Name of cache file. Must be provided if use_cached or cache_dir is True. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: materials segmentation.
        """
        path_root = cls._get_cache_path_root(
            cache_name,
            cache_dir=cache_dir,
        )
        log.info(f"Cache path root: {path_root}")

        if path_root is None:
            log.info(f"segmenting materials in volume")
            materials = cls._segment_materials(
                hu_values, use_thresholding=use_thresholding
            )
            return materials

        materials_path_npz = path_root.with_suffix(".npz")
        materials_record_path = path_root.with_suffix(".json")
        materials_path_nifti = path_root.with_suffix(".nii.gz")
        if use_cached and materials_path_npz.exists():
            log.info(f"using cached materials segmentation at {materials_path_npz}")
            materials = dict(np.load(materials_path_npz))
        elif (
            use_cached
            and materials_record_path.exists()
            and materials_path_nifti.exists()
        ):
            log.info(f"using cached materials segmentation at {materials_path_nifti}")
            material_names = data_utils.load_json(materials_record_path)
            materal_data = nib.load(materials_path_nifti).get_fdata()
            materials = {}
            for name, label in material_names.items():
                materials[name] = materal_data == label
        else:
            log.info(f"segmenting materials in volume")
            materials = cls._segment_materials(
                hu_values, use_thresholding=use_thresholding
            )

            if save_cache and not materials_path_nifti.exists():
                log.debug(f"saving materials segmentation to {materials_path_nifti}")
                material_names = dict((m, i) for i, m in enumerate(materials.keys()))
                material_data = np.zeros(hu_values.shape, dtype=np.int16)
                for m in material_names:
                    material_data[materials[m]] = material_names[m]
                img = nib.Nifti1Image(material_data, geo.get_data(anatomical_from_ijk))
                nib.save(img, materials_path_nifti)
                data_utils.save_json(materials_record_path, material_names)

        return materials

    @property
    def anatomical_from_ijk(self) -> geo.FrameTransform:
        return self.anatomical_from_IJK

    @anatomical_from_ijk.setter
    def anatomical_from_ijk(self, value: geo.FrameTransform):
        self.anatomical_from_IJK = value

    @property
    def LPS_from_IJK(self) -> geo.FrameTransform:
        """Get the LPS_from_IJK transform."""
        if self.anatomical_coordinate_system == "LPS":
            return self.anatomical_from_IJK
        elif self.anatomical_coordinate_system == "RAS":
            return geo.LPS_from_RAS @ self.anatomical_from_IJK
        else:
            raise ValueError(
                f"Unknown anatomical coordinate system {self.anatomical_coordinate_system}"
            )

    @property
    def IJK_from_LPS(self) -> geo.FrameTransform:
        return self.LPS_from_IJK.inverse()

    @property
    def RAS_from_IJK(self):
        """Get the RAS_from_IJK transform."""
        if self.anatomical_coordinate_system == "RAS":
            return self.anatomical_from_IJK
        elif self.anatomical_coordinate_system == "LPS":
            return geo.RAS_from_LPS @ self.anatomical_from_IJK
        else:
            raise ValueError(
                f"Unknown anatomical coordinate system {self.anatomical_coordinate_system}"
            )

    @property
    def IJK_from_RAS(self):
        return self.RAS_from_IJK.inverse()

    def save(self, output_dir: Path, segmentation: bool = False):
        """Save the volume to disk as a nifti file.

        Args:
            output_dir (Path): a directory to save the volume and segmentations to.
            segmentation (bool, optional): if the volume is a segmentation, there's
                no need to save the materials.
        """
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # save the volume
        img = nib.Nifti1Image(
            self.data,
            geo.get_data(self.RAS_from_IJK),
        )
        nib.save(img, output_dir / "data.nii.gz")

        self.world_from_anatomical.save(output_dir / "world_from_anatomical.txt")
        if segmentation:
            return

        # Save the material segmentations
        for name, segmentation in self.materials.items():
            img = nib.Nifti1Image(
                segmentation.astype(np.int32), geo.get_data(self.RAS_from_IJK)
            )
            nib.save(img, output_dir / f"{name}.nii.gz")

    @classmethod
    def load(cls: Type[Volume], path: Path, segmentation: bool = False) -> Volume:
        """Load a volume from disk.

        Args:
            path (Path): a directory containing the volume and segmentations.
            segmentation (bool, optional): if the volume is a segmentation,
                populate the materials from that.

        Returns: Volume.
        """

        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")

        # load the volume
        img = nib.load(path / "data.nii.gz")
        data = img.get_fdata()

        if segmentation:
            materials = dict(bone=data > 0)
        else:
            # load the segmentations
            materials = {}
            for p in path.glob("*.nii.gz"):
                if p.name == "data.nii.gz":
                    continue
                materials[p.stem.split(".")[0]] = nib.load(p).get_fdata()
            if len(materials) == 0:
                raise ValueError(
                    f"No materials found in {path}. Did you mean to load a segmentation?"
                )

        world_from_anatomical = geo.F.load(path / "world_from_anatomical.txt")

        return cls(
            data=data,
            materials=materials,
            anatomical_from_IJK=geo.F(img.affine),
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system="RAS",
        )

    @classmethod
    def from_nifti(
        cls,
        path: Path,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        use_thresholding: bool = True,
        use_cached: bool = True,
        save_cache: bool = False,
        cache_dir: Optional[Path] = None,
        materials: Optional[Dict[str, np.ndarray]] = None,
        segmentation: bool = False,
        label: Union[None, int, List[int]] = None,
        density_kwargs: dict = {},
        **kwargs,
    ):
        """Load a volume from NiFti file.

        Args:
            path (Path): path to the .nii.gz file.
            use_thresholding (bool, optional): segment the materials using thresholding (faster but less accurate). Defaults to True.
            world_from_anatomical (Optional[geo.FrameTransform], optional): position the volume in world space. If None, uses identity. Defaults to None.
            use_cached (bool, optional): Use a cached segmentation if available. Defaults to True.
            cache_dir (Optional[Path], optional): Where to load/save the cached segmentation. If None, use a "cache" directory
                in the same location as the nifti file. Defaults to None.
            materials: Optional material segmentation, as a dictionary mapping material name to binary segmentation.
                If not provided, materials are segmented from the CT. Defaults to None.
                Can also provide a dictionary mapping material names to Nifti files containing the segmentations.
            segmentation (bool, optional) If the file is a segmentation file, then its "materials" correspond to a high density material (bone),
                where the values are >0. Defaults to false. Overrides provided materials.
            label: which labels to treat as solid. If None, then all nonzero labels are treated as solid. Defaults to None.
            density_kwargs: Additional kwargs passed to convert_hounsfield_to_density.

        Returns:
            Volume: A new volume object.
        """
        config = dict(
            path=path,
            use_thresholding=use_thresholding,
            use_cached=use_cached,
            save_cache=save_cache,
            cache_dir=cache_dir,
            segmentation=segmentation,
            density_kwargs=density_kwargs,
        )
        path = Path(path)

        if cache_dir is None:
            cache_dir = path.parent / "cache"

            if not cache_dir.exists():
                cache_dir.mkdir()

        log.info(f"loading NiFti volume from {path}")
        img = nib.load(path)
        if img.header.get_xyzt_units()[0] not in ["mm", "unknown"]:
            log.warning(
                f'got NifTi xyz units: {img.header.get_xyzt_units()[0]}. (Expected "mm").'
            )

        anatomical_from_IJK = geo.FrameTransform(img.affine)

        if segmentation:
            data = img.get_fdata()
            if label is None:
                seg = data > 0
            elif isinstance(label, int):
                seg = data == label
            elif isinstance(label, list):
                seg = np.isin(data, label)
            else:
                raise ValueError(f"Invalid label: {label}")
            materials = dict(bone=seg)
            data = seg.astype(np.float32)
        else:
            hu_values = img.get_fdata()
            data = cls._convert_hounsfield_to_density(hu_values, **density_kwargs)
            if materials is None:
                materials = cls.segment_materials(
                    hu_values,
                    anatomical_from_IJK,
                    use_thresholding=use_thresholding,
                    use_cached=use_cached,
                    save_cache=save_cache,
                    cache_dir=cache_dir,
                    cache_name=path.name.split(".")[0],
                )
            else:
                # Check if the materials provided are path names.
                for m in materials:
                    if isinstance(materials[m], (str, Path)):
                        if Path(materials[m]).exists():
                            materials[m] = nib.load(materials[m]).get_fdata() > 0
                        else:
                            raise ValueError(
                                f"Could not find material {m} at {materials[m]}"
                            )
                    else:
                        materials[m] = materials[m].astype(bool)

        return cls(
            data,
            materials,
            anatomical_from_IJK=anatomical_from_IJK,
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system="RAS",
            cache_dir=cache_dir,
            config=config,
            **kwargs,
        )

    @classmethod
    def from_dicom(
        cls,
        path: Path,
        use_thresholding: bool = True,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        use_cached: bool = True,
        cache_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        load a volume from a dicom file and compute the anatomical_from_ijk transform from metadata
        https://www.slicer.org/wiki/Coordinate_systems
        Args:
            path: path-like to a multi-frame dicom file. (Currently only Multi-Frame from Siemens supported)
            use_thresholding (bool, optional): segment the materials using thresholding (faster but less accurate). Defaults to True.
            world_from_anatomical (Optional[geo.FrameTransform], optional): position the volume in world space. If None, uses identity. Defaults to None.
            use_cached (bool, optional): [description]. Use a cached segmentation if available. Defaults to True.
            cache_dir (Optional[Path], optional): Where to load/save the cached segmentation. If None, use the parent dir of `path`. Defaults to None.

        Returns:
            Volume: an instance of a deepdrr volume
        """
        path = Path(path)
        stem = path.name.split(".")[0]

        if cache_dir is None:
            cache_dir = path.parent // "cache"

        # Multi-frame dicoms store all slices of a volume in one file.
        # they must specify the necessary dicom tags under
        # https://dicom.innolitics.com/ciods/enhanced-ct-image/enhanced-ct-image-multi-frame-functional-groups
        assert (
            path.is_file()
        ), "Currently only multi-frame dicoms are supported. Path must refer to a file."
        log.info(f"loading Dicom volume from {path}")

        # reading the dicom dataset object
        ds = dcmread(path)

        # slice specific tags
        frames = ds.PerFrameFunctionalGroupsSequence
        num_slices = len(frames)
        first_slice_position = np.array(
            frames[0].PlanePositionSequence[0].ImagePositionPatient
        )
        last_slice_position = np.array(
            frames[-1].PlanePositionSequence[0].ImagePositionPatient
        )

        # volume specific tags
        shared = ds.SharedFunctionalGroupsSequence[0]
        RC = (
            np.array(shared.PlaneOrientationSequence[0].ImageOrientationPatient)
            .reshape(2, 3)
            .T
        )
        PixelSpacing = np.array(shared.PixelMeasuresSequence[0].PixelSpacing)
        SliceThickness = np.array(shared.PixelMeasuresSequence[0].SliceThickness)
        offset = shared.PixelValueTransformationSequence[0].RescaleIntercept
        scale = shared.PixelValueTransformationSequence[0].RescaleSlope

        # make user aware that this is only tested on windows
        if ds.Manufacturer != "SIEMENS":
            log.warning(
                "Multi-frame loading has only been tested on Siemens Enhanced CT DICOMs."
                "Please verify everything works as expected."
            )

        # read the 'raw' data array
        raw_data = ds.pixel_array.astype(np.float32)
        hu_values = raw_data * scale + offset

        """
        EXPLANATION - indexing conventions

        According to dicom (C.7.6.3.1.4 - Pixel Data) slices are of shape (Rows, Columns) 
        => must be (j, i) indexed if we define i == horizontal and j == vertical.
        => we want to conform to the (i, j, k) layout and therefore move the axis of the data array
        """

        # convert data to our indexing convention (k, j, i) -> (j, i, k)
        hu_values = hu_values.transpose((2, 1, 0)).copy()

        # transform the volume in HU to densities
        data = load_dicom.conv_hu_to_density(hu_values)

        # obtain materials analogous to nifti
        if use_thresholding:
            materials_path = cache_dir / f"{stem}_materials_thresholding.npz"
            if use_cached and materials_path.exists():
                log.info(f"found materials segmentation at {materials_path}.")
                # TODO: recover from EOFError
                materials = dict(np.load(materials_path))
            else:
                log.info(f"segmenting materials in volume")
                materials = load_dicom.conv_hu_to_materials_thresholding(hu_values)
                np.savez(materials_path, **materials)
        else:
            materials_path = cache_dir / f"{stem}_materials.npz"
            if use_cached and materials_path.exists():
                log.info(f"found materials segmentation at {materials_path}.")
                materials = dict(np.load(materials_path))
            else:
                log.info(f"segmenting materials in volume")
                materials = load_dicom.conv_hu_to_materials(hu_values)
                np.savez(materials_path, **materials)

        """
        EXPLANATION - 3d affine transform
        
        DICOM does not offer a 3d transform to locate the voxel data in world space for historic reasons.
        However we can construct it from some related DICOM tags. See this resource for more information:
        https://nipy.org/nibabel/dicom/dicom_orientation.html
        
        Note, that we do not modify the affine transform to account for the differences in indexing, but 
        instead modified the data in memory to be in (i, j, k) order.
        """
        # construct column for index k
        k = np.array(
            (last_slice_position - first_slice_position) / (num_slices - 1)
        ).reshape(3, 1)

        # check if the calculated increment matches the SliceThickness (allow .1 millimeters deviations)
        assert np.allclose(np.abs(k[2]), SliceThickness, atol=0.1, rtol=0)

        # apply scaling to mm
        RC_scaled = RC * PixelSpacing

        # construct rotation matrix from three columns for (i, j, k)
        rot = np.hstack((RC_scaled, k))

        # construct affine matrix
        affine = np.zeros((4, 4))
        affine[:3, :3] = rot  # rotation and scaling
        affine[:3, 3] = first_slice_position  # translation
        affine[3, 3] = 1  # homogenous component

        # log affine matrix in debug mode
        log.debug(f"manually constructed affine matrix: \n{affine}")
        log.debug(
            f"volume_center_xyz : {np.mean([affine @ np.array([*data.shape, 1]), affine @ [0, 0, 0, 1]], axis=0)}"
        )

        # cast to FrameTransform
        lps_from_ijk = geo.FrameTransform(affine)

        # constructing the volume
        return cls(data, materials, lps_from_ijk, world_from_anatomical, **kwargs)

    @classmethod
    def from_nrrd(
        cls,
        path: str,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        use_thresholding: bool = True,
        use_cached: bool = True,
        cache_dir: Optional[Path] = None,
        **kwargs,
    ):
        """Load a volume from a nrrd file.

        Args:
            path (str): path to the file.
            use_thresholding (bool, optional): segment the materials using thresholding (faster but less accurate). Defaults to True.
            world_from_anatomical (Optional[geo.FrameTransform], optional): position the volume in world space. If None, uses identity. Defaults to None.
            use_cached (bool, optional): Use a cached segmentation if available. Defaults to True.
            cache_dir (Optional[Path], optional): Where to load/save the cached segmentation. If None, use the parent dir of `path`. Defaults to None.

        Returns:
            Volume: A volume formed from the NRRD.
        """
        path = Path(path)
        hu_values, header = nrrd.read(path)
        ijk_from_anatomical = np.concatenate(
            [
                header["space directions"],
                header["space origin"].reshape(-1, 1),
            ],
            axis=1,
        )
        log.debug("TODO: double check this transform.")
        anatomical_from_ijk = np.concatenate(
            [ijk_from_anatomical, [[0, 0, 0, 1]]], axis=0
        )
        data = cls._convert_hounsfield_to_density(hu_values)
        materials = cls.segment_materials(
            hu_values,
            anatomical_from_ijk=anatomical_from_ijk,
            use_thresholding=use_thresholding,
            use_cached=use_cached,
            cache_dir=cache_dir,
            cache_name=path.stem,
        )

        anatomical_coordinate_system = {
            "right-anterior-superior": "RAS",
            "left-posterior-superior": "LPS",
        }.get(header.get("space", "right-anterior-superior"))

        return cls(
            data,
            materials,
            anatomical_from_ijk,
            world_from_anatomical,
            anatomical_coordinate_system=anatomical_coordinate_system,
            **kwargs,
        )

    @classmethod
    def from_meshes(
        cls,
        voxel_size: float = 0.1,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        surfaces: List[
            Tuple[str, float, pv.PolyData]
        ] = [],  # material, density, surface
    ):
        volume_args = mesh_utils.voxelize_multisurface(
            voxel_size=voxel_size, surfaces=surfaces
        )
        return cls(
            world_from_anatomical=world_from_anatomical,
            anatomical_coordinate_system=None,
            **volume_args,
        )

    def get_bounding_box_in_world(self) -> Tuple[geo.Point3D, geo.Point3D]:
        """Get the corners of a bounding box enclosing the volume in world coordinates.

        Assumes cell-centered sampling.

        Returns:
            geo.Point3D: The lower corner of the bounding box.
            geo.Point3D: The upper corner of the bounding box.
        """
        x, y, z = np.array(self.shape) - 0.5
        corners_ijk = [
            geo.point(-0.5, -0.5, -0.5),
            geo.point(-0.5, -0.5, z),
            geo.point(-0.5, y, -0.5),
            geo.point(-0.5, y, z),
            geo.point(x, -0.5, -0.5),
            geo.point(x, -0.5, z),
            geo.point(x, y, -0.5),
            geo.point(x, y, z),
        ]

        corners = np.array([np.array(self.world_from_ijk @ p) for p in corners_ijk])
        min_corner = geo.point(corners.min(0))
        max_corner = geo.point(corners.max(0))
        return min_corner, max_corner

    @property
    def spacing(self) -> geo.Vector3D:
        """The spacing of the voxels."""
        return geo.vector(np.abs(np.array(self.anatomical_from_ijk.R)).max(axis=0))

    def _format_materials(
        self,
        materials: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Standardize the input material segmentation."""
        for mat in materials:
            materials[mat] = np.array(materials[mat]).astype(np.float32)

        return materials

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def __array__(self) -> np.ndarray:
        return self.data

    def place_center(self, x: geo.Point3D) -> None:
        """Translate the volume so that its center is located at world-space point x.

        Only changes the translation elements of the world_from_anatomical transform. Preserves the current rotation of the

        Args:
            x (geo.Point3D): the world-space point.

        """

        x = geo.point(x)
        center_anatomical = self.anatomical_from_ijk @ geo.point(
            np.array(self.shape) / 2
        )
        center_world = self.world_from_anatomical @ center_anatomical
        self.place(center_anatomical, x)

    translate_center_to = place_center

    def place(
        self, point_in_anatomical: geo.Point3D, desired_point_in_world: geo.Point3D
    ) -> None:
        """Translate the volume so that x_in_anatomical corresponds to x_in_world."""
        p_A = np.array(point_in_anatomical)
        p_W = np.array(desired_point_in_world)
        r_WA = self.world_from_anatomical.R
        t_WA = p_W - r_WA @ p_A
        self.world_from_anatomical.t = t_WA  # fancy setter

    def copy_pose(self, other: Volume) -> None:
        """Copy the pose of another volume."""
        self.world_from_anatomical = other.world_from_anatomical.copy()

    def translate(self, t: geo.Vector3D) -> Volume:
        """Translate the volume by `t`.

        Args:
            t (geo.Vector3D): The vector to translate by, in world space.
        """
        t = geo.vector(t)
        T = geo.FrameTransform.from_translation(t)
        self.world_from_anatomical = T @ self.world_from_anatomical
        return self

    def rotate(
        self,
        rotation: Union[geo.Vector3D, Rotation],
        center: Optional[geo.Point3D] = None,
    ) -> Volume:
        """Rotate the volume by `rotation` about `center`.

        Args:
            rotation (Union[geo.Vector3D, Rotation]): the rotation in world-space. If it is a vector, `Rotation.from_rotvec(rotation)` is used.
            center (geo.Point3D, optional): the center of rotation in world space coordinates. If None, the center of the volume is used.
        """

        if isinstance(rotation, Rotation):
            R = geo.FrameTransform.from_rotation(rotation.as_matrix())
        else:
            r = geo.vector(rotation)
            R = geo.FrameTransform.from_rotation(Rotation.from_rotvec(r).as_matrix())

        if center is None:
            center = self.center_in_world

        T = geo.FrameTransform.from_translation(center)
        self.world_from_anatomical = T @ R @ T.inv @ self.world_from_anatomical
        return self

    def supine(self):
        """Turns the volume to be face up.

        This aligns the patient so that, in world space,
        the anterior side is toward +Z, inferior is toward +X,
        and left is toward +Y.

        Raises:
            NotImplementedError: If the anatomical coordinate system is not "RAS".

        """
        if self.anatomical_coordinate_system == "RAS":
            self.world_from_anatomical = geo.FrameTransform.from_rt(
                rotation=Rotation.from_euler("xz", [90, -90], degrees=True)
                .as_matrix()
                .squeeze(),
            )
        else:
            raise NotImplementedError

    faceup = supine

    def prone(self):
        """Turns the volume to be face down.

        This aligns the patient so that, in world space,
        the posterior side is toward +Z, inferior is toward +X,
        and right is toward +Y.

        Raises:
            NotImplementedError: If the anatomical coordinate system is not "RAS".

        """
        if self.anatomical_coordinate_system == "RAS":
            self.world_from_anatomical = geo.FrameTransform.from_rt(
                rotation=Rotation.from_euler("xz", [-90, 90], degrees=True)
                .as_matrix()
                .squeeze(),
            )
        else:
            raise NotImplementedError

    facedown = prone

    def orient_patient(
        self,
        head_first: bool = True,
        supine: bool = True,
        world_from_device: Optional[geo.FrameTransform] = None,
    ) -> None:
        """Orient the patient with the given orientation, aligning with the Loop-X coordinates.

        Args:
            head_first: If True, the patient is oriented with head (superior axis) pointing in the -Y direction. Defaults to True.
            supine: If True, the patient is oriented so that the anterior axis (stomach) points toward +Z. Defaults to True.
        """

        # R for the RAS_from_world
        if head_first and supine:
            # R <- x, A <- z, S <- -y
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        elif head_first and not supine:
            # R <- -x, A <- -z, S <- -y
            R = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        elif not head_first and supine:
            # R <- -x, A <- z, S <- y
            R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        elif not head_first and not supine:
            # R <- x, A <- -z, S <- y
            R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        else:
            raise ValueError("Invalid patient orientation.")

        if world_from_device is None:
            # Invert the rotation to go from RAS to world, keep the same translation
            self.world_from_anatomical = geo.FrameTransform.from_rt(
                R.T, self.world_from_anatomical.t
            )
        else:
            device_from_anatomical = geo.FrameTransform.from_rt(
                R.T, self.world_from_anatomical.t
            )
            self.world_from_anatomical = world_from_device @ device_from_anatomical

    def interpolate(self, *x: geo.Point3D, method: str = "linear") -> np.ndarray:
        """Interpolate the value of the volume at the point.

        This is a *slow* version of interpolation, using scipy under the hood. DeepDRR uses cubic
        spline interpolation on the GPU for rendering. This function is provided as a convenience.

        Args:
            x (geo.Point3D): The point or points in world-space.
            method (str): The interpolation method to be used.
                Accepted values are "linear" and "nearest".
                Defaults to "linear."

        Returns:
            Union[float, np.ndarray]: The interpolated value(s) of the point(s)
                in the Volume. If a point is outside the volume, the value is NaN.
        """
        if not hasattr(self, "_interpolator"):
            self._interpolator = RegularGridInterpolator(
                (range(self.shape[0]), range(self.shape[1]), range(self.shape[2])),
                self.data,
                bounds_error=False,
                fill_value=0,
            )

        ps = np.array([self.ijk_from_world @ geo.point(p) for p in x])
        out = self._interpolator(ps, method=method)
        if out.shape[0] == 1:
            return float(out[0])
        else:
            return out

    def __contains__(self, x: geo.Point3D) -> bool:
        """Determine whether the point x is inside the volume.

        Args:
            x (geo.Point3D)world': [-1346.4464, -65.99151, 8.187973], 'fractured': False,
                             'cortical_breach': 'TODO'}: world-space point.

        """
        x_ijk = self.ijk_from_world @ geo.point(x)
        return np.all(0 <= np.array(x_ijk) <= np.array(self.shape) - 1)

    def isosurface(
        self,
        value: float = 0.5,
        label: Optional[int] = None,
        node_centered: bool = True,
        smooth: bool = True,
        decimation: float = 0.01,
        decimation_points: Optional[int] = None,
        smooth_iter: int = 200,
        relaxation_factor: float = 0.25,
        convert_to_LPS: bool = False,
    ) -> pv.PolyData:
        """Make an isosurface from the volume's data, transforming to anatomical_coordinates.

        Accepts arguments passed to :func:`deepdrr.utils.mesh_utils.isosurface`.

        Args:
            value (float): The value at which to make the isosurface.
            label (int): The label of the isosurface.
            node_centered (bool): If True, the isosurface is centered at the node.
                If False, the isosurface is centered at the cell.
            smooth (bool): If True, the isosurface is smoothed.
            decimation (float): The decimation factor (how many points to remove).
            smooth_iter (int): The number of smoothing iterations.
            relaxation_factor (float): The relaxation factor.
            convert_to_LPS (bool): If True, the isosurface is converted to LPS coordinates. (Recommended)
                Future versions will set this to True.

        Returns:
            pv.PolyData: The surface mesh in anatomical coordinates.
        """
        # Pad data with 0s to avoid open surfaces at the edges.
        data = np.pad(self.data, 1, mode="constant", constant_values=0)

        surface = mesh_utils.isosurface(
            data,
            value=value,
            label=label,
            node_centered=node_centered,
            smooth=smooth,
            decimation=decimation,
            smooth_iter=smooth_iter,
            relaxation_factor=relaxation_factor,
        )
        # Subtract 1 in all directions to remove pad
        surface.transform(
            np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]]),
            inplace=True,
        )

        # Convert to anatomical coordinates
        surface.transform(geo.get_data(self.anatomical_from_ijk), inplace=True)

        if self.anatomical_coordinate_system == "RAS" and convert_to_LPS:
            # Convert to LPS
            surface.transform(geo.get_data(geo.RAS_from_LPS), inplace=True)

        if decimation_points is not None and surface.n_points > decimation_points:
            # Decimate the surface to the desired number of points
            surface = surface.decimate(1 - decimation_points / surface.n_points)

        return surface

    def _make_surface(self, material: str = "bone"):
        """Make a surface for the boolean segmentation"""
        assert (
            material in self.materials
        ), f'"{material}" not in {self.materials.keys()}'

        segmentation = self.materials[material]
        R = self.anatomical_from_ijk.R
        t = self.anatomical_from_ijk.t

        vol = vtk.vtkStructuredPoints()
        vol.SetDimensions(
            segmentation.shape[0], segmentation.shape[1], segmentation.shape[2]
        )
        vol.SetOrigin(
            -np.sign(R[0, 0]) * t[0],  # negate?
            np.sign(R[1, 1]) * t[1],  # negate?
            np.sign(R[2, 2]) * t[2],
        )
        vol.SetSpacing(
            -abs(R[0, 0]),
            abs(R[1, 1]),
            abs(R[2, 2]),  # negate?
        )

        segmentation = segmentation.astype(np.uint8)
        scalars = nps.numpy_to_vtk(segmentation.ravel(order="F"), deep=True)
        vol.GetPointData().SetScalars(scalars)

        log.debug("isolating bone surface for visualization...")
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(vol)
        dmc.GenerateValues(1, 1, 1)
        dmc.ComputeGradientsOff()
        dmc.ComputeNormalsOff()
        dmc.Update()

        surface = pv.wrap(dmc.GetOutput())
        if surface.is_all_triangles:
            surface.triangulate(inplace=True)

        surface.decimate_pro(
            0.01,
            feature_angle=60,
            splitting=False,
            preserve_topology=True,
            inplace=True,
        )
        surface.smooth(
            n_iter=30,
            relaxation_factor=0.25,
            feature_angle=70,
            boundary_smoothing=False,
            inplace=True,
        )
        surface.compute_normals(inplace=True)

        return surface

    def get_surface(
        self,
        material: str = "bone",
        use_cached: bool = True,
    ):
        log.info(f"cache_dir: {self.cache_dir}")
        cache_path = (
            None
            if self.cache_dir is None
            else self.cache_dir / f"cached_{material}_mesh.vtp"
        )
        log.info(f"cache_path: {cache_path}")
        if use_cached and cache_path is not None and cache_path.exists():
            log.info(f"reading cached {material} mesh from {cache_path}")
            surface = pv.read(cache_path)
        else:
            log.info(f"meshing {material} segmentation...")
            surface = self._make_surface(material)
            if cache_path is not None:
                if not self.cache_dir.exists():
                    self.cache_dir.mkdir()
                log.info(f"caching {material} surface to {cache_path}.")
                surface.save(cache_path)
        return surface

    _mesh_material = "bone"

    def get_mesh_in_world(
        self,
        full: bool = False,
        use_cached: bool = True,
    ) -> pv.PolyData:
        """Get a pyvista mesh of the outline in world-space.

        Args:
            full (bool): Whether to render the full volume or just a wireframe. Defaults to False.
            cache_dir (Optional[Path], optional): a location to cache the bone surface.
            use_cached (bool): If False, don't use the cached bone surface but re-create it (expensive). Defaults to True.

        Returns:
            pv.PolyData: pyvista mesh.
        """

        x, y, z = np.array(self.shape) - 1
        points = [
            [0, 0, 0],
            [0, 0, z],
            [0, y, 0],
            [0, y, z],
            [x, 0, 0],
            [x, 0, z],
            [x, y, 0],
            [x, y, z],
        ]

        points = [list(self.world_from_ijk @ geo.point(p)) for p in points]
        mesh = pv.Line(points[0], points[1])
        mesh += pv.Line(points[0], points[2])
        mesh += pv.Line(points[3], points[1])
        mesh += pv.Line(points[3], points[2])
        mesh += pv.Line(points[4], points[5])
        mesh += pv.Line(points[4], points[6])
        mesh += pv.Line(points[7], points[5])
        mesh += pv.Line(points[7], points[6])
        mesh += pv.Line(points[0], points[4])
        mesh += pv.Line(points[1], points[5])
        mesh += pv.Line(points[2], points[6])
        mesh += pv.Line(points[3], points[7])

        if full:
            log.debug(f"getting full surface mesh for volume")
            material_mesh = self.get_surface(
                material=self._mesh_material,
                use_cached=use_cached,
            )

            material_mesh.transform(geo.get_data(self.world_from_anatomical))
            mesh += material_mesh

        return mesh

    def crop(self, crop_box: Tuple[Tuple[float, float], ...]) -> Volume:
        """Crop the volume to a given bounding box.

        Args:
            crop_box (Tuple[Tuple[float, float], ...]): The bounding box to crop to, in IJK.

        Returns:
            Volume: The cropped volume.
        """
        crop_box = np.array(crop_box)
        crop_box = np.round(crop_box).astype(int)
        crop_box = np.clip(crop_box, 0, np.array(self.shape)[:, np.newaxis])

        cropped_data = self.data[
            crop_box[0, 0] : crop_box[0, 1],
            crop_box[1, 0] : crop_box[1, 1],
            crop_box[2, 0] : crop_box[2, 1],
        ]

        cropped_materials = {}
        for m, d in self.materials.items():
            cropped_materials[m] = d[
                crop_box[0, 0] : crop_box[0, 1],
                crop_box[1, 0] : crop_box[1, 1],
                crop_box[2, 0] : crop_box[2, 1],
            ]

        cropped_anatomical_from_IJK = self.anatomical_from_ijk.copy()
        cropped_anatomical_from_IJK.t = self.anatomical_from_ijk @ geo.p(crop_box[:, 0])

        return type(self)(
            cropped_data,
            materials=cropped_materials,
            anatomical_from_IJK=cropped_anatomical_from_IJK,
            world_from_anatomical=self.world_from_anatomical,
            anatomical_coordinate_system=self.anatomical_coordinate_system,
            cache_dir=self.cache_dir,
            config=self.config,
        )

    def get_bbox_IJK(self) -> Optional[np.ndarray]:
        """Get the bounding box of the materials in IJK.

        Returns:
            np.ndarray: The bounding box as a [3, 2] array.
            None, if the volume is empty.
        """
        any_material = np.zeros_like(self.data, dtype=bool)
        for d in self.materials.values():
            any_material = np.logical_or(any_material, d)
        indices = np.nonzero(any_material)
        if len(indices[0]) == 0:
            return None
        bbox = np.array(
            [
                [np.min(indices[0]), np.max(indices[0])],
                [np.min(indices[1]), np.max(indices[1])],
                [np.min(indices[2]), np.max(indices[2])],
            ]
        )
        return bbox

    def shrink(self) -> Volume:
        """Crop the volume to remove empty space.

        Returns:
            Volume: The cropped volume.
        """

        bbox = self.get_bbox_IJK()
        if bbox is None:
            log.warning("shrink called on empty volume")
            return self
        return self.crop(bbox)


class MetalVolume(Volume):
    """Same as a volume, but with a different segmentation for the materials."""

    @staticmethod
    def _convert_hounsfield_to_density(hu_values: np.ndarray):
        # TODO: verify
        return 30 * hu_values

    @staticmethod
    def _segment_materials(
        hu_values: np.ndarray, use_thresholding: bool = True
    ) -> Dict[str, np.ndarray]:
        if not use_thresholding:
            raise NotImplementedError

        return dict(
            air=(hu_values == 0),
            bone=(hu_values > 0),
            titanium=(hu_values > 0),
        )
