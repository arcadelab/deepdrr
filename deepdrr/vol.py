"""Volume class for CT volume.
"""

from __future__ import annotations
from typing import Union, Tuple, Literal, List, Optional, Dict

import logging
import numpy as np
from pathlib import Path
import nibabel as nib
from pydicom.filereader import dcmread, InvalidDicomError

from . import load_dicom
from . import geo


logger = logging.getLogger(__name__)


class Volume(object):
    data: np.ndarray
    materials: Dict[str, np.ndarray]
    anatomical_from_ijk: geo.FrameTransform
    world_from_anatomical: geo.FrameTransform

    def __init__(
        self,
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        anatomical_from_ijk: geo.FrameTransform,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ) -> None:
        """A deepdrr Volume object with materials segmentation and orientation in world-space.

        The recommended way to create a Volume is to load from a NifTi file using the `Volume.from_nifti(path)` class method.

        Args:
            data (np.ndarray): the density data (a 3D array)
            materials (Dict[str, np.ndarray]): material segmentation of the volume, mapping material name to binary segmentation.
            anatomical_from_ijk (geo.FrameTransform): transformation from IJK space to anatomical (RAS or LPS).
            world_from_anatomical (Optional[geo.FrameTransform], optional): transformation from the anatomical space to world coordinates. If None, assumes identity. Defaults to None.
        """
        self.data = np.array(data).astype(np.float32)
        self.materials = self._format_materials(materials)
        self.anatomical_from_ijk = anatomical_from_ijk
        self.world_from_anatomical = geo.FrameTransform.identity(3) if world_from_anatomical is None else world_from_anatomical

    @classmethod
    def from_parameters(
        cls,
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        origin: geo.Point3D,
        spacing: Optional[geo.Vector3D] = [1, 1, 1],
        anatomical_coordinate_system: Optional[Literal['LPS', 'RAS', 'none']] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ):
        """Create a volume object with a segmentation of the materials, with its own anatomical coordinate space, from parameters.

        Note that the anatomical coordinate system is not the world coordinate system (which is cartesian).
        
        Suggested anatomical coordinate space units is milimeters. 
        A helpful introduction to the geometry is can be found [here](https://www.slicer.org/wiki/Coordinate_systems).

        Args:
            volume (np.ndarray): the volume density data.
            materials (dict[str, np.ndarray]): mapping from material names to binary segmentation of that material.
            origin (Point3D): Location of the volume's origin in the anatomical coordinate system.
            spacing (Tuple[float, float, float], optional): Spacing of the volume in the anatomical coordinate system. Defaults to (1, 1, 1).
            anatomical_coordinate_system (Literal['LPS', 'RAS', 'none']): anatomical coordinate system convention. Defaults to 'none'.
            world_from_anatomical (FrameTransform, optional): Optional transformation from anatomical to world coordinates. 
                If None, then identity is used. Defaults to None.
        """
        origin = geo.point(origin)
        spacing = geo.vector(spacing)

        assert spacing.dim == 3

        # define anatomical_from_ijk FrameTransform
        if anatomical_coordinate_system is None or anatomical_coordinate_system == 'none':
            anatomical_from_ijk = geo.FrameTransform.from_scaling(scaling=spacing, translation=origin)
        elif anatomical_coordinate_system == 'LPS':
            # IJKtoLPS = LPS_from_IJK
            rotation = [
                [spacing[0], 0, 0],
                [0, 0, spacing[2]],
                [0, -spacing[1], 0],
            ]
            anatomical_from_ijk = geo.FrameTransform.from_rt(rotation=rotation, translation=origin)
        else:
            raise NotImplementedError("conversion from RAS (not hard, look at LPS example)")

        return cls(
            data=data,
            materials=materials,
            anatomical_from_ijk=anatomical_from_ijk,
            world_from_anatomical=world_from_anatomical,
        )

    @classmethod
    def from_nifti(
        cls,
        path: Path,
        use_thresholding: bool = True,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        use_cached: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """Load a volume from NiFti file.

        Args:
            path (Path): path to the .nii.gz file.
            use_thresholding (bool, optional): segment the materials using thresholding (faster but less accurate). Defaults to True.
            world_from_anatomical (Optional[geo.FrameTransform], optional): position the volume in world space. If None, uses identity. Defaults to None.
            use_cached (bool, optional): [description]. Use a cached segmentation if available. Defaults to True.
            cache_dir (Optional[Path], optional): Where to load/save the cached segmentation. If None, use the parent dir of `path`. Defaults to None.

        Returns:
            [type]: [description]
        """
        path = Path(path)
        stem = path.name.split('.')[0]

        if cache_dir is None:
            cache_dir = path.parent

        logger.info(f'loading NiFti volume from {path}')
        img = nib.load(path)
        assert img.header.get_xyzt_units() == ('mm', 'sec'), 'TODO: NiFti image != (mm, sec)'

        anatomical_from_ijk = geo.FrameTransform(img.affine)
        hu_values = img.get_fdata()
        data = load_dicom.conv_hu_to_density(hu_values)

        if use_thresholding:
            materials_path = cache_dir / f'{stem}_materials_thresholding.npz'
            if use_cached and materials_path.exists():
                logger.info(f'found materials segmentation at {materials_path}.')
                materials = dict(np.load(materials_path))
            else:
                logger.info(f'segmenting materials in volume')
                materials = load_dicom.conv_hu_to_materials_thresholding(hu_values)
                np.savez(materials_path, **materials)
        else:
            materials_path = cache_dir / f'{stem}_materials.npz'
            if use_cached and materials_path.exists():
                logger.info(f'found materials segmentation at {materials_path}.')
                materials = dict(np.load(materials_path))
            else:
                logger.info(f'segmenting materials in volume')
                materials = load_dicom.conv_hu_to_materials(hu_values)
                np.savez(materials_path, **materials)

        return cls(
            data,
            materials,
            anatomical_from_ijk,
            world_from_anatomical,
        )

    @classmethod
    def from_dicom(
            cls,
            path: Path,
            origin: geo.Point3D = None,
            use_thresholding: bool = True,
            world_from_anatomical: Optional[geo.FrameTransform] = None,
            use_cached: bool = True,
            cache_dir: Optional[Path] = None
    ):
        """
        load a volume from a dicom file and set the anatomical_from_ijk transform from metadata
        https://www.slicer.org/wiki/Coordinate_systems
        Args:
            path:
            use_thresholding:
            world_from_anatomical:
            use_cached:
            cache_dir:

        Returns:

        """
        path = Path(path)
        stem = path.name.split('.')[0]

        if cache_dir is None:
            cache_dir = path.parent

        # Multi-frame dicoms store all slices of a volume in one file.
        # they must specify the necessary dicom tags under
        # https://dicom.innolitics.com/ciods/enhanced-ct-image/enhanced-ct-image-multi-frame-functional-groups
        assert path.is_file(), 'Currently only multi-frame dicoms are supported. Path must refer to a file.'
        logger.info(f'loading Dicom volume from {path}')

        # reading the dicom dataset object
        ds = dcmread(path)

        # extracting all needed tags TODO add try exepts
        frames = ds.PerFrameFunctionalGroupsSequence
        shared = ds.SharedFunctionalGroupsSequence[0]
        num_slices = len(frames)
        first_slice_position = np.array(frames[0].PlanePositionSequence[0].ImagePositionPatient)
        last_slice_position = np.array(frames[-1].PlanePositionSequence[0].ImagePositionPatient)
        RC = np.array(shared.PlaneOrientationSequence[0].ImageOrientationPatient).reshape(2, 3).T
        PixelSpacing = np.array(ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing)
        SliceThickness = np.array(ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
        offset = ds.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleIntercept
        scale = ds.SharedFunctionalGroupsSequence[0].PixelValueTransformationSequence[0].RescaleSlope

        # make user aware that this is only tested on windows
        if ds.Manufacturer != "SIEMENS":
            logger.warning("Multi-frame loading has only been tested on Siemens Enhanced CT DICOMs."
                           "Please verify everything works as expected.")

        # read the 'raw' data array
        raw_data = ds.pixel_array.astype(np.float32)
        hu_values = raw_data * scale + offset

        # move slice index axis to back (k, j, i) -> (j, i, k)
        hu_values = hu_values.transpose((1, 2, 0))

        '''
        EXPLANATION
        
        According to dicom (C.7.6.3.1.4 - Pixel Data) slices are of shape (Rows, Columns) 
        => must be (j, i) indexed if we define i == horizontal and j == vertical.
        This is taken care of in the definition of the affine transform 
        Further reading: https://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
        '''

        # transform the volume in HU to densities
        data = load_dicom.conv_hu_to_density(hu_values)

        # obtain materials analogous to nifti
        if use_thresholding:
            materials_path = cache_dir / f'{stem}_materials_thresholding.npz'
            if use_cached and materials_path.exists():
                logger.info(f'found materials segmentation at {materials_path}.')
                materials = dict(np.load(materials_path))
            else:
                logger.info(f'segmenting materials in volume')
                materials = load_dicom.conv_hu_to_materials_thresholding(hu_values)
                np.savez(materials_path, **materials)
        else:
            materials_path = cache_dir / f'{stem}_materials.npz'
            if use_cached and materials_path.exists():
                logger.info(f'found materials segmentation at {materials_path}.')
                materials = dict(np.load(materials_path))
            else:
                logger.info(f'segmenting materials in volume')
                materials = load_dicom.conv_hu_to_materials(hu_values)
                np.savez(materials_path, **materials)

        # manually composing affine transform lps_from_ijk

        # construct column for index k
        k = np.array((last_slice_position - first_slice_position) / num_slices).reshape(3, 1)

        # check if the calculated increment matches the SliceThickness (allow .1 millimeters deviations)
        assert np.allclose(np.abs(k[2]), SliceThickness, atol=0.1, rtol=0)

        # flip because dicom convention indexes a slice as [Columns, Rows]. see explanation above
        CR = np.fliplr(RC)
        CR_scaled = CR * PixelSpacing

        # construct rotation matrix from three columns for (j, i, k)
        rot = np.hstack((CR_scaled, k))

        # construct affine matrix
        affine = np.zeros((4, 4))
        affine[:3, :3] = rot
        affine[:3, 3] = first_slice_position
        affine[3, 3] = 1

        # log affine matrix in debug mode
        logger.debug(f"manually constructed affine matrix: \n{affine}")
        logger.debug(
            f"volume_center_xyz : {np.mean([affine @ np.array([*data.shape, 1]), affine @ [0, 0, 0, 1]], axis=0)}")

        # cast to FrameTransform
        lps_from_ijk = geo.FrameTransform(affine)

        # constructing the volume
        return cls(
            data,
            materials,
            lps_from_ijk,
            world_from_anatomical,
        )

    def to_dicom(self, path: Union[str, Path]):
        """Write the volume to a DICOM file.

        Args:
            path (str): the path to the file.
        """
        path = Path(path)

        raise NotImplementedError('save volume to dicom file')

    @property
    def origin(self) -> geo.Point3D:
        """The origin of the volume in anatomical space."""
        return geo.point(self.anatomical_from_ijk.t)

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

    @property
    def world_from_ijk(self) -> geo.FrameTransform:
        return self.world_from_anatomical @ self.anatomical_from_ijk

    @property
    def ijk_from_world(self) -> geo.FrameTransform:
        return self.world_from_ijk.inv

    def __array__(self) -> np.ndarray:
        return self.data

