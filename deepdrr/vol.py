"""Volume class for CT volume.
"""

from __future__ import annotations
from typing import Union, Tuple, Literal, List, Optional, Dict

import logging
import numpy as np
from pathlib import Path
import nibabel as nib

from . import load_dicom
from . import geo
from . import utils

pv, pv_available = utils.try_import_pyvista()
vtk, nps, vtk_available = utils.try_import_vtk()


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

        Note that the anatomical coordinate system is not the world coordinate system (which is cartesion). 
        
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

        # define anatomical_from_indices FrameTransform
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
    def from_hu(
        cls,
        hu_values: np.ndarray,
        origin: geo.Point3D,
        use_thresholding: bool = True,
        spacing: Optional[geo.Vector3D] = (1, 1, 1),
        anatomical_coordinate_system: Optional[Literal['LPS', 'RAS', 'none']] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
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
        )

    @staticmethod
    def _get_cache_path(
        use_thresholding: bool = True,
        cache_dir: Optional[Path] = None,
        prefix: str = '',
    ) -> Optional[Path]:
        return None if cache_dir is None else Path(cache_dir) / '{}materials{}.npz'.format(prefix, '_with_thresholding' if use_thresholding else '')

    @staticmethod
    def _convert_hounsfield_to_density(hu_values: np.ndarray):
        return load_dicom.conv_hu_to_density(hu_values)

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
        use_thresholding: bool = True,
        use_cached: bool = True,
        cache_dir: Optional[Path] = None,
        prefix: str = '',
    ) -> Dict[str, np.ndarray]:
        """Segment the materials in a volume, potentially caching.

       

        Args:
            hu_values (np.ndarray): volume data in Hounsfield Units.
            use_thretholding (bool, optional): whether to segment with thresholding (true) or a DNN. Defaults to True.
            use_cached (bool, optional): use the cached segmentation, if it exists. Defaults to True.
            cache_dir (Optional[Path], optional): where to look for the segmentation cache. If None, no caching performed. Defaults to None.
            prefix (str, optional): Optional prefix to prepend to the cache names. Defaults to ''.

        Returns:
            Dict[str, np.ndarray]: materials segmentation.
        """

        materials_path = cls._get_cache_path(use_thresholding=use_thresholding, cache_dir=cache_dir, prefix=prefix)

        if materials_path is not None and materials_path.exists() and use_cached:
            logger.info(f'using cached materials segmentation at {materials_path}')
            materials = dict(np.load(materials_path))
        else:
            logger.info(f'segmenting materials in volume')
            materials = cls._segment_materials(hu_values, use_thresholding=use_thresholding)
        
            if materials_path is not None:
                np.savez(materials_path, **materials)

        return materials


    @classmethod
    def from_nifti(
        cls,
        path: Path,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        use_thresholding: bool = True,
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

        if cache_dir is None:
            cache_dir = path.parent

        logger.info(f'loading NiFti volume from {path}')
        img = nib.load(path)
        if img.header.get_xyzt_units() != ('mm', 'sec'):
            logger.warning(f'got NifTi units: {img.header.get_xyzt_units()}')

        anatomical_from_ijk = geo.FrameTransform(img.affine)
        hu_values = img.get_fdata()
        
        data = cls._convert_hounsfield_to_density(hu_values)
        materials = cls.segment_materials(hu_values, use_thresholding=use_thresholding, use_cached=use_cached, cache_dir=cache_dir)
        
        return cls(
            data,
            materials,
            anatomical_from_ijk,
            world_from_anatomical,
        )

    @classmethod
    def from_dicom(
        cls,
        path: Union[str, Path],
    ) -> Volume:
        """Create the volume from a DICOM file."""
        raise NotImplementedError('load a volume from a dicom file')

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
    def origin_in_world(self) -> geo.Point3D:
        """The origin of the volume in world space."""
        return geo.point(self.world_from_ijk.t)

    @property
    def center_in_world(self) -> geo.Point3D:
        """The center of the volume in world coorindates. Useful for debugging."""
        return self.world_from_ijk @ geo.point(np.array(self.shape) / 2)

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

    def translate_center_to(self, x: geo.Point3D) -> None:
        """Translate the volume so that its center is located at world-space point x.

        Only changes the translation elements of the world_from_anatomical transform. Preserves the current rotation of the 

        Args:
            x (geo.Point3D): the world-space point.

        """
        x = geo.point(x)
        center_anatomical = self.anatomical_from_ijk @ geo.point(np.array(self.shape) / 2)
        self.world_from_anatomical = geo.FrameTransform.from_rt(self.world_from_anatomical.R) @ geo.FrameTransform.from_origin(center_anatomical)

    def __contains__(self, x: geo.Point3D) -> bool:
        """Determine whether the point x is inside the volume.

        Args:
            x (geo.Point3D): world-space point.

        """
        x_ijk = self.ijk_from_world @ geo.point(x)
        return np.all(0 <= np.array(x_ijk) <= np.array(self.shape) - 1)

    def _make_surface(self, material: str = 'bone'):
        """Make a surface for the boolean segmentation"""
        assert vtk_available and pv_available
        assert material in self.materials, f'"{material}" not in {self.materials.keys()}'

        segmentation = self.materials[material]
        R = self.anatomical_from_ijk.R
        t = self.anatomical_from_ijk.t

        vol = vtk.vtkStructuredPoints()
        vol.SetDimensions(segmentation.shape[0], segmentation.shape[1], segmentation.shape[2])
        vol.SetOrigin(
            -np.sign(R[0, 0]) * t[0],
            np.sign(R[1, 1]) * t[1],
            np.sign(R[2, 2]) * t[2],
        )
        vol.SetSpacing(
            -abs(R[0, 0]),
            abs(R[1, 1]),
            abs(R[2, 2]),
        )

        segmentation = segmentation.astype(np.uint8)
        scalars = nps.numpy_to_vtk(segmentation.ravel(order='F'), deep=True)
        vol.GetPointData().SetScalars(scalars)

        logger.debug('isolating bone surface for visualization...')
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(vol)
        dmc.GenerateValues(1, 1, 1)
        dmc.ComputeGradientsOff()
        dmc.ComputeNormalsOff()
        dmc.Update()

        surface = pv.wrap(dmc.GetOutput())
        if surface.is_all_triangles():
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
        material: str = 'bone',
        cache_dir: Optional[Path] = None,
        use_cached: bool = True,
    ):
        cache_path = None if cache_dir is None else Path(cache_dir) / f'{material}_mesh.vtp'
        if use_cached and cache_path is not None and cache_path.exists():
            logger.info(f'reading cached {material} mesh from {cache_path}')
            surface = pv.read(cache_path)
        else:
            logger.info(f'meshing {material} segmentation...')
            surface = self._make_surface(material)
            if cache_path is not None:
                if not cache_dir.exists():
                    cache_dir.mkdir()
                logger.info(f'caching {material} surface to {cache_path}.')
                surface.save(cache_path)
        return surface

    def get_mesh_in_world(self, full: bool = False, cache_dir: Optional[Path] = None) -> pv.PolyData:
        """Get a pyvista mesh of the outline in world-space.

        Returns:
            pv.PolyData: pyvist mesh.
        """

        assert pv_available, f'PyVista not available for obtaining Volume mesh. Try: `pip install pyvista`'

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
            material_mesh = self.get_surface(material='bone', cache_dir=cache_dir)
            material_mesh.transform(geo.get_data(self.world_from_anatomical))
            mesh += material_mesh

        return mesh


class MetalVolume(Volume):
    """Same as a volume, but with a different segmentation for the materials.

    """
    @staticmethod
    def _convert_hounsfield_to_density(hu_values: np.ndarray):
        # TODO: verify
        return 30 * hu_values

    @staticmethod
    def _segment_materials(hu_values: np.ndarray, use_thresholding: bool = True) -> Dict[str, np.ndarray]:
        if not use_thresholding:
            raise NotImplementedError

        return dict(
            air=(hu_values == 0),
            bone=(hu_values > 0),
            titanium=(hu_values > 0),
        )
        