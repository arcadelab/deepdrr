"""Legacy code for loading DICOM files. See vol.Volume.from_dicom."""

import glob

import logging
import numpy as np
from skimage.transform import resize
import pydicom as dicom

from . import segmentation


log = logging.getLogger(__name__)


materials = {1: "air", 2: "soft tissue", 3: "cortical bone"}


def load_dicom(
    source_path=r"./*/*/",
    fixed_slice_thinckness=None,
    new_resolution=None,
    truncate=None,
    smooth_air=False,
    use_thresholding_segmentation=False,
    file_extension=".dcm",
):
    source_path += "*" + file_extension
    files = np.array(glob.glob(source_path))
    one_slice = dicom.read_file(files[0])
    if hasattr(one_slice, "InstanceNumber"):
        sliceOrder = [dicom.read_file(curDCM).InstanceNumber for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]
    else:
        sliceOrder = [dicom.read_file(curDCM).SliceLocation for curDCM in files]
        files = files[np.argsort(sliceOrder).astype(np.int32)]

    files = list(files)

    # Get ref file
    refDs = dicom.read_file(files[0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    volume_size = [int(refDs.Rows), int(refDs.Columns), files.__len__()]

    if not hasattr(refDs, "SliceThickness"):
        log.debug(
            "Volume has no attribute Slice Thickness, please provide it manually!"
        )
        log.debug("using fixed slice thickness of:", fixed_slice_thinckness)
        voxel_size = [
            float(refDs.PixelSpacing[1]),
            float(refDs.PixelSpacing[0]),
            fixed_slice_thinckness,
        ]
    else:
        voxel_size = [
            float(refDs.PixelSpacing[1]),
            float(refDs.PixelSpacing[0]),
            float(refDs.SliceThickness),
        ]

    # The array is sized based on 'PixelDims'
    volume = np.zeros(volume_size, dtype=np.float64)

    # loop through all the DICOM files
    for filenameDCM in files:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        if files.index(filenameDCM) < volume.shape[2]:
            volume[:, :, files.index(filenameDCM)] = ds.pixel_array.astype(np.int32)

    # use intercept point
    if hasattr(refDs, "RescaleIntercept"):
        volume += int(refDs.RescaleIntercept)

    volume = np.moveaxis(volume, [0, 1, 2], [1, 0, 2]).copy()

    # truncate
    if truncate:
        volume = volume[
            truncate[0][0] : truncate[0][1],
            truncate[1][0] : truncate[1][1],
            truncate[2][0] : truncate[2][1],
        ]

    # volume = np.flip(volume,2)
    # upsample Volume
    if new_resolution:
        volume, volume_size, voxel_size = upsample(volume, new_resolution, voxel_size)

    # convert hu_values to density
    densities = conv_hu_to_density(volume, smoothAir=smooth_air)

    # convert hu_values to materials
    if not use_thresholding_segmentation:
        materials = conv_hu_to_materials(volume)
    else:
        materials = conv_hu_to_materials_thresholding(volume)

    return (
        densities.astype(np.float32),
        materials,
        np.array(voxel_size, dtype=np.float32),
    )


def upsample(volume, newResolution, voxelSize):
    upsampled_voxel_size = list(
        np.array(voxelSize) * np.array(volume.shape) / newResolution
    )
    upsampled_volume = resize(volume, newResolution, order=1, cval=-1000)
    return upsampled_volume, upsampled_voxel_size, upsampled_voxel_size


def conv_hu_to_density(hu_values, smoothAir=False):
    # Use two linear interpolations from data: (HU,g/cm^3)
    # use for lower HU: density = 0.001029*HU + 1.03
    # use for upper HU: density = 0.0005886*HU + 1.03

    # set air densities
    if smoothAir:
        hu_values[hu_values <= -900] = -1000
    # hu_values[hu_values > 600] = 5000;
    densities = np.maximum(
        np.minimum(0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0
    )
    return densities


def conv_hu_to_materials_thresholding(hu_values):
    log.info("segmenting volume with thresholding...")
    materials = {}
    # Air
    materials["air"] = hu_values <= -800
    # Soft Tissue
    materials["soft tissue"] = (-800 < hu_values) * (hu_values <= 350)
    # Bone
    materials["bone"] = 350 < hu_values
    log.info("done.")

    return materials


def conv_hu_to_materials(hu_values):
    log.info("segmenting volume with Vnet")
    segmentation_network = segmentation.SegmentationNet()
    materials = segmentation_network.segment(hu_values)
    segmentation_network = None
    return materials
