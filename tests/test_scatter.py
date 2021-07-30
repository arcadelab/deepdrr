import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
import numpy as np
import logging

log = logging.getLogger(__name__)

def test_scatter_single_volume_aligned():
    """A single volume, aligned with the world XYZ planes"""
    file_path = test_utils.download_sampledata("CT-chest")
    volume = deepdrr.Volume.from_nrrd(file_path)

    carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
    with deepdrr.Projector(
        volume=volume,
        carm=carm,
        step=0.1,
        mode="linear",
        max_block_index=200,
        spectrum="90KV_AL40",
        photon_count=100000,
        scatter_num=10e8,
        threads=8,
        neglog=True,
    ) as projector:
        image = projector.project()

    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save("output/test_multivolume.png")
    
    output_dir = test_utils.get_output_dir()

def test_scatter_single_volume_rotated():
    """A single volume, rotated so its bounding surfaces do not align with the world XYZ planes"""
    pass

def test_scatter_same_spacings():
    """Two volumes, but voxel sizes are identical"""
    pass

def test_scatter_diff_spacings():
    """Two volumes, but voxel sizes differ"""
    pass

def test_scatter_many_volumes():
    """A large number of volumes"""
    pass

if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.DEBUG)
    test_scatter_single_volume_aligned()
    test_scatter_single_volume_rotated()
    test_scatter_same_spacings()
    test_scatter_diff_spacings()
    test_scatter_many_volumes()
