import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
import numpy as np
import logging

log = logging.getLogger(__name__)

def test_scatter_geometry():
    file_path = test_utils.download_sampledata("CT-chest")
    volume = deepdrr.Volume.from_nrrd(file_path)

    carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)

    print(f"volume center in world: {volume.center_in_world}")
    print(f"volume spacing: {volume.spacing}")
    print(f"volume ijk_from_world\n{volume.ijk_from_world}")

    with deepdrr.Projector(
        volume=volume,
        carm=carm,
        step=0.1,
        mode="linear",
        max_block_index=200,
        spectrum="90KV_AL40",
        photon_count=100000,
        scatter_num=10e7,
        threads=8,
        neglog=True,
    ) as projector:
        image = projector.project()

    image = (image * 255).astype(np.uint8)
    #Image.fromarray(image).save("output/test_multivolume.png")
    
    output_dir = test_utils.get_output_dir()

if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.DEBUG)
    test_scatter_geometry()
