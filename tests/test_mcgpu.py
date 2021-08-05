import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils
import numpy as np
import logging

log = logging.getLogger(__name__)

# NOTE on requirements for running this test file:
# This test file assumes that MCGPU is installed on the machine in the following way:
#
#   some-directory
#       |
#       |___ deepdrr
#       |       |
#       |       |___ deepdrr
#       |       |       |
#       |       |       |___ projector
#       |       |       |
#       |       |       |___ geo
#       |       |       |
#       |       |      ...
#       |       |
#       |       |___ examples
#       |       |
#       |       |___ tests
#       |       |
#       |      ...
#       |
#       |___  MCGPU
#       |       |
#      ...      |___ TODO
#               |
#               |___ TODO
#               |
#               |___ MC-GPU_v1.3.x
#               |
#               |___ TODO
#               |
#
#

def test_mcgpu():
    file_path = test_utils.download_sampledata("CT-chest")
    volume = deepdrr.Volume.from_nrrd(file_path)

    # TODO list
    # 1. check that MCGPU is installed in correct location. If not, automatically pass the test
    # 2. load volume and specify test's geometry
    # 3. Convert to MCGPU format and write out to file in proper location
    # 4. Run DeepDRR
    # 5. Run MCGPU
    # 6. Compare in intelligent ways

    carm = deepdrr.MobileCArm(isocenter=volume.center_in_world)
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
    test_mcgpu()
