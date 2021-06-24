from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils

# TODO: create a test case possibly using the new dataset, along with some annotations, that tests the KWire alignment code.
# This will create a test case, demo that the annotations are correct, etc.


def test_kwire():
    output_dir = test_utils.get_output_dir()
    data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
    volume = deepdrr.Volume.from_nifti(
        data_dir / "dataset6_CLINIC_0001_data.nii.gz", use_thresholding=True)
    volume.rotate(Rotation.from_euler("xz", [90, 0], degrees=True))
    carm = deepdrr.MobileCArm(volume.center_in_world)

    with deepdrr.Projector(volume, carm=carm) as projector:
        image = projector()

    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(output_dir / "test_kwire_empty.png")


if __name__ == "__main__":
    test_kwire()
