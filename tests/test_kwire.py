from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation
import json

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
    volume.rotate(Rotation.from_euler("x", 90, degrees=True))
    annotation_paths = sorted(list(data_dir.glob("*.mrk.json")))

    with open(annotation_paths[0], 'r') as file:
        ann = json.load(file)
    control_points = ann["markups"][0]["controlPoints"]
    control_points = dict((cp['label'], geo.point(cp['position']))
                          for cp in control_points)
    points = [control_points['entry'], control_points['exit']]

    if ann["markups"][0]["coordinateSystem"] == "LPS":
        points = [geo.RAS_from_LPS @ p for p in points]
    elif ann["markups"][0]["coordinateSystem"] == "RAS":
        pass
    else:
        raise TypeError(
            "annotation in unknown coordinate system: {}".format(
                ann["markups"][0]["coordinateSystem"]
            )
        )

    carm = deepdrr.MobileCArm()
    with deepdrr.Projector(volume, carm=carm) as projector:
        image = projector()

    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(output_dir / "test_kwire_empty.png")


if __name__ == "__main__":
    test_kwire()
