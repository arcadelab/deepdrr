from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation
import json
import logging

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr import vis

# TODO: create a test case possibly using the new dataset, along with some annotations, that tests the KWire alignment code.
# This will create a test case, demo that the annotations are correct, etc.


def test_kwire():
    output_dir = test_utils.get_output_dir()
    data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
    volume = deepdrr.Volume.from_nifti(
        data_dir / "dataset6_CLINIC_0001_data.nii.gz", use_thresholding=False)
    volume.rotate(Rotation.from_euler("x", 90, degrees=True))
    annotation_paths = sorted(list(data_dir.glob("*.mrk.json")))

    with open(annotation_paths[1], 'r') as file:
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

    points_in_world = list(map(volume.world_from_anatomical, points))
    print(points_in_world)
    carm = deepdrr.MobileCArm(points_in_world[0] + geo.point(30, -40, 0))

    # first, just do the CT volume on its own
    # with deepdrr.Projector(volume, carm=carm) as projector:
    #     image = projector()
    #     image_utils.save(output_dir / "test_kwire_empty.png", image)

    # Then add a kwire
    kwire = deepdrr.vol.KWire.from_example()
    kwire.align(*points_in_world, 0.5)

    vis.show(volume, kwire)

    with deepdrr.Projector([volume, kwire], carm=carm) as projector:
        image = projector()
        image = np.stack([image, image, image], axis=-1)

        for p in points_in_world:
            print(p)
            i, j = carm.get_camera_projection().index_from_world @ p
            print(i, j)
            image[int(i), int(j)] = [1, 0, 0]
        image_utils.save(output_dir / "test_kwire.png", image)


if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.DEBUG)
    test_kwire()
