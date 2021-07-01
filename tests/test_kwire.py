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
        data_dir / "dataset6_CLINIC_0001_data.nii.gz", use_thresholding=True
    )
    volume.rotate(Rotation.from_euler("xz", [90, -90], degrees=True))

    # load the line annotation for the trajectory
    annotation_path = sorted(list(data_dir.glob("*.mrk.json")))[1]
    annotation = deepdrr.LineAnnotation.from_markup(annotation_path, volume)

    # define the simulated C-arm
    carm = deepdrr.MobileCArm(
        annotation.startpoint_in_world.lerp(annotation.endpoint_in_world, 0.3)
    )

    # first, just do the CT volume on its own
    with deepdrr.Projector(volume, carm=carm) as projector:
        image = projector()
        image_utils.save(output_dir / "test_kwire_empty.png", image)

    # Then add a kwire
    kwire = deepdrr.vol.KWire.from_example()
    kwire.align(annotation.startpoint_in_world, annotation.endpoint_in_world, 1)

    # Then, do them both
    # with deepdrr.Projector([volume, kwire], carm=carm) as projector:
    #     image = projector()
    #     image_utils.save(output_dir / "test_kwire.png", image)

    # screenshot = vis.show(
    #     volume, carm, annotation, kwire, full=[True, True, True, True]
    # )
    # image_utils.save(output_dir / "test_kwire_screenshot.png", screenshot)

    output_dir = output_dir / "test_kwire"
    output_dir.mkdir(exist_ok=True)
    with deepdrr.Projector([volume, kwire], carm=carm) as projector:
        for alpha in [-30, -15, 0, 15, 30, 45, 60, 75, 90]:
            carm.move_to(alpha=alpha, degrees=True)
            for progress in np.arange(0, 1, 0.2):
                kwire.align(
                    annotation.startpoint_in_world,
                    annotation.endpoint_in_world,
                    progress,
                )
                image = projector()
                image_utils.save(
                    output_dir
                    / f"test_kwire_only_alpha={alpha}_progress={int(100 * progress)}.png",
                    image,
                )


if __name__ == "__main__":
    logging.getLogger("deepdrr").setLevel(logging.DEBUG)
    test_kwire()
