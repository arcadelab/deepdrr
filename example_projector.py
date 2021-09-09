#! python3

import logging
import os
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rich.logging import RichHandler
from time import time

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils

# set up fancy logging
log = logging.getLogger().handlers.clear()
log = logging.getLogger('deepdrr')
log.addHandler(RichHandler())
log.setLevel(logging.INFO)

def main():
    output_dir = test_utils.get_output_dir()
    data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
    patient = deepdrr.Volume.from_nifti(
        data_dir / "dataset6_CLINIC_0001_data.nii.gz", use_thresholding=True
    )
    patient.faceup()

    # define the simulated C-arm
    carm = deepdrr.MobileCArm(patient.center_in_world)

    # project in the AP view
    with deepdrr.Projector(patient, carm=carm) as projector:
        carm.move_to(alpha=0, beta=-15)
        image = projector()

    path = output_dir / "example_projector.png"
    image_utils.save(path, image)
    log.info(f"saved example projection image to {path.absolute()}")

if __name__ == "__main__":
    main()
