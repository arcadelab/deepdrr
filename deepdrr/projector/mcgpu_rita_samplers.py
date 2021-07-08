from .mcgpu_rita_params.adipose_rita_params import (
    adipose_ICRP110_RITA_PARAMS as ADIPOSE_RITA_PARAMS,
)
from .mcgpu_rita_params.air_rita_params import air_RITA_PARAMS as AIR_RITA_PARAMS
from .mcgpu_rita_params.blood_rita_params import (
    blood_ICRP110_RITA_PARAMS as BLOOD_RITA_PARAMS,
)
from .mcgpu_rita_params.bone_rita_params import (
    bone_ICRP110_RITA_PARAMS as BONE_RITA_PARAMS,
)
from .mcgpu_rita_params.brain_rita_params import (
    brain_ICRP110_RITA_PARAMS as BRAIN_RITA_PARAMS,
)
from .mcgpu_rita_params.breast_rita_params import (
    breast_RITA_PARAMS as BREAST_RITA_PARAMS,
)
from .mcgpu_rita_params.cartilage_rita_params import (
    cartilage_ICRP110_RITA_PARAMS as CARTILAGE_RITA_PARAMS,
)
from .mcgpu_rita_params.connective_rita_params import (
    connective_Woodard_RITA_PARAMS as CONNECTIVE_RITA_PARAMS,
)
from .mcgpu_rita_params.glands_others_rita_params import (
    glands_others_ICRP110_RITA_PARAMS as GLANDS_OTHERS_RITA_PARAMS,
)
from .mcgpu_rita_params.liver_rita_params import (
    liver_ICRP110_RITA_PARAMS as LIVER_RITA_PARAMS,
)
from .mcgpu_rita_params.lung_rita_params import (
    lung_ICRP110_RITA_PARAMS as LUNG_RITA_PARAMS,
)
from .mcgpu_rita_params.muscle_rita_params import (
    muscle_ICRP110_RITA_PARAMS as MUSCLE_RITA_PARAMS,
)
from .mcgpu_rita_params.PMMA_rita_params import PMMA_RITA_PARAMS
from .mcgpu_rita_params.red_marrow_rita_params import (
    red_marrow_Woodard_RITA_PARAMS as RED_MARROW_RITA_PARAMS,
)
from .mcgpu_rita_params.skin_rita_params import (
    skin_ICRP110_RITA_PARAMS as SKIN_RITA_PARAMS,
)
from .mcgpu_rita_params.soft_tissue_rita_params import (
    soft_tissue_ICRP110_RITA_PARAMS as SOFT_TISSUE_RITA_PARAMS,
)
from .mcgpu_rita_params.stomach_intestines_rita_params import (
    stomach_intestines_ICRP110_RITA_PARAMS as STOMACH_INTESTINES_RITA_PARAMS,
)
from .mcgpu_rita_params.titanium_rita_params import (
    titanium_RITA_PARAMS as TITANIUM_RITA_PARAMS,
)
from .mcgpu_rita_params.water_rita_params import water_RITA_PARAMS as WATER_RITA_PARAMS
from .rita import RITA

import logging
import numpy as np

log = logging.getLogger(__name__)

saved_rita_params = {
    "adipose": ADIPOSE_RITA_PARAMS,
    "air": AIR_RITA_PARAMS,
    "blood": BLOOD_RITA_PARAMS,
    "bone": BONE_RITA_PARAMS,
    "brain": BRAIN_RITA_PARAMS,
    "breast": BREAST_RITA_PARAMS,
    "cartilage": CARTILAGE_RITA_PARAMS,
    "connective tissue": CONNECTIVE_RITA_PARAMS,
    "glands": GLANDS_OTHERS_RITA_PARAMS,
    "liver": LIVER_RITA_PARAMS,
    "lung": LUNG_RITA_PARAMS,
    "muscle": MUSCLE_RITA_PARAMS,
    "PMMA": PMMA_RITA_PARAMS,
    "red marrow": RED_MARROW_RITA_PARAMS,
    "skin": SKIN_RITA_PARAMS,
    "soft tissue": SOFT_TISSUE_RITA_PARAMS,
    "stomach intestines": STOMACH_INTESTINES_RITA_PARAMS,
    "titanium": TITANIUM_RITA_PARAMS,
    "water": WATER_RITA_PARAMS,
}

rita_samplers = {
    "adipose": RITA.from_saved_params(saved_rita_params["adipose"]),
    "air": RITA.from_saved_params(saved_rita_params["air"]),
    "blood": RITA.from_saved_params(saved_rita_params["blood"]),
    "bone": RITA.from_saved_params(saved_rita_params["bone"]),
    "brain": RITA.from_saved_params(saved_rita_params["brain"]),
    "breast": RITA.from_saved_params(saved_rita_params["breast"]),
    "cartilage": RITA.from_saved_params(saved_rita_params["cartilage"]),
    "connective tissue": RITA.from_saved_params(saved_rita_params["connective tissue"]),
    "glands": RITA.from_saved_params(saved_rita_params["glands"]),
    "liver": RITA.from_saved_params(saved_rita_params["liver"]),
    "lung": RITA.from_saved_params(saved_rita_params["lung"]),
    "muscle": RITA.from_saved_params(saved_rita_params["muscle"]),
    "PMMA": RITA.from_saved_params(saved_rita_params["PMMA"]),
    "red marrow": RITA.from_saved_params(saved_rita_params["red marrow"]),
    "skin": RITA.from_saved_params(saved_rita_params["skin"]),
    "soft tissue": RITA.from_saved_params(saved_rita_params["soft tissue"]),
    "stomach intenstines": RITA.from_saved_params(
        saved_rita_params["stomach intestines"]
    ),
    "titanium": RITA.from_saved_params(saved_rita_params["titanium"]),
    "water": RITA.from_saved_params(saved_rita_params["water"]),
}


def sanity_check_saved_rita_params():
    mats = list(saved_rita_params.keys())
    NUM_MATS = len(mats)
    for i in range(NUM_MATS - 1):
        for j in range(i + 1, NUM_MATS):
            data_1 = saved_rita_params[mats[i]]
            data_2 = saved_rita_params[mats[j]]

            assert 2 == data_1.ndims
            assert 2 == data_2.ndims

            # Should be 128 gridpoints
            assert 128 == data_1.shape[0]
            assert 128 == data_2.shape[0]

            # Should be 6 categories for each gridpoint
            assert 6 == data_1.shape[1]
            assert 6 == data_2.shape[1]

    for i in range(NUM_MATS):
        data = saved_rita_params[mats[i]]

        # CDF-ness
        assert 0 == data[0, 1]
        assert 1 == data[-1, 1]

    log.info("The saved RITA parameters' sanity has been checked!")
