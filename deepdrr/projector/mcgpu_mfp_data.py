import logging
import numpy as np

from .mcgpu_mean_free_path_data.adipose_mfp import adipose_ICRP110_MFP as ADIPOSE_MFP
from .mcgpu_mean_free_path_data.air_mfp import air_MFP as AIR_MFP
from .mcgpu_mean_free_path_data.blood_mfp import blood_ICRP110_MFP as BLOOD_MFP
from .mcgpu_mean_free_path_data.bone_mfp import bone_ICRP110_MFP as BONE_MFP
from .mcgpu_mean_free_path_data.brain_mfp import brain_ICRP110_MFP as BRAIN_MFP
from .mcgpu_mean_free_path_data.breast_mfp import breast_MFP as BREAST_MFP
from .mcgpu_mean_free_path_data.cartilage_mfp import (
    cartilage_ICRP110_MFP as CARTILAGE_MFP,
)
from .mcgpu_mean_free_path_data.connective_mfp import (
    connective_Woodard_MFP as CONNECTIVE_MFP,
)
from .mcgpu_mean_free_path_data.glands_others_mfp import (
    glands_others_ICRP110_MFP as GLANDS_OTHERS_MFP,
)
from .mcgpu_mean_free_path_data.liver_mfp import liver_ICRP110_MFP as LIVER_MFP
from .mcgpu_mean_free_path_data.lung_mfp import lung_ICRP110_MFP as LUNG_MFP
from .mcgpu_mean_free_path_data.muscle_mfp import muscle_ICRP110_MFP as MUSCLE_MFP
from .mcgpu_mean_free_path_data.PMMA_mfp import PMMA_MFP
from .mcgpu_mean_free_path_data.red_marrow_mfp import (
    red_marrow_Woodard_MFP as RED_MARROW_MFP,
)
from .mcgpu_mean_free_path_data.skin_mfp import skin_ICRP110_MFP as SKIN_MFP
from .mcgpu_mean_free_path_data.soft_tissue_mfp import (
    soft_tissue_ICRP110_MFP as SOFT_TISSUE_MFP,
)
from .mcgpu_mean_free_path_data.stomach_intestines_mfp import (
    stomach_intestines_ICRP110_MFP as STOMACH_INTESTINES_MFP,
)
from .mcgpu_mean_free_path_data.titanium_mfp import titanium_MFP as TITANIUM_MFP
from .mcgpu_mean_free_path_data.water_mfp import water_MFP as WATER_MFP

log = logging.getLogger(__name__)


def _convert_to_millimeters(mfp_data_cm: np.ndarray) -> np.ndarray:
    """Transforms the MFP data, given in centimeters, to millimeters
    """
    mfp_data_mm = np.copy(mfp_data_cm)
    mfp_data_mm[:, 1:5] *= 10
    return mfp_data_mm


MFP_DATA = {
    "adipose": _convert_to_millimeters(ADIPOSE_MFP),
    "air": _convert_to_millimeters(AIR_MFP),
    "blood": _convert_to_millimeters(BLOOD_MFP),
    "bone": _convert_to_millimeters(BONE_MFP),
    "brain": _convert_to_millimeters(BRAIN_MFP),
    "breast": _convert_to_millimeters(BREAST_MFP),
    "cartilage": _convert_to_millimeters(CARTILAGE_MFP),
    "connective tissue": _convert_to_millimeters(CONNECTIVE_MFP),
    "glands": _convert_to_millimeters(GLANDS_OTHERS_MFP),
    "liver": _convert_to_millimeters(LIVER_MFP),
    "lung": _convert_to_millimeters(LUNG_MFP),
    "muscle": _convert_to_millimeters(MUSCLE_MFP),
    "PMMA": _convert_to_millimeters(PMMA_MFP),
    "red marrow": _convert_to_millimeters(RED_MARROW_MFP),
    "skin": _convert_to_millimeters(SKIN_MFP),
    "soft tissue": _convert_to_millimeters(SOFT_TISSUE_MFP),
    "stomach intestines": _convert_to_millimeters(STOMACH_INTESTINES_MFP),
    "titanium": _convert_to_millimeters(TITANIUM_MFP),
    "water": _convert_to_millimeters(WATER_MFP),
}


def sanity_check_mfps():
    mats = list(MFP_DATA.keys())
    NUM_MATS = len(mats)
    for i in range(NUM_MATS - 1):
        for j in range(i + 1, NUM_MATS):
            data_1 = MFP_DATA[mats[i]]
            data_2 = MFP_DATA[mats[j]]

            assert 2 == data_1.ndims
            assert 2 == data_2.ndims

            # Should be 23001 energy entries
            assert 23001 == data_1.shape[0]
            assert 23001 == data_2.shape[0]

            # Should be 6 categories for each energy level
            assert 6 == data_1.shape[1]
            assert 6 == data_2.shape[1]

            # Check that the energy categories for data_1 and data_2 match each other
            assert np.all(np.equal(data_1[:, 0], data_2[:, 0]))

            # Check that the other data categories for data_1 and data_2 DON'T match
            for i in range(1, 6):
                assert np.all(np.not_equal(data_1[:, 0], data_2[:, 0]))

    for i in range(NUM_MATS):
        data = MFP_DATA[mats[i]]
        # Check the inverse-MFP sum equation: \sum_{interaction type i} (MFP_{i})^{-1} = MFP_{total}
        for energy_bin in range(data.shape[0]):
            Ra_inv = 1 / data[energy_bin, 1]
            Co_inv = 1 / data[energy_bin, 2]
            PE_inv = 1 / data[energy_bin, 3]
            Tot_inv = 1 / data[energy_bin, 4]
            assert (Ra_inv + Co_inv + PE_inv) <= Tot_inv

    log.info("MFP sanity has been checked!")
