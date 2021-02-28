from .mcgpu_mean_free_path_data.adipose_mfp import adipose_ICRP110_MFP as ADIPOSE_MFP
from .mcgpu_mean_free_path_data.air_mfp import air_MFP as AIR_MFP
from .mcgpu_mean_free_path_data.blood_mfp import blood_ICRP110_MFP as BLOOD_MFP
from .mcgpu_mean_free_path_data.bone_mfp import bone_ICRP110_MFP as BONE_MFP
from .mcgpu_mean_free_path_data.brain_mfp import brain_ICRP110_MFP as BRAIN_MFP
from .mcgpu_mean_free_path_data.breast_mfp import breast_MFP as BREAST_MFP
from .mcgpu_mean_free_path_data.cartilage_mfp import cartilage_ICRP110_MFP as CARTILAGE_MFP
from .mcgpu_mean_free_path_data.connective_mfp import connective_Woodard_MFP as CONNECTIVE_MFP
from .mcgpu_mean_free_path_data.glands_others_mfp import glands_others_ICRP110_MFP as GLANDS_OTHERS_MFP
from .mcgpu_mean_free_path_data.liver_mfp import liver_ICRP110_MFP as LIVER_MFP
from .mcgpu_mean_free_path_data.lung_mfp import lung_ICRP110_MFP as LUNG_MFP
from .mcgpu_mean_free_path_data.muscle_mfp import muscle_ICRP110_MFP as MUSCLE_MFP
from .mcgpu_mean_free_path_data.PMMA_mfp import PMMA_MFP
from .mcgpu_mean_free_path_data.red_marrow_mfp import red_marrow_Woodard_MFP as RED_MARROW_MFP
from .mcgpu_mean_free_path_data.skin_mfp import skin_ICRP110_MFP as SKIN_MFP
from .mcgpu_mean_free_path_data.soft_tissue_mfp import soft_tissue_ICRP110_MFP as SOFT_TISSUE_MFP
from .mcgpu_mean_free_path_data.stomach_intestines_mfp import stomach_intestines_ICRP110_MFP as STOMACH_INTESTINES_MFP
from .mcgpu_mean_free_path_data.titanium_mfp import titanium_MFP as TITANIUM_MFP
from .mcgpu_mean_free_path_data.water_mfp import water_MFP as WATER_MFP

import numpy as np

mfp_data = {
    "adipose": ADIPOSE_MFP,
    "air": AIR_MFP,
    "blood": BLOOD_MFP,
    "bone": BONE_MFP,
    "brain": BRAIN_MFP,
    "breast": BREAST_MFP,
    "cartilage": CARTILAGE_MFP,
    "connective tissue": CONNECTIVE_MFP,
    "glands": GLANDS_OTHERS_MFP,
    "liver": LIVER_MFP,
    "lung": LUNG_MFP,
    "muscle": MUSCLE_MFP,
    "PMMA": PMMA_MFP,
    "red marrow": RED_MARROW_MFP,
    "skin": SKIN_MFP,
    "soft tissue": SOFT_TISSUE_MFP,
    "stomach intestines": STOMACH_INTESTINES_MFP,
    "titanium": TITANIUM_MFP,
    "water": WATER_MFP
}

def sanity_check_mfps():
    mats = list(mfp_data.keys())
    NUM_MATS = len(mats)
    for i in range(NUM_MATS - 1):
        for j in range(i + 1, NUM_MATS):
            data_1 = mfp_data[mats[i]]
            data_2 = mfp_data[mats[j]]

            assert 2 == data_1.ndims
            assert 2 == data_2.ndims

            # Should be 23001 energy entries
            assert 23001 == data_1.shape[0]
            assert 23001 == data_2.shape[0]

            # Should be 6 categories for each energy level
            assert 6 == data_1.shape[1]
            assert 6 == data_2.shape[1]

            # Check that the energy categories for data_1 and data_2 match each other
            assert np.all(np.equal(data_1[:,0], data_2[:,0]))

            # Check that the other data categories for data_1 and data_2 DON'T match
            for i in range(1, 6):
                assert np.all(np.not_equal(data_1[:,0], data_2[:,0]))

    for i in range(NUM_MATS):
        data = mfp_data[mats[i]]
        # Check the inverse-MFP sum equation: \sum_{interaction type i} (MFP_{i})^{-1} = MFP_{total}
        for energy_bin in range(data.shape[0]):
            Ra_inv = 1 / data[energy_bin, 1]
            Co_inv = 1 / data[energy_bin, 2]
            PE_inv = 1 / data[energy_bin, 3]
            Tot_inv = 1 / data[energy_bin, 4]
            assert (Ra_inv + Co_inv + PE_inv) <= Tot_inv

    print("MFP sanity has been checked!")
