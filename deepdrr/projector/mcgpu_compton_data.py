# NUM_SHELLS imports
import logging

log = logging.getLogger(__name__)

from .mcgpu_incoherent_scatter_data.adipose_compton_data import (
    adipose_ICRP110_NUM_SHELLS as ADIPOSE_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.air_compton_data import (
    air_NUM_SHELLS as AIR_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.blood_compton_data import (
    blood_ICRP110_NUM_SHELLS as BLOOD_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.bone_compton_data import (
    bone_ICRP110_NUM_SHELLS as BONE_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.brain_compton_data import (
    brain_ICRP110_NUM_SHELLS as BRAIN_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.breast_compton_data import (
    breast_NUM_SHELLS as BREAST_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.cartilage_compton_data import (
    cartilage_ICRP110_NUM_SHELLS as CARTILAGE_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.connective_compton_data import (
    connective_Woodard_NUM_SHELLS as CONNECTIVE_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.glands_others_compton_data import (
    glands_others_ICRP110_NUM_SHELLS as GLANDS_OTHERS_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.liver_compton_data import (
    liver_ICRP110_NUM_SHELLS as LIVER_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.lung_compton_data import (
    lung_ICRP110_NUM_SHELLS as LUNG_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.muscle_compton_data import (
    muscle_ICRP110_NUM_SHELLS as MUSCLE_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.PMMA_compton_data import PMMA_NUM_SHELLS
from .mcgpu_incoherent_scatter_data.red_marrow_compton_data import (
    red_marrow_Woodard_NUM_SHELLS as RED_MARROW_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.skin_compton_data import (
    skin_ICRP110_NUM_SHELLS as SKIN_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.soft_tissue_compton_data import (
    soft_tissue_ICRP110_NUM_SHELLS as SOFT_TISSUE_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.stomach_intestines_compton_data import (
    stomach_intestines_ICRP110_NUM_SHELLS as STOMACH_INTESTINES_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.titanium_compton_data import (
    titanium_NUM_SHELLS as TITANIUM_NUM_SHELLS,
)
from .mcgpu_incoherent_scatter_data.water_compton_data import (
    water_NUM_SHELLS as WATER_NUM_SHELLS,
)

# Compton shell data imports

from .mcgpu_incoherent_scatter_data.adipose_compton_data import (
    adipose_ICRP110_compton_data as ADIPOSE_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.air_compton_data import (
    air_compton_data as AIR_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.blood_compton_data import (
    blood_ICRP110_compton_data as BLOOD_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.bone_compton_data import (
    bone_ICRP110_compton_data as BONE_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.brain_compton_data import (
    brain_ICRP110_compton_data as BRAIN_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.breast_compton_data import (
    breast_compton_data as BREAST_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.cartilage_compton_data import (
    cartilage_ICRP110_compton_data as CARTILAGE_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.connective_compton_data import (
    connective_Woodard_compton_data as CONNECTIVE_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.glands_others_compton_data import (
    glands_others_ICRP110_compton_data as GLANDS_OTHERS_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.liver_compton_data import (
    liver_ICRP110_compton_data as LIVER_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.lung_compton_data import (
    lung_ICRP110_compton_data as LUNG_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.muscle_compton_data import (
    muscle_ICRP110_compton_data as MUSCLE_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.PMMA_compton_data import (
    PMMA_compton_data as PMMA_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.red_marrow_compton_data import (
    red_marrow_Woodard_compton_data as RED_MARROW_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.skin_compton_data import (
    skin_ICRP110_compton_data as SKIN_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.soft_tissue_compton_data import (
    soft_tissue_ICRP110_compton_data as SOFT_TISSUE_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.stomach_intestines_compton_data import (
    stomach_intestines_ICRP110_compton_data as STOMACH_INTESTINES_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.titanium_compton_data import (
    titanium_compton_data as TITANIUM_COMPTON_DATA,
)
from .mcgpu_incoherent_scatter_data.water_compton_data import (
    water_compton_data as WATER_COMPTON_DATA,
)

MAX_NSHELLS = 30

MATERIAL_NSHELLS = {
    "adipose": ADIPOSE_NUM_SHELLS,
    "air": AIR_NUM_SHELLS,
    "blood": BLOOD_NUM_SHELLS,
    "bone": BONE_NUM_SHELLS,
    "brain": BRAIN_NUM_SHELLS,
    "breast": BREAST_NUM_SHELLS,
    "cartilage": CARTILAGE_NUM_SHELLS,
    "connective tissue": CONNECTIVE_NUM_SHELLS,
    "glands": GLANDS_OTHERS_NUM_SHELLS,
    "liver": LIVER_NUM_SHELLS,
    "lung": LUNG_NUM_SHELLS,
    "muscle": MUSCLE_NUM_SHELLS,
    "PMMA": PMMA_NUM_SHELLS,
    "red marrow": RED_MARROW_NUM_SHELLS,
    "skin": SKIN_NUM_SHELLS,
    "soft tissue": SOFT_TISSUE_NUM_SHELLS,
    "stomach intestines": STOMACH_INTESTINES_NUM_SHELLS,
    "titanium": TITANIUM_NUM_SHELLS,
    "water": WATER_NUM_SHELLS,
}

COMPTON_DATA = {
    "adipose": ADIPOSE_COMPTON_DATA,
    "air": AIR_COMPTON_DATA,
    "blood": BLOOD_COMPTON_DATA,
    "bone": BONE_COMPTON_DATA,
    "brain": BRAIN_COMPTON_DATA,
    "breast": BREAST_COMPTON_DATA,
    "cartilage": CARTILAGE_COMPTON_DATA,
    "connective tissue": CONNECTIVE_COMPTON_DATA,
    "glands": GLANDS_OTHERS_COMPTON_DATA,
    "liver": LIVER_COMPTON_DATA,
    "lung": LUNG_COMPTON_DATA,
    "muscle": MUSCLE_COMPTON_DATA,
    "PMMA": PMMA_COMPTON_DATA,
    "red marrow": RED_MARROW_COMPTON_DATA,
    "skin": SKIN_COMPTON_DATA,
    "soft tissue": SOFT_TISSUE_COMPTON_DATA,
    "stomach intestines": STOMACH_INTESTINES_COMPTON_DATA,
    "titanium": TITANIUM_COMPTON_DATA,
    "water": WATER_COMPTON_DATA,
}


def sanity_check_compton_data():
    for mat in list(COMPTON_DATA.keys()):
        assert MATERIAL_NSHELLS[mat] == COMPTON_DATA[mat].shape[0]

    log.info("Compton data sanity has been checked!")
