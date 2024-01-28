from enum import auto
from strenum import StrEnum


class PatientPose(StrEnum):
    """Enum for patient pose.

    See https://dicom.innolitics.com/ciods/ct-image/general-series/00185100.

    """

    HFP = auto()
    HFS = auto()
    HFDR = auto()
    HFDL = auto()
    FFDR = auto()
    FFDL = auto()
    FFP = auto()
    FFS = auto()
    LFP = auto()
    LFS = auto()
    RFP = auto()
    RFS = auto()
    AFDR = auto()
    AFDL = auto()
    PFDR = auto()
    PFDL = auto()
