from typing import Dict, Optional
import logging
import numpy as np
from pathlib import Path

from . import Volume
from .. import geo
from ..utils import data_utils

logger = logging.getLogger(__name__)


class KWire(Volume):
    """A provided volume representing a steel K wire, based on a straight cylinder with a conical tip.

    Args:
        world_from_anatomical (Optional[geo.FrameTransform], optional): transformation from the anatomical space to world coordinates. If None, assumes identity. Defaults to None.
    """
    url = "https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EUd4AAG7svFJotzyWjcjd7cBc-iuAfHL819iO8u0bCYN0A?e=rDWrM7"
    filename = "Kwire2.nii.gz"

    def __new__(cls, world_from_anatomical: Optional[geo.FrameTransform] = None):
        path = data_utils.download(cls.url, filename=cls.filename)
        return cls.from_nifti(path, world_from_anatomical=world_from_anatomical, use_thresholding=True)

    def __init__(
        self,
        data: np.ndarray,
        materials: Dict[str, np.ndarray],
        anatomical_from_ijk: geo.FrameTransform,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
    ) -> None:
        super(KWire, self).__init__(data, materials,
                                    anatomical_from_ijk, world_from_anatomical)

    @staticmethod
    def _convert_hounsfield_to_density(hu_values: np.ndarray):
        # TODO: verify coefficient.
        return 30 * hu_values

    @staticmethod
    def _segment_materials(
        hu_values: np.ndarray, use_thresholding: bool = True
    ) -> Dict[str, np.ndarray]:
        if not use_thresholding:
            raise NotImplementedError

        return dict(iron=(hu_values > 0))
