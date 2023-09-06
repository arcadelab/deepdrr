try:
    import cupy
except ImportError:
    raise ImportError(
        """CuPy must be installed to use DeepDRR.
Please install the version of CuPy for your CUDA Toolkit version by following the instructions here: https://cupy.dev/
Or by installing deepdrr with the optional CuPy extra for your CUDA Toolkit version:
pip install deepdrr[cuda102] # for CUDA 10.2
pip install deepdrr[cuda110] # for CUDA 11.0
pip install deepdrr[cuda111] # for CUDA 11.1
pip install deepdrr[cuda11x] # for CUDA 11.2 - 11.8
pip install deepdrr[cuda12x] # for CUDA 12.x

You can find which version of CUDA you have by running `nvcc --version` in your terminal.
"""
    )

from . import vis, geo, projector, device, annotations, utils
from .projector import Projector
from .vol import Volume, Mesh
from .device import CArm, MobileCArm, SimpleDevice
from .annotations import LineAnnotation
from .logging import setup_log


__all__ = [
    "MobileCArm",
    "CArm",
    "SimpleDevice",
    "Volume",
    "Mesh",
    "Projector",
    "vis",
    "geo",
    "projector",
    "device",
    "annotations",
    "utils",
    "setup_log",
]
