#!/usr/bin/env python3

from pydicom.data import get_testdata_file
import deepdrr
from deepdrr import geo


def test_simple():
    file_path = get_testdata_file("CT_small.dcm")
    volume = deepdrr.Volume.from_dicom(file_path)
    print(volume)
