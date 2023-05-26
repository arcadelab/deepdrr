#============================================================================
#Copyright (c) 2023, Raymond Leung
#All rights reserved.
#
#This source code is licensed under the BSD-3-clause license.
#See the LICENSE.md file in the root directory for details.
#
#Purpose: Encapsulate the cuda module source code
#This version brings in new changes from commit 9ec0642ccc596b33
#============================================================================
import os

def get_cuda_template():
    # read pycuda_source.cu
    with open(os.path.join(os.path.dirname(__file__), 'pycuda_source.cu'), 'r') as f:
        cuda_template = f.read()
    return cuda_template
