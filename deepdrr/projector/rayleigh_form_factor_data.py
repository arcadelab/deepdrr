#
# TODO: find and "upload" the form factor coefficient data
#

from typing import Callable
import numpy as np

from material_composition_data import material_compositions

def build_form_factor_func(mat: str) -> Callable[np.float32, np.float32]:
    """Generates and returns, for the given material, the Rayleigh form factor function \\pi(x^2), per Eqn 2.16

    Args:
        mat (str): the material, e.g. 'bone', 'air', 'soft tissue'
    
    Returns:
        Callable[np.float32, np.float32]: the function \\pi(x^2), returning a value on the interval [0,1], as specified by Eqn 2.16
    """
    return NotImplemented
