#
# TODO: cite the papers that form the basis of this code
#

from typing import Optional

import logging
import numpy as np
from . import spectral_data
from .. import geo


logger = logging.getLogger(__name__)


def simulate_scatter_no_vr(
    volume: np.ndarray,
    source: geo.Point3D,
    spectrum: Optional[np.ndarray] = spectral_data.spectrums['90KV_AL40'], 
    photon_count: Optional[int] = 10000000, # 10^7
    Eabs: Optional[float] = 1000
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter during an X-Ray, 
    without using VR (variance reduction) techniques.

    Args:
        volume (np.ndarray): the volume density data.
        source (Point3D): the source point for rays in the camera's IJK space
        spectrum (Optional[np.ndarray], optional): spectrum array.  Defaults to 90KV_AL40 spectrum.
        photon_count (Optional[int], optional): the number of photons simulated.  Defaults to 10^7 photons.
        Eabs (Optional[float], optional): the energy (in keV) at or below which photons are assumed to be absorbed by the materials.  Defaults to 1000 (keV).

    Returns:
        np.ndarray: intensity image of the photon scatter
    """
    return NotImplemented

def track_single_photon_no_vr(
    initial_pos: geo.Point3D,
    initial_dir: geo.Vector3D,
    initial_E: np.float32,
    N_vals: np.ndarray,
    sigma_C_vals: np.ndarray,
    sigma_R_vals: np.ndarray
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter of a single photon 
    during an X-Ray, without using VR (variance reduction) techniques.

    Args:
        initial_pos (geo.Point3D): the initial position of the photon, which is the X-Ray source
        initial_dir (geo.Vector3D): the initial direction of travel of the photon
        initital_E (np.float32): the initial energy of the photon
        N_vals (np.ndarray): a field over the volume of 'N', the number of molecules per unit volume.  Each cell in N_vals contains the information for the corresponding voxel.
        sigma_C_vals (np.ndarray): a field over the volume of 'sigma_C', the interactional cross-section for Compton scatter.  Each cell in sigma_C_vals contains the information for the corresponding voxel.
        sigma_R_vals (np.ndarray): a field over the volume of 'sigma_R', the interactional cross-section for Rayleigh scatter.  Each cell in sigma_R_vals contains the information for the corresponding voxel.

    Returns:
        np.ndarray: intensity image of the photon scatter
    """
    return NotImplemented

def get_scattered_dir(
    dir: geo.Vector3D,
    theta: np.float32,
    phi: np.float32
) -> geo.Vector3D:
    """Determine the new direction of travel after getting scattered

    Args:
        dir (geo.Vector3D): the incoming direction of travel
        theta (np.float32): the polar scattering angle, i.e. the angle dir and dir_prime
        phi (np.float32): the azimuthal angle, i.e. how dir_prime is rotated about the axis 'dir'.

    Returns:
        geo.Vector3D: the outgoing direction of travel
    """
    return NotImplemented

def sample_U01() -> np.float32:
    """Returns a value uniformly sampled from the interval [0,1]"""
    return NotImplemented

def sample_Rayleigh_theta_W(
    N_val: np.float32, 
    sigma_val: np.float32 # TODO: are these parameters actually the relevant ones?
) -> np.float32:
    """Randomly sample values of theta and W for a given Rayleigh scatter interaction

    Args:
        N_val: the number of molecules per unit volume at the location of the photon interaction
        sigma_val: the interactional cross-sectional area at the location of the photon interation

    Returns:
        np.float32: theta, the polar scattering angle 
        np.float32: W, the energy loss (will always be zero)
    """
    theta = NotImplemented
    W = 0 # by definition of Rayleigh scatter, E_incident = E_outgoing
    return theta, W

def sample_Compton_theta_W(
    N_val: np.float32, 
    sigma_val: np.float32 # TODO: are these parameters actually the relevant ones?
) -> np.float32:
    """Randomly sample values of theta and W for a given Compton scatter interaction

    Args:
        N_val: the number of molecules per unit volume at the location of the photon interaction
        sigma_val: the interactional cross-sectional area at the location of the photon interation

    Returns:
        np.float32: theta, the polar scattering angle 
        np.float32: W, the energy loss (will always be zero)
    """
    theta = NotImplemented
    W = NotImplemented
    return theta, W
