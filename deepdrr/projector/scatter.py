#
# TODO: cite the papers that form the basis of this code
#

from typing import Tuple, Optional, Dict, Callable

import logging
import numpy as np
import spectral_data
from deepdrr import geo
from deepdrr import vol
from rita import RITA

from .mcgpu_mfp_data import mfp_data
from .mcgpu_compton_data import MAX_NSHELLS as COMPTON_MAX_NSHELLS
from .mcgpu_compton_data import material_nshells, compton_data

from .mcgpu_rita_samplers import rita_samplers

import math # for 'count_milestones'


logger = logging.getLogger(__name__)


#
# USEFUL CONSTANTS
#

LIGHT_C_METERS = 2.99792458e08 # m/s
ELECTRON_MASS_KILOS = 9.1093826e-31 # kg

LIGHT_C = 2.99792458e10 # cm/s
ELECTRON_MASS = 9.1093826e-28 # g

ELECTRON_REST_ENERGY = 510998.918 # eV


def simulate_scatter_no_vr(
    volume: vol.Volume,
    source: geo.Point3D,
    output_shape: Tuple[int, int],
    spectrum: Optional[np.ndarray] = spectral_data.spectrums['90KV_AL40'], 
    photon_count: Optional[int] = 10000000, # 10^7
    E_abs: Optional[np.float32] = 5000
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter during an X-Ray, 
    without using VR (variance reduction) techniques.

    Args:
        volume (np.ndarray): the volume density data.
        source (Point3D): the source point for rays in the camera's IJK space
        spectrum (Optional[np.ndarray], optional): spectrum array.  Defaults to 90KV_AL40 spectrum.
        photon_count (Optional[int], optional): the number of photons simulated.  Defaults to 10^7 photons.
        E_abs (Optional[np.float32], optional): the energy (in keV) at or below which photons are assumed to be absorbed by the materials.  Defaults to 1000 (eV).
        output_shape (Tuple[int, int]): the {height}x{width} dimensions of the output image

    Returns:
        np.ndarray: intensity image of the photon scatter
    """
    count_milestones = [int(math.pow(10, i)) for i in range(int(1 + math.ceil(math.log10(photon_count))))] # [1, 10, 100, ..., 10^7] in default case

    accumulator = np.zeros(output_shape).astype(np.float32)

    material_ids = {}
    for i, mat_name in enumerate(volume.materials.keys()):
        material_ids[i] = mat_name
    assert len(material_ids) > 0
    
    # Convert the volume segmentation data from one-hot to [0..N-1]-labeled
    assert np.all(np.equal(volume.data.shape, volume.materials[material_ids[0]].shape))
    labeled_seg = np.empty_like(volume.data)

    for ilabel in material_ids.keys():
        labeled_seg = np.add(labeled_seg, ilabel * volume.materials[material_ids[ilabel]])

    for i in range(photon_count):
        if (i+1) in count_milestones:
            print(f"Simulating photon history {i+1} / {photon_count}")
        BLAH = 0
        initial_dir = BLAH # Random sampling is a TODO -- need to figure out this geometry
        initial_E = sample_initial_energy(spectrum)
        single_scatter = track_single_photon_no_vr(source, initial_dir, initial_E, E_abs, labeled_seg, volume.data, material_ids)
        accumulator = accumulator + single_scatter
    
    print(f"Finished simulating {photon_count} photon histories")
    return NotImplemented

def track_single_photon_no_vr(
    initial_pos: geo.Point3D,
    initial_dir: geo.Vector3D,
    initial_E: np.float32,
    E_abs: np.float32,
    labeled_seg: np.ndarray,
    density_vol: np.ndarray,
    material_ids: Dict[int, str]
) -> np.ndarray:
    """Produce a grayscale (intensity-based) image representing the photon scatter of a single photon 
    during an X-Ray, without using VR (variance reduction) techniques.

    Args:
        initial_pos (geo.Point3D): the initial position of the photon, which is the X-Ray source
        initial_dir (geo.Vector3D): the initial direction of travel of the photon
        initital_E (np.float32): the initial energy of the photon
        E_abs (np.float32): the energy (in keV) at or below which photons are assumed to be absorbed by the materials.
        labeled_seg (np.ndarray): a [0..N-1]-labeled segmentation of the volume
        density_vol (np.ndarray): the density information of the volume
        material_ids (Dict[int,str]): a dictionary mapping an integer material ID-label to the name of the material
    Returns:
        np.ndarray: intensity image of the photon scatter
    """
    BLAH = 0
    # find the point on the boundary of the volume that the photon from the source first reaches
    pos = BLAH
    dir = initial_dir

    photon_energy = initial_E # tracker variable throughout the duration of the photon history

    s = 0 # So that the photon doesn't move upon entering the loop

    while True: # emulate a do-while loop
        voxel_coords = VOXEL_FROM_POS(pos) # TODO: implement this
        mat_label = labeled_seg[voxel_coords]
        mat_name = material_ids[mat_label]

        # NOTE: I could perform some interpolation to get slighter more accurate mfp data
        mfp_data_E_bin = CHOOSE_ENERGY_BIN(photon_energy) # TODO: implement

        mfp_Ra = mfp_data[mat_name][mfp_data_E_bin, 1]
        mfp_Co = mfp_data[mat_name][mfp_data_E_bin, 2]
        mfp_Tot = mfp_data[mat_name][mfp_data_E_bin, 4]

        # simulate moving the photon
        s = -1 * mfp_Tot * np.log(sample_U01())
        pos = pos + (s * dir)

        # Handle crossings of segementation boundary
        BLAH = 0 # TODO -- handle crossing boundaries

        will_exit_volume = BLAH
        if will_exit_volume:
            break

        # Sample the photon interaction type
        #
        # (1 / mfp_Tot) * (1 / molecules_per_vol) ==    total interaction cross section =: sigma_Tot
        # (1 / mfp_Ra ) * (1 / molecules_per_vol) == Rayleigh interaction cross section =: sigma_Ra
        # (1 / mfp_Co ) * (1 / molecules_per_vol) ==  Compton interaction cross section =: sigma_Co
        #
        # SAMPLING RULE: Let rnd be a uniformly selected number on [0,1]
        # 
        # if rnd > simga_Co / sigma_Tot: # if rnd > (mfp_Tot / mfp_Co)
        #   COMPTON INTERACTION
        # elif rnd > (sigma_Ra + sigma_Co) / sigma_Tot: # if rnd > mfp_Tot * ((1 / mfp_Co) + (1 / mfp_Ra))
        #   RAYLEIGH INTERACTION
        # else:
        #   DELTA INTERACTION (AKA no interaction, continue on course)
        # 
        cos_theta, E_prime = None, None
        rnd = sample_U01()
        if rnd > (mfp_Tot / mfp_Co):
            cos_theta, E_prime = sample_Compton_theta_E_prime(photon_energy, material_nshells[mat_name], compton_data[mat_name])
        elif rnd > mfp_Tot * ((1 / mfp_Co) + (1 / mfp_Ra)):
            cos_theta = sample_Rayleigh_theta(photon_energy, rita_samplers[mat_name])
            E_prime = photon_energy
        else:
            # "delta" interaction
            cos_theta = 0
            E_prime = photon_energy
        
        photon_energy = E_prime
        if photon_energy <= E_abs:
            break
        
        phi = 2 * np.pi * sample_U01()
        dir = get_scattered_dir(dir, cos_theta, phi)

        # END WHILE
    
    # final processing
    BLAH = BLAH

    return NotImplemented

def get_scattered_dir(
    dir: geo.Vector3D,
    cos_theta: np.float32,
    phi: np.float32
) -> geo.Vector3D:
    """Determine the new direction of travel after getting scattered

    Args:
        dir (geo.Vector3D): the incoming direction of travel
        cos_theta (np.float32): the cosine of the polar scattering angle, i.e. the angle dir and dir_prime
        phi (np.float32): the azimuthal angle, i.e. how dir_prime is rotated about the axis 'dir'.

    Returns:
        geo.Vector3D: the outgoing direction of travel
    """
    dx = dir.data[0]
    dy = dir.data[1]
    dz = dir.data[2]

    # since \theta is restricted to [0, \pi], sin_theta is restricted to [0,1]
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    tmp = np.sqrt(1 - dz * dz)

    # See PENELOPE-2006 Eqn 1.131
    new_dx = dx * cos_theta + sin_theta * (dx * dz * cos_phi - dy * sin_phi) / tmp
    new_dy = dy * cos_theta + sin_theta * (dy * dz * cos_phi - dx * sin_phi) / tmp
    new_dz = dz * cos_theta - tmp * sin_theta * cos_phi

    # Normalize the new direction vector
    new_mag = np.sqrt((new_dx * new_dx) + (new_dy * new_dy) + (new_dz * new_dz))

    new_dx = new_dx / new_mag
    new_dy = new_dy / new_mag
    new_dz = new_dz / new_mag

    return geo.Vector3D.from_array(np.array([new_dx, new_dy, new_dz]))

def sample_U01() -> np.float32:
    """Returns a value uniformly sampled from the interval [0,1]"""
    # TODO: implement RANECU? Could be useful for validating translation to CUDA
    return np.random.random_sample()

def sample_initial_energy(spectrum: np.ndarray) -> np.float32:
    """Determine the energy (in keV) of a photon emitted by an X-Ray source with the given spectrum

    Args:
        spectrum (np.ndarray): the data associated with the spectrum.  Cross-reference spectral_data.py
    
    Returns:
        np.float32: the energy of a photon, in eV
    """
    total_count = sum(spectrum[:,1])
    threshold = sample_U01() * total_count
    accumulator = 0
    for i in range(spectrum.shape[0] - 1):
        accumulator = accumulator + spectrum[i, 1]
        if accumulator >= threshold:
            return spectrum[i, 0]
    
    # If threshold hasn't been reached yet, we must have sampled the highest energy level
    return spectrum[-1, 0]

def sample_Rayleigh_theta(
    photon_energy: np.float32,
    rayleigh_sampler: RITA
) -> np.float32:
    """Randomly sample values of theta for a given Rayleigh scatter interaction
    Based on page 49 of paper 'PENELOPE-2006: A Code System for Monte Carlo Simulation of Electron and Photon Transport'

    Note that the materials files distributed with MC-GPU_v1.3 (https://code.google.com/archive/p/mcgpu/downloads) uses
    Form Factor data from PENELOPE-2006 files.  Accordingly, the (unnormalized) PDF is factored as:
            p_{Ra}(\\cos \\theta) = g(\\cos \\theta)[F(x,Z)]^2
    not
            p_{Ra}(\\cos \\theta) = g(\\cos \\theta)[F(q,Z)]^2
    Accordingly, we compute cos(theta) using the x-values, not the q-values

    Args:
        photon_energy (np.float32): the energy of the incoming photon
        rayleigh_sampler (RITA): the RITA sampler object for the material at the location of the interaction

    Returns:
        np.float32: cos(theta), where theta is the polar scattering angle 
    """
    kappa = photon_energy / ELECTRON_REST_ENERGY
    # Sample a random value of x^2 from the distribution pi(x^2), restricted to the interval (0, x_{max}^2)
    x_max = 20.6074 * 2 * kappa
    x_max2 = x_max * x_max
    x2 = rayleigh_sampler.sample_rita()
    while (x2 > x_max2):
        # Resample until x^2 is in the interval (0, x_{max}^2)
        x2 = rayleigh_sampler.sample_rita()

    while True:
        # Set cos_theta
        cos_theta = 1 - (2 * x2 / x_max2)

        # Test cost_theta
        g = (1 + cos_theta * cos_theta) / 2

        if sample_U01() <= g:
            break

    return cos_theta

def sample_Compton_theta_E_prime(
    photon_energy: np.float32,
    mat_nshells: np.int32,
    mat_compton_data: np.ndarray
) -> np.float32:
    """Randomly sample values of theta and W for a given Compton scatter interaction

    Args:
        photon_energy (np.float32): the energy of the incoming photon
        mat_nshells (np.int32): the number of electron shells in the material being interacted with
        mat_compton_data (np.ndarray): the Compton scatter data for the material being interacted with.  See mcgpu_compton_data.py for more details 

    Returns:
        np.float32: cos_theta, the polar scattering angle 
        np.float32: E_prime, the energy of the outgoing photon
    """
    kappa = photon_energy / ELECTRON_REST_ENERGY

    a_1 = np.log1p(2 * kappa)
    one_p2k = 1 + 2 * kappa
    a_2 = 2 * kappa * (1 + kappa) / (one_p2k * one_p2k)

    tau_min = 1 / one_p2k

    ### Sample cos_theta

    # Compute S(E, \theta=\pi) here, since it does not depend on cos_theta
    s_pi = 0
    for shell in range(mat_nshells):
        U_i = mat_compton_data[shell, 1]
        if photon_energy > U_i: # this serves as the Heaviside function
            left_term = photon_energy * (photon_energy - U_i) * 2 # since (1 - \cos(\theta=\pi)) == 2
            p_i_max = (left_term - ELECTRON_REST_ENERGY * U_i) / (LIGHT_C * np.sqrt(2 * left_term + U_i * U_i))
            # NOTE: the above line uses LIGHT_C in the denominator on the assumption that 
            # mat_compton_data[shell,2], from the Compton data files, contains the value J_{i,0}.
            # If the data files instead store the value (J_{i,0} m_{e} c), I should replace the
            # instance of LIGHT_C with ELECTRON_REST_ENERGY
            
            # Use several steps to calculate n_{i}(p_{i,max})
            tmp = mat_compton_data[shell, 2] * p_i_max # J_{i,0} p_{i,max}
            tmp = (1 - tmp - tmp) if (p_i_max < 0) else (1 + tmp + tmp)
            exponent = 0.5 - 0.5 * tmp * tmp
            tmp = 0.5 * np.exp(exponent)
            if (p_i_max > 0):
                tmp = 1 - tmp
            # 'tmp' now holds n_{i}(p_{i,max})

            s_pi = s_pi + mat_compton_data[shell, 0] * tmp # Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
    # s_pi is now set

    cos_theta = None
    # local storage for the results of calculating n_{i}(p_{i,max})
    n_p_i_max_vals = [0 for i in range(COMPTON_MAX_NSHELLS)]

    while True: # emulate do-while loop
        i = 1 if sample_U01() < (a_1 / (a_1 + a_2)) else 2 # in CUDA code, we will be able to avoid using a variable to store i
        trnd = sample_U01() # random number for calculating tau
        tau = np.power(tau_min, trnd) if (1 == i) else np.sqrt(trnd + tau_min * tau_min * (1 - trnd))
        cos_theta = 1 - (1 - tau) / (kappa * tau)

        # Compute S(E, \theta)
        s_theta = 0
        one_minus_cos = 1 - cos_theta
        for shell in range(mat_nshells):
            U_i = mat_compton_data[shell, 1]
            if photon_energy > U_i: # this serves as the Heaviside function
                left_term = photon_energy * (photon_energy - U_i) * one_minus_cos
                p_i_max = (left_term - ELECTRON_REST_ENERGY * U_i) / (LIGHT_C * np.sqrt(2 * left_term + U_i * U_i))
                # NOTE: the above line uses LIGHT_C in the denominator on the assumption that 
                # mat_compton_data[shell,2], from the Compton data files, contains the value J_{i,0}.
                # If the data files instead store the value (J_{i,0} m_{e} c), I should replace the
                # instance of LIGHT_C with ELECTRON_REST_ENERGY
                
                # Use several steps to calculate n_{i}(p_{i,max})
                tmp = mat_compton_data[shell, 2] * p_i_max # J_{i,0} p_{i,max}
                tmp = (1 - tmp - tmp) if (p_i_max < 0) else (1 + tmp + tmp)
                exponent = 0.5 - 0.5 * tmp * tmp
                tmp = 0.5 * np.exp(exponent)
                if (p_i_max > 0):
                    tmp = 1 - tmp
                # 'tmp' now holds n_{i}(p_{i,max})

                n_p_i_max_vals[shell] = tmp # for later use in sampling E_prime

                s_theta = s_theta + mat_compton_data[shell, 0] * tmp # Equivalent to: s_pi += f_{i} n_{i}(p_{i,max})
            else:
                n_p_i_max_vals[shell] = 0

        # s_theta is now set

        # Compute the term of T(cos_theta) that does not involve S(E,\theta)
        T_tau_term = 1 - ((1 - tau) * ((2 * kappa + 1) * tau - 1)) / (kappa * kappa * tau * (1 + tau * tau))

        # Test for acceptance
        if (s_pi * sample_U01()) <= (T_tau_term * s_theta):
            break
    
    # cos_theta is set by now

    # Choose the active shell
    while True: # emulate do-while loop
        #
        # Steps:
        #   1. Choose a threshold value in range [0, s_theta]
        #   2. Accumulate the partial sum of f_{i} \Theta(E - U_i) n_{i}(p_{i,max}) over the electron shells
        #   3. Once the partial sum reaches the threshold value, we 'return' the most recently considered 
        #       shell. In this manner, we select the active electron shell with relative probability equal 
        #       to f_{i} \Theta(E - U_i) n_{i}(p_{i,max}).
        #   4. Calculate a random value of p_z
        #   5. Reject p_z and start over if p_z < -1 * m_{e} * c
        #   6. Calculate F_{max} and F_{p_z} and reject appropriately
        #
        threshold = sample_U01() * s_theta
        accumulator = 0
        active_shell = None
        for shell in range(mat_nshells): 
            accumulator += mat_compton_data[shell, 0] * n_p_i_max_vals[shell]
            if (accumulator >= threshold):
                active_shell = shell
                break
        # active_shell is now set

        p_z = None
        two_A = sample_U01() * 2 * n_p_i_max_vals[active_shell]
        if two_A < 1:
            p_z = 0.5 - np.sqrt(0.25 - 0.5 * np.log(two_A))
        else:
            p_z = np.sqrt(0.25 - 0.5 * np.log(2 - two_A)) - 0.5
        p_z = p_z / mat_compton_data[active_shell,2] # Equivalent to: p_z = p_z / J_{i,0}, completing the calculation

        if p_z < -1:
            continue
            # NOTE: the above if-statement condition uses -1 on the assumption that 
            # mat_compton_data[shell,2], from the Compton data files, contains the value 
            # (J_{i,0} m_{e} c). If the data files instead store the value J_{i,0}, I should 
            # replace the instance of (-1) with (-1 * m_{e} * c)
        
        # Calculate F(p_z), where p_z is the PENELOPE-2006 'p_z' divided by (m_{e} c)
        # NOTE: I think the MC-GPU variable 'pzomc' might mean "p_z over (m_{e} c)"
        beta2 = 1 + (tau * tau) - (2 * tau * cos_theta) # beta2 = (\beta)^2, where \beta := (c q_{C}) / E
        beta_tau_factor = np.sqrt(beta2) * (1 + tau * (tau - cos_theta) / beta2)
        F_p_z = 1 + beta_tau_factor * p_z
        F_max = 1 + beta_tau_factor * (0.2 * (-1 if p_z < 0 else 1))
        # NOTE: when converting to CUDA, I will want to see what happens when I "multiply everything through" by beta2.
        # That way, when comparing F_p_z with (\xi * F_max), there will only be multiplications and no divisions

        if sample_U01() * F_max < F_p_z:
            break # p_z is accepted
    
    # p_z is now set. Calculate E_ratio = E_prime / E
    t = p_z * p_z
    term_tau = 1 - t * tau * tau
    term_cos = 1 - t * tau * cos_theta
    
    tmp = np.sqrt(term_cos * term_cos - term_tau * (1 - t))
    if p_z < 0:
        tmp = -1 * tmp
    tmp = term_cos + tmp

    E_ratio = tau * tmp / term_tau

    return cos_theta, E_ratio * photon_energy
