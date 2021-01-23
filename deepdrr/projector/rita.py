#
# Implementation of RITA algorithm as described in 'PENELOPE-2006: A Code System for Monte Carlo Simulation of Electron and Photon Transport'
#
from typing import Callable
import numpy as np

def make_rita_params(
    x_min: np.float32,
    x_max: np.float32,
    pdf_func: Callable[np.float32, np.float32],
    cdf_func: Callable[np.float32, np.float32],
    n_grid_points: np.int32
):
    """Determine the RITA parameters: x_i, y_i = CDF(x_i), a_i, b_i for i = 1, ..., n_grid_points

    We require:
        1. cdf_func(x_min) == 0.0
        2. cdf_func(x_max) == 1.0

    Args:
        x_min (np.float32): the lower bound of the interval to sample from
        x_max (np.float32): the upper bound of the interval to sample from
        pdf_func (Callable[np.float32, np.float32]): the analytical PDF (float -> float) of the function whose PDF we are sampling from
        cdf_func (Callable[np.float32, np.float32]): the analytical CDF (float -> float) of the function whose PDF we are sampling from
        n_grid_points (np.int32): the number of grid points to finish with.  Must be at least 10.
    
    Returns:
        np.ndarray: the x_i's (grid points) for RITA
        np.ndarray: the y_i's (Greek 'xi' values) for RITA
        np.ndarray: the a_i's for RITA
        np.ndarray: the b_i's for RITA
    """
    NUM_INITIAL_GRID_POINTS = 10

    assert NUM_INITIAL_GRID_POINTS <= n_grid_points
    
    # Initial grid setup
    delta_x = (x_max - x_min) / (NUM_INITIAL_GRID_POINTS - 1) # the initial size of the intervals
    x_arr = [(x_min + (k * delta_x)) for k in range(NUM_INITIAL_GRID_POINTS)]
    y_arr = [cdf_func(x_arr[i]) for i in range(len(x_arr))]
    pdf_arr = [pdf_func(x_arr[i]) for i in range(len(x_arr))] # not actually something that's returned. Just useful to store these values
    a_arr = []
    b_arr = []
    for i in range(len(x_arr) - 1):
        a, b = _rita_calc_ab_for_idx(x_arr, y_arr, pdf_arr, i)
        a_arr.append(a)
        b_arr.append(b)
    a_arr.append(0)
    b_arr.append(0)
    
    assert len(x_arr) == NUM_INITIAL_GRID_POINTS
    assert len(y_arr) == len(x_arr)
    assert len(a_arr) == len(x_arr)
    assert len(b_arr) == len(a_arr)

    errors = []
    for i in range(len(x_arr) - 1):
        epsilon = _rita_calc_interp_error(x_arr, y_arr, a_arr, b_arr, pdf_func, i) # TODO: implement this -- it's a numerically evaluated integral
        errors.append(epsilon)
    errors.append(0)
    assert len(errors) == len(x_arr)
    
    for i in range(NUM_INITIAL_GRID_POINTS, n_grid_points): 
        assert len(errors) == i
        _rita_add_gridpoint(x_arr, y_arr, pdf_arr, a_arr, b_arr, errors, pdf_arr, cdf_func)
    
    assert n_grid_points == len(x_arr)
    assert n_grid_points == len(y_arr)
    assert n_grid_points == len(a_arr)
    assert n_grid_points == len(b_arr)

    np_x = np.array(x_arr, dtype=np.float32)
    np_y = np.array(y_arr, dtype=np.float32)
    np_a = np.array(a_arr, dtype=np.float32)
    np_b = np.array(b_arr, dtype=np.float32)

    return np_x, np_y, np_a, np_b

def _rita_calc_ab_for_idx(
    x_arr,
    y_arr,
    pdf_arr,
    idx: np.int32
):
    """Calculate the RITA parameters a_{idx} and b_{idx}, given the gridpoint arrays x_arr and y_arr

    Args:
        x_arr (raw Python array): the x-values for RITA
        y_arr (raw Python array): the y-values for RITA
        pdf_arr (raw Python array): the PDF we are sampling from, applied to element-wise to each item in x_arr
        idx (np.int32): the index in the gridpoint arrays of the interval we are concerned with: [x_{idx}, x_{idx+1})

    Returns:
        np.float32: the RITA parameter a_{idx}
        np.float32: the RITA parameter b_{idx}
    """
    tmp = (y_arr[idx + 1] - y_arr[idx]) / (x_arr[idx + 1] - x_arr[idx])
    b = 1 - (tmp * tmp) / (pdf_arr[idx + 1] * pdf_arr[idx])
    a = (tmp / pdf_arr[idx]) - b - 1
    return a, b

def _rita_calc_interp_error(
    x_arr,
    y_arr,
    a_arr,
    b_arr,
    pdf_func: Callable[np.float32, np.float32],
    idx: np.int32,
):
    """Calculate the interpolation error \\epsilon_{idx}.  The formula for interpolation error is given in Eqn 1.57:

    \\epsilon_i = \\int_{x_i}^{x_{i+1}} |p(x) - \\widetilde{p}(x)| dx,

    where \\widetilde{p}(x) is defined by Eqns 1.55 and 1.56.  We compute this integral numerically.

    Args:
        x_arr (raw Python array): the x-values for RITA
        y_arr (raw Python array): the y-values for RITA
        a_arr (raw Python array): the 'a_i' parameters for RITA
        b_arr (raw Python array): the 'b_i' parameters for RITA
        pdf_func (Callable[np.float32, np.float32]): the analytical PDF (float -> float) of the function whose PDF we are sampling from
        idx (np.int32): the index of the interval we are calculating for
    """
    x_i = a_arr[idx]
    a_i = a_arr[idx]
    b_i = b_arr[idx]
    def calc_nu(x):
        tau = (x - x_i) / (x_arr[idx + 1] - x_i)
        big_term = 1 + a_i + b_i - (a_i * tau)
        inside_sqrt_term = (4 * b_i * tau * tau) / (big_term * big_term)
        right_factor = 1 - np.sqrt(1 - inside_sqrt_term)
        return (big_term * right_factor) / (2 * b_i * tau)
    
    def p_tilde(x):
        nu = calc_nu(x)
        b_nu2 = b_i * nu * nu
        partial_numer = 1 + (a_i * nu) + b_nu2
        numerator = (partial_numer * partial_numer) * (y_arr[idx + 1] - y_arr[idx])
        denominator = (1 + a_i + b_i) * (1 - b_nu2) * (x_arr[idx + 1] - x_i)
        return numerator / denominator

    # TODO: decide on a numerical integration methodology
    return NotImplemented

def _rita_add_gridpoint(
    x_arr,
    y_arr,
    pdf_arr,
    a_arr,
    b_arr,
    eps_arr,
    pdf_func: Callable[np.float32, np.float32],
    cdf_func: Callable[np.float32, np.float32]
):
    """Add a gridpoint for the RITA algorithm (within the interval with the currently-largest interpolation error)

    Args:
        x_arr (raw Python array): the x-values for RITA
        y_arr (raw Python array): the y-values for RITA
        pdf_arr (raw Python array): the PDF we are sampling from, applied to element-wise to each item in x_arr
        a_arr (raw Python array): the 'a_i' parameters for RITA
        b_arr (raw Python array): the 'b_i' parameters for RITA
        eps_arr (raw Python array): the interpolation errors for each interval
        pdf_func (Callable[np.float32, np.float32]): the analytical PDF (float -> float) of the function whose PDF we are sampling from
        cdf_func (Callable[np.float32, np.float32]): the analytical CDF (float -> float) of the function whose PDF we are sampling from
    
    Returns:
        Upon return, data and parameters for a new gridpoint will have been placed in the arrays
    """
    # Find the interval with the largest interpolation error thus far
    max_idx = eps_arr.index(max(eps_arr))
    # The interval [x_{max_idx}, x_{max_idx+1}] has the largest error
    
    # Split the interval with the largest interpolation error
    x_j = (x_arr[max_idx] + x_arr[max_idx + 1]) / 2

    # Emplace the new (x_j, y_j) gridpoint
    x_arr   =   x_arr[:max_idx + 1] + [         x_j ] +   x_arr[max_idx + 1:]
    y_arr   =   y_arr[:max_idx + 1] + [cdf_func(x_j)] +   y_arr[max_idx + 1:]
    pdf_arr = pdf_arr[:max_idx + 1] + [pdf_func(x_j)] + pdf_arr[max_idx + 1:]

    # Emplace the a_i and b_i for the upper of the two new intervals
    # (This corresponds to a_j and b_j, associated with x_j. j == max_idx+1)
    a, b = _rita_calc_ab_for_idx(x_arr, y_arr, pdf_arr, max_idx + 1)
    a_arr = a_arr[:max_idx + 1] + [a] + a_arr[max_idx + 1:]
    b_arr = b_arr[:max_idx + 1] + [b] + b_arr[max_idx + 1:]

    # Replace (emphasis on the 're-'!) the a_i and b_i for the lower of the 
    # two new intervals. (This corresponds to index max_idx)
    a, b = _rita_calc_ab_for_idx(x_arr, y_arr, pdf_arr, max_idx)
    a_arr[max_idx] = a
    b_arr[max_idx] = b

    # Emplace the interpolation error for the upper of the two new intervals
    epsilon_j = _rita_calc_interp_error(x_arr, y_arr, a_arr, b_arr, pdf_func, max_idx + 1)
    eps_arr = eps_arr[:max_idx + 1] + [epsilon_j] + eps_arr[max_idx + 1:]

    # Replace the interpolation error for the lower of the two new intervals
    eps_arr[max_idx] = _rita_calc_interp_error(x_arr, y_arr, a_arr, b_arr, pdf_func, max_idx)
    return

def sample_rita(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    a_arr: np.ndarray,
    b_arr: np.ndarray
) -> np.float32:
    """Using the provided RITA parameters, return an x-value based on the RITA-approximated PDF

    Args:
        x_arr (np.ndarray): the x_i's (grid points) for RITA
        y_arr (np.ndarray): the y_i's (Greek 'xi' values) for RITA
        a_arr (np.ndarray): the a_i's for RITA
        b_arr (np.ndarray): the b_i's for RITA

    Returns:
        np.float32: a randomly sampled x-value
    """
    assert 1 == x_arr.ndim
    assert 1 == y_arr.ndim
    assert 1 == a_arr.ndim
    assert 1 == b_arr.ndim

    N = x_arr.shape[0]
    assert N == y_arr.shape[0]
    assert N == a_arr.shape[0]
    assert N == b_arr.shape[0]

    y = np.random.random_sample() # U[0,1]

    # Binary search to find the interval [y_i, y_{i+1}] that contains y
    lo_idx = np.int32(0) # inclusive
    hi_idx = np.int32(N) # exclusive
    i = None            # the index of the interval we find y in
    while lo_idx < hi_idx:
        mid_idx = np.floor_divide(lo_idx + hi_idx, np.int32(2))

        # Check if mid_idx is the lower bound of the correct interval
        if y < y_arr[mid_idx]:
            # Need to check lower intervals
            hi_idx = mid_idx
        elif y < y_arr[mid_idx + 1]:
            # found correct interval
            i = mid_idx
            break
        else:
            # Need to check higher intervals
            lo_idx = mid_idx + 1
    
    assert (y_arr[i] <= y) and (y < y_arr[i + 1])
    
    nu = y - y_arr[i]
    delta_i = y_arr[i + 1] - y_arr[i]

    delta_i_nu = delta_i * nu
    numerator = (1 + a_arr[i] + b_arr[i]) * delta_i_nu
    denominator = (delta_i * delta_i) + (a_arr[i] * delta_i_nu) + (b_arr[i] * nu * nu)

    return x_arr[i] + (x_arr[i+1] - x_arr[i]) * numerator / denominator
