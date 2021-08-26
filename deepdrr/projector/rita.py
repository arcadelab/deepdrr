#
# Implementation of RITA algorithm as described in 'PENELOPE-2006: A Code System for Monte Carlo Simulation of Electron and Photon Transport'
#
from typing import Callable, Optional
import numpy as np

class RITA:

    dtype = np.float64

    def __init__(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        a_arr: np.ndarray,
        b_arr: np.ndarray
    ):
        """Creates a RITA object, which stores the tables of RITA parameters

        Args:
            x_arr (np.ndarray): the x_i's (grid points) for RITA
            y_arr (np.ndarray): the y_i's (Greek 'xi' values) for RITA
            a_arr (np.ndarray): the a_i's for RITA
            b_arr (np.ndarray): the b_i's for RITA
        """
        assert 1 == x_arr.ndim
        assert 1 == y_arr.ndim
        assert 1 == a_arr.ndim
        assert 1 == b_arr.ndim

        self.n_grid_points = x_arr.shape[0]
        assert self.n_grid_points == x_arr.size
        assert self.n_grid_points == y_arr.size
        assert self.n_grid_points == a_arr.size
        assert self.n_grid_points == b_arr.size

        self.x_arr = x_arr.astype(self.dtype)
        self.y_arr = y_arr.astype(self.dtype)
        self.a_arr = a_arr.astype(self.dtype)
        self.b_arr = b_arr.astype(self.dtype)
    
    @classmethod
    def from_saved_params(
        cls,
        params: np.ndarray
    ):
        """Creates and returns a RITA object based on the saved RITA parameters.

        Args:
            params (np.ndarray): the saved parameters.  See mcgpu_rita_samplers.py:saved_rita_params dictionary for available options
        """
        np_x = np.ascontiguousarray(params[:,0])
        np_y = np.ascontiguousarray(params[:,1])
        np_a = np.ascontiguousarray(params[:,2])
        np_b = np.ascontiguousarray(params[:,3])

        # the saved RITA params should have 128 gridpoints
        assert 128 == np_x.size
        assert 128 == np_y.size
        assert 128 == np_a.size
        assert 128 == np_b.size

        return cls(np_x, np_y, np_a, np_b)

    @classmethod
    def from_pdf(
        cls, 
        x_min: dtype,
        x_max: dtype,
        pdf_func: Callable[[dtype], dtype],
        n_grid_points: Optional[np.int32] = 128
    ):
        """Creates and returns a RITA object from the provided PDF over the provided interval, using the specified number of gridpoints.

        Args:
            x_min (np.float64): the lower bound of the interval to sample from
            x_max (np.float64): the upper bound of the interval to sample from
            pdf_func (Callable[[np.float64], np.float64]): the analytical PDF (float -> float) of the function whose PDF we are sampling from
            n_grid_points (Optional[np.int32], optional): the number of grid points to finish with.  Must be at least 10.  Defaults to 128
        """
        def cdf_func(x):
            return numerically_integrate(pdf_func, x_min, x)
        NUM_INITIAL_GRID_POINTS = 10

        assert NUM_INITIAL_GRID_POINTS <= n_grid_points
        
        # Initial grid setup
        delta_x = (x_max - x_min) / (NUM_INITIAL_GRID_POINTS - 1) # the initial size of the intervals
        x_arr = [(x_min + (k * delta_x)) for k in range(NUM_INITIAL_GRID_POINTS)]
        y_arr = [cdf_func(x_arr[i]) for i in range(NUM_INITIAL_GRID_POINTS)]
        pdf_arr = [pdf_func(x_arr[i]) for i in range(NUM_INITIAL_GRID_POINTS)] # not actually something that's returned. Just useful to store these values
        a_arr = []
        b_arr = []
        for i in range(NUM_INITIAL_GRID_POINTS - 1):
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
        for i in range(NUM_INITIAL_GRID_POINTS - 1):
            epsilon = _rita_calc_interp_error(x_arr, y_arr, a_arr, b_arr, pdf_func, i)
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

        np_x = np.array(x_arr)
        np_y = np.array(y_arr)
        np_a = np.array(a_arr)
        np_b = np.array(b_arr)

        return cls(np_x, np_y, np_a, np_b)

    def sample_rita(
        self,
    ) -> np.float64:
        """Using the provided RITA parameters, return an x-value based on the RITA-approximated PDF

        Args:
            x_arr (np.ndarray): the x_i's (grid points) for RITA
            y_arr (np.ndarray): the y_i's (Greek 'xi' values) for RITA
            a_arr (np.ndarray): the a_i's for RITA
            b_arr (np.ndarray): the b_i's for RITA

        Returns:
            np.float64: a randomly sampled x-value
        """
        assert 1 == self.x_arr.ndim
        assert 1 == self.y_arr.ndim
        assert 1 == self.a_arr.ndim
        assert 1 == self.b_arr.ndim

        N = self.n_grid_points
        assert N == self.y_arr.shape[0]
        assert N == self.a_arr.shape[0]
        assert N == self.b_arr.shape[0]

        y = np.random.random_sample() # U[0,1]

        # Binary search to find the interval [y_i, y_{i+1}] that contains y
        lo_idx = np.int32(0) # inclusive
        hi_idx = np.int32(N) # exclusive
        i = None            # the index of the interval we find y in
        while lo_idx < hi_idx:
            mid_idx = np.floor_divide(lo_idx + hi_idx, np.int32(2))

            # Check if mid_idx is the lower bound of the correct interval
            if y < self.y_arr[mid_idx]:
                # Need to check lower intervals
                hi_idx = mid_idx
            elif y < self.y_arr[mid_idx + 1]:
                # found correct interval
                i = mid_idx
                break
            else:
                # Need to check higher intervals
                lo_idx = mid_idx + 1
        
        assert (self.y_arr[i] <= y) and (y < self.y_arr[i + 1])
        
        nu = y - self.y_arr[i]
        delta_i = self.y_arr[i + 1] - self.y_arr[i]

        delta_i_nu = delta_i * nu
        numerator = (1 + self.a_arr[i] + self.b_arr[i]) * delta_i_nu
        denominator = (delta_i * delta_i) + (self.a_arr[i] * delta_i_nu) + (self.b_arr[i] * nu * nu)

        return self.x_arr[i] + (self.x_arr[i+1] - self.x_arr[i]) * numerator / denominator

#
# HELPER FUNCTIONS for RITA initialization
#

def numerically_integrate(
    func: Callable[[np.float64], np.float64],
    x_min: np.float64,
    x_max: np.float64
):
    """Numerically integrates function 'func' on the interval [x_min,x_max] using the 20-point Gauss method (see page 261 of 'PENELOPE-2006')

    Args:
        func (Callable[[np.float64], np.float64]): the function to integrate
        x_min (np.float64): the lower integration bound
        x_max (np.float64): the upper integration bound
    
    Returns:
        np.float64: the result of numerically integrating 'func'
    """
    abscissas = [
        +7.6526521133497334e-02,
        -7.6526521133497334e-02,
        +2.2778585114164508e-01,
        -2.2778585114164508e-01,
        +3.7370608871541956e-01,
        -3.7370608871541956e-01,
        +5.1086700195082710e-01,
        -5.1086700195082710e-01,
        +6.3605368072651503e-01,
        -6.3605368072651503e-01,
        +7.4633190646015079e-01,
        -7.4633190646015079e-01,
        +8.3911697182221882e-01,
        -8.3911697182221882e-01,
        +9.1223442825132591e-01,
        -9.1223442825132591e-01,
        +9.6397192727791379e-01,
        -9.6397192727791379e-01,
        +9.9312859918509492e-01
        -9.9312859918509492e-01
    ]
    weights = [
        1.5275338713072585e-01,
        1.5275338713072585e-01,
        1.4917298647260375e-01,
        1.4917298647260375e-01,
        1.4209610931838205e-01,
        1.4209610931838205e-01,
        1.3168863844917663e-01,
        1.3168863844917663e-01,
        1.1819453196151842e-01,
        1.1819453196151842e-01,
        1.0193011981724044e-01,
        1.0193011981724044e-01,
        8.3276741576704749e-02,
        8.3276741576704749e-02,
        6.2672048334109064e-02,
        6.2672048334109064e-02,
        4.0601429800386941e-02,
        4.0601429800386941e-02,
        1.7614007139152118e-02,
        1.7614007139152118e-02
    ]

    assert len(abscissas) == 20
    assert len(weights) == 20
    for i in range(10):
        assert abscissas[2 * i] == (-1 * abscissas[2 * i + 1])
        assert weights[2 * i] == weights[2 * i + 1]

    avg = (x_min + x_max) / 2
    diff = (x_max - x_min) / 2
    f_z_i = [func(diff * abscissas[i] + avg) for i in range(len(abscissas))]
    np_w = np.array(weights).astype(np.float64)
    np_f = np.array(f_z_i).astype(np.float64)
    return diff * np.dot(np_w, np_f)

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
        np.float64: the RITA parameter a_{idx}
        np.float64: the RITA parameter b_{idx}
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
    pdf_func: Callable[[np.float64], np.float64],
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
        pdf_func (Callable[[np.float64], np.float64]): the analytical PDF (float -> float) of the function whose PDF we are sampling from
        idx (np.int32): the index of the interval we are calculating for
    """
    x_i = a_arr[idx]
    a_i = a_arr[idx]
    b_i = b_arr[idx]

    interval = x_arr[idx + 1] - x_i # the length of the interval
    def calc_nu(x):
        tau = (x - x_i) / interval
        big_term = 1 + a_i + b_i - (a_i * tau)
        inside_sqrt_term = (4 * b_i * tau * tau) / (big_term * big_term)
        right_factor = 1 - np.sqrt(1 - inside_sqrt_term)
        return (big_term * right_factor) / (2 * b_i * tau)
    
    def p_tilde(x):
        nu = calc_nu(x)
        b_nu2 = b_i * nu * nu
        partial_numer = 1 + (a_i * nu) + b_nu2
        numerator = (partial_numer * partial_numer) * (y_arr[idx + 1] - y_arr[idx])
        denominator = (1 + a_i + b_i) * (1 - b_nu2) * interval
        return numerator / denominator
    
    def integrand(x):
        return np.absolute(pdf_func(x) - p_tilde(x))

    # Using extended Simpson's rule with 51 points
    h = interval / 50
    odd_sum = sum([integrand(x_i + k * h) for k in range(1, 50, 2)]) # 1, 3, ..., 49
    even_sum = sum([integrand(x_i + k * h) for k in range(2, 49, 2)]) # 2, 4, ..., 48
    # error_term = (h * h * h * h * h) * fourth_derivative(x_star) * 25 / 90 # don't actually need to figure out how to handle this
    return (h / 3) * (integrand(x_i) + (4 * odd_sum) + (2 * even_sum) + integrand(x_i + interval)) # - error_term

def _rita_add_gridpoint(
    x_arr,
    y_arr,
    pdf_arr,
    a_arr,
    b_arr,
    eps_arr,
    pdf_func: Callable[[np.float64], np.float64],
    cdf_func: Callable[[np.float64], np.float64]
):
    """Add a gridpoint for the RITA algorithm (within the interval with the currently-largest interpolation error)

    Args:
        x_arr (raw Python array): the x-values for RITA
        y_arr (raw Python array): the y-values for RITA
        pdf_arr (raw Python array): the PDF we are sampling from, applied to element-wise to each item in x_arr
        a_arr (raw Python array): the 'a_i' parameters for RITA
        b_arr (raw Python array): the 'b_i' parameters for RITA
        eps_arr (raw Python array): the interpolation errors for each interval
        pdf_func (Callable[[np.float64], np.float64]): the analytical PDF (float -> float) of the function whose PDF we are sampling from
        cdf_func (Callable[[np.float64], np.float64]): the analytical CDF (float -> float) of the function whose PDF we are sampling from
    
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
