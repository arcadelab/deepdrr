from typing import Union
import numpy as np
from numpy.typing import NDArray


def log_interp(
    xInterp: Union[float, NDArray[np.float_]],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
) -> Union[float, NDArray[np.float_]]:
    """
    Performs logarithmic interpolation of y values at given x values.
    The function takes the logarithm of the x values and performs linear interpolation in the log space.
    It then returns the interpolated y values in the original space.
    Args:
        xInterp: The x values at which to interpolate (can be a single value or an array).
        x: The x values of the known data points (must be a 1D array).
        y: The y values of the known data points (must be a 1D array).
    Returns:
        The interpolated y values at the specified xInterp values (same type as xInterp).
    """
    if isinstance(xInterp, np.ndarray):
        xInterp = np.log10(xInterp.copy())
    else:
        xInterp = np.log10(xInterp)
    x = np.log10(x.copy())
    y = np.log10(y.copy())
    yInterp = np.power(
        10, np.interp(xInterp, x, y)
    )  # np.interp is 1-D linear interpolation
    return yInterp
