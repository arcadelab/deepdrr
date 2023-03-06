import numpy as np
from typing import overload


def get_threshold(h: np.ndarray, fraction: float = 0.5) -> float:
    """Get the threshold for a heatmap.

    Args:
        h (np.ndarray): A 2D array
        fraction (float): Fraction of the heatmap range to set the threshold at. Higher values keeps fewer pixels.

    """
    hmin = h.min()
    hmax = h.max()
    return hmin + (hmax - hmin) * fraction
