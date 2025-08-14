from __future__ import annotations
import numpy as np

def circular_mask(nrows: int, ncols: int, cx: float, cy: float, radius_pix: float) -> np.ndarray:
    y, x = np.ogrid[:nrows, :ncols]
    return (x - cx)**2 + (y - cy)**2 > radius_pix**2  # True outside circle