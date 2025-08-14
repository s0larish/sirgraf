from __future__ import annotations
import numpy as np
import cv2

def cartesian_to_polar(image: np.ndarray, cx: float, cy: float) -> np.ndarray:
    radius = float(np.hypot(image.shape[0]-cx, image.shape[1]-cy))
    return cv2.linearPolar(image, (cx, cy), radius, cv2.WARP_FILL_OUTLIERS)

def polar_to_cartesian(image_polar: np.ndarray, cx: float, cy: float, out_shape: tuple[int,int]) -> np.ndarray:
    radius = float(np.hypot(out_shape[0]-cx, out_shape[1]-cy))
    return cv2.linearPolar(image_polar, (cx, cy), radius,
                           cv2.WARP_FILL_OUTLIERS | cv2.WARP_INVERSE_MAP)

def azimuthal_profile_from_polar(polar_img: np.ndarray, y_nonneg_mask: np.ndarray) -> np.ndarray:
    # Average over azimuth (rows) for columns corresponding to y >= 0 in original
    # If mask is 1D over y, we just select columns by index.
    cols = np.where(y_nonneg_mask)[0]
    prof = np.empty(cols.size, dtype=float)
    # Mean along rows (axis=0) for each selected column
    prof[:] = polar_img[:, cols].mean(axis=0)
    return prof