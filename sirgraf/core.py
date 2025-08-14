from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from astropy.visualization import ZScaleInterval

from .io import read_stack
from .instruments import infer_instrument_spec
from .background import compute_min_image, compute_uniform_background
from .mask import circular_mask

@dataclass
class ProcessResult:
    minimum: np.ndarray
    uniform: np.ndarray
    filtered: np.ndarray  # (t, y, x)
    mask: np.ndarray
    cmap_name: str
    inner_radius_rsun: float
    rsun_pix: float
    x_rsun: np.ndarray
    y_rsun: np.ndarray
    avg_profile_posY: np.ndarray
    dates: list[str]
    times: list[str]
    zscale_limits: tuple[float, float]

def _xy_arrays_rsun(nrows: int, ncols: int, cx: float, cy: float, rsun_pix: float):
    xpix = np.arange(ncols)
    ypix = np.arange(nrows)
    x_rsun = (xpix - cx) / rsun_pix
    y_rsun = (ypix - cy) / rsun_pix
    return x_rsun, y_rsun

def process_directory(path: str) -> ProcessResult:
    stack, metas, smeta = read_stack(path)

    spec = infer_instrument_spec(smeta.instrume, smeta.detector)

    # LASCO C3: flip Y orientation if requested
    if spec.flip_y:
        stack = stack[:, ::-1, :]

    min_img = compute_min_image(stack)
    uniform_img, avg_prof = compute_uniform_background(min_img, smeta.crpix1, smeta.crpix2)

    # Normalize each frame
    # Avoid division by zero
    safe_uniform = uniform_img.copy()
    safe_uniform[safe_uniform == 0] = np.nan

    filtered = (stack - min_img[None, :, :]) / safe_uniform[None, :, :]

    # Outer mask: circle at max |y| in Rsun
    nrows, ncols = min_img.shape
    x_rsun, y_rsun = _xy_arrays_rsun(nrows, ncols, smeta.crpix1, smeta.crpix2, smeta.rsun_pix)
    rr_pix = np.max(np.abs(y_rsun)) * smeta.rsun_pix
    mask = circular_mask(nrows, ncols, smeta.crpix1, smeta.crpix2, rr_pix)

    # ZScale limits on a representative median-blurred frame
    interval = ZScaleInterval()
    mid_idx = filtered.shape[0] // 2
    frame_mid = np.nan_to_num(filtered[mid_idx], nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = interval.get_limits(frame_mid)

    dates = [m.date for m in metas]
    times = [m.time for m in metas]

    return ProcessResult(
        minimum=min_img,
        uniform=uniform_img,
        filtered=filtered,
        mask=mask,
        cmap_name=spec.cmap_name,
        inner_radius_rsun=spec.inner_radius_rsun,
        rsun_pix=smeta.rsun_pix,
        x_rsun=x_rsun,
        y_rsun=y_rsun,
        avg_profile_posY=avg_prof,
        dates=dates,
        times=times,
        zscale_limits=(vmin, vmax),
    )