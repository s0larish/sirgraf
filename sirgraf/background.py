from __future__ import annotations
import numpy as np
from skimage import util as skutil
from .polar import cartesian_to_polar, polar_to_cartesian

def compute_min_image(stack: np.ndarray) -> np.ndarray:
    # Robust min ignoring non-positive values: use masked min along time axis
    # Replace non-positive with +inf to avoid biasing the min
    safe = stack.copy()
    safe[safe <= 0] = np.inf
    min_img = np.nanmin(safe, axis=0)
    # If a pixel was all <=0, set to 0
    min_img[np.isinf(min_img)] = 0.0
    return min_img

def compute_uniform_background(img: np.ndarray, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (uniform_image, average_profile_ypos)."""
    image = skutil.img_as_float(img)
    polar = cartesian_to_polar(image, cx, cy)

    # Make a uniform image by averaging each column (azimuthal average)
    col_means = np.nanmean(polar, axis=0)
    polar_uniform = np.tile(col_means, (polar.shape[0], 1))

    uniform = polar_to_cartesian(polar_uniform, cx, cy, out_shape=image.shape)

    # For convenience also return the positive-Y profile (center outward)
    # Build a y array in pixels and take y >= 0 indices
    nrows, ncols = image.shape
    y = np.arange(nrows) - cy
    y_nonneg_mask = (y >= 0)
    avg_profile = col_means[y_nonneg_mask[:col_means.size]] if col_means.size <= y_nonneg_mask.size else col_means

    return uniform, avg_profile


# from __future__ import annotations
# from typing import Iterable, List, Literal, Optional, Tuple, Union
# import numpy as np
#
# try:
#     from astropy.io import fits  # optional: only needed for FITS paths
#     _HAS_ASTROPY = True
# except Exception:
#     _HAS_ASTROPY = False
#
# __all__ = ["create_static_background", "create_uniform_background"]
#
# ArrayLike3D = np.ndarray
# PathList = List[str]
# Method = Literal["percentile", "min", "block_median_min", "block_min_median"]
#
# def _to_dtype(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
#     """NumPy 2.0-safe cast: allow copy if needed; avoid copy if already correct dtype."""
#     a = np.asarray(a)
#     return a.astype(dtype, copy=False) if a.dtype != dtype else a
#
#
# # ---------------------------
# # Core background estimator
# # ---------------------------
# def _as_cube(
#     data_or_paths: Union[ArrayLike3D, PathList, np.ndarray],
#     hdu: int = 0,
#     dtype: np.dtype = np.float32,
# ) -> Tuple[Iterable[np.ndarray], int, int, int]:
#     """Yield frames and report (T,H,W). Accepts 2D (treated as T=1), 3D, or FITS paths."""
#     if isinstance(data_or_paths, np.ndarray):
#         if data_or_paths.ndim == 2:
#             H, W = data_or_paths.shape
#             def gen():
#                 yield _to_dtype(data_or_paths, dtype)
#             return gen(), 1, H, W
#         if data_or_paths.ndim == 3:
#             T, H, W = data_or_paths.shape
#             def gen():
#                 for i in range(T):
#                     yield _to_dtype(data_or_paths[i], dtype)
#             return gen(), T, H, W
#         raise ValueError("Array input must be 2D (H,W) or 3D (T,H,W).")
#
#     if not isinstance(data_or_paths, (list, tuple)) or len(data_or_paths) == 0:
#         raise ValueError("Path list must be non-empty.")
#
#     if not _HAS_ASTROPY:
#         raise ImportError("astropy is required for FITS path input.")
#
#     with fits.open(data_or_paths[0], memmap=True) as hdul:
#         H, W = np.asarray(hdul[hdu].data).shape
#     T = len(data_or_paths)
#
#     def gen():
#         for p in data_or_paths:
#             with fits.open(p, memmap=True) as hdul:
#                 yield _to_dtype(hdul[hdu].data, dtype)
#     return gen(), T, H, W
#
#
# def _chunk_edges(n: int, k: int) -> List[Tuple[int, int]]:
#     return [(s, min(n, s + k)) for s in range(0, n, k)]
#
#
# def create_static_background(
#     data_or_paths: Union[ArrayLike3D, PathList],
#     *,
#     method: Method = "percentile",
#     percentile: float = 10.0,
#     block_size: int = 20,
#     hdu: int = 0,
#     dtype: np.dtype = np.float32,
#     nan_policy: Literal["omit", "propagate"] = "omit",
#     mask: Optional[np.ndarray] = None,
#     return_masked: bool = False,
# ) -> np.ndarray:
#     """
#     Make a background image using:
#       - 'percentile': per-pixel percentile
#       - 'min': per-pixel running min (streaming)
#       - 'block_median_min': median in each block, then min across blocks
#       - 'block_min_median': min in each block, then median across those mins
#     """
#     frames, T, H, W = _as_cube(data_or_paths, hdu=hdu, dtype=dtype)
#     use_nan = (nan_policy == "omit")
#
#     def _apply_mask(a: np.ndarray) -> np.ndarray:
#         if mask is not None:
#             a = a.copy()
#             a[mask] = np.nan
#         return a
#
#     if method == "percentile":
#         cube = np.empty((T, H, W), dtype=dtype)
#         for i, f in enumerate(frames):
#             cube[i] = _apply_mask(f)
#         func = np.nanpercentile if use_nan else np.percentile
#         return func(cube, percentile, axis=0).astype(dtype, copy=False)
#
#     if method == "min":
#         bg = None
#         for f in frames:
#             a = _apply_mask(f)
#             if use_nan:
#                 a = np.where(np.isnan(a), np.inf, a)
#             bg = a if bg is None else np.minimum(bg, a)
#         if use_nan:
#             bg = np.where(np.isinf(bg), np.nan if return_masked else 0.0, bg)
#         return bg.astype(dtype, copy=False)
#
#     # Block helpers (support ndarray or paths without double-reading)
#     if isinstance(data_or_paths, np.ndarray):
#         def get_block(s, e):
#             blk = data_or_paths[s:e].astype(dtype, copy=False)
#             if mask is not None:
#                 blk = blk.copy()
#                 blk[:, mask] = np.nan
#             return blk
#     else:
#         paths: List[str] = data_or_paths
#         def get_block(s, e):
#             blk = np.empty((e - s, H, W), dtype=dtype)
#             for j, p in enumerate(paths[s:e]):
#                 with fits.open(p, memmap=True) as hdul:
#                     a = _to_dtype(hdul[hdu].data, dtype)
#                 if mask is not None:
#                     a = a.copy(); a[mask] = np.nan
#                 blk[j] = a
#             return blk
#
#     edges = _chunk_edges(T, block_size)
#
#     if method == "block_median_min":
#         bg = None
#         for s, e in edges:
#             blk = get_block(s, e)
#             med = (np.nanmedian if use_nan else np.median)(blk, axis=0)
#             tmp = np.where(np.isnan(med), np.inf, med) if use_nan else med
#             bg = tmp if bg is None else np.minimum(bg, tmp)
#         if use_nan:
#             bg = np.where(np.isinf(bg), np.nan if return_masked else 0.0, bg)
#         return bg.astype(dtype, copy=False)
#
#     if method == "block_min_median":
#         mins = []
#         for s, e in edges:
#             blk = get_block(s, e)
#             if use_nan:
#                 blk = np.where(np.isnan(blk), np.inf, blk)
#                 bmin = np.min(blk, axis=0)
#                 bmin = np.where(np.isinf(bmin), np.nan, bmin)
#             else:
#                 bmin = np.min(blk, axis=0)
#             mins.append(bmin.astype(dtype, copy=False))
#         stack = np.stack(mins, axis=0)
#         return ((np.nanmedian if use_nan else np.median)(stack, axis=0)
#                 .astype(dtype, copy=False))
#
#     raise ValueError(f"Unknown method: {method}")
#
#
# # ---------------------------------------------
# # Radial profile & uniform radial reconstruction
# # ---------------------------------------------
# def _azimuthal_mean(
#     img: np.ndarray,
#     *,
#     center: Optional[Tuple[float, float]] = None,
#     dr: float = 1.0,
#     rmax: Optional[float] = None,
#     nan_omit: bool = True,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Fast azimuthal average using bincount (NaN-aware)."""
#     if img.ndim != 2:
#         raise ValueError("img must be 2D")
#     H, W = img.shape
#     cy, cx = center if center is not None else ((H - 1) * 0.5, (W - 1) * 0.5)
#
#     yy = np.arange(H, dtype=np.float64)[:, None] - cy
#     xx = np.arange(W, dtype=np.float64)[None, :] - cx
#     r = np.hypot(yy, xx)
#
#     if rmax is None:
#         rmax = float(r.max())
#     nbins = int(np.floor(rmax / dr)) + 1
#     bin_idx = np.minimum((r / dr).astype(np.int32), nbins - 1)
#
#     if nan_omit:
#         finite = np.isfinite(img)
#         vals = np.where(finite, img, 0.0)
#         w = finite.astype(np.float64)
#     else:
#         vals = np.nan_to_num(img, nan=0.0)
#         w = np.ones_like(img, dtype=np.float64)
#
#     s = np.bincount(bin_idx.ravel(), weights=vals.ravel(), minlength=nbins)
#     c = np.bincount(bin_idx.ravel(), weights=w.ravel(), minlength=nbins)
#     with np.errstate(invalid="ignore", divide="ignore"):
#         prof = s / c
#     prof[c == 0] = np.nan
#
#     r_centers = (np.arange(nbins, dtype=np.float64) + 0.5) * dr
#     return r_centers, prof
#
#
# def _radialize(
#     r_vals: np.ndarray,
#     prof_vals: np.ndarray,
#     *,
#     shape: Tuple[int, int],
#     center: Optional[Tuple[float, float]] = None,
#     fill: Literal["edge", "zero", "nan"] = "edge",
# ) -> np.ndarray:
#     """Build a radially symmetric image from a 1D profile."""
#     H, W = shape
#     cy, cx = center if center is not None else ((H - 1) * 0.5, (W - 1) * 0.5)
#
#     yy = np.arange(H, dtype=np.float64)[:, None] - cy
#     xx = np.arange(W, dtype=np.float64)[None, :] - cx
#     rmap = np.hypot(yy, xx)
#
#     left, right = prof_vals[0], prof_vals[-1]
#     if fill == "zero":
#         left = right = 0.0
#
#     out = np.interp(rmap.ravel(), r_vals, prof_vals, left=left, right=right).reshape(H, W)
#     if fill == "nan":
#         out[(rmap < r_vals.min()) | (rmap > r_vals.max())] = np.nan
#     return out
#
#
# def create_uniform_background(
#     data_or_paths: Union[np.ndarray, List[str]],
#     *,
#     method: Method = "percentile",
#     percentile: float = 10.0,
#     block_size: int = 20,
#     hdu: int = 0,
#     dtype: np.dtype = np.float32,
#     nan_policy: Literal["omit", "propagate"] = "omit",
#     mask: Optional[np.ndarray] = None,
#     return_masked: bool = False,
#     center: Optional[Tuple[float, float]] = None,
#     dr: float = 1.0,
#     rmax: Optional[float] = None,
#     fill: Literal["edge", "zero", "nan"] = "edge",
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     1) Compute background via create_static_background.
#     2) Compute azimuthally averaged radial profile.
#     3) Reconstruct radially uniform background having that profile.
#
#     Returns (bg, r, prof, uniform_bg).
#     """
#     bg = create_static_background(
#         data_or_paths,
#         method=method,
#         percentile=percentile,
#         block_size=block_size,
#         hdu=hdu,
#         dtype=dtype,
#         nan_policy=nan_policy,
#         mask=mask,
#         return_masked=return_masked,
#     )
#     r, prof = _azimuthal_mean(bg, center=center, dr=dr, rmax=rmax, nan_omit=(nan_policy == "omit"))
#     uniform_bg = _radialize(r, prof, shape=bg.shape, center=center, fill=fill)
#     return bg, r, prof, uniform_bg