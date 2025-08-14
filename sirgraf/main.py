from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Union, Literal
import argparse, glob, os, sys
import numpy as np

from background import create_static_background, create_uniform_background

try:
    from astropy.io import fits
    _HAS_ASTROPY = True
except Exception:
    _HAS_ASTROPY = False

Array2D = np.ndarray
Array3D = np.ndarray
PathList = List[str]

def _fits_shape(path: str) -> Tuple[int, int]:
    """
    Read image shape from header without loading data.
    Works for standard and compressed FITS.
    """
    hdr = fits.getheader(path, 0)
    # NAXIS ordering is (x=W, y=H) for 2D primary HDU
    W = int(hdr.get("NAXIS1"))
    H = int(hdr.get("NAXIS2"))
    if not (H and W):
        raise ValueError(f"Could not read image shape from header: {path}")
    return H, W

# def _has_scaling_or_blank(path: str) -> bool:
#     """Detects headers that force data scaling (no memmap)."""
#     hdr = fits.getheader(path, 0)
#     return any(k in hdr for k in ("BSCALE", "BZERO", "BLANK"))

def _load_fits(path: str) -> np.ndarray:
    """
    Load FITS data safely:
      - if scaling keywords present -> memmap=False so astropy can scale
      - use uint=True to properly handle unsigned ints via BZERO
      - squeeze and ensure 2D
    """
    # We choose memmap=False always to be safe across instruments;
    # If you later want conditional memmap for speed, you can switch:
    # use_memmap = not _has_scaling_or_blank(path)
    # use_memmap = False
    arr = fits.getdata(path, ext=0, memmap=False, uint=True)
    arr = np.asarray(arr)
    # Many solar FITS are (H, W) but some are 3D with singleton dimension.
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image in {path}, got shape {arr.shape}")
    return arr

def _normalize_input(
    data_or_paths: Union[Array2D, Array3D, PathList, Iterable]
) -> Tuple[Iterable[np.ndarray], int, int, int, np.dtype]:
    """
    Normalize input types to a unified iterable of 2D frames.
    Returns: (frames_iterable, T, H, W, dtype)
    """
    # Case 1: already an ndarray
    if isinstance(data_or_paths, np.ndarray):
        arr = np.asarray(data_or_paths)
        if arr.ndim == 2:
            H, W = arr.shape
            T = 1
            frames = (arr,)
            return frames, T, H, W, arr.dtype
        elif arr.ndim == 3:
            T, H, W = arr.shape[:3]
            frames = (arr[i] for i in range(T))
            return frames, T, H, W, arr.dtype
        else:
            raise ValueError(f"Expected 2D or 3D ndarray, got shape {arr.shape}")

    # Case 2: iterable of arrays
    if hasattr(data_or_paths, "__iter__") and not isinstance(data_or_paths, (str, bytes, list, tuple)):
        # Peek first frame to infer shape
        it = iter(data_or_paths)
        first = next(it)
        first = np.asarray(first)
        if first.ndim != 2:
            raise ValueError(f"Iterable must yield 2D arrays, got {first.shape}")
        H, W = first.shape
        def gen():
            yield first
            for f in it:
                a = np.asarray(f)
                if a.ndim != 2 or a.shape != (H, W):
                    raise ValueError(f"All frames must be 2D and same shape {(H,W)}, got {a.shape}")
                yield a
        # We can’t know T without consuming; keep dtype from first
        return gen(), -1, H, W, first.dtype  # T=-1 indicates unknown/stream

    # Case 3: list/tuple of paths
    if isinstance(data_or_paths, (list, tuple)) and data_or_paths and isinstance(data_or_paths[0], (str, bytes)):
        if not _HAS_ASTROPY:
            raise ImportError("astropy is required for FITS path inputs.")
        H, W = _fits_shape(data_or_paths[0])
        T = len(data_or_paths)
        def gen():
            for p in data_or_paths:
                a = _load_fits(p)
                if a.shape != (H, W):
                    raise ValueError(f"All frames must share shape {(H,W)}; {p} has {a.shape}")
                yield a
        # dtype only known after first load; assume float32 pipeline later
        return gen(), T, H, W, np.float32

    raise TypeError("Unsupported input type for data_or_paths.")

def _filter_frame(
    frame: np.ndarray, static_bg: np.ndarray, uniform_bg: np.ndarray, eps: float
) -> np.ndarray:
    denom = np.where(np.isfinite(uniform_bg), uniform_bg, 0.0)
    denom = np.sign(denom) * np.maximum(np.abs(denom), eps)
    out = (frame - static_bg) / denom
    return out.astype(np.float32, copy=False)

def process_images(
    data_or_paths: Union[Array2D, Array3D, PathList, Iterable],
    *,
    method: Literal["percentile", "min", "block_median_min", "block_min_median"] = "percentile",
    percentile: float = 10.0,
    block_size: int = 20,
    nan_policy: Literal["omit", "propagate"] = "omit",
    mask: Optional[np.ndarray] = None,
    center: Optional[Tuple[float, float]] = None,
    dr: float = 1.0,
    rmax: Optional[float] = None,
    fill: Literal["edge", "zero", "nan"] = "edge",
    return_static: bool = False,
    return_uniform: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Core pipeline:
      1) compute static + uniform backgrounds once
      2) filter: (frame - static) / uniform
      3) return filtered image(s); optional return static/uniform.
    """
    frames, T, H, W, _ = _normalize_input(data_or_paths)

    static_bg, _, _, uniform_bg = create_uniform_background(
        data_or_paths,
        method=method,
        percentile=percentile,
        block_size=block_size,
        nan_policy=nan_policy,
        mask=mask,
        center=center,
        dr=dr,
        rmax=rmax,
        fill=fill,
    )
    eps = np.finfo(np.float32).eps

    if T == 1:
        f = next(iter(frames))
        filtered = _filter_frame(f, static_bg, uniform_bg, eps)
    else:
        filtered = np.empty((T, H, W), dtype=np.float32)
        for i, f in enumerate(frames):
            filtered[i] = _filter_frame(f, static_bg, uniform_bg, eps)

    if return_static or return_uniform:
        return filtered, static_bg, uniform_bg
    return filtered


# ---------------------------
# Simple CLI
# ---------------------------
def _parse_center(center_str: Optional[str]) -> Optional[Tuple[float, float]]:
    if not center_str:
        return None
    yx = center_str.split(",")
    if len(yx) != 2:
        raise argparse.ArgumentTypeError("center must be 'y,x' in pixels")
    return float(yx[0]), float(yx[1])

def cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Static + uniform background filtering pipeline.")
    p.add_argument("inputs", nargs="+", help="FITS glob(s) or paths, or a .npy cube path")
    p.add_argument("-o", "--output", required=True, help="Output .npy file (stack) or directory for FITS")
    p.add_argument("--method", choices=["percentile","min","block_median_min","block_min_median"], default="percentile")
    p.add_argument("--percentile", type=float, default=10.0)
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--nan-policy", choices=["omit","propagate"], default="omit")
    p.add_argument("--dr", type=float, default=1.0)
    p.add_argument("--rmax", type=float, default=None)
    p.add_argument("--center", type=_parse_center, default=None, metavar="Y,X")
    p.add_argument("--save-format", choices=["npy","fits"], default="npy")
    p.add_argument("--save-backgrounds", action="store_true", help="Also save static/uniform backgrounds")

    args = p.parse_args(argv)

    # Gather inputs
    paths: List[str] = []
    for pat in args.inputs:
        if pat.endswith(".npy") and os.path.isfile(pat):
            # Load cube directly
            cube = np.load(pat)
            data = cube
            break
        matched = sorted(glob.glob(pat))
        paths.extend(matched)
    else:
        data = paths if paths else args.inputs  # fall back to raw list if user passed explicit paths

    # Process
    if args.save_backgrounds:
        out, static_bg, uniform_bg = process_images(
            data,
            method=args.method,
            percentile=args.percentile,
            block_size=args.block_size,
            nan_policy=args.nan_policy,
            center=args.center,
            dr=args.dr,
            rmax=args.rmax,
            return_static=True,
            return_uniform=True,
        )
    else:
        out = process_images(
            data,
            method=args.method,
            percentile=args.percentile,
            block_size=args.block_size,
            nan_policy=args.nan_policy,
            center=args.center,
            dr=args.dr,
            rmax=args.rmax,
        )

    # Save
    if args.save_format == "npy":
        np.save(args.output, out)
        if args.save_backgrounds:
            base, _ = os.path.splitext(args.output)
            np.save(base + "_static.npy", static_bg)
            np.save(base + "_uniform.npy", uniform_bg)
    else:
        try:
            from astropy.io import fits
        except Exception as e:
            print("astropy required to save FITS.", file=sys.stderr)
            return 2
        os.makedirs(args.output, exist_ok=True)
        if out.ndim == 2:
            fits.PrimaryHDU(out).writeto(os.path.join(args.output, "filtered.fits"), overwrite=True)
        else:
            for i in range(out.shape[0]):
                fits.PrimaryHDU(out[i]).writeto(os.path.join(args.output, f"filtered_{i:04d}.fits"), overwrite=True)
        if args.save_backgrounds:
            fits.PrimaryHDU(static_bg).writeto(os.path.join(args.output, "static_bg.fits"), overwrite=True)
            fits.PrimaryHDU(uniform_bg).writeto(os.path.join(args.output, "uniform_bg.fits"), overwrite=True)

    return 0

if __name__ == "__main__":
    raise SystemExit(cli())