# main.py
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Union, Literal
import argparse, glob, os, sys
import numpy as np

from background import create_static_background, create_uniform_background

Array2D = np.ndarray
Array3D = np.ndarray
PathList = List[str]

def _is_map(obj) -> bool:
    return getattr(obj, "__class__", None).__module__.startswith("sunpy.map")

def _normalize_input(
    data_or_paths: Union[Array2D, Array3D, PathList, Iterable],
) -> Tuple[Iterable[np.ndarray], int, int, int, bool]:
    """
    Returns (iter_frames, T, H, W, is_paths). Frames are float32 arrays.
    Supports: 2D array, 3D cube, list[str] FITS paths, SunPy Map or list/iterable of Maps.
    """
    if isinstance(data_or_paths, np.ndarray) and data_or_paths.ndim == 2:
        H, W = data_or_paths.shape
        def gen(): yield data_or_paths.astype(np.float32, copy=False)
        return gen(), 1, H, W, False

    if isinstance(data_or_paths, np.ndarray) and data_or_paths.ndim == 3:
        T, H, W = data_or_paths.shape
        def gen():
            for i in range(T):
                yield np.asarray(data_or_paths[i], dtype=np.float32, copy=False)
        return gen(), T, H, W, False

    try:
        import sunpy.map as smap  # noqa: F401
        if _is_map(data_or_paths):
            img = np.asarray(data_or_paths.data, dtype=np.float32)
            H, W = img.shape
            def gen(): yield img
            return gen(), 1, H, W, False
        if hasattr(data_or_paths, "__iter__") and not isinstance(data_or_paths, (str, bytes)):
            first = None
            frames = []
            for m in data_or_paths:
                if _is_map(m):
                    a = np.asarray(m.data, dtype=np.float32)
                    frames.append(a)
                    if first is None:
                        first = a.shape
                    elif a.shape != first:
                        raise ValueError("All SunPy maps must have same shape.")
                else:
                    frames = None
                    break
            if frames is not None:
                T, (H, W) = len(frames), frames[0].shape
                def gen():
                    for a in frames: yield a
                return gen(), T, H, W, False
    except Exception:
        pass

    if isinstance(data_or_paths, (list, tuple)) and data_or_paths and isinstance(data_or_paths[0], str):
        try:
            from astropy.io import fits
        except Exception as e:
            raise ImportError("astropy is required for FITS path inputs.") from e
        with fits.open(data_or_paths[0], memmap=True) as hdul:
            H, W = np.asarray(hdul[0].data).shape
        T = len(data_or_paths)
        def gen():
            from astropy.io import fits
            for p in data_or_paths:
                with fits.open(p, memmap=True) as hdul:
                    yield np.asarray(hdul[0].data, dtype=np.float32)
        return gen(), T, H, W, True

    raise ValueError("Unsupported input type.")

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