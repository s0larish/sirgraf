from __future__ import annotations
import glob, os
from dataclasses import dataclass
import numpy as np
from astropy.io import fits

@dataclass
class FrameMeta:
    date: str
    time: str

@dataclass
class StackMeta:
    crpix1: float
    crpix2: float
    rsun_pix: float   # RSUN / CDELT1
    instrume: str
    detector: str | None

def discover_fits(path: str) -> list[str]:
    exts = {os.path.splitext(f)[-1].lower() for f in os.listdir(path)}
    if ".fits" in exts:
        return sorted(glob.glob(os.path.join(path, "*.fits")))
    if ".fts" in exts:
        return sorted(glob.glob(os.path.join(path, "*.fts")))
    raise FileNotFoundError("No .fits or .fts files found.")

def read_primary_header(fname: str):
    with fits.open(fname, memmap=False) as hdul:
        h, d = hdul[0].header, hdul[0].data
        return h, d

def read_stack(path: str):
    files = discover_fits(path)
    h0, d0 = read_primary_header(files[0])

    crpix1 = float(h0["CRPIX1"])
    crpix2 = float(h0["CRPIX2"])
    rsun_pix = float(h0["RSUN"]) / float(h0["CDELT1"])
    instrume = str(h0.get("INSTRUME", ""))
    detector = h0.get("DETECTOR")

    dates, times, data = [], [], []
    for f in files:
        h, d = read_primary_header(f)
        # Flip Y for compatibility (we’ll handle C3 later as a special case)
        dates.append(h["DATE-OBS"].split("T")[0])
        times.append(h["DATE-OBS"].split("T")[1].split(".")[0])
        data.append(np.flipud(np.asarray(d, dtype=float)))

    stack = np.stack(data, axis=0)
    metas = [FrameMeta(date=d, time=t) for d, t in zip(dates, times)]
    return stack, metas, StackMeta(crpix1, crpix2, rsun_pix, instrume, detector)