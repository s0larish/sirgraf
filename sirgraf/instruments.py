from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class InstrumentSpec:
    inner_radius_rsun: float
    cmap_name: str
    flip_y: bool = False

def infer_instrument_spec(instrume: str, detector: str | None) -> InstrumentSpec:
    i = (instrume or "").upper()
    d = (detector or "").upper() if detector else ""

    if i == "LASCO":
        if "C2" in d:
            return InstrumentSpec(2.25, "soholasco2")
        if "C3" in d:
            # LASCO C3 often needs Y flip relative to some conventions
            return InstrumentSpec(4.0, "soholasco3", flip_y=True)

    if i == "COSMO K-CORONAGRAPH":
        return InstrumentSpec(1.15, "kcor")

    if i == "SECCHI":
        if "COR1" in d:
            return InstrumentSpec(1.57, "stereocor1")
        if "COR2" in d:
            return InstrumentSpec(3.0, "stereocor2")

    # Default fallback
    return InstrumentSpec(1.5, "gray")