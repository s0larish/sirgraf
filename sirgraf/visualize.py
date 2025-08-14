from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pa
from matplotlib import colormaps
from .core import ProcessResult

def plot_quicklook(res: ProcessResult):
    cmap = colormaps.get(res.cmap_name, colormaps["gray"])
    idx = res.filtered.shape[0] // 2
    vmin, vmax = res.zscale_limits

    # Prepare scaled backgrounds
    bgU = np.log(np.maximum(res.uniform, 1e-12))
    bgM = np.log(np.maximum(res.minimum, 1e-12))

    bgU[res.mask] = 0.0
    bgM[res.mask] = 0.0

    X0, X1 = res.x_rsun[0], res.x_rsun[-1]
    Y0, Y1 = res.y_rsun[0], res.y_rsun[-1]
    extent = [X0, X1, Y0, Y1]

    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    # Minimum
    ax[0,0].imshow(bgM / np.nanmax(bgM), extent=extent, cmap=cmap, origin="lower")
    ax[0,0].add_patch(pa.Circle((0,0), res.inner_radius_rsun, color="black"))
    ax[0,0].add_patch(pa.Circle((0,0), 1.0, color="white", fill=False))
    ax[0,0].set_title("Minimum Intensity Image")
    ax[0,0].set_xlabel("Solar X (R$_\\odot$)")
    ax[0,0].set_ylabel("Solar Y (R$_\\odot$)")
    plt.colorbar(ax[0,0].images[0], ax=ax[0,0], shrink=0.85, pad=0.01, extend="both", label="Log(Intensity)")

    # Azimuthal profile (positive Y)
    g = np.log10(np.clip(res.avg_profile_posY, 1e-20, None))
    g = g[np.isfinite(g)]
    rpow = int(np.round(np.mean(g))) if g.size else 0
    ax[0,1].plot(res.y_rsun[res.y_rsun >= 0], res.avg_profile_posY[:np.sum(res.y_rsun>=0)] / (10**rpow), color="black")
    ax[0,1].set_xlabel("Solar Y (R$_\\odot$)")
    ax[0,1].set_ylabel(f"Average Intensity (10$^{{{rpow}}}$)")
    ax[0,1].set_box_aspect(1)

    # Uniform
    ax[1,0].imshow(bgU / np.nanmax(bgU), extent=extent, cmap=cmap, origin="lower")
    ax[1,0].add_patch(pa.Circle((0,0), res.inner_radius_rsun, color="black"))
    ax[1,0].add_patch(pa.Circle((0,0), 1.0, color="white", fill=False))
    ax[1,0].set_title("Uniform Intensity Image")
    ax[1,0].set_xlabel("Solar X (R$_\\odot$)")
    ax[1,0].set_ylabel("Solar Y (R$_\\odot$)")
    plt.colorbar(ax[1,0].images[0], ax=ax[1,0], shrink=0.85, pad=0.01, extend="both", label="Log(Intensity)")

    # Filtered representative frame
    im = ax[1,1].imshow(res.filtered[idx], extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    patch = pa.Circle((0,0), radius=np.max(np.abs(res.y_rsun)), transform=ax[1,1].transData)
    im.set_clip_path(patch)
    ax[1,1].add_patch(pa.Circle((0,0), res.inner_radius_rsun, color="black"))
    ax[1,1].add_patch(pa.Circle((0,0), 1.0, color="white", fill=False))
    ax[1,1].set_facecolor("black")
    ax[1,1].set_xlabel("Solar X (R$_\\odot$)")
    ax[1,1].set_ylabel("Solar Y (R$_\\odot$)")
    plt.colorbar(im, ax=ax[1,1], shrink=0.85, pad=0.01, extend="both", label="Normalized Intensity")

    plt.tight_layout()
    plt.show()
