import numpy as np
import pytest

from sirgraf.background import create_static_background, create_uniform_background


# ---------- Fixtures ----------

@pytest.fixture
def toy_cube():
    """6 frames, each constant: [5, 2, 7, 3, 9, 4]."""
    H, W = 8, 10
    vals = np.array([5, 2, 7, 3, 9, 4], dtype=np.float32)
    cube = np.stack([np.full((H, W), v, np.float32) for v in vals], axis=0)
    return cube, vals

@pytest.fixture
def fits_paths(tmp_path, toy_cube):
    """Write toy cube frames to FITS and return paths."""
    cube, _ = toy_cube
    pytest.importorskip("astropy.io.fits")
    from astropy.io import fits

    paths = []
    for i, frame in enumerate(cube):
        p = tmp_path / f"f{i:02d}.fits"
        fits.PrimaryHDU(frame).writeto(p, overwrite=True)
        paths.append(str(p))
    return paths


# ---------- create_static_background ----------

@pytest.mark.parametrize("method,kwargs,expected", [
    ("percentile", {"percentile": 50.0}, 4.5),  # median of [2,3,4,5,7,9]
    ("min", {}, 2.0),
    ("block_median_min", {"block_size": 2}, 3.5),   # medians [3.5,5,6.5] -> min 3.5
    ("block_min_median", {"block_size": 2}, 3.0),   # mins [2,3,4] -> median 3
])
def test_static_background_methods_ndarray(toy_cube, method, kwargs, expected):
    cube, _ = toy_cube
    bg = create_static_background(cube, method=method, **kwargs)
    assert bg.shape == cube.shape[1:]
    assert np.allclose(bg, expected, atol=1e-6)

def test_static_background_accepts_2d_array():
    img = np.full((16, 16), 5.0, np.float32)
    bg = create_static_background(img, method="min")
    assert bg.shape == img.shape
    assert np.allclose(bg, 5.0, atol=1e-7)

def test_static_background_mask_nan_policy(toy_cube):
    cube, _ = toy_cube
    mask = np.zeros(cube.shape[1:], dtype=bool)
    mask[3, 4] = True
    bg_masked = create_static_background(cube, method="min",
                                         nan_policy="omit", mask=mask, return_masked=True)
    assert np.isnan(bg_masked[3, 4])
    m = bg_masked.copy(); m[3, 4] = 2.0
    assert np.allclose(m, 2.0, atol=1e-6)

@pytest.mark.parametrize("method,kwargs", [
    ("percentile", {"percentile": 50.0}),
    ("min", {}),
    ("block_median_min", {"block_size": 2}),
    ("block_min_median", {"block_size": 2}),
])
def test_static_background_ndarray_vs_fits(toy_cube, fits_paths, method, kwargs):
    cube, _ = toy_cube
    bg_arr = create_static_background(cube, method=method, **kwargs)
    bg_fits = create_static_background(fits_paths, method=method, **kwargs)
    assert np.allclose(bg_arr, bg_fits, atol=1e-6)


# ---------- create_uniform_background ----------

def test_uniform_background_rotational_invariance(toy_cube):
    cube, _ = toy_cube
    bg, r, prof, uni = create_uniform_background(cube, method="min", dr=1.0)

    H, W = uni.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W), dtype=np.float64)
    radii = np.hypot(yy - cy, xx - cx)
    dr = float(r[1] - r[0]) if len(r) > 1 else 1.0

    probe_rs = np.linspace(r.min(), r.max(), 6)
    for rad in probe_rs:
        ring = (radii >= rad - 0.5 * dr) & (radii < rad + 0.5 * dr)
        if not np.any(ring):
            continue
        vals = uni[ring]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        # match the interpolated profile value at this radius
        v = np.interp(rad, r, prof, left=prof[0], right=prof[-1])
        assert np.allclose(vals, v, atol=1e-6)

def test_uniform_background_center_matches_profile(toy_cube):
    cube, _ = toy_cube
    bg, r, prof, uni = create_uniform_background(cube, method="block_median_min", block_size=2, dr=1.0)
    cy, cx = (uni.shape[0]-1)//2, (uni.shape[1]-1)//2
    assert np.isclose(uni[cy, cx], prof[0], equal_nan=True)

@pytest.mark.parametrize("method,kwargs", [
    ("percentile", {"percentile": 50.0}),
    ("min", {}),
    ("block_median_min", {"block_size": 2}),
    ("block_min_median", {"block_size": 2}),
])
def test_uniform_background_ndarray_vs_fits(toy_cube, fits_paths, method, kwargs):
    cube, _ = toy_cube
    bg_a, r_a, p_a, u_a = create_uniform_background(cube, method=method, dr=1.0, **kwargs)
    bg_f, r_f, p_f, u_f = create_uniform_background(fits_paths, method=method, dr=1.0, **kwargs)
    assert np.allclose(bg_a, bg_f, atol=1e-6)
    assert np.allclose(r_a, r_f, atol=1e-12, equal_nan=True)
    assert np.allclose(p_a, p_f, atol=1e-6, equal_nan=True)
    assert np.allclose(u_a, u_f, atol=1e-6, equal_nan=True)

def test_paths_shape_mismatch_raises(tmp_path):
    pytest.importorskip("astropy.io.fits")
    from astropy.io import fits
    a = np.zeros((10, 10), np.float32)
    b = np.zeros((12, 10), np.float32)
    p1 = tmp_path / "a.fits"; p2 = tmp_path / "b.fits"
    fits.PrimaryHDU(a).writeto(p1, overwrite=True)
    fits.PrimaryHDU(b).writeto(p2, overwrite=True)
    with pytest.raises(ValueError):
        create_static_background([str(p1), str(p2)], method="min")