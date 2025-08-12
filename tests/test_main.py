import numpy as np
import pytest

from main import process_images
from background import create_uniform_background


# ---------- Fixtures ----------

@pytest.fixture
def test_cube():
    """6 frames, each constant: [5, 2, 7, 3, 9, 4]."""
    H, W = 8, 10
    vals = np.array([5, 2, 7, 3, 9, 4], dtype=np.float32)
    cube = np.stack([np.full((H, W), v, np.float32) for v in vals], axis=0)
    return cube, vals

@pytest.fixture
def fits_paths(tmp_path, test_cube):
    cube, _ = test_cube
    pytest.importorskip("astropy.io.fits")
    from astropy.io import fits
    paths = []
    for i, frame in enumerate(cube):
        p = tmp_path / f"f{i:02d}.fits"
        fits.PrimaryHDU(frame).writeto(p, overwrite=True)
        paths.append(str(p))
    return paths


# ---------- Tests ----------

def test_single_image_constant_goes_to_zero():
    """Single constant image ⇒ filtered ≈ 0."""
    img = np.full((16, 16), 5.0, np.float32)
    out = process_images(img, method="min")
    assert out.shape == img.shape
    assert np.allclose(out, 0.0, atol=1e-7)

def test_stack_matches_manual_formula(test_cube):
    """Verify (frame - static)/uniform equals process_images output."""
    cube, vals = test_cube
    static_bg, r, prof, uniform_bg = create_uniform_background(cube, method="min", dr=1.0)
    out = process_images(cube, method="min")

    exp = np.empty_like(out)
    for i, v in enumerate(vals):
        exp[i] = (v - static_bg) / uniform_bg
    assert np.allclose(out, exp, atol=1e-6)

def test_return_backgrounds_shapes(test_cube):
    cube, _ = test_cube
    out, static_bg, uniform_bg = process_images(
        cube, method="block_median_min", block_size=2,
        return_static=True, return_uniform=True
    )
    assert out.shape == cube.shape
    assert static_bg.shape == cube.shape[1:]
    assert uniform_bg.shape == cube.shape[1:]

@pytest.mark.parametrize("method,kwargs", [
    ("percentile", {"percentile": 50.0}),
    ("min", {}),
    ("block_median_min", {"block_size": 2}),
    ("block_min_median", {"block_size": 2}),
])
def test_ndarray_vs_fits_parity(test_cube, fits_paths, method, kwargs):
    cube, _ = test_cube
    out_arr = process_images(cube, method=method, **kwargs)
    out_fits = process_images(fits_paths, method=method, **kwargs)
    assert out_arr.shape == out_fits.shape == cube.shape
    assert np.allclose(out_arr, out_fits, atol=1e-6)
