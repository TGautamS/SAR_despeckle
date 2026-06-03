"""
classical_despeckle.py
======================
Two things you get for free here so you don't have to hunt for code:

1. add_speckle()  -- the SAR-Trans speckle model (paper Eqs. 1-2): y = x * n,
   n ~ Gamma(shape=L, scale=1/L), unit mean, variance 1/L, single-look L=1.
   Use it to (re)generate synthetic test sets so every method sees identical noise.
   NOTE: if your uploaded generator differs (amplitude vs intensity, normalisation),
   replace ONLY this function -- nothing else in the harness changes.

2. Classical, TRAINING-FREE despeckling baselines: Lee, Refined Lee, Frost, Kuan.
   These run on CPU in seconds. Use them directly as baselines.

For the non-local classical baselines (PPB, SAR-BM3D) there is no clean pure-Python
version; use the authors' released code / known ports and feed their *outputs*
into run_eval.py like any other method.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import uniform_filter, variance as ndi_variance

EPS = 1e-8


# ---------------------------------------------------------------------------
# speckle generation (SAR-Trans Eqs. 1-2)
# ---------------------------------------------------------------------------
def add_speckle(clean: np.ndarray, L: int = 1, seed: int | None = None) -> np.ndarray:
    """Multiply a clean image by Gamma speckle with unit mean, variance 1/L.
    FLOAT version, no clipping -- keeps the full intensity range.

    clean : float image (any non-negative range; [0,1] or [0,255] both fine)
    L     : number of looks (L=1 = single-look, the hardest / SAR-Trans setting)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(clean, dtype=np.float64)
    # Gamma with shape=L, scale=1/L  ->  E[n]=1, Var[n]=1/L
    n = rng.gamma(shape=L, scale=1.0 / L, size=x.shape)
    return x * n


def add_speckle_div2k_style(clean_uint8: np.ndarray, L: int = 1,
                            seed: int | None = None) -> np.ndarray:
    """EXACT reproduction of your prepare_div2k_for_training.py pipeline:

        /255  ->  Gamma(shape=L, scale=1/L) multiply  ->  *255  ->  clip[0,255]  ->  uint8

    Use THIS (not add_speckle) to generate the Set12 synthetic TEST set, so the
    test noise matches how your DIV2K TRAINING set was made -- same clip at 255
    and same uint8 quantisation. Using the unclipped float version for testing
    would create a subtle train/test mismatch (your net learned on clipped data).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(clean_uint8, dtype=np.float64) / 255.0
    n = rng.gamma(shape=L, scale=1.0 / L, size=x.shape)
    return np.clip(x * n * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# classical filters  (operate on linear INTENSITY)
# ---------------------------------------------------------------------------
def _local_mean_var(img, size):
    mean = uniform_filter(img, size)
    mean_sq = uniform_filter(img * img, size)
    var = mean_sq - mean * mean
    return mean, np.clip(var, 0, None)


def lee_filter(img: np.ndarray, size: int = 7, L: int = 1) -> np.ndarray:
    """Classic Lee filter. cu^2 = 1/L is the speckle noise variation coefficient."""
    img = np.asarray(img, dtype=np.float64)
    mean, var = _local_mean_var(img, size)
    cu2 = 1.0 / L
    ci2 = var / (mean * mean + EPS)
    w = 1.0 - cu2 / (ci2 + EPS)
    w = np.clip(w, 0, 1)
    return mean + w * (img - mean)


def kuan_filter(img: np.ndarray, size: int = 7, L: int = 1) -> np.ndarray:
    """Kuan filter -- like Lee but a slightly different weighting."""
    img = np.asarray(img, dtype=np.float64)
    mean, var = _local_mean_var(img, size)
    cu2 = 1.0 / L
    ci2 = var / (mean * mean + EPS)
    w = (1.0 - cu2 / (ci2 + EPS)) / (1.0 + cu2)
    w = np.clip(w, 0, 1)
    return mean + w * (img - mean)


def frost_filter(img: np.ndarray, size: int = 7, damping: float = 2.0) -> np.ndarray:
    """Frost filter: locally adaptive exponential weighting by distance & contrast."""
    img = np.asarray(img, dtype=np.float64)
    mean, var = _local_mean_var(img, size)
    sigma2 = var
    out = np.empty_like(img)
    pad = size // 2
    padded = np.pad(img, pad, mode="reflect")

    ax = np.arange(-pad, pad + 1)
    dx, dy = np.meshgrid(ax, ax)
    dist = np.sqrt(dx * dx + dy * dy)

    H, W = img.shape
    for i in range(H):
        for j in range(W):
            window = padded[i:i + size, j:j + size]
            local_mean = window.mean()
            local_var = window.var()
            alpha = damping * local_var / (local_mean * local_mean + EPS)
            wgt = np.exp(-alpha * dist)
            wgt /= wgt.sum() + EPS
            out[i, j] = (window * wgt).sum()
    return out


def refined_lee_filter(img: np.ndarray, size: int = 7, L: int = 1) -> np.ndarray:
    """A practical Refined-Lee approximation (edge-aware Lee).

    Full Refined Lee uses 8 directional edge-aligned subwindows; this version
    keeps the edge-adaptive spirit with much less code by shrinking the filter
    weight where local contrast is high. Good enough as a baseline; swap for the
    full directional version if a reviewer asks."""
    img = np.asarray(img, dtype=np.float64)
    mean, var = _local_mean_var(img, size)
    cu2 = 1.0 / L
    ci2 = var / (mean * mean + EPS)
    cmax = np.sqrt(1.0 + 2.0 * cu2)            # heterogeneity threshold
    ci = np.sqrt(np.clip(ci2, 0, None))
    w = np.where(ci <= np.sqrt(cu2), 0.0,
                 np.where(ci >= cmax, 1.0,
                          (ci - np.sqrt(cu2)) / (cmax - np.sqrt(cu2) + EPS)))
    return mean + w * (img - mean)


FILTERS = {
    "lee": lee_filter,
    "refined_lee": refined_lee_filter,
    "frost": frost_filter,
    "kuan": kuan_filter,
}
