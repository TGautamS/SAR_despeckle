"""
sar_metrics.py
==============
Model-AGNOSTIC metric library for the SAR despeckling + flood-mapping benchmark.

Nothing in here imports torch. You feed it numpy arrays (the *outputs* any
despeckler already produced) and it returns numbers. That is the whole point:
every baseline -- Lee, PPB, SAR-BM3D, ID-CNN, MERLIN, SAR-Trans, and later your
own model -- goes through the EXACT same code, so the comparison is fair.

Metric blocks
-------------
1. Synthetic, paired (needs clean ground truth):   psnr(), ssim_metric()
2. Real, no-reference (needs a homogeneous ROI):   enl(), cx(), mean_preservation()
3. Edge preservation (needs a reference image):    epi()
4. Thresholding for unsupervised flood mapping:    threshold_otsu_safe(),
                                                    threshold_kittler_illingworth()
5. Change detection:                               log_ratio(), detect_change()
6. Segmentation scores (needs GT mask):            seg_scores()
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import threshold_otsu
from skimage.morphology import disk

EPS = 1e-8


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected foreground components smaller than min_size (scipy-based,
    version-stable)."""
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl, dtype=np.float64), lbl, range(1, n + 1))
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    keep[1:] = sizes >= min_size
    return keep[lbl]


def _fill_small_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill enclosed background holes smaller than max_hole; leave the large
    border-connected background alone."""
    holes = ~mask
    lbl, n = ndimage.label(holes)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl, dtype=np.float64), lbl, range(1, n + 1))
    fill = np.zeros(n + 1, dtype=bool)
    fill[1:] = sizes < max_hole
    return mask | fill[lbl]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_float(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a


def infer_data_range(*imgs) -> float:
    """Guess data_range for PSNR/SSIM from the value range of the images.
    [0,1] images -> 1.0 ; [0,255] images -> 255.0. Pass it explicitly if unsure."""
    m = max(float(np.nanmax(_as_float(i))) for i in imgs)
    return 255.0 if m > 1.5 else 1.0


# ---------------------------------------------------------------------------
# 1. synthetic, paired
# ---------------------------------------------------------------------------
def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float | None = None) -> float:
    pred, gt = _as_float(pred), _as_float(gt)
    if data_range is None:
        data_range = infer_data_range(pred, gt)
    return float(peak_signal_noise_ratio(gt, pred, data_range=data_range))


def ssim_metric(pred: np.ndarray, gt: np.ndarray, data_range: float | None = None) -> float:
    pred, gt = _as_float(pred), _as_float(gt)
    if data_range is None:
        data_range = infer_data_range(pred, gt)
    return float(structural_similarity(gt, pred, data_range=data_range))


# ---------------------------------------------------------------------------
# 2. real, no-reference  (computed on a HOMOGENEOUS region)
# ---------------------------------------------------------------------------
def enl(region: np.ndarray) -> float:
    """Equivalent Number of Looks = mean^2 / variance over a homogeneous patch.
    HIGHER = more smoothing. A *correct* despeckler lands in the tens-to-low-hundreds;
    ENL in the thousands means you have flattened the image (a pathology, not a win)."""
    r = _as_float(region)
    mu, var = r.mean(), r.var()
    return float(mu * mu / (var + EPS))


def cx(region: np.ndarray) -> float:
    """Coefficient of variation = std/mean over a homogeneous patch. LOWER = better."""
    r = _as_float(region)
    return float(r.std() / (r.mean() + EPS))


def mean_preservation(pred: np.ndarray, ref: np.ndarray,
                      region: tuple | None = None) -> float:
    """mean(pred)/mean(ref). Should sit at ~1.0. Your earlier 1.5 = radiometric bias.
    `region` is an optional (r0, r1, c0, c1) box; default = whole image."""
    pred, ref = _as_float(pred), _as_float(ref)
    if region is not None:
        r0, r1, c0, c1 = region
        pred, ref = pred[r0:r1, c0:c1], ref[r0:r1, c0:c1]
    return float(pred.mean() / (ref.mean() + EPS))


# ---------------------------------------------------------------------------
# 3. edge preservation index (correlation-of-Laplacian, Sattar-style)
# ---------------------------------------------------------------------------
def epi(pred: np.ndarray, ref: np.ndarray) -> float:
    """Edge Preservation Index in [~0, 1]. Pearson correlation between the
    high-pass (Laplacian) content of `pred` and `ref`.

    - Synthetic: pass ref = CLEAN ground truth  -> meaningful, higher is better.
    - Real SAR : pass ref = NOISY original       -> 'relative edge retention'.
      Interpret with care: speckle is itself high-frequency, so a good filter that
      removes speckle but keeps true edges will NOT score ~1.0 here. Report it
      alongside the qualitative figures, never alone."""
    p = ndimage.laplace(_as_float(pred))
    r = ndimage.laplace(_as_float(ref))
    p, r = p.ravel(), r.ravel()
    p, r = p - p.mean(), r - r.mean()
    denom = np.sqrt((p * p).sum() * (r * r).sum())
    return float((p * r).sum() / (denom + EPS))


# ---------------------------------------------------------------------------
# 4. thresholding
# ---------------------------------------------------------------------------
def threshold_otsu_safe(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu, guarded against degenerate (single-value) inputs."""
    v = _as_float(values).ravel()
    if np.allclose(v.min(), v.max()):
        return float(v.min())
    return float(threshold_otsu(v, nbins=nbins))


def threshold_kittler_illingworth(values: np.ndarray, nbins: int = 256) -> float:
    """Kittler & Illingworth minimum-error thresholding.

    Models the two classes as Gaussians and minimises the misclassification
    criterion J(t). Handles the NON-bimodal / small-flood case far better than
    Otsu -- which is exactly Otsu's failure mode and part of why a robust
    threshold matters for the two-stage pipeline."""
    v = _as_float(values).ravel()
    if np.allclose(v.min(), v.max()):
        return float(v.min())

    h, edges = np.histogram(v, bins=nbins)
    h = h.astype(np.float64)
    g = 0.5 * (edges[:-1] + edges[1:])           # bin centres

    c1 = np.cumsum(h)                            # counts in class 1 (<= t)
    m1 = np.cumsum(h * g)
    s1 = np.cumsum(h * g * g)
    tot_c = c1[-1]
    tot_m = m1[-1]
    tot_s = s1[-1]

    best_J, best_t = np.inf, g[len(g) // 2]
    for t in range(nbins - 1):
        p1 = c1[t]
        p2 = tot_c - p1
        if p1 < EPS or p2 < EPS:
            continue
        mu1 = m1[t] / p1
        mu2 = (tot_m - m1[t]) / p2
        var1 = s1[t] / p1 - mu1 * mu1
        var2 = (tot_s - s1[t]) / p2 - mu2 * mu2
        if var1 < EPS or var2 < EPS:
            continue
        P1, P2 = p1 / tot_c, p2 / tot_c
        J = (1.0
             + P1 * np.log(var1) + P2 * np.log(var2)
             - 2.0 * (P1 * np.log(P1) + P2 * np.log(P2)))
        if J < best_J:
            best_J, best_t = J, g[t]
    return float(best_t)


# ---------------------------------------------------------------------------
# 5. change detection (the unsupervised flood step)
# ---------------------------------------------------------------------------
def log_ratio(pre: np.ndarray, post: np.ndarray, domain: str = "linear") -> np.ndarray:
    """Log-ratio change image.

    domain='linear' : inputs are linear INTENSITY -> r = log((post+eps)/(pre+eps))
    domain='db'     : inputs are already in dB (log scale) -> r = post - pre

    Open-water flooding lowers backscatter, so newly flooded pixels have post<pre
    and therefore r < 0."""
    pre, post = _as_float(pre), _as_float(post)
    if domain == "db":
        return post - pre
    if domain == "linear":
        return np.log((post + EPS) / (pre + EPS))
    raise ValueError("domain must be 'linear' or 'db'")


def detect_change(ratio: np.ndarray,
                  method: str = "ki",
                  direction: str = "decrease",
                  open_radius: int = 1,
                  close_radius: int = 1,
                  min_size: int = 25,
                  fill_holes: bool = True) -> tuple[np.ndarray, float]:
    """Turn a log-ratio image into a binary flood mask.

    method     : 'ki' (Kittler-Illingworth) or 'otsu'
    direction  : 'decrease' (open water, default) | 'increase' | 'magnitude'
    morphology : opening removes speckle-induced false positives; small-object
                 removal and hole-filling clean the blobs. All tunable.

    Returns (mask_bool, threshold_used).
    """
    r = _as_float(ratio)

    if direction == "magnitude":
        score = np.abs(r)
        thr = (threshold_kittler_illingworth(score) if method == "ki"
               else threshold_otsu_safe(score))
        mask = score > thr
    else:
        thr = (threshold_kittler_illingworth(r) if method == "ki"
               else threshold_otsu_safe(r))
        mask = r < thr if direction == "decrease" else r > thr

    # spatial regularisation (scipy.ndimage = stable API)
    if open_radius > 0:
        mask = ndimage.binary_opening(mask, structure=disk(open_radius))
    if close_radius > 0:
        mask = ndimage.binary_closing(mask, structure=disk(close_radius))
    if min_size > 0:
        mask = _remove_small_objects(mask, min_size)
    if fill_holes and min_size > 0:
        mask = _fill_small_holes(mask, min_size)

    return mask.astype(bool), float(thr)


# ---------------------------------------------------------------------------
# 6. segmentation scores
# ---------------------------------------------------------------------------
def seg_scores(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """F1, IoU, Precision, Recall for a binary flood map vs binary ground truth."""
    pred = np.asarray(pred_mask).astype(bool).ravel()
    gt = np.asarray(gt_mask).astype(bool).ravel()

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)

    return {"F1": float(f1), "IoU": float(iou),
            "Precision": float(precision), "Recall": float(recall),
            "TP": int(tp), "FP": int(fp), "FN": int(fn)}
