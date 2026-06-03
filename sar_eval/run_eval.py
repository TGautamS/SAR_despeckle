#!/usr/bin/env python3
"""
run_eval.py  --  one harness, identical pipeline for every method.

Subcommands
-----------
  denoise     Apply a classical filter (lee/refined_lee/frost/kuan) to a folder.
              (For learned models, run their own code; just save the outputs.)

  synthetic   PSNR / SSIM on synthetic speckled test set (needs clean GT).
              -> use Set12 to match SAR-Trans Table 1.

  real        ENL / Cx / mean-preservation / EPI on real SAR (Mendeley).
              No-reference: metrics on homogeneous ROIs vs the noisy original.

  flood       Unsupervised flood map from bitemporal pairs (S1GFloods):
              log-ratio -> Otsu/KI threshold -> morphology -> F1/IoU/Prec/Recall.
              Run it once on RAW pre/post, once on DENOISED pre/post -> the
              raw-vs-denoised comparison is your headline table.

Everything writes a per-image CSV + prints a mean summary.

Image I/O: .png/.jpg/.tif/.tiff via PIL, .npy via numpy. Grayscale assumed
(SAR is single-channel); RGB is averaged to luma.
"""

from __future__ import annotations
import argparse, csv, os, sys, glob, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sar_metrics as M
import classical_despeckle as C


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_gray(path: str) -> np.ndarray:
    if path.lower().endswith(".npy"):
        a = np.load(path).astype(np.float64)
    else:
        from PIL import Image
        im = Image.open(path)
        a = np.asarray(im, dtype=np.float64)
    if a.ndim == 3:                       # RGB/RGBA -> luma
        a = a[..., :3].mean(axis=2)
    return a


def save_array(path: str, arr: np.ndarray):
    """Radiometry-PRESERVING save for denoised outputs.

    .npy  -> exact float (RECOMMENDED: preserves mean & ratios perfectly).
    .png/.tif -> clip to [0,255] uint8 with FIXED scaling (NOT min-max), so the
    radiometry survives. NEVER min-max here -- it would break mean-preservation
    and the log-ratio flood step.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".npy"):
        np.save(path, arr.astype(np.float32)); return
    from PIL import Image
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def save_mask(path: str, mask: np.ndarray):
    """Binary mask -> 0/255 PNG (visualisation only)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from PIL import Image
    Image.fromarray((np.asarray(mask).astype(bool) * 255).astype(np.uint8)).save(path)


def to_unit(img: np.ndarray) -> np.ndarray:
    """Bring an image to [0,1] with FIXED /255 scaling if it looks like [0,255].
    Fixed scaling (not min-max) keeps pred and reference on the same radiometric
    footing -- required for a valid PSNR and mean-preservation."""
    a = np.asarray(img, dtype=np.float64)
    return a / 255.0 if np.nanmax(a) > 1.5 else a


def list_images(folder: str) -> list[str]:
    fs = [f for f in sorted(glob.glob(os.path.join(folder, "*")))
          if f.lower().endswith(IMG_EXT) or f.lower().endswith(".npy")]
    if not fs:
        sys.exit(f"[error] no images found in {folder}")
    return fs


def match_by_name(dir_a: str, dir_b: str):
    """Pair files in two dirs by basename (ignoring extension)."""
    a = {os.path.splitext(os.path.basename(f))[0]: f for f in list_images(dir_a)}
    b = {os.path.splitext(os.path.basename(f))[0]: f for f in list_images(dir_b)}
    keys = sorted(set(a) & set(b))
    if not keys:
        sys.exit(f"[error] no matching filenames between {dir_a} and {dir_b}")
    miss = (set(a) | set(b)) - set(keys)
    if miss:
        print(f"[warn] {len(miss)} unmatched files skipped (e.g. {sorted(miss)[:3]})")
    return [(k, a[k], b[k]) for k in keys]


def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {path}")


def summarise(rows, keys):
    print("\n==== MEAN over {} images ====".format(len(rows)))
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            print(f"  {k:>18s} : {np.mean(vals):.4f}")
    print("=" * 34)


# ---------------------------------------------------------------------------
# ROI handling for the 'real' block
# ---------------------------------------------------------------------------
def auto_roi(img: np.ndarray, win: int = 32) -> tuple[int, int, int, int]:
    """Pick the lowest-variance win x win patch (a quick stand-in for a hand-drawn
    homogeneous ROI). For the FINAL paper, define fixed ROIs with --roi-json."""
    from scipy.ndimage import uniform_filter
    m = uniform_filter(img, win)
    v = uniform_filter(img * img, win) - m * m
    v[:win, :] = v[-win:, :] = v[:, :win] = v[:, -win:] = np.inf
    i, j = np.unravel_index(np.argmin(v), v.shape)
    h = win // 2
    return (i - h, i + h, j - h, j + h)


def get_rois(name, img, args):
    if args.roi_json:
        with open(args.roi_json) as fh:
            spec = json.load(fh)
        boxes = spec.get(name) or spec.get(os.path.basename(name)) or spec.get("__default__")
        if boxes is None:
            sys.exit(f"[error] no ROI for {name} in {args.roi_json}")
        return [tuple(b) for b in boxes]
    if args.roi and args.roi != "auto":
        return [tuple(int(x) for x in args.roi.split(","))]
    return [auto_roi(img, args.roi_win)]


# ---------------------------------------------------------------------------
# subcommand: denoise (classical filters)
# ---------------------------------------------------------------------------
def cmd_denoise(args):
    fn = C.FILTERS[args.filter]
    for f in list_images(args.input):
        img = load_gray(f)
        kw = {"size": args.size}
        if args.filter in ("lee", "refined_lee", "kuan"):
            kw["L"] = args.looks
        if args.filter == "frost":
            kw["damping"] = args.damping
        out = fn(img, **kw)
        # save exact float (.npy) so mean & log-ratio survive; basename preserved
        base = os.path.splitext(os.path.basename(f))[0]
        save_array(os.path.join(args.output, base + ".npy"), out)
    print(f"[done] {args.filter} -> {args.output} (saved as .npy, exact float)")


def cmd_speckle(args):
    for i, f in enumerate(list_images(args.clean)):
        clean = load_gray(f)
        clean_u8 = np.clip(clean, 0, 255).astype(np.uint8) if clean.max() > 1.5 \
            else (clean * 255).clip(0, 255).astype(np.uint8)
        noisy = C.add_speckle_div2k_style(clean_u8, L=args.looks, seed=args.seed + i)
        save_array(os.path.join(args.output, os.path.basename(f)), noisy)
    print(f"[done] speckle L={args.looks} -> {args.output} "
          f"(div2k-style: /255, Gamma, *255, clip, uint8)")


# ---------------------------------------------------------------------------
# subcommand: synthetic (PSNR / SSIM)
# ---------------------------------------------------------------------------
def cmd_synthetic(args):
    rows = []
    for name, cf, pf in match_by_name(args.clean, args.pred):
        clean, pred = load_gray(cf), load_gray(pf)
        # put both on the SAME [0,1] footing with fixed scaling -> valid PSNR
        clean, pred = to_unit(clean), to_unit(pred)
        rows.append({"image": name,
                     "PSNR": M.psnr(pred, clean, data_range=1.0),
                     "SSIM": M.ssim_metric(pred, clean, data_range=1.0)})
    write_csv(rows, args.out_csv)
    summarise(rows, ["PSNR", "SSIM"])


# ---------------------------------------------------------------------------
# subcommand: real (ENL / Cx / mean-preservation / EPI)
# ---------------------------------------------------------------------------
def cmd_real(args):
    rows = []
    for name, nf, pf in match_by_name(args.noisy, args.pred):
        noisy, pred = load_gray(nf), load_gray(pf)
        rois = get_rois(name, noisy, args)
        enls, cxs = [], []
        for (r0, r1, c0, c1) in rois:
            patch = pred[r0:r1, c0:c1]
            enls.append(M.enl(patch))      # ENL & Cx are scale-invariant
            cxs.append(M.cx(patch))
        # mean-preservation needs a common scale (fixed, not min-max)
        mp = M.mean_preservation(to_unit(pred), to_unit(noisy))
        rows.append({"image": name,
                     "ENL": float(np.mean(enls)),
                     "Cx": float(np.mean(cxs)),
                     "mean_preservation": mp,
                     "EPI_vs_noisy": M.epi(pred, noisy)})
    write_csv(rows, args.out_csv)
    summarise(rows, ["ENL", "Cx", "mean_preservation", "EPI_vs_noisy"])


# ---------------------------------------------------------------------------
# subcommand: flood (log-ratio -> threshold -> morphology -> scores)
# ---------------------------------------------------------------------------
def cmd_flood(args):
    pre = {os.path.splitext(os.path.basename(f))[0]: f for f in list_images(args.pre)}
    post = {os.path.splitext(os.path.basename(f))[0]: f for f in list_images(args.post)}
    mask = {os.path.splitext(os.path.basename(f))[0]: f for f in list_images(args.mask)}
    keys = sorted(set(pre) & set(post) & set(mask))
    if not keys:
        sys.exit("[error] no triplets matched across pre/post/mask by basename")
    print(f"[info] {len(keys)} pre/post/mask triplets")

    rows = []
    for k in keys:
        pr, po = load_gray(pre[k]), load_gray(post[k])
        gt = load_gray(mask[k]) > 0
        ratio = M.log_ratio(pr, po, domain=args.domain)
        pred_mask, thr = M.detect_change(
            ratio, method=args.method, direction=args.direction,
            open_radius=args.open_radius, close_radius=args.close_radius,
            min_size=args.min_size, fill_holes=not args.no_fill)
        sc = M.seg_scores(pred_mask, gt)
        sc.update({"image": k, "threshold": thr})
        rows.append(sc)
        if args.save_maps:
            save_mask(os.path.join(args.save_maps, k + ".png"), pred_mask)
    # reorder columns nicely
    ordered = [{"image": r["image"], "F1": r["F1"], "IoU": r["IoU"],
                "Precision": r["Precision"], "Recall": r["Recall"],
                "threshold": r["threshold"],
                "TP": r["TP"], "FP": r["FP"], "FN": r["FN"]} for r in rows]
    write_csv(ordered, args.out_csv)
    summarise(ordered, ["F1", "IoU", "Precision", "Recall"])


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("denoise", help="apply a classical filter to a folder")
    d.add_argument("--filter", required=True, choices=list(C.FILTERS))
    d.add_argument("--input", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--size", type=int, default=7)
    d.add_argument("--looks", type=int, default=1)
    d.add_argument("--damping", type=float, default=2.0)
    d.set_defaults(func=cmd_denoise)

    s = sub.add_parser("synthetic", help="PSNR/SSIM vs clean GT")
    s.add_argument("--clean", required=True, help="clean ground-truth dir")
    s.add_argument("--pred", required=True, help="denoised-output dir")
    s.add_argument("--out-csv", default="results_synthetic.csv")
    s.set_defaults(func=cmd_synthetic)

    sp = sub.add_parser("speckle",
                        help="generate synthetic test set (div2k-style: clip+uint8)")
    sp.add_argument("--clean", required=True, help="clean image dir (e.g. Set12)")
    sp.add_argument("--output", required=True, help="output dir for noisy images")
    sp.add_argument("--looks", type=int, default=1)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_speckle)

    r = sub.add_parser("real", help="ENL/Cx/mean/EPI on real SAR")
    r.add_argument("--noisy", required=True, help="original noisy SAR dir")
    r.add_argument("--pred", required=True, help="denoised-output dir")
    r.add_argument("--roi", default="auto",
                   help="'auto', or a single box 'r0,r1,c0,c1' for all images")
    r.add_argument("--roi-json", default=None,
                   help="JSON {filename:[[r0,r1,c0,c1],...]} for fixed ROIs (paper)")
    r.add_argument("--roi-win", type=int, default=32, help="auto-ROI patch size")
    r.add_argument("--out-csv", default="results_real.csv")
    r.set_defaults(func=cmd_real)

    f = sub.add_parser("flood", help="unsupervised flood map -> F1/IoU/Prec/Recall")
    f.add_argument("--pre", required=True, help="pre-flood dir (raw OR denoised)")
    f.add_argument("--post", required=True, help="post-flood dir (raw OR denoised)")
    f.add_argument("--mask", required=True, help="ground-truth flood mask dir")
    f.add_argument("--domain", choices=["linear", "db"], default="linear",
                   help="'linear' intensity (log-ratio) or 'db' (difference)")
    f.add_argument("--method", choices=["ki", "otsu"], default="ki")
    f.add_argument("--direction", choices=["decrease", "increase", "magnitude"],
                   default="decrease", help="open-water flood = decrease (default)")
    f.add_argument("--open-radius", type=int, default=1)
    f.add_argument("--close-radius", type=int, default=1)
    f.add_argument("--min-size", type=int, default=25)
    f.add_argument("--no-fill", action="store_true", help="disable hole filling")
    f.add_argument("--save-maps", default=None, help="dir to dump predicted masks")
    f.add_argument("--out-csv", default="results_flood.csv")
    f.set_defaults(func=cmd_flood)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
