# SAR Despeckling → Flood-Mapping Evaluation Harness

One model-agnostic pipeline. Every method — Lee, PPB, SAR-BM3D, ID-CNN, MERLIN,
SAR-Trans, and later YOUR model — is scored by the *same* code, so the comparison
is fair. The harness never trains anything; it evaluates **outputs**.

```
sar_eval/
  sar_metrics.py          # all metrics: PSNR/SSIM, ENL/Cx/mean/EPI, Otsu+KI, log-ratio, F1/IoU
  classical_despeckle.py  # speckle generator (matches your DIV2K script) + Lee/RefinedLee/Frost/Kuan
  run_eval.py             # CLI: denoise | speckle | synthetic | real | flood
  profile_model.py        # params / FLOPs / inference-time  (the column where Mamba wins)
```

Dependencies: `numpy scipy scikit-image pillow` (+ `torch` and optionally `thop`
only for `profile_model.py`).

---

## The protocol (decide once, keep fixed)

| Split | Data | Used for | Report it? |
|-------|------|----------|-----------|
| Train | DIV2K + L=1 speckle (your `prepare_div2k_for_training.py`) | fit the denoiser | no |
| **Validation** | **BSD68 + L=1 speckle** | checkpoint / early-stopping | no |
| Test (synthetic) | **Set12 + L=1 speckle** | PSNR/SSIM table | **yes** — matches SAR-Trans Table 1 |
| Test (real) | Mendeley SAR | ENL/Cx/mean/EPI | **yes** |
| Downstream | S1GFloods (bitemporal) | flood F1/IoU/Prec/Recall | **yes — the headline** |

**Your BSD68-as-validation question: YES.** DIV2K and BSD68 are disjoint, so no
leakage. One rule: because BSD68 picks your checkpoint, do **not** also report it
as a test number — report synthetic test results on **Set12** (keeps you
comparable to the SAR-Trans paper). Clean three-way split.

Generate the Set12 test set with the SAME clip+uint8 pipeline as your training
(this matters — your net learned on clipped data):
```bash
python run_eval.py speckle --clean /path/Set12_clean --output data/Set12_noisy_L1 --looks 1 --seed 0
```

---

## ⚠️ The one mistake that will silently corrupt your numbers

**Never save denoised outputs with per-image min-max normalization.** It rescales
each image differently, which:
- breaks **mean-preservation** (your 1.5-bias metric becomes meaningless), and
- destroys the **log-ratio** (the flood step divides post by pre — independent
  rescaling = garbage ratio = F1 collapse).

Save denoised outputs as **`.npy` float** (exact) or as PNG with **fixed** scaling.
The classical `denoise` subcommand already saves `.npy`. For the learned models
(ID-CNN / MERLIN / SAR-Trans), make their inference code save `.npy` or fixed-scale
PNG — **not** `imsave` with auto-contrast. ENL/Cx/EPI are scale-invariant and
survive either way; mean-preservation and flood do not.

---

## Step-by-step: run all 5 baselines through the full grid

### Classical baselines (training-free, run here)
```bash
for F in lee refined_lee frost kuan; do
  # synthetic test set
  python run_eval.py denoise --filter $F --input data/Set12_noisy_L1 --output out/$F/set12
  # real SAR
  python run_eval.py denoise --filter $F --input data/Mendeley --output out/$F/mendeley
  # S1GFloods pre & post (despeckle each date)
  python run_eval.py denoise --filter $F --input data/S1G/pre  --output out/$F/s1g_pre
  python run_eval.py denoise --filter $F --input data/S1G/post --output out/$F/s1g_post
done
```
For **PPB** and **SAR-BM3D**: run the authors' released code, save outputs as
`.npy`/fixed-scale PNG into `out/ppb/...` and `out/sarbm3d/...`, same folder layout.

### Learned baselines (ID-CNN, MERLIN, SAR-Trans)
Run each model's own inference on the same four input folders
(`Set12_noisy_L1`, `Mendeley`, `S1G/pre`, `S1G/post`), saving outputs to
`out/<model>/<set>`. Then they plug into the eval commands below identically.

### Score every method (identical commands, just change the path)
```bash
M=refined_lee   # or lee/frost/kuan/ppb/sarbm3d/idcnn/merlin/sartrans

# 1) synthetic PSNR/SSIM  (vs clean Set12)
python run_eval.py synthetic --clean data/Set12_clean --pred out/$M/set12 \
       --out-csv results/$M.synthetic.csv

# 2) real ENL/Cx/mean/EPI  (vs noisy Mendeley; define ROIs — see below)
python run_eval.py real --noisy data/Mendeley --pred out/$M/mendeley \
       --roi-json rois.json --out-csv results/$M.real.csv

# 3) DOWNSTREAM flood  (despeckled pre/post -> F1/IoU/Prec/Recall)
python run_eval.py flood --pre out/$M/s1g_pre --post out/$M/s1g_post \
       --mask data/S1G/mask --domain linear --method ki --direction decrease \
       --save-maps maps/$M --out-csv results/$M.flood.csv
```

### The two reference rows you MUST include
```bash
# RAW floor: flood on undenoised S1GFloods (shows speckle's damage)
python run_eval.py flood --pre data/S1G/pre --post data/S1G/post --mask data/S1G/mask \
       --domain linear --method ki --direction decrease --out-csv results/RAW.flood.csv
```
The RAW row is what every despeckler must beat — it *is* the "is despeckling worth
it?" answer. (In the self-test: raw F1=0.00 → despeckled F1=0.98.)

---

## Things you must verify for YOUR data

1. **S1GFloods pixel domain.** If the PNGs encode **linear intensity**, use
   `--domain linear` (log-ratio). If they encode **dB** (byte-scaled), use
   `--domain db` (difference). Check the repo/readme or histogram a few tiles.
   Wrong choice tanks F1 — try both on 5 images and keep the better.
2. **Flood direction.** Open-water flooding lowers backscatter → `--direction
   decrease` (default). If your event is inundated vegetation (backscatter
   *increase* via double-bounce), use `--direction increase`. If unsure,
   `--direction magnitude`.
3. **Fixed ROIs for the real-SAR table.** `--roi auto` is fine for a quick look,
   but for the paper define homogeneous regions per image in `rois.json`:
   ```json
   { "mendeley_img1": [[40,72,40,72],[150,182,200,232]],
     "__default__":   [[10,42,10,42]] }
   ```
   (boxes are `[r0,r1,c0,c1]`).

---

## Efficiency block (run in your training env, has torch)

```python
from profile_model import profile_model
# load each model, then:
print(profile_model(model, input_shape=(1,1,256,256), device="cuda"))
# -> {'params_M':.., 'gflops':.., 'time_ms_mean':.., 'fps':..}
```
For **SAR-DDPM** (dropped from quality tables): measure ONE timing run and report
it as an efficiency-only row. Expected story: *diffusion = seconds–minutes/scene;
Mamba = milliseconds at comparable quality.*

---

## What "good" looks like (calibration, from the self-test)

| Metric | Raw single-look | A correct despeckler | Pathology |
|--------|----------------|----------------------|-----------|
| ENL (homogeneous ROI) | ≈ 1 | tens → low hundreds | **thousands = over-smoothed (your old 5000)** |
| Cx | ≈ 1 | ≪ 1 (e.g. 0.1–0.3) | — |
| mean-preservation | 1.0 | **≈ 1.0** | **1.5 = radiometric bias bug** |
| EPI vs noisy | 1.0 (trivially) | lower; rank across methods | — |

Report **ENL and EPI together** (a trade-off curve), never ENL alone — a method at
ENL 60 / high-EPI beats one at ENL 5000 / EPI 0.38 for every downstream purpose.
