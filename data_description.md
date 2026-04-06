# Data Description: NeurIPS 2025 FAIR Universe Weak Lensing Uncertainty Challenge (Phase 2)

## Objective

**Maximize the leaderboard score on Codabench competition #10902, and produce a scientific paper describing the method and results.**

This is a full research cycle: for each iteration, we (1) develop an OoD detection method, (2) produce a competition submission file, and (3) write a paper describing the approach and findings. The goal is both to achieve the highest possible competition score and to contribute scientifically to the field of simulation-based inference and OoD detection for cosmology.

The task is **out-of-distribution (OoD) detection** on weak gravitational lensing convergence maps. Given 10,000 test maps (mix of in-distribution [InD] and out-of-distribution [OoD]), assign a continuous OoD score t(x_i) ∈ ℝ to each test sample such that OoD samples receive higher scores than InD samples.

**Scoring metric (to maximize):**
```
score = (1/100) * sum_i TPR(FPR_i)
```
where FPR_i are 100 log-spaced values in [0.001, 0.05]. This is the mean TPR over a partial ROC curve in the low-FPR regime. A perfect classifier scores 1.0; a random classifier scores ~0.05. The metric rewards high OoD detection sensitivity at very low false positive rates (0.1% to 5%).

**Primary deliverable — submission file:** A ZIP file saved at `/home/node/work/weak_lensing_phase2/submission.zip` containing `submission.json` with two keys:
- `"means"`: list of 10,000 floats (the OoD scores, one per test sample, in the same order as the test file)
- `"errorbars"`: list of 10,000 floats (uncertainty estimates, can be zeros)

This file must be produced at the end of every analysis run. It is the primary output that will be uploaded to Codabench.

**Secondary deliverable — validation score:** Ground-truth test labels are withheld by Codabench. Use a held-out validation split of the training data (hold out 20% of cosmologies or realizations) to estimate the partial AUC locally. Report this as `val_score` in the results.

**Compute budget — 30 minutes per pipeline step (hard limit).** Each step has a 1800-second timeout. Design for:
- Feature extraction (all 25,856 training maps): ≤ 10 min
- Model training (flow, CNN, etc.): ≤ 15 min
- Test set scoring (10,000 maps): ≤ 5 min
- Use GPU (device='cuda') wherever possible
- If a step risks timeout, reduce model complexity or subsample rather than failing

---

## Nature of the OoD Signal — Critical Context

The training data is generated with one specific hydrodynamical simulation code implementing baryonic feedback (AGN feedback, stellar winds, etc.). The OoD test samples are generated with the **same cosmological parameters and same physical model in principle, but using a different hydrodynamical simulation engine** — a different numerical implementation of the same baryonic physics.

This means:
- The OoD signal is **not** a different cosmology (Ω_m, S_8 values are the same distribution)
- The OoD signal is **not** the known nuisance parameters (T_AGN, f_0, Δz — these are already varied in training)
- The OoD signal is a **subtle difference in the non-Gaussian small-scale structure** arising from how different hydro codes implement baryonic feedback numerically
- The signal is expected to appear primarily at **small scales (high ℓ, baryonic feedback scales ~1-10 arcmin)** — in the non-Gaussian features of the convergence field, not in the Gaussian (power spectrum) part
- Features sensitive to **higher-order statistics, peak distributions, void statistics, Minkowski functionals, and wavelet/scattering coefficients** are expected to be more discriminative than the power spectrum alone
- The method must be robust to the known nuisances (T_AGN, f_0, Δz) already present in training — these must NOT trigger false OoD flags

This is a classic **simulation model mismatch** problem: can we detect when the data comes from a different hydro code, given that we only trained on one?

---

## Data Files (all at absolute paths)

### 1. Training convergence maps
**Path:** `/home/node/work/weak_lensing_phase2/data/public_data/WIDE12H_bin2_2arcmin_kappa_newrealization.npy`
- **Shape:** `(101, 256, 132019)` — dtype: `float16`
- 101 cosmological models × 256 systematic realizations × 132,019 valid pixels per map
- **IMPORTANT:** Data is stored as **flattened valid pixels only** (not full 2D arrays). To reconstruct 2D maps, use the mask (see below):
  ```python
  mask = np.load(".../WIDE12H_bin2_2arcmin_mask.npy")  # shape (1424, 176), dtype bool
  full_map = np.zeros((1424, 176), dtype=np.float32)
  full_map[mask] = kappa[cosmo_idx, sys_idx].astype(np.float32)
  ```
- Training maps are **noiseless**. Gaussian noise must be added before use:
  ```python
  ng = 30  # galaxies per arcmin²
  pixel_size = 2.0  # arcmin
  noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5  # ≈ 0.02582
  noisy_map = noiseless_map + np.random.randn(*noiseless_map.shape) * noise_sigma * mask
  ```
- Memory: 6.8 GB as float16 (as stored), 13.7 GB if cast to float32
- Convergence range: approximately [-0.05, 0.23]

### 2. Test convergence maps
**Path:** `/home/node/work/weak_lensing_phase2/data/public_data/WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy`
- **Shape:** `(10000, 132019)` — dtype: `float64`
- 10,000 noisy test maps stored as flattened valid pixels
- **Already noisy** (noise added with ng=30, pixel_size=2 arcmin)
- Reconstruct 2D with `full_map[mask] = kappa_test[i]`
- Memory: 10.6 GB
- Mix of InD (label=0) and OoD (label=1) — ground truth labels withheld

### 3. Survey mask
**Path:** `/home/node/work/weak_lensing_phase2/data/public_data/WIDE12H_bin2_2arcmin_mask.npy`
- **Shape:** `(1424, 176)` — dtype: `bool`
- 132,019 of 250,624 pixels are valid (52.7%)
- Apply when adding noise: zero noise outside survey area

### 4. Training labels
**Path:** `/home/node/work/weak_lensing_phase2/data/public_data/label.npy`
- **Shape:** `(101, 256, 5)` — dtype: `float64`
- `[..., 0]` = Ω_m: range [0.091, 0.619], mean 0.290 — cosmological parameter
- `[..., 1]` = S_8: range [0.680, 0.955], mean 0.814 — cosmological parameter
- `[..., 2]` = T_AGN ∈ [7.20, 8.50], uniform — **nuisance, marginalize**
- `[..., 3]` = f_0 ∈ [0.00004, 0.0264], uniform — **nuisance, marginalize**
- `[..., 4]` = Δz ~ N(0, 0.022) — **nuisance, marginalize**
- Cosmological parameters vary across 101 cosmologies (axis 0)
- Nuisance parameters vary across 256 realizations (axis 1)

---

## Key Utility Code

```python
import numpy as np

DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"

# Load all files
mask       = np.load(f"{DATA_DIR}/WIDE12H_bin2_2arcmin_mask.npy")           # (1424, 176) bool
kappa      = np.load(f"{DATA_DIR}/WIDE12H_bin2_2arcmin_kappa_newrealization.npy", mmap_mode='r')  # (101, 256, 132019) float16
kappa_test = np.load(f"{DATA_DIR}/WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy", mmap_mode='r')  # (10000, 132019) float64
label      = np.load(f"{DATA_DIR}/label.npy")                                # (101, 256, 5) float64

# Add noise to a flat training map
def add_noise(flat_map, mask, ng=30, pixel_size=2.0):
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5  # ≈ 0.02582
    return flat_map + np.random.randn(*flat_map.shape) * noise_sigma * mask[mask]

# Reconstruct 2D map from flat valid pixels
def to_2d(flat_map, mask):
    full = np.zeros(mask.shape, dtype=np.float32)
    full[mask] = flat_map.astype(np.float32)
    return full  # shape (1424, 176)

# Scoring metric (partial AUC, FPR in [0.001, 0.05])
from sklearn.metrics import roc_curve
def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)

# Save submission
import json, zipfile
def save_submission(ood_scores, errorbars=None, save_dir="/home/node/work/weak_lensing_phase2"):
    if errorbars is None:
        errorbars = [0.0] * len(ood_scores)
    data = {"means": list(map(float, ood_scores)), "errorbars": list(map(float, errorbars))}
    json_path = f"{save_dir}/submission.json"
    zip_path  = f"{save_dir}/submission.zip"
    with open(json_path, 'w') as f:
        json.dump(data, f)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(json_path, "submission.json")
    return zip_path
```

### Power spectrum utility (from starting kit)
```python
import scipy.stats
def power_spectrum(x, pixsize, kedge):
    """Azimuthally averaged 2D power spectrum. x: 2D array, pixsize: pixel size in radians."""
    assert x.ndim == 2
    xk = np.fft.rfft2(x)
    xk2 = (xk * xk.conj()).real
    Nmesh = x.shape
    k = np.zeros((Nmesh[0], Nmesh[1]//2+1))
    k += np.fft.fftfreq(Nmesh[0], d=pixsize).reshape(-1, 1) ** 2
    k += np.fft.rfftfreq(Nmesh[1], d=pixsize).reshape(1, -1) ** 2
    k = np.sqrt(k)
    power, _, _ = scipy.stats.binned_statistic(k.ravel(), xk2.ravel(), statistic='mean', bins=kedge)
    k_centers, _, _ = scipy.stats.binned_statistic(k.ravel(), k.ravel(), statistic='mean', bins=kedge)
    norm = x.size * pixsize**2
    return k_centers, power / norm
```

---

## Baselines (known to be suboptimal — to be improved upon)

Three baselines from the competition starting kit use power spectrum χ², CNN parameter estimation χ², and autoencoder reconstruction error. All rely on Gaussian or low-dimensional summary statistics and are known to miss the non-Gaussian OoD signal. Any method exploiting higher-order statistics or learned density estimation should substantially outperform them.

---

## Hardware Available

- **CPU:** 32 vCPUs (AMD Threadripper PRO 9995WX), cgroup limit
- **RAM:** 64 GB (cgroup hard limit) — use mmap_mode='r' and float16 to stay within budget
- **GPU:** NVIDIA RTX PRO 6000 Blackwell, 97.9 GB VRAM, CUDA 12.8
- **Python:** `/opt/denario-venv/bin/python3.12`
- **Installed:** torch 2.11+cu128, torchvision, numpy, scipy, scikit-learn, kymatio (wavelet scattering), zuko (normalizing flows), emcee, h5py, joblib, matplotlib

---

## Important Notes

1. **Data is NOT 2D images in the files.** Stored as 1D arrays of 132,019 valid pixels. Always reconstruct to 2D before 2D operations (FFT, convolutions, scattering transforms).

2. **Training data is noiseless; test data is noisy.** Always add noise to training maps before computing features.

3. **Multiple noise realizations are free.** Add noise with different seeds to augment the training set.

4. **No inter-sample information at test time.** OoD scores must be independent per sample (disqualification rule).

5. **Nuisance parameters must be marginalized.** Methods that flag T_AGN, f_0, or Δz variation as OoD will score poorly. The 256 realizations per cosmology cover this variation fully.
