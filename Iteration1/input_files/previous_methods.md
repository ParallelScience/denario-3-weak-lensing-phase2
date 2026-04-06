1. **Data Preprocessing and WST Feature Extraction**: Reconstruct 2D maps (1424×176) from flattened arrays using the boolean mask (`full_map[mask] = flat_pixels`). Add Gaussian noise (σ≈0.02582, applied only within the survey mask) to match the test set noise level. Compute Wavelet Scattering Transform (WST) features using `kymatio` with **J=3, L=8** on GPU (device='cuda') in **batches of 64** maps. Apply **global average pooling** over the spatial dimensions of the scattering output → **217 features per map** (one per scattering channel). *Benchmarked: J=3/L=8 processes all 25,856 training maps in ~0.6 minutes on GPU.* Save pooled features to disk as `wst_features_train.npy` (shape: 25856 × 217, float32). Repeat for test set: `wst_features_test.npy` (shape: 10000 × 217).

2. **Training Set Preparation**: Z-score normalize WST features using mean and std computed over the training set only (apply same normalization to test features). Hold out 20 of the 101 cosmological models (fixed seed) as the validation set. Flatten labels to shape (25856, 5) matching the flattened map order (row-major: cosmology axis first, then realizations).

3. **Conditional Normalizing Flow (CNF) Training**: Construct a Masked Autoregressive Flow (MAF) using `zuko.flows.MAF` with 5 transforms and hidden features [256, 256]. Train to learn the conditional density $p(\text{WST features} | \Omega_m, S_8, T_{\rm AGN}, f_0, \Delta z)$ using ground-truth labels from `label.npy`. Train on GPU with Adam (lr=1e-3), batch size 512, up to 50 epochs with early stopping on validation NLL. *Budget: ≤15 minutes.* Conditioning on all 5 parameters ensures the flow captures legitimate InD variation and only flags true structural anomalies as OoD.

4. **MLP Parameter Regressor Training**: Train a 3-layer MLP (217 → 512 → 256 → 5) to predict the five parameters from WST features (normalized inputs and outputs). Use MSE loss, Adam (lr=1e-3), GPU, up to 100 epochs with early stopping. This provides the conditioning vector $\hat{\theta}$ for test maps at inference time. Note: if nuisance parameter prediction (T_AGN, f_0, Δz) proves unreliable from a single noisy map, fall back to conditioning only on (Ω_m, S_8).

5. **Inference and OoD Scoring**: For each of the 10,000 test maps: (a) reconstruct 2D map, (b) compute WST features on GPU (batched), (c) apply avg pooling and z-score normalization, (d) pass through MLP to predict $\hat{\theta}$, (e) evaluate NLL = $-\log p(\text{features} | \hat{\theta})$ under the CNF. NLL is the OoD score — higher = more likely OoD. Process all maps in batches; do NOT use inter-sample statistics at test time.

6. **Validation and Performance Estimation**: Construct a synthetic OoD validation set to estimate the competition metric locally: use the 20 held-out InD cosmologies as the negative class (InD, label=0), and construct pseudo-OoD samples by applying a signal rescaling perturbation (multiply the noiseless kappa by 1.3 before adding noise, simulating a different baryonic suppression amplitude) as the positive class (OoD, label=1). Compute `score_phase2` (partial AUC, FPR in [0.001, 0.05]) on this proxy set and report as `val_score`. Note: this proxy may not perfectly reflect the true OoD type, but provides a useful relative comparison across iterations.

7. **Submission Generation**: Save the 10,000 NLL scores as `submission.json` with keys `"means"` (NLL values, list of 10,000 floats) and `"errorbars"` (zeros). Compress to `/home/node/work/weak_lensing_phase2/submission.zip`.

8. **Scientific Documentation**: Report validation partial AUC, WST architecture (J, L, feature dim), CNF architecture and training details, MLP regressor performance, and analysis of sensitivity to nuisance parameters vs. the OoD signal.

---

**Environment note:** kymatio 0.3.0 has a scipy 1.17 incompatibility (`sph_harm` renamed to `sph_harm_y`). This has already been patched at `/opt/denario-venv/lib/python3.12/site-packages/kymatio/scattering3d/filter_bank.py`. If kymatio fails to import with `ImportError: cannot import name 'sph_harm'`, apply this fix:
```bash
sed -i 's/from scipy.special import sph_harm, factorial/from scipy.special import sph_harm_y as sph_harm, factorial/' /opt/denario-venv/lib/python3.12/site-packages/kymatio/scattering3d/filter_bank.py
```
