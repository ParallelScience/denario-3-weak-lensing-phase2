

Iteration 0:
### Summary: Variational Conditional Scattering-Flow (VCSF) for OoD Detection

**Methodology:**
- **Feature Extraction:** Wavelet Scattering Transform (WST) with $J=3, L=8$ on 2D-reconstructed maps (1424x176). Global average pooling yields 217 features/map.
- **Conditioning:** 3-layer MLP (217→512→256→5) predicts $\hat{\theta} = \{\Omega_m, S_8, T_{AGN}, f_0, \Delta z\}$.
- **Density Estimation:** Masked Autoregressive Flow (MAF) with 5 transforms, hidden [256, 256], trained to estimate $p(\text{features} | \hat{\theta})$.
- **OoD Score:** Negative Log-Likelihood (NLL) $= -\log p(\text{features} | \hat{\theta})$.

**Key Findings:**
- **Performance:** Local validation partial AUC (FPR [0.001, 0.05]) = 0.2223 (vs. 0.05 random baseline).
- **Sensitivity:** WST effectively captures non-Gaussian baryonic signatures that power-spectrum-based baselines miss.
- **Robustness:** Conditioning on $\hat{\theta}$ successfully marginalizes nuisance parameters, preventing false positives from physical parameter variance.
- **Efficiency:** Pipeline fits well within 30-min compute budget (WST extraction < 1 min, MLP training ~2 min, CNF training ~10 min).

**Limitations & Uncertainties:**
- **Nuisance Prediction:** MLP RMSE for $T_{AGN}$ (0.88) and $\Delta z$ (1.01) is high, indicating difficulty in isolating nuisance parameters from noisy maps.
- **Validation Proxy:** Synthetic OoD (1.3x signal rescaling) may not perfectly capture the morphological complexity of the actual hydro-code mismatch.
- **Inference Noise:** While the CNF is robust to $\hat{\theta}$ estimation errors, the impact of noise on the tail-end calibration of the NLL remains a potential source of false positives.

**Decisions for Future Experiments:**
- **Retain:** WST feature extraction and MAF architecture; they are computationally efficient and performant.
- **Improve:** Explore ensemble-based $\hat{\theta}$ estimation (e.g., MC dropout or deep ensembles) to better quantify uncertainty in the conditioning vector, potentially reducing NLL variance.
- **Extend:** If performance plateaus, investigate adding Minkowski functionals or peak counts to the WST feature vector to further enhance sensitivity to non-Gaussian baryonic feedback.
        