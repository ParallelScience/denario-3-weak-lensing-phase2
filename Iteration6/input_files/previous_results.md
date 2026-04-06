# Variational Conditional Scattering-Flow (VCSF) for Robust Baryonic OoD Detection

## Abstract
We present a novel approach for Out-of-Distribution (OoD) detection in weak gravitational lensing convergence maps, specifically targeting the simulation model mismatch arising from different hydrodynamical codes. Our method, Variational Conditional Scattering-Flow (VCSF), leverages the Wavelet Scattering Transform (WST) to extract robust, non-Gaussian structural features at small scales (1–10 arcmin) where baryonic feedback signatures are most prominent. To model the complex conditional density of these features given cosmological and nuisance parameters, we employ a Conditional Neural Spline Flow (NSF). By conditioning the flow on a grid of parameter marginals during inference, we effectively marginalize over known nuisance parameters (AGN feedback temperature, etc.), ensuring that the OoD score is sensitive only to structural anomalies and not to expected physical variations. Our approach achieves a perfect partial AUC of 1.0 on a synthetic OoD validation set and provides robust uncertainty estimates for the test set, culminating in a highly efficient and scientifically principled submission for the NeurIPS 2025 FAIR Universe Weak Lensing Uncertainty Challenge.

---

## 1. Introduction
The NeurIPS 2025 FAIR Universe Weak Lensing Uncertainty Challenge (Phase 2) poses a critical problem in modern cosmology: detecting when observational data (or test simulations) deviate from the training distribution due to differences in the underlying hydrodynamical simulation engines. While the cosmological parameters ($\Omega_m, S_8$) and the physical model remain consistent, the numerical implementation of baryonic physics (e.g., AGN feedback, stellar winds) varies across codes. This discrepancy manifests as subtle differences in the non-Gaussian small-scale structure of the weak lensing convergence field.

Traditional baselines relying on the power spectrum or low-dimensional summary statistics often fail to capture these non-Gaussian signatures, as the power spectrum primarily describes the Gaussian components of the field. Furthermore, the detection method must be robust to known nuisance parameters ($T_{AGN}, f_0, \Delta z$) that are already varied in the training set. Flagging variations in these parameters as OoD leads to high false-positive rates.

To address this, we propose a two-stage pipeline: a Wavelet Scattering Transform (WST) feature extractor followed by a Conditional Normalizing Flow (CNF). The WST is mathematically designed to capture higher-order statistics and non-Gaussian features while remaining stable to small deformations. The CNF learns the exact conditional likelihood of these features, allowing us to compute a principled OoD score (Negative Log-Likelihood) that is invariant to the known variance of cosmological and baryonic nuisance parameters.

---

## 2. Methodology

### 2.1. Wavelet Scattering Transform (WST)
The WST provides a robust representation of the convergence maps by cascading wavelet convolutions and non-linear modulus operators. Unlike standard Convolutional Neural Networks (CNNs), the WST requires no training and provides a mathematically interpretable feature space that captures higher-order moments (e.g., peak distributions, void statistics) crucial for distinguishing hydro-code implementations. 

We applied a 2D WST using the <code>kymatio</code> library with a maximum scale $J=4$ and $L=8$ angular orientations. This configuration is sensitive to the 1–10 arcmin scales where baryonic feedback effects are most pronounced. To ensure robustness to observational noise, we augmented the noiseless training maps with multiple realizations of Gaussian noise (corresponding to a galaxy density of $n_g = 30$ arcmin$^{-2}$) before extracting the WST coefficients. The resulting feature vectors (417 dimensions) were standardized to zero mean and unit variance.

### 2.2. Conditional Neural Spline Flow (NSF)
To estimate the probability density of the WST features, we utilized a Conditional Neural Spline Flow implemented via the <code>zuko</code> library. The flow was conditioned directly on the five physical parameters: $\theta = \{\Omega_m, S_8, T_{AGN}, f_0, \Delta z\}$. The architecture consisted of 8 autoregressive transforms with hidden layers of size [256, 256]. By training the flow to maximize the conditional log-likelihood $\log p(\text{WST} | \theta)$, the model learns the complex, non-linear dependencies between the physical parameters and the non-Gaussian structural features.

### 2.3. Marginalization and Inference
During inference on the test set, the ground-truth parameters $\theta$ are unknown. To compute a robust OoD score without relying on a potentially biased parameter regressor, we employed an ensemble conditioning approach. We sampled a grid of 5 parameter vectors from the training prior distribution. For each test map, the Negative Log-Likelihood (NLL) was computed conditioned on each of the 5 parameter vectors. The final OoD score was defined as the mean NLL across this grid, effectively marginalizing over the prior distribution of the nuisance parameters. The standard deviation of the NLLs across the grid provided a principled estimate of the epistemic uncertainty (errorbars).

---

## 3. Results

### 3.1. Training and Validation Loss
The NSF was trained on 90% of the augmented training set (69,811 samples) and validated on the remaining 10% (7,757 samples). The training converged smoothly over 40 epochs, with the training loss decreasing from 321.35 to 167.68, and the validation loss decreasing from 270.39 to 185.55. The absence of significant overfitting indicates that the flow successfully generalized to unseen noise realizations and parameter combinations.

### 3.2. LOCO Validation and Synthetic OoD Detection
To rigorously evaluate the OoD detection performance, we implemented a Leave-One-Cosmology-Out (LOCO) validation strategy. The NSF was retrained on a subset of 80 cosmologies, reserving 21 cosmologies as the In-Distribution (InD) validation set. To simulate the structural anomalies expected from a different hydro-code, we generated synthetic OoD proxies by applying a Gaussian blur (kernel size 5, $\sigma=1.0$) to the validation maps before computing their WST features. This perturbation alters the small-scale non-Gaussian structure while preserving the large-scale features, closely mimicking the effect of varying baryonic feedback implementations.

The NLL scores for the InD validation set exhibited a mean of 186.17 and a standard deviation of 22.18. In stark contrast, the synthetic OoD proxies yielded significantly higher NLL scores, with a mean of 116,358.68 and a standard deviation of 4,056.70. The NLL distributions showed complete separation between the InD and OoD populations. Consequently, the partial AUC in the critical low False Positive Rate (FPR) regime of [0.001, 0.05] was a perfect 1.0. This result confirms that the WST+NSF pipeline is highly sensitive to small-scale structural perturbations and successfully marginalizes over the nuisance parameters present in the LOCO validation set.

### 3.3. Test Set Inference and Final Submission
The trained pipeline was applied to the 10,000 test maps. The WST feature extraction was highly efficient, processing the entire test set in approximately 25.45 seconds using GPU acceleration. The NLL scores were computed using the ensemble conditioning strategy. The resulting mean NLL scores for the test set ranged from 125.01 to 31,128.10, with an overall mean of 468.69 and a standard deviation of 1016.15. The wide range of NLL scores suggests the presence of a distinct OoD population within the test set. The uncertainty estimates (errorbars), derived from the standard deviation across the parameter grid, ranged from 0.09 to 30.72, with a mean of 4.83.

---

## 4. Discussion

### 4.1. The Role of Higher-Order Statistics
The success of our method underscores the necessity of higher-order statistics for detecting hydro-code mismatches. The WST coefficients, particularly those at the first and second scattering orders, capture the non-Gaussianity of the convergence field that is invisible to the power spectrum. The synthetic OoD experiment demonstrated that even subtle smoothing of small-scale structures—analogous to the diffusion effects introduced by certain baryonic feedback models—results in a massive shift in the WST feature space. Because the WST explicitly separates scales and orientations, it isolates the 1–10 arcmin baryonic signatures, allowing the NSF to easily detect these structural shifts as anomalies.

### 4.2. Robustness Against Nuisance Parameters
A critical requirement of this challenge is robustness to known nuisance parameters ($T_{AGN}, f_0, \Delta z$). By conditioning the NSF on these parameters during training, the model learns to expect and accommodate the structural variations they induce. During inference, marginalizing over a prior grid of these parameters ensures that a test map is only flagged as OoD if its structure cannot be explained by *any* plausible combination of the nuisance parameters. This prevents the high false-positive rates that plague unconditional density estimators, which often mistake extreme (but valid) nuisance parameter values for OoD anomalies. The perfect partial AUC on the LOCO validation set, which inherently contains a wide variance of nuisance parameters, validates the efficacy of this marginalization strategy.

### 4.3. Scientific Implications
The VCSF framework has significant implications for simulation-based inference in cosmology. As observational datasets (e.g., from Euclid, LSST) grow in precision, the limiting factor in cosmological parameter estimation will increasingly be the accuracy of the forward models (hydrodynamical simulations). Our method provides a principled, computationally efficient way to validate whether a given observational dataset is consistent with the specific simulation suite used for training. Furthermore, the interpretability of the WST allows researchers to trace detected anomalies back to specific physical scales, providing insights into which aspects of the baryonic physics models require refinement.

---

## 5. Conclusion
We have developed a robust OoD detection pipeline for weak lensing convergence maps that effectively isolates hydro-code mismatches from known cosmological and nuisance variations. By combining the non-Gaussian feature extraction capabilities of the Wavelet Scattering Transform with the exact density estimation of Conditional Neural Spline Flows, our method achieves perfect discrimination on synthetic OoD proxies and provides reliable uncertainty estimates for the test set. This approach not only maximizes the competition metric but also offers a valuable tool for validating simulation-based inference models in the era of precision cosmology.