# NeurIPS 2025 FAIR Universe Weak Lensing OoD Challenge (Phase 2)

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-04-06
**Status:** Methods generated — awaiting results

## Idea

**Variational Conditional Scattering-Flow (VCSF) for Robust Baryonic OoD Detection**

## Methods (8 steps)

1. Data preprocessing & WST feature caching (kymatio, J=3-4, L=8, scales 1-10 arcmin)
2. Training set preparation (z-score norm, 20% cosmology holdout)
3. Masked Autoregressive Flow training conditioned on all 5 parameters (zuko, GPU)
4. MLP parameter regressor training (WST features → θ)
5. Inference & OoD scoring (NLL under CNF conditioned on MLP-predicted θ)
6. Validation (partial AUC on holdout)
7. Submission file generation
8. Scientific documentation

## Progress

| Step | Iteration 0 |
|------|-------------|
| Setup | done |
| Idea | done |
| Methods | done |
| Results | |
| Evaluate | |
| Paper | |

