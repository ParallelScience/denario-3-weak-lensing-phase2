# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from kymatio.torch import Scattering2D
import inspect
from step_2 import *

plt.rcParams['text.usetex'] = False

def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)

def add_noise_batch(flat_maps, noise_sigma):
    noise = np.random.randn(*flat_maps.shape).astype(np.float32) * noise_sigma
    return flat_maps + noise

def reconstruct_2d_batch(flat_maps, mask):
    n_sys = flat_maps.shape[0]
    maps_2d = np.zeros((n_sys, *mask.shape), dtype=np.float32)
    maps_2d[:, mask] = flat_maps
    return maps_2d

def compute_nll(features, mlp_model, gmm_model, f_mean, f_std):
    features_scaled = (features - f_mean) / f_std
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
    with torch.no_grad():
        out = mlp_model(features_tensor)
        theta_hat = out[0] if isinstance(out, tuple) else out[:, :5]
    theta_hat = theta_hat.cpu().numpy()
    x_gmm = np.hstack([features_scaled, theta_hat])
    return -gmm_model.score_samples(x_gmm)

if __name__ == '__main__':
    DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
    OUTPUT_DIR = "data"
    mask = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"), mmap_mode='r')
    val_idx_path = os.path.join(OUTPUT_DIR, "val_cosmo_idx.npy")
    val_cosmo_idx = np.load(val_idx_path)
    top_100_idx = np.load(os.path.join(OUTPUT_DIR, "top_100_wst_indices.npy"))
    wst_features = np.load(os.path.join(OUTPUT_DIR, "wst_features.npy"))
    train_idx_path = os.path.join(OUTPUT_DIR, "train_cosmo_idx.npy")
    train_cosmo_idx = np.load(train_idx_path)
    train_features = wst_features[train_cosmo_idx][:, :, top_100_idx].reshape(-1, 100)
    f_mean = np.mean(train_features, axis=0)
    f_std = np.std(train_features, axis=0)
    f_std[f_std == 0] = 1e-6
    gmm = joblib.load(os.path.join(OUTPUT_DIR, "gmm_model.joblib"))
    mlp = torch.load(os.path.join(OUTPUT_DIR, "mlp_regressor.pth"), map_location='cpu')
    mlp.eval()
    if torch.cuda.is_available():
        mlp = mlp.cuda()
    val_features = wst_features[val_cosmo_idx][:, :, top_100_idx].reshape(-1, 100)
    scattering = Scattering2D(J=4, shape=mask.shape, L=8)
    if torch.cuda.is_available():
        scattering = scattering.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    n_ood_cosmo = min(5, len(val_cosmo_idx))
    ood_cosmo_idx = val_cosmo_idx[:n_ood_cosmo]
    n_sys = kappa.shape[1]
    batch_size = 64
    noise_sigma = 0.4 / (2 * 30 * 2.0**2)**0.5
    ood_features_list = []
    for i in ood_cosmo_idx:
        kappa_cosmo = kappa[i].astype(np.float32)
        kappa_ood = kappa_cosmo + 3.0 * (kappa_cosmo ** 2)
        kappa_noisy = add_noise_batch(kappa_ood, noise_sigma)
        maps_2d = reconstruct_2d_batch(kappa_noisy, mask)
        cosmo_features = []
        for b in range(0, n_sys, batch_size):
            batch_maps = torch.tensor(maps_2d[b:b+batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                wst = scattering(batch_maps).mean(dim=(2, 3))
            cosmo_features.append(wst.cpu().numpy())
        ood_features_list.append(np.concatenate(cosmo_features, axis=0))
    ood_features = np.concatenate(ood_features_list, axis=0)[:, top_100_idx]
    nll_ind = compute_nll(val_features, mlp, gmm, f_mean, f_std)
    nll_ood = compute_nll(ood_features, mlp, gmm, f_mean, f_std)
    mean_nll, std_nll = np.mean(nll_ind), np.std(nll_ind)
    calibrated_nll_ind = (nll_ind - mean_nll) / std_nll
    calibrated_nll_ood = (nll_ood - mean_nll) / std_nll
    labels = np.concatenate([np.zeros(len(calibrated_nll_ind)), np.ones(len(calibrated_nll_ood))])
    scores = np.concatenate([calibrated_nll_ind, calibrated_nll_ood])
    p_auc = score_phase2(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    np.save(os.path.join(OUTPUT_DIR, "nll_calibration.npy"), np.array([mean_nll, std_nll]))
    np.savez(os.path.join(OUTPUT_DIR, "validation_results.npz"), nll_ind=nll_ind, nll_ood=nll_ood, calibrated_nll_ind=calibrated_nll_ind, calibrated_nll_ood=calibrated_nll_ood, labels=labels, scores=scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = ' + str(round(roc_auc, 4)) + ')')
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_" + str(int(time.time())) + ".png"))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.hist(calibrated_nll_ind, bins=50, alpha=0.5, density=True, label='InD')
    plt.hist(calibrated_nll_ood, bins=50, alpha=0.5, density=True, label='OoD')
    plt.savefig(os.path.join(OUTPUT_DIR, "nll_distribution_" + str(int(time.time())) + ".png"))
    plt.close()