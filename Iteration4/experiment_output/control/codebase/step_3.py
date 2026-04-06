# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from kymatio.torch import Scattering2D

plt.rcParams['text.usetex'] = False

class MLPRegressor(nn.Module):
    def __init__(self, input_dim=100, output_dim=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2), 
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), 
            nn.Linear(128, 64), nn.ReLU()
        )
        self.mean_head = nn.Linear(64, output_dim)
        self.logvar_head = nn.Linear(64, output_dim)
        
    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

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

def compute_nll(features, mlp_model, gmm_model, f_mean, f_std, z_mean, z_std):
    features_scaled = (features - f_mean) / f_std
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    if torch.cuda.is_available():
        features_tensor = features_tensor.cuda()
    with torch.no_grad():
        out = mlp_model(features_tensor)
        theta_hat = out[0] if isinstance(out, tuple) else out[:, :5]
    theta_hat = theta_hat.cpu().numpy()
    Z = np.concatenate([features_scaled, theta_hat], axis=1)
    Z_norm = (Z - z_mean) / z_std
    return -gmm_model.score_samples(Z_norm)

if __name__ == '__main__':
    DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
    OUTPUT_DIR = "data"
    mask = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"), mmap_mode='r')
    val_idx_path = os.path.join(OUTPUT_DIR, "val_cosmo_idx.npy")
    val_cosmo_idx = np.load(val_idx_path)
    top_100_idx = np.load(os.path.join(OUTPUT_DIR, "top_100_wst_indices.npy"))
    wst_features = np.load(os.path.join(OUTPUT_DIR, "wst_features.npy"))
    feature_scaler = np.load(os.path.join(OUTPUT_DIR, "feature_scaler.npy"))
    f_mean, f_std = feature_scaler[0], feature_scaler[1]
    z_scaler = np.load(os.path.join(OUTPUT_DIR, "z_scaler.npy"))
    z_mean, z_std = z_scaler[0], z_scaler[1]
    gmm = joblib.load(os.path.join(OUTPUT_DIR, "gmm_model.joblib"))
    mlp_state_dict = torch.load(os.path.join(OUTPUT_DIR, "mlp_regressor.pth"), map_location='cpu')
    mlp = MLPRegressor(input_dim=100, output_dim=5)
    mlp.load_state_dict(mlp_state_dict)
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
    nll_ind = compute_nll(val_features, mlp, gmm, f_mean, f_std, z_mean, z_std)
    nll_ood = compute_nll(ood_features, mlp, gmm, f_mean, f_std, z_mean, z_std)
    mean_nll = np.mean(nll_ind)
    std_nll = np.std(nll_ind)
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Classifier (AUC = ' + str(round(roc_auc, 4)) + ')')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: In-Distribution vs. Synthetic OoD')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path_roc = os.path.join(OUTPUT_DIR, "roc_curve_1_" + str(int(time.time())) + ".png")
    plt.savefig(plot_path_roc, dpi=300)
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.hist(calibrated_nll_ind, bins=50, alpha=0.5, density=True, label='In-Distribution')
    plt.hist(calibrated_nll_ood, bins=50, alpha=0.5, density=True, label='Synthetic OoD')
    plt.xlabel('Calibrated OoD Score (Z-score)')
    plt.ylabel('Density')
    plt.title('Distribution of Calibrated OoD Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path_dist = os.path.join(OUTPUT_DIR, "nll_distribution_2_" + str(int(time.time())) + ".png")
    plt.savefig(plot_path_dist, dpi=300)
    plt.close()
    print("="*50)
    print("Validation Results")
    print("="*50)
    print("InD NLL - Mean: " + str(round(mean_nll, 4)) + ", Std: " + str(round(std_nll, 4)))
    print("Synthetic OoD NLL - Mean: " + str(round(float(np.mean(nll_ood)), 4)) + ", Std: " + str(round(float(np.std(nll_ood)), 4)))
    print("Partial AUC (FPR 0.1% - 5%): " + str(round(p_auc, 4)))
    print("ROC AUC: " + str(round(roc_auc, 4)))
    print("="*50)