# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import json
import zipfile
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
DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
OUTPUT_DIR = 'data'
SAVE_DIR = '/home/node/work/weak_lensing_phase2'
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=100, output_dim=5):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
        self.mean_head = nn.Linear(64, output_dim)
        self.logvar_head = nn.Linear(64, output_dim)
    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar
def reconstruct_2d_batch(flat_maps, mask):
    n_sys = flat_maps.shape[0]
    maps_2d = np.zeros((n_sys, *mask.shape), dtype=np.float32)
    maps_2d[:, mask] = flat_maps
    return maps_2d
def save_submission(ood_scores, errorbars=None, save_dir='/home/node/work/weak_lensing_phase2'):
    if errorbars is None:
        errorbars = [0.0] * len(ood_scores)
    data = {'means': list(map(float, ood_scores)), 'errorbars': list(map(float, errorbars))}
    json_path = os.path.join(save_dir, 'submission.json')
    zip_path = os.path.join(save_dir, 'submission.zip')
    with open(json_path, 'w') as f:
        json.dump(data, f)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(json_path, 'submission.json')
    return zip_path
if __name__ == '__main__':
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    kappa_test = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy'), mmap_mode='r')
    n_test = kappa_test.shape[0]
    top_100_idx = np.load(os.path.join(OUTPUT_DIR, 'top_100_wst_indices.npy'))
    feature_scaler = np.load(os.path.join(OUTPUT_DIR, 'feature_scaler.npy'))
    f_mean, f_std = feature_scaler[0], feature_scaler[1]
    z_scaler = np.load(os.path.join(OUTPUT_DIR, 'z_scaler.npy'))
    z_mean, z_std = z_scaler[0], z_scaler[1]
    nll_calib = np.load(os.path.join(OUTPUT_DIR, 'nll_calibration.npy'))
    calib_mean, calib_std = nll_calib[0], nll_calib[1]
    gmm = joblib.load(os.path.join(OUTPUT_DIR, 'gmm_model.joblib'))
    mlp = MLPRegressor(input_dim=100, output_dim=5)
    mlp.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'mlp_regressor.pth'), map_location='cpu'))
    mlp.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        mlp = mlp.cuda()
    scattering = Scattering2D(J=4, shape=mask.shape, L=8)
    if torch.cuda.is_available():
        scattering = scattering.cuda()
    batch_size = 64
    test_features_list = []
    start_time = time.time()
    for b in range(0, n_test, batch_size):
        batch_flat = kappa_test[b:b+batch_size].astype(np.float32)
        batch_2d = reconstruct_2d_batch(batch_flat, mask)
        batch_tensor = torch.tensor(batch_2d, dtype=torch.float32).to(device)
        with torch.no_grad():
            wst = scattering(batch_tensor).mean(dim=(2, 3))
        test_features_list.append(wst.cpu().numpy())
    test_features_full = np.concatenate(test_features_list, axis=0)
    test_features = test_features_full[:, top_100_idx]
    test_features_scaled = (test_features - f_mean) / f_std
    test_features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        mean_pred, logvar_pred = mlp(test_features_tensor)
    mean_pred = mean_pred.cpu().numpy()
    std_pred = np.exp(0.5 * logvar_pred.cpu().numpy())
    n_samples = 5
    nll_samples = np.zeros((n_test, n_samples))
    for s in range(n_samples):
        theta_sample = mean_pred + std_pred * np.random.randn(*mean_pred.shape)
        Z = np.concatenate([test_features_scaled, theta_sample], axis=1)
        Z_norm = (Z - z_mean) / z_std
        nll_samples[:, s] = -gmm.score_samples(Z_norm)
    mean_nll = np.mean(nll_samples, axis=1)
    std_nll = np.std(nll_samples, axis=1)
    calibrated_scores = (mean_nll - calib_mean) / calib_std
    errorbars = std_nll / calib_std + 1e-3
    zip_path = save_submission(calibrated_scores, errorbars, save_dir=SAVE_DIR)
    val_res = np.load(os.path.join(OUTPUT_DIR, 'validation_results.npz'))
    labels, scores = val_res['labels'], val_res['scores']
    calibrated_nll_ind, calibrated_nll_ood = val_res['calibrated_nll_ind'], val_res['calibrated_nll_ood']
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Classifier (AUC = ' + str(round(roc_auc, 4)) + ')')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: In-Distribution vs. Synthetic OoD')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path_roc = os.path.join(OUTPUT_DIR, 'roc_curve_1_' + str(int(time.time())) + '.png')
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
    plot_path_dist = os.path.join(OUTPUT_DIR, 'nll_distribution_2_' + str(int(time.time())) + '.png')
    plt.savefig(plot_path_dist, dpi=300)
    plt.close()
    wst_features = np.load(os.path.join(OUTPUT_DIR, 'wst_features.npy'))
    n_train_cosmo = wst_features.shape[0] // 2
    features_train = wst_features[:n_train_cosmo]
    mean_per_cosmo = np.mean(features_train, axis=1)
    var_cosmo = np.var(mean_per_cosmo, axis=0)
    var_per_cosmo = np.var(features_train, axis=1)
    var_nuisance = np.clip(np.mean(var_per_cosmo, axis=0), a_min=1e-12, a_max=None)
    snr = var_cosmo / var_nuisance
    plt.figure(figsize=(10, 6))
    plt.plot(snr, marker='o', linestyle='-', markersize=4, alpha=0.7, label='SNR')
    plt.axhline(y=1.0, color='r', linestyle='--', label='SNR = 1')
    plt.yscale('log')
    plt.xlabel('WST Coefficient Index')
    plt.ylabel('Signal-to-Noise Ratio (Cosmo Var / Nuisance Var)')
    plt.title('SNR Spectrum of WST Coefficients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path_snr = os.path.join(OUTPUT_DIR, 'snr_spectrum_3_' + str(int(time.time())) + '.png')
    plt.savefig(plot_path_snr, dpi=300)
    plt.close()