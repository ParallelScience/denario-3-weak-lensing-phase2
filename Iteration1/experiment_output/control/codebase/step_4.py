# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
import torch.nn as nn
from kymatio.torch import Scattering2D
import joblib
import zuko
from sklearn.metrics import roc_curve
import json
import matplotlib.pyplot as plt
import datetime

class ProbabilisticRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

def apply_filter_batch(flat_maps, mask_tensor, F_tensor):
    batch_size = flat_maps.shape[0]
    full = torch.zeros((batch_size, mask_tensor.shape[0], mask_tensor.shape[1]), dtype=torch.float32, device='cuda')
    full[:, mask_tensor] = flat_maps
    xk = torch.fft.fft2(full)
    xk_filtered = xk * F_tensor
    full_filtered = torch.fft.ifft2(xk_filtered).real
    filtered_flat = full_filtered[:, mask_tensor]
    return filtered_flat

def compute_nll(features_scaled, regressor, cnf, device, num_samples=5):
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        mean, logvar = regressor(features_tensor)
        std = torch.exp(0.5 * logvar)
        nll_list = []
        for _ in range(num_samples):
            epsilon = torch.randn_like(std)
            theta_sample = mean + std * epsilon
            log_prob = cnf(theta_sample).log_prob(features_tensor)
            nll_list.append(-log_prob)
        nll_stack = torch.stack(nll_list, dim=0)
        avg_nll = nll_stack.mean(dim=0).cpu().numpy()
    return avg_nll

def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)

def main():
    start_time = time.time()
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    OUTPUT_DIR = 'data/'
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    kappa = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy'), mmap_mode='r')
    pipeline = joblib.load(os.path.join(OUTPUT_DIR, 'feature_pipeline.pkl'))
    selected_indices = pipeline['selected_indices']
    pca_scaler = pipeline['scaler']
    pca = pipeline['pca']
    noise_sigma = pipeline['noise_sigma']
    scalers = joblib.load(os.path.join(OUTPUT_DIR, 'regressor_scalers.pkl'))
    feature_scaler = scalers['feature_scaler']
    input_dim = pca.n_components_
    output_dim = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regressor = ProbabilisticRegressor(input_dim=input_dim, output_dim=output_dim, hidden_dim=256).to(device)
    regressor.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'regressor_model.pth'), weights_only=True))
    regressor.eval()
    cnf = zuko.flows.MAF(features=input_dim, context=output_dim, transforms=4, hidden_features=[128, 128]).to(device)
    cnf.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'cnf_model.pth'), weights_only=True))
    cnf.eval()
    J = 3
    L = 8
    shape = (mask.shape[0], mask.shape[1])
    scattering = Scattering2D(J=J, shape=shape, L=L).to(device)
    pixsize = 2.0 / 60.0 * np.pi / 180.0
    kx = np.fft.fftfreq(shape[0], d=pixsize).reshape(-1, 1)
    ky = np.fft.fftfreq(shape[1], d=pixsize).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)
    ell = 2 * np.pi * k
    ell_0 = 5000
    sigma_ell = 2000
    A = 0.5
    F = 1 - A * np.exp(-(ell - ell_0)**2 / (2 * sigma_ell**2))
    F_tensor = torch.tensor(F, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask, device=device)
    N_samples = 1000
    np.random.seed(42)
    N_cosmo, N_sys, _ = kappa.shape
    N_total = N_cosmo * N_sys
    flat_indices = np.random.choice(N_total, N_samples, replace=False)
    cosmo_indices = flat_indices // N_sys
    sys_indices = flat_indices % N_sys
    batch_size = 50
    ind_features_list = []
    ood_features_list = []
    for i in range(0, N_samples, batch_size):
        end = min(i + batch_size, N_samples)
        batch_cosmo = cosmo_indices[i:end]
        batch_sys = sys_indices[i:end]
        flat_maps_np = kappa[batch_cosmo, batch_sys].astype(np.float32)
        flat_maps = torch.tensor(flat_maps_np, device=device)
        ood_flat_maps = apply_filter_batch(flat_maps, mask_tensor, F_tensor)
        noise = torch.randn_like(flat_maps) * noise_sigma
        ind_noisy = flat_maps + noise
        ood_noisy = ood_flat_maps + noise
        ind_full = torch.zeros((end - i, shape[0], shape[1]), dtype=torch.float32, device=device)
        ind_full[:, mask_tensor] = ind_noisy
        ood_full = torch.zeros((end - i, shape[0], shape[1]), dtype=torch.float32, device=device)
        ood_full[:, mask_tensor] = ood_noisy
        with torch.no_grad():
            ind_wst = scattering(ind_full).mean(dim=(-1, -2))
            ood_wst = scattering(ood_full).mean(dim=(-1, -2))
        ind_features_list.append(ind_wst.cpu().numpy())
        ood_features_list.append(ood_wst.cpu().numpy())
    ind_features = np.concatenate(ind_features_list, axis=0)
    ood_features = np.concatenate(ood_features_list, axis=0)
    ind_selected = ind_features[:, selected_indices]
    ood_selected = ood_features[:, selected_indices]
    ind_pca = pca.transform(pca_scaler.transform(ind_selected))
    ood_pca = pca.transform(pca_scaler.transform(ood_selected))
    ind_scaled = feature_scaler.transform(ind_pca)
    ood_scaled = feature_scaler.transform(ood_pca)
    ind_nll = compute_nll(ind_scaled, regressor, cnf, device)
    ood_nll = compute_nll(ood_scaled, regressor, cnf, device)
    y_true = np.concatenate([np.zeros(N_samples), np.ones(N_samples)])
    y_scores = np.concatenate([ind_nll, ood_nll])
    val_score = score_phase2(y_true, y_scores)
    shift = float(np.mean(ind_nll))
    calibrated_ind_nll = ind_nll - shift
    calibrated_ood_nll = ood_nll - shift
    calibration_params = {'shift': shift}
    with open(os.path.join(OUTPUT_DIR, 'calibration.json'), 'w') as f:
        json.dump(calibration_params, f)
    plt.figure(figsize=(10, 6))
    plt.hist(calibrated_ind_nll, bins=50, alpha=0.6, label='InD (Calibrated)', density=True)
    plt.hist(calibrated_ood_nll, bins=50, alpha=0.6, label='Synthetic OoD (Calibrated)', density=True)
    plt.xlabel('Calibrated NLL Score')
    plt.ylabel('Density')
    plt.title('OoD Score Distributions (Val Score: ' + str(round(val_score, 4)) + ')')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(OUTPUT_DIR, 'ood_score_distributions_1_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)

if __name__ == '__main__':
    main()