# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import zuko
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from step_4 import MLPRegressor
try:
    import kymatio
except ImportError as e:
    if "sph_harm" in str(e):
        import subprocess
        subprocess.run("sed -i 's/from scipy.special import sph_harm, factorial/from scipy.special import sph_harm_y as sph_harm, factorial/' /opt/denario-venv/lib/python3.12/site-packages/kymatio/scattering3d/filter_bank.py", shell=True, check=True)
        import kymatio
    else:
        raise e
from kymatio.torch import Scattering2D
def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)
def to_2d_batch(flat_maps, mask):
    batch_size = flat_maps.shape[0]
    full = np.zeros((batch_size, mask.shape[0], mask.shape[1]), dtype=np.float32)
    full[:, mask] = flat_maps.astype(np.float32)
    return full
if __name__ == '__main__':
    data_dir = 'data/'
    DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
    val_indices = np.load(os.path.join(data_dir, 'val_indices.npy'))
    mask = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"), mmap_mode='r')
    num_cosmo, num_sys, num_pixels = kappa.shape
    kappa_flat = kappa.reshape(num_cosmo * num_sys, num_pixels)
    wst_mean = np.load(os.path.join(data_dir, 'wst_mean.npy'))
    wst_std = np.load(os.path.join(data_dir, 'wst_std.npy'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fallback = np.load(os.path.join(data_dir, 'cnf_fallback.npy'))[0]
    output_dim = 2 if fallback else 5
    mlp = MLPRegressor(input_dim=217, output_dim=output_dim).to(device)
    mlp.load_state_dict(torch.load(os.path.join(data_dir, 'mlp_weights.pth'), map_location=device))
    mlp.eval()
    flow = zuko.flows.MAF(features=217, context=output_dim, transforms=5, hidden_features=[256, 256]).to(device)
    flow.load_state_dict(torch.load(os.path.join(data_dir, 'cnf_weights.pth'), map_location=device))
    flow.eval()
    print('--- Validation and Performance Estimation ---')
    train_features_norm = np.load(os.path.join(data_dir, 'wst_features_train_normalized.npy'))
    val_features_norm = train_features_norm[val_indices]
    X_val = torch.tensor(val_features_norm, dtype=torch.float32)
    val_dataset = TensorDataset(X_val)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    ind_nll = []
    with torch.no_grad():
        for (x_batch,) in val_loader:
            x_batch = x_batch.to(device)
            theta_hat = mlp(x_batch)
            log_probs = flow(theta_hat).log_prob(x_batch)
            ind_nll.extend((-log_probs).cpu().numpy())
    ind_nll = np.array(ind_nll)
    scattering = Scattering2D(J=3, shape=mask.shape, L=8)
    if torch.cuda.is_available():
        scattering = scattering.cuda()
    noise_sigma = 0.4 / (2 * 30 * 2.0**2)**0.5
    ood_features = []
    batch_size = 64
    for i in range(0, len(val_indices), batch_size):
        batch_idx = val_indices[i:i+batch_size]
        batch_flat = kappa_flat[batch_idx]
        batch_flat_perturbed = batch_flat.astype(np.float32) * 1.3
        noise = np.random.randn(*batch_flat_perturbed.shape) * noise_sigma
        batch_noisy = batch_flat_perturbed + noise
        batch_2d = to_2d_batch(batch_noisy, mask)
        batch_tensor = torch.tensor(batch_2d, dtype=torch.float32)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        with torch.no_grad():
            s_out = scattering(batch_tensor)
            s_pooled = s_out.mean(dim=(-2, -1))
        ood_features.append(s_pooled.cpu().numpy())
    ood_features = np.concatenate(ood_features, axis=0)
    ood_features_norm = (ood_features - wst_mean) / wst_std
    X_ood = torch.tensor(ood_features_norm, dtype=torch.float32)
    ood_dataset = TensorDataset(X_ood)
    ood_loader = DataLoader(ood_dataset, batch_size=512, shuffle=False)
    ood_nll = []
    with torch.no_grad():
        for (x_batch,) in ood_loader:
            x_batch = x_batch.to(device)
            theta_hat = mlp(x_batch)
            log_probs = flow(theta_hat).log_prob(x_batch)
            ood_nll.extend((-log_probs).cpu().numpy())
    ood_nll = np.array(ood_nll)
    labels = np.concatenate([np.zeros_like(ind_nll), np.ones_like(ood_nll)])
    scores = np.concatenate([ind_nll, ood_nll])
    if np.isnan(scores).any() or np.isinf(scores).any():
        finite_vals = scores[np.isfinite(scores)]
        max_val = np.max(finite_vals) if len(finite_vals) > 0 else 0.0
        min_val = np.min(finite_vals) if len(finite_vals) > 0 else 0.0
        scores = np.nan_to_num(scores, nan=max_val, posinf=max_val, neginf=min_val)
    val_score = score_phase2(labels, scores)
    print('Validation Score (partial AUC):', val_score)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label='ROC curve')
    plt.axvspan(0.001, 0.05, color='red', alpha=0.2, label='Scoring Region [0.001, 0.05]')
    plt.xscale('log')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Validation ROC Curve (Synthetic OoD)')
    plt.legend(loc='lower right')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(data_dir, 'roc_curve_' + timestamp + '.png'))
    print('Saved plot to ' + os.path.join(data_dir, 'roc_curve_' + timestamp + '.png'))