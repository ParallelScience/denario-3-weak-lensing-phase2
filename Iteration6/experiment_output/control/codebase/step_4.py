# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from kymatio.torch import Scattering2D
import joblib
import zuko
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams['text.usetex'] = False

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim * 2)
        )
    def forward(self, x):
        out = self.net(x)
        mean, log_var = torch.chunk(out, 2, dim=-1)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mean, log_var

def add_noise_batch(flat_maps, mask, ng=30, pixel_size=2.0):
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5
    flat_maps_f32 = flat_maps.astype(np.float32)
    noise = np.random.randn(*flat_maps.shape).astype(np.float32) * noise_sigma
    return flat_maps_f32 + noise

def to_2d_batch(flat_maps, mask):
    batch_size = flat_maps.shape[0]
    full = np.zeros((batch_size, mask.shape[0], mask.shape[1]), dtype=np.float32)
    full[:, mask] = flat_maps
    return full

def blur_batch(maps_2d, sigma=1.5):
    blurred = np.zeros_like(maps_2d)
    for i in range(maps_2d.shape[0]):
        blurred[i] = gaussian_filter(maps_2d[i], sigma=sigma)
    return blurred

def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)

if __name__ == '__main__':
    print('Starting Step 4: Validation Evaluation')
    start_time = time.time()
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    OUTPUT_DIR = 'data'
    print('Loading data and models...')
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    kappa = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy'), mmap_mode='r')
    label = np.load(os.path.join(DATA_DIR, 'label.npy'))
    num_cosmo, num_sys, num_pixels = kappa.shape
    total_samples = num_cosmo * num_sys
    kappa_flat = kappa.reshape(total_samples, num_pixels)
    label_flat = label.reshape(total_samples, 5)
    val_start = 80 * 256
    val_end = total_samples
    val_flat = kappa_flat[val_start:val_end]
    val_labels = label_flat[val_start:val_end]
    num_val = val_end - val_start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    J = 4
    L = 8
    scattering = Scattering2D(J=J, shape=mask.shape, L=L).to(device)
    pca = joblib.load(os.path.join(OUTPUT_DIR, 'pca_model.joblib'))
    inv_sqrt_cov = np.load(os.path.join(OUTPUT_DIR, 'inv_sqrt_cov.npy'))
    batch_size = 128
    wst_ind = []
    wst_ood = []
    print('Computing WST features for ' + str(num_val) + ' validation samples...')
    for i in range(0, num_val, batch_size):
        end = min(i + batch_size, num_val)
        flat_batch = val_flat[i:end]
        noisy_ind_flat = add_noise_batch(flat_batch, mask)
        maps_ind_2d = to_2d_batch(noisy_ind_flat, mask)
        tensor_ind = torch.tensor(maps_ind_2d, device=device)
        wst_i = scattering(tensor_ind).mean(dim=(-2, -1)).cpu().numpy()
        wst_ind.append(wst_i)
        clean_2d = to_2d_batch(flat_batch, mask)
        blurred_2d = blur_batch(clean_2d, sigma=1.5)
        blurred_flat = blurred_2d[:, mask]
        noisy_ood_flat = add_noise_batch(blurred_flat, mask)
        maps_ood_2d = to_2d_batch(noisy_ood_flat, mask)
        tensor_ood = torch.tensor(maps_ood_2d, device=device)
        wst_o = scattering(tensor_ood).mean(dim=(-2, -1)).cpu().numpy()
        wst_ood.append(wst_o)
        if (i // batch_size) % 10 == 0:
            print('Processed ' + str(end) + '/' + str(num_val) + ' validation samples...')
    wst_ind = np.concatenate(wst_ind, axis=0)
    wst_ood = np.concatenate(wst_ood, axis=0)
    wst_ind_pca = pca.transform(wst_ind)
    x_ind = wst_ind_pca @ inv_sqrt_cov
    wst_ood_pca = pca.transform(wst_ood)
    x_ood = wst_ood_pca @ inv_sqrt_cov
    input_dim = x_ind.shape[1]
    output_dim = 5
    mlp = MLPRegressor(input_dim, output_dim).to(device)
    mlp.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'mlp_regressor.pth')))
    mlp.eval()
    flow = zuko.flows.NSF(features=input_dim, context=output_dim, transforms=5, hidden_features=[128, 128]).to(device)
    flow.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'nsf_flow.pth')))
    flow.eval()
    def get_scores(x_features):
        scores = []
        dataset = TensorDataset(torch.tensor(x_features, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(device)
                mean_theta, _ = mlp(x_batch)
                nll = -flow(mean_theta).log_prob(x_batch)
                scores.append(nll.cpu().numpy())
        return np.concatenate(scores, axis=0)
    scores_ind = get_scores(x_ind)
    scores_ood = get_scores(x_ood)
    np.save(os.path.join(OUTPUT_DIR, 'val_scores_ind.npy'), scores_ind)
    np.save(os.path.join(OUTPUT_DIR, 'val_scores_ood.npy'), scores_ood)
    labels = np.concatenate([np.zeros(num_val), np.ones(num_val)])
    all_scores = np.concatenate([scores_ind, scores_ood])
    if np.isnan(all_scores).any() or np.isinf(all_scores).any():
        max_val = np.nanmax(all_scores[np.isfinite(all_scores)])
        all_scores[np.isnan(all_scores) | np.isinf(all_scores)] = max_val
    auc = roc_auc_score(labels, all_scores)
    p_auc = score_phase2(labels, all_scores)
    print('\n--- Validation Results ---')
    print('Validation AUC: ' + str(round(auc, 4)))
    print('Validation Partial AUC: ' + str(round(p_auc, 4)))
    t_agn_val = val_labels[:, 2]
    high_t_agn_mask = t_agn_val > 8.3
    scores_high_t_agn = scores_ind[high_t_agn_mask]
    scores_normal_t_agn = scores_ind[~high_t_agn_mask]
    print('\n--- Robustness Analysis ---')
    print('Mean InD score (Validation): ' + str(round(np.mean(scores_normal_t_agn), 4)))
    print('Mean InD score (High T_AGN): ' + str(round(np.mean(scores_high_t_agn), 4)))
    print('Mean OoD score (Blurred): ' + str(round(np.mean(scores_ood), 4)))
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.hist(scores_normal_t_agn, bins=50, alpha=0.5, label='InD (Validation)', density=True)
    ax1.hist(scores_high_t_agn, bins=50, alpha=0.5, label='InD (High T_AGN)', density=True)
    ax1.hist(scores_ood, bins=50, alpha=0.5, label='OoD (Blurred)', density=True)
    ax1.set_xlabel('OoD Score (Negative Log-Likelihood)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Validation Score Distributions', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    fpr, tpr, _ = roc_curve(labels, all_scores)
    ax2.plot(fpr, tpr, label='Full AUC = ' + str(round(auc, 4)) + '\nPartial AUC = ' + str(round(p_auc, 4)), linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.axvspan(0.001, 0.05, color='gray', alpha=0.2, label='Eval Range (0.001-0.05)')
    ax2.set_xscale('log')
    ax2.set_xlim([1e-4, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'validation_results_2_' + timestamp + '.png')
    fig.savefig(plot_path, dpi=300)
    print('Validation plots saved to ' + plot_path)
    print('Step 4 completed in ' + str(round(time.time() - start_time, 2)) + ' seconds.')