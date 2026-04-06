# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import zuko
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import time

def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)

if __name__ == '__main__':
    data_dir = 'data/'
    Z = np.load(os.path.join(data_dir, 'Z_latent.npy'))
    y_labels = np.load(os.path.join(data_dir, 'y_labels.npy'))
    y_mean = np.load(os.path.join(data_dir, 'y_mean.npy'))
    y_std = np.load(os.path.join(data_dir, 'y_std.npy'))
    if np.isnan(Z).any() or np.isinf(Z).any():
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    y_scaled = (y_labels - y_mean) / y_std
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = Z.shape[1]
    context_dim = y_labels.shape[1]
    flow = zuko.flows.MAF(features=latent_dim, context=context_dim, transforms=5, hidden_features=[128, 128])
    flow.load_state_dict(torch.load(os.path.join(data_dir, 'cnf_model.pth'), map_location=device, weights_only=True))
    flow = flow.to(device)
    flow.eval()
    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(Z_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    all_nll = []
    with torch.no_grad():
        for batch_Z, batch_y in dataloader:
            batch_Z = batch_Z.to(device)
            batch_y = batch_y.to(device)
            nll = -flow(batch_y).log_prob(batch_Z)
            all_nll.append(nll.cpu().numpy())
    all_nll = np.concatenate(all_nll)
    t_agn = y_labels[:, 2]
    f_0 = y_labels[:, 3]
    t_agn_p5, t_agn_p95 = np.percentile(t_agn, [5, 95])
    f_0_p5, f_0_p95 = np.percentile(f_0, [5, 95])
    extreme_mask = (t_agn < t_agn_p5) | (t_agn > t_agn_p95) | (f_0 < f_0_p5) | (f_0 > f_0_p95)
    n_cosmo = 101
    n_sys = 256
    val_cosmo_start = int(n_cosmo * 0.8)
    val_start_idx = val_cosmo_start * n_sys
    train_nll = all_nll[:val_start_idx]
    train_extreme = extreme_mask[:val_start_idx]
    val_nll = all_nll[val_start_idx:]
    val_extreme = extreme_mask[val_start_idx:]
    try:
        p_auc_train = score_phase2(train_extreme.astype(int), train_nll)
        print('Diagnostic Partial AUC: ' + str(round(p_auc_train, 4)))
    except Exception as e:
        print('Could not compute diagnostic partial AUC: ' + str(e))
    try:
        p_auc_val = score_phase2(val_extreme.astype(int), val_nll)
        print('Validation Partial AUC: ' + str(round(p_auc_val, 4)))
    except Exception as e:
        print('Could not compute validation partial AUC: ' + str(e))
    train_normal_nll = train_nll[~train_extreme]
    nll_p5 = np.percentile(train_normal_nll, 5)
    nll_median = np.median(train_normal_nll)
    nll_p95 = np.percentile(train_normal_nll, 95)
    np.save(os.path.join(data_dir, 'calibration_params.npy'), np.array([nll_p5, nll_median, nll_p95]))
    plt.figure(figsize=(10, 6))
    plt.hist(train_normal_nll, bins=50, alpha=0.5, density=True, label='Normal Nuisance (Train)')
    plt.hist(train_nll[train_extreme], bins=50, alpha=0.5, density=True, label='Extreme Nuisance (Train)')
    plt.xlabel('Negative Log-Likelihood (NLL)')
    plt.ylabel('Density')
    plt.title('NLL Distribution: Normal vs Extreme Nuisance')
    plt.legend()
    plt.tight_layout()
    plot_filepath = os.path.join(data_dir, 'nll_distribution_' + str(int(time.time())) + '.png')
    plt.savefig(plot_filepath, dpi=300)
    print('Histogram saved to ' + plot_filepath)