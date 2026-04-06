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
import torch.optim as optim
from kymatio.torch import Scattering2D
import joblib
import zuko

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

def to_2d_batch(flat_maps, mask):
    batch_size = flat_maps.shape[0]
    full = np.zeros((batch_size, mask.shape[0], mask.shape[1]), dtype=np.float32)
    full[:, mask] = flat_maps
    return full

def save_submission(ood_scores, errorbars=None, save_dir="/home/node/work/weak_lensing_phase2"):
    if errorbars is None:
        errorbars = [0.0] * len(ood_scores)
    data = {"means": list(map(float, ood_scores)), "errorbars": list(map(float, errorbars))}
    json_path = os.path.join(save_dir, "submission.json")
    zip_path = os.path.join(save_dir, "submission.zip")
    with open(json_path, 'w') as f:
        json.dump(data, f)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(json_path, "submission.json")
    return zip_path

if __name__ == '__main__':
    start_time = time.time()
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    OUTPUT_DIR = 'data'
    SAVE_DIR = '/home/node/work/weak_lensing_phase2'
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    kappa_test = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy'), mmap_mode='r')
    num_test = kappa_test.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    J = 4
    L = 8
    scattering = Scattering2D(J=J, shape=mask.shape, L=L).to(device)
    pca = joblib.load(os.path.join(OUTPUT_DIR, 'pca_model.joblib'))
    inv_sqrt_cov = np.load(os.path.join(OUTPUT_DIR, 'inv_sqrt_cov.npy'))
    input_dim = inv_sqrt_cov.shape[0]
    output_dim = 5
    mlp = MLPRegressor(input_dim, output_dim).to(device)
    mlp.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'mlp_regressor.pth')))
    mlp.eval()
    flow = zuko.flows.NSF(features=input_dim, context=output_dim, transforms=5, hidden_features=[128, 128]).to(device)
    flow.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'nsf_flow.pth')))
    flow.eval()
    batch_size = 200
    all_scores = []
    all_errorbars = []
    try:
        for i in range(0, num_test, batch_size):
            end = min(i + batch_size, num_test)
            flat_batch = kappa_test[i:end].astype(np.float32)
            maps_2d = to_2d_batch(flat_batch, mask)
            tensor_maps = torch.tensor(maps_2d, device=device)
            wst = scattering(tensor_maps).mean(dim=(-2, -1)).cpu().numpy()
            wst_pca = pca.transform(wst)
            x_features = wst_pca @ inv_sqrt_cov
            x_tensor = torch.tensor(x_features, dtype=torch.float32, device=device)
            with torch.no_grad():
                mean_theta, log_var_theta = mlp(x_tensor)
            theta = mean_theta.clone().detach().requires_grad_(True)
            optimizer = optim.Adam([theta], lr=0.05)
            best_nll = torch.full((end - i,), float('inf'), device=device)
            with torch.no_grad():
                initial_nll = -flow(theta).log_prob(x_tensor)
                best_nll = torch.minimum(best_nll, initial_nll)
            for step in range(10):
                optimizer.zero_grad()
                nll = -flow(theta).log_prob(x_tensor)
                loss = nll.mean()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    current_nll = -flow(theta).log_prob(x_tensor)
                    improved = current_nll < best_nll
                    best_nll[improved] = current_nll[improved]
            all_scores.extend(best_nll.cpu().numpy().tolist())
            uncertainty = torch.exp(log_var_theta).mean(dim=-1).cpu().numpy().tolist()
            all_errorbars.extend(uncertainty)
    except Exception as e:
        remaining = num_test - len(all_scores)
        all_scores.extend([0.0] * remaining)
        all_errorbars.extend([0.0] * remaining)
    all_scores_np = np.array(all_scores)
    all_errorbars_np = np.array(all_errorbars)
    if np.isnan(all_scores_np).any() or np.isinf(all_scores_np).any():
        finite_mask = np.isfinite(all_scores_np)
        max_val = np.nanmax(all_scores_np[finite_mask]) if finite_mask.any() else 0.0
        all_scores_np[~finite_mask] = max_val
    if np.isnan(all_errorbars_np).any() or np.isinf(all_errorbars_np).any():
        all_errorbars_np[~np.isfinite(all_errorbars_np)] = 0.0
    zip_path = save_submission(all_scores_np.tolist(), all_errorbars_np.tolist(), save_dir=SAVE_DIR)
    np.save(os.path.join(OUTPUT_DIR, 'test_ood_scores.npy'), all_scores_np)
    np.save(os.path.join(OUTPUT_DIR, 'test_errorbars.npy'), all_errorbars_np)