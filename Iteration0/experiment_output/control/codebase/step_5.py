# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import zuko
import matplotlib.pyplot as plt
import json
import zipfile
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from step_4 import MLPRegressor

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
    data_dir = 'data/'
    test_features_norm = np.load(os.path.join(data_dir, 'wst_features_test_normalized.npy'))
    fallback = np.load(os.path.join(data_dir, 'cnf_fallback.npy'))[0]
    output_dim = 2 if fallback else 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp = MLPRegressor(input_dim=217, output_dim=output_dim).to(device)
    mlp.load_state_dict(torch.load(os.path.join(data_dir, 'mlp_weights.pth'), map_location=device))
    mlp.eval()
    flow = zuko.flows.MAF(features=217, context=output_dim, transforms=5, hidden_features=[256, 256]).to(device)
    flow.load_state_dict(torch.load(os.path.join(data_dir, 'cnf_weights.pth'), map_location=device))
    flow.eval()
    X_test = torch.tensor(test_features_norm, dtype=torch.float32)
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    nll_scores = []
    print('--- Inference and OoD Scoring ---')
    print('Evaluating NLL for 10,000 test maps...')
    with torch.no_grad():
        for (x_batch,) in test_loader:
            x_batch = x_batch.to(device)
            theta_hat = mlp(x_batch)
            log_probs = flow(theta_hat).log_prob(x_batch)
            nll = -log_probs
            nll_scores.extend(nll.cpu().numpy())
    nll_scores = np.array(nll_scores)
    if np.isnan(nll_scores).any() or np.isinf(nll_scores).any():
        print('Warning: NaNs or Infs found in NLL scores. Replacing with max finite value.')
        finite_vals = nll_scores[np.isfinite(nll_scores)]
        max_val = np.max(finite_vals) if len(finite_vals) > 0 else 0.0
        min_val = np.min(finite_vals) if len(finite_vals) > 0 else 0.0
        nll_scores = np.nan_to_num(nll_scores, nan=max_val, posinf=max_val, neginf=min_val)
    np.save(os.path.join(data_dir, 'test_nll_scores.npy'), nll_scores)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    p99 = np.percentile(nll_scores, 99)
    truncated_scores = nll_scores[nll_scores <= p99]
    outliers_count = len(nll_scores) - len(truncated_scores)
    plt.hist(truncated_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Main Distribution (up to 99th percentile)')
    plt.xlabel('Negative Log-Likelihood (OoD Score)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.text(0.5, 0.9, 'Excluded ' + str(outliers_count) + ' outliers > ' + str(round(p99, 1)), transform=plt.gca().transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    plt.subplot(1, 2, 2)
    min_val = max(1.0, np.min(nll_scores))
    max_val = np.max(nll_scores)
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
    plt.hist(nll_scores, bins=bins, alpha=0.7, color='red', edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Full Distribution (Log-Log Scale)')
    plt.xlabel('Negative Log-Likelihood (OoD Score)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = os.path.join(data_dir, 'nll_histogram_1_' + timestamp + '.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print('Histogram saved to ' + plot_filename)
    save_dir = '/home/node/work/weak_lensing_phase2'
    zip_path = save_submission(nll_scores, save_dir=save_dir)
    print('Submission saved to ' + zip_path)
    print('\nNLL Scores Statistics:')
    print('  Min: ' + str(np.min(nll_scores)))
    print('  Max: ' + str(np.max(nll_scores)))
    print('  Mean: ' + str(np.mean(nll_scores)))
    print('  Median: ' + str(np.median(nll_scores)))
    print('  Std: ' + str(np.std(nll_scores)))