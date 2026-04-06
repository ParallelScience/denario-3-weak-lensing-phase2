# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
import json
import zipfile
import scipy.ndimage
from sklearn.metrics import roc_curve
from kymatio.torch import Scattering2D
import zuko

def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)

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

def main():
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    OUTPUT_DIR = 'data'
    SAVE_DIR = '/home/node/work/weak_lensing_phase2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wst_mean = np.load(os.path.join(OUTPUT_DIR, 'wst_mean.npy'))
    wst_std = np.load(os.path.join(OUTPUT_DIR, 'wst_std.npy'))
    label_mean = np.load(os.path.join(OUTPUT_DIR, 'label_mean.npy'))
    label_std = np.load(os.path.join(OUTPUT_DIR, 'label_std.npy'))
    features_dim = len(wst_mean)
    context_dim = 5
    flow = zuko.flows.MAF(features=features_dim, context=context_dim, transforms=8, hidden_features=[256, 256])
    flow.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_maf_model.pth'), map_location=device, weights_only=True))
    flow = flow.to(device)
    flow.eval()
    kappa_test = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy'), mmap_mode='r')
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
    J = 4
    L = 8
    shape = mask.shape
    scattering = Scattering2D(J=J, shape=shape, L=L).to(device)
    batch_size = 64
    n_test = kappa_test.shape[0]
    test_wst_features = []
    for i in range(0, n_test, batch_size):
        end_idx = min(i + batch_size, n_test)
        batch_flat = kappa_test[i:end_idx].astype(np.float32)
        batch_tensor = torch.tensor(batch_flat, device=device)
        B = batch_tensor.shape[0]
        full_maps = torch.zeros((B, shape[0], shape[1]), device=device, dtype=torch.float32)
        full_maps[:, mask_tensor] = batch_tensor
        with torch.no_grad():
            wst = scattering(full_maps)
            wst_avg = wst.mean(dim=(-2, -1))
        test_wst_features.append(wst_avg.cpu().numpy())
    test_wst_features = np.concatenate(test_wst_features, axis=0)
    normalized_test_wst = (test_wst_features - wst_mean) / wst_std
    train_labels = np.load(os.path.join(DATA_DIR, 'label.npy')).reshape(-1, 5)
    normalized_train_labels = (train_labels - label_mean) / label_std
    n_passes = 5
    test_nlls = np.zeros((n_test, n_passes))
    wst_tensor = torch.tensor(normalized_test_wst, dtype=torch.float32, device=device)
    for p in range(n_passes):
        idx = np.random.randint(0, len(normalized_train_labels), size=n_test)
        cond = torch.tensor(normalized_train_labels[idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            nlls = []
            for i in range(0, n_test, 1000):
                batch_wst = wst_tensor[i:i+1000]
                batch_cond = cond[i:i+1000]
                nll = -flow(batch_cond).log_prob(batch_wst)
                nlls.append(nll.cpu().numpy())
            test_nlls[:, p] = np.concatenate(nlls)
    mean_nll = test_nlls.mean(axis=1)
    std_nll = test_nlls.std(axis=1)
    save_submission(mean_nll, std_nll, save_dir=SAVE_DIR)
    val_indices = np.load(os.path.join(OUTPUT_DIR, 'val_indices.npy'))
    kappa = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy'), mmap_mode='r')
    n_sys = kappa.shape[1]
    n_val_eval = min(1000, len(val_indices))
    eval_indices = val_indices[:n_val_eval]
    val_cosmo_eval = eval_indices // n_sys
    val_sys_eval = eval_indices % n_sys
    val_maps_noiseless_eval = kappa[val_cosmo_eval, val_sys_eval].astype(np.float32)
    ng = 30
    pixel_size = 2.0
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5
    ood_maps_noiseless = np.zeros_like(val_maps_noiseless_eval)
    for i in range(n_val_eval):
        full = np.zeros(mask.shape, dtype=np.float32)
        full[mask] = val_maps_noiseless_eval[i]
        blurred = scipy.ndimage.gaussian_filter(full, sigma=1.0)
        ood_maps_noiseless[i] = blurred[mask]
    val_maps_ood = ood_maps_noiseless + np.random.randn(*ood_maps_noiseless.shape) * noise_sigma
    ood_wst_features = []
    for i in range(0, n_val_eval, batch_size):
        end_idx = min(i + batch_size, n_val_eval)
        batch_flat = val_maps_ood[i:end_idx]
        batch_tensor = torch.tensor(batch_flat, device=device, dtype=torch.float32)
        B = batch_tensor.shape[0]
        full_maps = torch.zeros((B, shape[0], shape[1]), device=device, dtype=torch.float32)
        full_maps[:, mask_tensor] = batch_tensor
        with torch.no_grad():
            wst = scattering(full_maps)
            wst_avg = wst.mean(dim=(-2, -1))
        ood_wst_features.append(wst_avg.cpu().numpy())
    ood_wst_features = np.concatenate(ood_wst_features, axis=0)
    normalized_ood_wst = (ood_wst_features - wst_mean) / wst_std
    wst_features = np.load(os.path.join(OUTPUT_DIR, 'wst_features.npy'))
    val_wst_ind_eval = wst_features[eval_indices]
    val_cond_eval = train_labels[eval_indices]
    normalized_val_cond_eval = (val_cond_eval - label_mean) / label_std
    ind_nlls = []
    with torch.no_grad():
        for i in range(0, n_val_eval, 1000):
            end_idx = min(i + 1000, n_val_eval)
            batch_wst = torch.tensor(val_wst_ind_eval[i:end_idx], dtype=torch.float32, device=device)
            batch_cond = torch.tensor(normalized_val_cond_eval[i:end_idx], dtype=torch.float32, device=device)
            nll = -flow(batch_cond).log_prob(batch_wst)
            ind_nlls.append(nll.cpu().numpy())
    ind_nlls = np.concatenate(ind_nlls)
    ood_nlls = []
    with torch.no_grad():
        for i in range(0, n_val_eval, 1000):
            end_idx = min(i + 1000, n_val_eval)
            batch_wst = torch.tensor(normalized_ood_wst[i:end_idx], dtype=torch.float32, device=device)
            batch_cond = torch.tensor(normalized_val_cond_eval[i:end_idx], dtype=torch.float32, device=device)
            nll = -flow(batch_cond).log_prob(batch_wst)
            ood_nlls.append(nll.cpu().numpy())
    ood_nlls = np.concatenate(ood_nlls)
    val_labels = np.concatenate([np.zeros(n_val_eval), np.ones(n_val_eval)])
    val_scores = np.concatenate([ind_nlls, ood_nlls])
    val_score = score_phase2(val_labels, val_scores)
    print('Validation Score (Partial AUC): ' + str(round(val_score, 4)))

if __name__ == '__main__':
    main()