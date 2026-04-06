# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import os
import time
from kymatio.torch import Scattering2D

DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
OUTPUT_DIR = "data"

def add_noise_batch(flat_maps, noise_sigma):
    noise = np.random.randn(*flat_maps.shape).astype(np.float32) * noise_sigma
    return flat_maps + noise

def reconstruct_2d_batch(flat_maps, mask):
    n_sys = flat_maps.shape[0]
    maps_2d = np.zeros((n_sys, *mask.shape), dtype=np.float32)
    maps_2d[:, mask] = flat_maps
    return maps_2d

if __name__ == '__main__':
    print("Starting Step 1: WST Feature Extraction and Selection")
    mask = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"), mmap_mode='r')
    n_cosmo, n_sys, n_pixels = kappa.shape
    J = 4
    L = 8
    shape = mask.shape
    scattering = Scattering2D(J=J, shape=shape, L=L)
    if torch.cuda.is_available():
        scattering = scattering.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dummy_input = torch.zeros((1, *shape), dtype=torch.float32).to(device)
    dummy_output = scattering(dummy_input)
    num_coeffs = dummy_output.shape[1]
    features = np.zeros((n_cosmo, n_sys, num_coeffs), dtype=np.float32)
    batch_size = 64
    noise_sigma = 0.4 / (2 * 30 * 2.0**2)**0.5
    start_time = time.time()
    for i in range(n_cosmo):
        kappa_cosmo = kappa[i].astype(np.float32)
        kappa_noisy = add_noise_batch(kappa_cosmo, noise_sigma)
        maps_2d = reconstruct_2d_batch(kappa_noisy, mask)
        for b in range(0, n_sys, batch_size):
            batch_maps = torch.tensor(maps_2d[b:b+batch_size], dtype=torch.float32).to(device)
            with torch.no_grad():
                wst = scattering(batch_maps)
                wst = wst.mean(dim=(2, 3))
            features[i, b:b+batch_size] = wst.cpu().numpy()
        if (i + 1) % 10 == 0 or (i + 1) == n_cosmo:
            elapsed = time.time() - start_time
            print("Processed " + str(i+1) + "/" + str(n_cosmo) + " cosmologies in " + str(round(elapsed, 1)) + "s")
    features_path = os.path.join(OUTPUT_DIR, "wst_features.npy")
    np.save(features_path, features)
    n_train_cosmo = n_cosmo // 2
    features_train = features[:n_train_cosmo]
    mean_per_cosmo = np.mean(features_train, axis=1)
    var_cosmo = np.var(mean_per_cosmo, axis=0)
    var_per_cosmo = np.var(features_train, axis=1)
    var_nuisance = np.mean(var_per_cosmo, axis=0)
    var_nuisance = np.clip(var_nuisance, a_min=1e-12, a_max=None)
    snr = var_cosmo / var_nuisance
    top_100_idx = np.argsort(snr)[-100:][::-1]
    top_100_idx_path = os.path.join(OUTPUT_DIR, "top_100_wst_indices.npy")
    np.save(top_100_idx_path, top_100_idx)
    print("Saved top 100 WST indices to " + top_100_idx_path)