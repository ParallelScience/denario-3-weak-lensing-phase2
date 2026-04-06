# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
from kymatio.torch import Scattering2D

def main():
    DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
    OUTPUT_DIR = "data"
    print("Loading data...")
    mask_path = os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy")
    kappa_path = os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_newrealization.npy")
    mask = np.load(mask_path)
    kappa = np.load(kappa_path, mmap_mode='r')
    n_cosmo, n_sys, n_pixels = kappa.shape
    total_maps = n_cosmo * n_sys
    np.random.seed(42)
    all_cosmo_indices = np.arange(n_cosmo)
    np.random.shuffle(all_cosmo_indices)
    val_cosmo_indices = all_cosmo_indices[:20]
    is_val = np.zeros(n_cosmo, dtype=bool)
    is_val[val_cosmo_indices] = True
    is_val_flat = np.repeat(is_val, n_sys)
    train_indices = np.where(~is_val_flat)[0]
    val_indices = np.where(is_val_flat)[0]
    print("Total maps: " + str(total_maps))
    print("Training maps: " + str(len(train_indices)))
    print("Validation maps: " + str(len(val_indices)))
    np.save(os.path.join(OUTPUT_DIR, "train_indices.npy"), train_indices)
    np.save(os.path.join(OUTPUT_DIR, "val_indices.npy"), val_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    J = 4
    L = 8
    shape = mask.shape
    scattering = Scattering2D(J=J, shape=shape, L=L).to(device)
    ng = 30
    pixel_size = 2.0
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5
    mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
    batch_size = 64
    n_realizations = 3
    all_wst_features = []
    print("Computing WST features...")
    start_time = time.time()
    for i in range(0, total_maps, batch_size):
        end_idx = min(i + batch_size, total_maps)
        batch_cosmo = np.arange(i, end_idx) // n_sys
        batch_sys = np.arange(i, end_idx) % n_sys
        batch_flat = kappa[batch_cosmo, batch_sys].astype(np.float32)
        batch_tensor = torch.tensor(batch_flat, device=device)
        B = batch_tensor.shape[0]
        batch_tensor_rep = batch_tensor.repeat(n_realizations, 1)
        noise = torch.randn_like(batch_tensor_rep) * noise_sigma
        noisy_batch_flat = batch_tensor_rep + noise
        full_maps = torch.zeros((n_realizations * B, shape[0], shape[1]), device=device, dtype=torch.float32)
        full_maps[:, mask_tensor] = noisy_batch_flat
        with torch.no_grad():
            wst = scattering(full_maps)
            wst_avg = wst.mean(dim=(-2, -1))
            wst_avg = wst_avg.view(n_realizations, B, -1).mean(dim=0)
        all_wst_features.append(wst_avg.cpu().numpy())
        if (i // batch_size) % 40 == 0:
            print("Processed " + str(end_idx) + "/" + str(total_maps) + " maps...")
    all_wst_features = np.concatenate(all_wst_features, axis=0)
    print("WST computation completed in " + str(round(time.time() - start_time, 2)) + " seconds.")
    print("Raw WST features shape: " + str(all_wst_features.shape))
    train_wst = all_wst_features[train_indices]
    wst_mean = np.mean(train_wst, axis=0)
    wst_std = np.std(train_wst, axis=0)
    wst_std[wst_std == 0] = 1e-8
    normalized_wst = (all_wst_features - wst_mean) / wst_std
    np.save(os.path.join(OUTPUT_DIR, "wst_features.npy"), normalized_wst)
    np.save(os.path.join(OUTPUT_DIR, "wst_mean.npy"), wst_mean)
    np.save(os.path.join(OUTPUT_DIR, "wst_std.npy"), wst_std)
    print("Normalized WST features saved to " + os.path.join(OUTPUT_DIR, 'wst_features.npy'))
    print("Normalization statistics saved.")
    print("Output feature dimensionality: " + str(normalized_wst.shape[1]))

if __name__ == '__main__':
    main()