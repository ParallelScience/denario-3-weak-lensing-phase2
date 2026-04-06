# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
from kymatio.torch import Scattering2D
import os
import time

DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
OUTPUT_DIR = "data/"

def add_noise(flat_map, mask, ng=30, pixel_size=2.0):
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5
    return flat_map + np.random.randn(*flat_map.shape) * noise_sigma * mask[mask]

def to_2d(flat_map, mask):
    full = np.zeros(mask.shape, dtype=np.float32)
    full[mask] = flat_map.astype(np.float32)
    return full

if __name__ == '__main__':
    print("Loading data...")
    mask = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"), mmap_mode='r')
    label = np.load(os.path.join(DATA_DIR, "label.npy"))
    
    n_cosmo, n_sys, n_pixels = kappa.shape
    n_realizations = 3
    
    J = 4
    L = 8
    shape = mask.shape
    scattering = Scattering2D(J=J, L=L, shape=shape)
    if torch.cuda.is_available():
        scattering = scattering.cuda()
        
    batch_size = 64
    total_augmented = n_cosmo * n_sys * n_realizations
    
    dummy_input = torch.zeros((1, *shape), dtype=torch.float32)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    dummy_output = scattering(dummy_input)
    n_features = dummy_output.shape[1]
    
    features = np.zeros((total_augmented, n_features), dtype=np.float32)
    aug_labels = np.zeros((total_augmented, 5), dtype=np.float32)
    
    print("Starting WST feature extraction...")
    start_time = time.time()
    
    idx = 0
    batch_maps = []
    batch_labels = []
    
    for c in range(n_cosmo):
        for s in range(n_sys):
            flat_map = kappa[c, s]
            lbl = label[c, s]
            for r in range(n_realizations):
                noisy_flat = add_noise(flat_map, mask)
                map_2d = to_2d(noisy_flat, mask)
                batch_maps.append(map_2d)
                batch_labels.append(lbl)
                
                if len(batch_maps) == batch_size:
                    batch_tensor = torch.tensor(np.stack(batch_maps), dtype=torch.float32)
                    if torch.cuda.is_available():
                        batch_tensor = batch_tensor.cuda()
                    with torch.no_grad():
                        wst_out = scattering(batch_tensor)
                        wst_features = wst_out.mean(dim=(2, 3)).cpu().numpy()
                    features[idx:idx+batch_size] = wst_features
                    aug_labels[idx:idx+batch_size] = np.array(batch_labels)
                    idx += batch_size
                    batch_maps = []
                    batch_labels = []
        if (c + 1) % 10 == 0:
            print("Processed " + str(c + 1) + " cosmologies.")
            
    if len(batch_maps) > 0:
        batch_tensor = torch.tensor(np.stack(batch_maps), dtype=torch.float32)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        with torch.no_grad():
            wst_out = scattering(batch_tensor)
            wst_features = wst_out.mean(dim=(2, 3)).cpu().numpy()
        features[idx:idx+len(batch_maps)] = wst_features
        aug_labels[idx:idx+len(batch_maps)] = np.array(batch_labels)
        
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    features_scaled = (features - mean) / std
    
    np.save(os.path.join(OUTPUT_DIR, "wst_features_scaled.npy"), features_scaled)
    np.save(os.path.join(OUTPUT_DIR, "wst_labels.npy"), aug_labels)
    np.save(os.path.join(OUTPUT_DIR, "wst_scaler.npy"), np.vstack((mean, std)))
    print("Saved to data/")