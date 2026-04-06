# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
from kymatio.torch import Scattering2D
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def extract_features():
    start_time = time.time()
    DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
    OUTPUT_DIR = "data/"
    print("Loading data...")
    mask = np.load(DATA_DIR + "/WIDE12H_bin2_2arcmin_mask.npy")
    kappa = np.load(DATA_DIR + "/WIDE12H_bin2_2arcmin_kappa_newrealization.npy", mmap_mode='r')
    label = np.load(DATA_DIR + "/label.npy")
    N_cosmo, N_sys, N_pix = kappa.shape
    N_total = N_cosmo * N_sys
    ng = 30
    pixel_size = 2.0
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5
    print("Noise sigma: " + str(round(noise_sigma, 5)))
    J = 3
    L = 8
    shape = (mask.shape[0], mask.shape[1])
    print("Initializing Scattering2D with J=" + str(J) + ", L=" + str(L) + ", shape=" + str(shape) + "...")
    scattering = Scattering2D(J=J, shape=shape, L=L).cuda()
    wst_features = []
    labels_flat = []
    print("Computing WST features in batches...")
    for i in range(N_cosmo):
        flat_maps = kappa[i]
        noise = np.random.randn(*flat_maps.shape) * noise_sigma
        noisy_flat = flat_maps + noise
        full_maps = np.zeros((N_sys, shape[0], shape[1]), dtype=np.float32)
        full_maps[:, mask] = noisy_flat.astype(np.float32)
        x = torch.tensor(full_maps, dtype=torch.float32, device='cuda')
        with torch.no_grad():
            out = scattering(x)
            out_mean = out.mean(dim=(-1, -2))
        wst_features.append(out_mean.cpu().numpy())
        labels_flat.append(label[i])
        del x, out, out_mean
        torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or (i + 1) == N_cosmo:
            print("Processed " + str(i + 1) + "/" + str(N_cosmo) + " cosmologies...")
    wst_features = np.concatenate(wst_features, axis=0)
    labels_flat = np.concatenate(labels_flat, axis=0)
    print("Extracted WST features shape: " + str(wst_features.shape))
    print("Performing feature selection using mutual_info_regression...")
    np.random.seed(42)
    subsample_idx = np.random.choice(N_total, min(5000, N_total), replace=False)
    mi_omega = mutual_info_regression(wst_features[subsample_idx], labels_flat[subsample_idx, 0])
    mi_s8 = mutual_info_regression(wst_features[subsample_idx], labels_flat[subsample_idx, 1])
    mi_total = mi_omega + mi_s8
    print("MI with Omega_m - Min: " + str(round(mi_omega.min(), 4)) + ", Max: " + str(round(mi_omega.max(), 4)) + ", Mean: " + str(round(mi_omega.mean(), 4)))
    print("MI with S_8 - Min: " + str(round(mi_s8.min(), 4)) + ", Max: " + str(round(mi_s8.max(), 4)) + ", Mean: " + str(round(mi_s8.mean(), 4)))
    print("Total MI - Min: " + str(round(mi_total.min(), 4)) + ", Max: " + str(round(mi_total.max(), 4)) + ", Mean: " + str(round(mi_total.mean(), 4)))
    k_best = min(150, wst_features.shape[1])
    selected_indices = np.argsort(mi_total)[-k_best:]
    selected_indices = np.sort(selected_indices)
    print("Selected " + str(k_best) + " features out of " + str(wst_features.shape[1]) + ".")
    selected_features = wst_features[:, selected_indices]
    print("Applying StandardScaler and PCA...")
    scaler = StandardScaler()
    selected_features_scaled = scaler.fit_transform(selected_features)
    n_components = 75
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(selected_features_scaled)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print("PCA reduced features to " + str(n_components) + " components.")
    print("Total explained variance: " + str(round(explained_variance, 4)))
    print("Explained variance by first 5 components: " + str(pca.explained_variance_ratio_[:5]))
    print("Saving outputs...")
    np.save(OUTPUT_DIR + "wst_pca_features.npy", pca_features)
    np.save(OUTPUT_DIR + "labels_flat.npy", labels_flat)
    joblib.dump({'selected_indices': selected_indices, 'scaler': scaler, 'pca': pca, 'noise_sigma': noise_sigma}, OUTPUT_DIR + "feature_pipeline.pkl")
    print("Step 1 completed in " + str(round(time.time() - start_time, 2)) + " seconds.")

if __name__ == "__main__":
    extract_features()