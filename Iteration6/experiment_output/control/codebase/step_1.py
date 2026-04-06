# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
from kymatio.torch import Scattering2D
from sklearn.decomposition import PCA
import joblib

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

if __name__ == '__main__':
    print('Starting Step 1: WST Feature Extraction and Whitening')
    start_time = time.time()
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    OUTPUT_DIR = 'data'
    print('Loading data...')
    mask = np.load(DATA_DIR + '/WIDE12H_bin2_2arcmin_mask.npy')
    kappa = np.load(DATA_DIR + '/WIDE12H_bin2_2arcmin_kappa_newrealization.npy', mmap_mode='r')
    label = np.load(DATA_DIR + '/label.npy')
    num_cosmo, num_sys, num_pixels = kappa.shape
    total_samples = num_cosmo * num_sys
    kappa_flat = kappa.reshape(total_samples, num_pixels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    J = 4
    L = 8
    scattering = Scattering2D(J=J, shape=mask.shape, L=L).to(device)
    dummy_input = torch.zeros((1, mask.shape[0], mask.shape[1]), device=device)
    dummy_output = scattering(dummy_input)
    num_wst_coeffs = dummy_output.shape[1]
    print('Number of WST coefficients per map: ' + str(num_wst_coeffs))
    wst_features = np.zeros((total_samples, num_wst_coeffs), dtype=np.float32)
    batch_size = 128
    print('Computing WST features in batches of ' + str(batch_size) + '...')
    for i in range(0, total_samples, batch_size):
        end = min(i + batch_size, total_samples)
        flat_maps = kappa_flat[i:end]
        wst_avg = torch.zeros((end - i, num_wst_coeffs), device=device)
        for _ in range(3):
            noisy_flat = add_noise_batch(flat_maps, mask)
            maps_2d = to_2d_batch(noisy_flat, mask)
            maps_tensor = torch.tensor(maps_2d, device=device)
            wst = scattering(maps_tensor)
            wst_mean = wst.mean(dim=(-2, -1))
            wst_avg += wst_mean
        wst_avg /= 3.0
        wst_features[i:end] = wst_avg.cpu().numpy()
        if (i // batch_size) % 20 == 0:
            print('Processed ' + str(end) + '/' + str(total_samples) + ' samples...')
    print('WST computation completed in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    print('Applying PCA to retain 95% variance...')
    pca = PCA(n_components=0.95, svd_solver='full')
    wst_pca = pca.fit_transform(wst_features)
    num_pca_components = wst_pca.shape[1]
    print('Original WST features shape: ' + str(wst_features.shape))
    print('PCA retained components (95% variance): ' + str(num_pca_components))
    print('Explained variance ratio sum: ' + str(round(np.sum(pca.explained_variance_ratio_), 4)))
    print('First 5 PCA explained variance ratios: ' + str(pca.explained_variance_ratio_[:5]))
    print('Computing intra-cosmology covariance for whitening...')
    wst_pca_reshaped = wst_pca.reshape(num_cosmo, num_sys, num_pca_components)
    mean_per_cosmo = wst_pca_reshaped.mean(axis=1, keepdims=True)
    centered_intra = wst_pca_reshaped - mean_per_cosmo
    centered_intra_flat = centered_intra.reshape(-1, num_pca_components)
    cov_nuisance = np.cov(centered_intra_flat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_nuisance)
    eigvals = np.maximum(eigvals, 0)
    print('Intra-cosmology covariance matrix shape: ' + str(cov_nuisance.shape))
    print('Top 5 eigenvalues of nuisance covariance: ' + str(eigvals[-5:][::-1]))
    print('Bottom 5 eigenvalues of nuisance covariance: ' + str(eigvals[:5]))
    epsilon = 1e-6
    inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + epsilon)) @ eigvecs.T
    print('Applying whitening transformation...')
    wst_whitened = wst_pca @ inv_sqrt_cov
    wst_whitened_reshaped = wst_whitened.reshape(num_cosmo, num_sys, num_pca_components)
    mean_whitened_per_cosmo = wst_whitened_reshaped.mean(axis=1, keepdims=True)
    centered_whitened = wst_whitened_reshaped - mean_whitened_per_cosmo
    centered_whitened_flat = centered_whitened.reshape(-1, num_pca_components)
    cov_whitened = np.cov(centered_whitened_flat, rowvar=False)
    print('Verification - Whitened intra-cosmology covariance max deviation from identity: ' + str(np.max(np.abs(cov_whitened - np.eye(num_pca_components)))))
    print('Saving processed features and transformation parameters...')
    np.save(os.path.join(OUTPUT_DIR, 'wst_whitened_features.npy'), wst_whitened)
    np.save(os.path.join(OUTPUT_DIR, 'inv_sqrt_cov.npy'), inv_sqrt_cov)
    joblib.dump(pca, os.path.join(OUTPUT_DIR, 'pca_model.joblib'))
    label_flat = label.reshape(total_samples, 5)
    np.save(os.path.join(OUTPUT_DIR, 'label_flat.npy'), label_flat)
    print('Step 1 completed successfully in ' + str(round(time.time() - start_time, 2)) + ' seconds.')
    print('Final whitened features shape: ' + str(wst_whitened.shape))