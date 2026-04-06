# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
from kymatio.torch import Scattering2D
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def add_noise(flat_map, mask, ng=30, pixel_size=2.0):
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5
    return flat_map + np.random.randn(*flat_map.shape) * noise_sigma * mask[mask]

def to_2d_batch(flat_maps, mask):
    batch_size = flat_maps.shape[0]
    full = np.zeros((batch_size, *mask.shape), dtype=np.float32)
    full[:, mask] = flat_maps.astype(np.float32)
    return full

if __name__ == '__main__':
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    OUTPUT_DIR = 'data/'
    print('Loading data...')
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    kappa = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_newrealization.npy'), mmap_mode='r')
    label = np.load(os.path.join(DATA_DIR, 'label.npy'))
    J = 4
    L = 8
    print('Initializing Scattering2D with J=' + str(J) + ', L=' + str(L) + '...')
    scattering = Scattering2D(J=J, shape=mask.shape, L=L)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scattering = scattering.to(device)
    n_cosmo, n_sys, n_pixels = kappa.shape
    total_maps = n_cosmo * n_sys
    dummy_map = torch.zeros((1, *mask.shape), dtype=torch.float32).to(device)
    dummy_wst = scattering(dummy_map)
    dummy_wst = dummy_wst.mean(dim=(-2, -1))
    n_coeffs = dummy_wst.shape[1]
    print('Number of WST coefficients per map (spatially averaged): ' + str(n_coeffs))
    X_wst = np.zeros((total_maps, n_coeffs), dtype=np.float32)
    y_labels = np.zeros((total_maps, 5), dtype=np.float32)
    print('Extracting WST features for ' + str(total_maps) + ' maps...')
    start_time = time.time()
    idx = 0
    for i in range(n_cosmo):
        t0 = time.time()
        flat_maps = kappa[i]
        noisy_flat_maps = np.zeros_like(flat_maps, dtype=np.float32)
        for j in range(n_sys):
            noisy_flat_maps[j] = add_noise(flat_maps[j], mask)
        maps_2d = to_2d_batch(noisy_flat_maps, mask)
        maps_tensor = torch.tensor(maps_2d, dtype=torch.float32).to(device)
        with torch.no_grad():
            wst_out = scattering(maps_tensor)
            wst_out = wst_out.mean(dim=(-2, -1))
        X_wst[idx:idx+n_sys] = wst_out.cpu().numpy()
        y_labels[idx:idx+n_sys] = label[i]
        idx += n_sys
        if (i + 1) % 20 == 0 or i == 0 or i == n_cosmo - 1:
            print('Processed cosmology ' + str(i+1) + '/' + str(n_cosmo) + ' in ' + str(round(time.time()-t0, 2)) + 's')
    print('WST extraction completed in ' + str(round(time.time()-start_time, 2)) + 's')
    print('Training Random Forest Regressor...')
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_wst, y_labels)
    y_pred = rf.predict(X_wst)
    print('RF training completed in ' + str(round(time.time()-t0, 2)) + 's')
    r2 = r2_score(y_labels, y_pred, multioutput='raw_values')
    print('RF R^2 scores for [Omega_m, S_8, T_AGN, f_0, Delta_z]:')
    print('  Omega_m: ' + str(round(r2[0], 4)))
    print('  S_8:     ' + str(round(r2[1], 4)))
    print('  T_AGN:   ' + str(round(r2[2], 4)))
    print('  f_0:     ' + str(round(r2[3], 4)))
    print('  Delta_z: ' + str(round(r2[4], 4)))
    print('Computing residuals...')
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(y_pred, X_wst)
    X_pred = lr.predict(y_pred)
    X_res = X_wst - X_pred
    print('Residual computation completed in ' + str(round(time.time()-t0, 2)) + 's')
    print('Residuals statistics:')
    print('  Mean: ' + str(np.mean(X_res)))
    print('  Std:  ' + str(np.std(X_res)))
    print('  Min:  ' + str(np.min(X_res)))
    print('  Max:  ' + str(np.max(X_res)))
    print('Saving data and models...')
    np.save(os.path.join(OUTPUT_DIR, 'X_wst.npy'), X_wst)
    np.save(os.path.join(OUTPUT_DIR, 'X_res.npy'), X_res)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y_labels)
    np.save(os.path.join(OUTPUT_DIR, 'y_pred.npy'), y_pred)
    joblib.dump(rf, os.path.join(OUTPUT_DIR, 'rf_regressor.joblib'))
    joblib.dump(lr, os.path.join(OUTPUT_DIR, 'lr_residual.joblib'))
    print('All tasks completed successfully.')