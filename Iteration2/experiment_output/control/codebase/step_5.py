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
import joblib
from kymatio.torch import Scattering2D
import zuko

class DynamicMLP(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.layers = nn.ModuleList()
        weight_keys = [k for k in state_dict.keys() if 'weight' in k and len(state_dict[k].shape) == 2]
        for i, w_key in enumerate(weight_keys):
            out_f, in_f = state_dict[w_key].shape
            linear = nn.Linear(in_f, out_f)
            linear.weight.data = state_dict[w_key]
            b_key = w_key.replace('weight', 'bias')
            if b_key in state_dict:
                linear.bias.data = state_dict[b_key]
            self.layers.append(linear)
            if i < len(weight_keys) - 1:
                self.layers.append(nn.ReLU())
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def to_2d_batch(flat_maps, mask):
    batch_size = flat_maps.shape[0]
    full = np.zeros((batch_size, *mask.shape), dtype=np.float32)
    full[:, mask] = flat_maps.astype(np.float32)
    return full

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
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    data_dir = 'data/'
    save_dir = '/home/node/work/weak_lensing_phase2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading test data and mask...')
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    kappa_test = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy'), mmap_mode='r')
    n_test = kappa_test.shape[0]
    J = 4
    L = 8
    scattering = Scattering2D(J=J, shape=mask.shape, L=L).to(device)
    batch_size = 100
    X_wst_test = []
    print('Extracting WST features for ' + str(n_test) + ' test maps (J=' + str(J) + ', L=' + str(L) + ')...')
    start_time = time.time()
    for i in range(0, n_test, batch_size):
        batch_flat = kappa_test[i:i+batch_size]
        batch_2d = to_2d_batch(batch_flat, mask)
        batch_tensor = torch.tensor(batch_2d, dtype=torch.float32).to(device)
        with torch.no_grad():
            wst_out = scattering(batch_tensor)
            wst_out = wst_out.mean(dim=(-2, -1))
        X_wst_test.append(wst_out.cpu().numpy())
        if (i + batch_size) % 1000 == 0 or (i + batch_size) == n_test:
            print('Processed ' + str(i + batch_size) + '/' + str(n_test) + ' maps in ' + str(round(time.time() - start_time, 2)) + 's')
    X_wst_test = np.concatenate(X_wst_test, axis=0)
    print('WST extraction completed in ' + str(round(time.time() - start_time, 2)) + 's')
    print('Predicting physical parameters using RF regressor...')
    t0 = time.time()
    rf = joblib.load(os.path.join(data_dir, 'rf_regressor.joblib'))
    y_pred_test = rf.predict(X_wst_test)
    print('Computing uncertainty estimates from RF trees...')
    preds = np.stack([tree.predict(X_wst_test) for tree in rf.estimators_])
    y_var = np.var(preds, axis=0)
    errorbars = np.mean(y_var, axis=1)
    errorbars = errorbars / (np.max(errorbars) + 1e-8)
    print('RF prediction and uncertainty estimation completed in ' + str(round(time.time() - t0, 2)) + 's')
    print('Computing residuals...')
    t0 = time.time()
    lr = joblib.load(os.path.join(data_dir, 'lr_residual.joblib'))
    X_pred_test = lr.predict(y_pred_test)
    X_res_test = X_wst_test - X_pred_test
    print('Residual computation completed in ' + str(round(time.time() - t0, 2)) + 's')
    print('Encoding residuals to latent space (VIB)...')
    t0 = time.time()
    Z_train = np.load(os.path.join(data_dir, 'Z_latent.npy'))
    X_res_train = np.load(os.path.join(data_dir, 'X_res.npy'))
    latent_dim = Z_train.shape[1]
    use_surrogate = False
    try:
        try:
            vib_state = torch.load(os.path.join(data_dir, 'vib_encoder.pth'), map_location='cpu', weights_only=True)
        except:
            vib_state = torch.load(os.path.join(data_dir, 'vib_encoder.pth'), map_location='cpu', weights_only=False)
        if isinstance(vib_state, dict):
            vib_encoder = DynamicMLP(vib_state)
        else:
            vib_encoder = vib_state
        vib_encoder.eval()
        with torch.no_grad():
            Z_pred = vib_encoder(torch.tensor(X_res_train[:1000], dtype=torch.float32))
            Z_pred = Z_pred[:, :latent_dim].numpy() if Z_pred.shape[1] == latent_dim * 2 else Z_pred.numpy()
        mse = np.mean((Z_pred - Z_train[:1000])**2)
        print('PyTorch VIB encoder MSE on train subset: ' + str(round(mse, 6)))
        if mse > 1e-2:
            print('MSE is too high. The architecture might be different. Falling back to surrogate.')
            use_surrogate = True
        else:
            print('Successfully verified PyTorch VIB encoder.')
            with torch.no_grad():
                Z_out = vib_encoder(torch.tensor(X_res_test, dtype=torch.float32))
                Z_test = Z_out[:, :latent_dim].numpy() if Z_out.shape[1] == latent_dim * 2 else Z_out.numpy()
    except Exception as e:
        print('Could not use PyTorch VIB encoder: ' + str(e))
        use_surrogate = True
    if use_surrogate:
        print('Training surrogate MLP to map X_res to Z_latent...')
        from sklearn.neural_network import MLPRegressor
        surrogate = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=100, random_state=42)
        surrogate.fit(X_res_train, Z_train)
        Z_test = surrogate.predict(X_res_test)
        print('Surrogate MLP training and prediction completed.')
    print('VIB encoding completed in ' + str(round(time.time() - t0, 2)) + 's')
    print('Computing OoD scores (NLL) using CNF...')
    t0 = time.time()
    context_dim = 5
    flow = zuko.flows.MAF(features=latent_dim, context=context_dim, transforms=5, hidden_features=[128, 128])
    flow.load_state_dict(torch.load(os.path.join(data_dir, 'cnf_model.pth'), map_location=device, weights_only=True))
    flow.to(device)
    flow.eval()
    y_mean = np.load(os.path.join(data_dir, 'y_mean.npy'))
    y_std = np.load(os.path.join(data_dir, 'y_std.npy'))
    y_scaled_test = (y_pred_test - y_mean) / y_std
    Z_tensor = torch.tensor(Z_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_scaled_test, dtype=torch.float32).to(device)
    batch_size_cnf = 1000
    nll_scores = []
    with torch.no_grad():
        for i in range(0, n_test, batch_size_cnf):
            batch_Z = Z_tensor[i:i+batch_size_cnf]
            batch_y = y_tensor[i:i+batch_size_cnf]
            nll = -flow(batch_y).log_prob(batch_Z)
            nll_scores.append(nll.cpu().numpy())
    nll_scores = np.concatenate(nll_scores)
    print('CNF scoring completed in ' + str(round(time.time() - t0, 2)) + 's')
    calib_params = np.load(os.path.join(data_dir, 'calibration_params.npy'))
    nll_p5 = calib_params[0]
    calibrated_scores = nll_scores - nll_p5
    print('\nScore Statistics (Calibrated):')
    print('  Min:  ' + str(round(float(np.min(calibrated_scores)), 4)))
    print('  Max:  ' + str(round(float(np.max(calibrated_scores)), 4)))
    print('  Mean: ' + str(round(float(np.mean(calibrated_scores)), 4)))
    print('  Std:  ' + str(round(float(np.std(calibrated_scores)), 4)))
    print('\nGenerating submission file...')
    zip_path = save_submission(calibrated_scores, errorbars, save_dir=save_dir)
    print('Submission saved to ' + zip_path)
    print('All tasks completed successfully.')