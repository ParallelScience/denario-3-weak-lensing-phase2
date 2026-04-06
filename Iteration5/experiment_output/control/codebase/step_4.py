# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import json
import zipfile
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import zuko
from kymatio.torch import Scattering2D

DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
OUTPUT_DIR = "data/"
SAVE_DIR = "/home/node/work/weak_lensing_phase2"

def to_2d(flat_map, mask):
    full = np.zeros(mask.shape, dtype=np.float32)
    full[mask] = flat_map.astype(np.float32)
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
    print("Loading test data and mask...")
    mask = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa_test = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy"), mmap_mode='r')
    n_test = kappa_test.shape[0]
    print("Number of test maps: " + str(n_test))
    print("Initializing WST...")
    J = 4
    L = 8
    shape = mask.shape
    scattering = Scattering2D(J=J, L=L, shape=shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scattering = scattering.to(device)
    batch_size = 128
    test_features = []
    print("Computing WST features for test maps...")
    start_time = time.time()
    batch_maps = []
    for i in range(n_test):
        flat_map = kappa_test[i]
        map_2d = to_2d(flat_map, mask)
        batch_maps.append(map_2d)
        if len(batch_maps) == batch_size:
            batch_tensor = torch.tensor(np.stack(batch_maps), dtype=torch.float32).to(device)
            with torch.no_grad():
                wst_out = scattering(batch_tensor)
                wst_features = wst_out.mean(dim=(2, 3)).cpu().numpy()
            test_features.append(wst_features)
            batch_maps = []
        if (i + 1) % 1000 == 0:
            print("Processed " + str(i + 1) + " test maps.")
    if len(batch_maps) > 0:
        batch_tensor = torch.tensor(np.stack(batch_maps), dtype=torch.float32).to(device)
        with torch.no_grad():
            wst_out = scattering(batch_tensor)
            wst_features = wst_out.mean(dim=(2, 3)).cpu().numpy()
        test_features.append(wst_features)
    test_features = np.vstack(test_features)
    print("Test features shape: " + str(test_features.shape))
    print("WST computation took " + str(round(time.time() - start_time, 2)) + " seconds.")
    print("Loading scaler and scaling test features...")
    scaler = np.load(os.path.join(OUTPUT_DIR, "wst_scaler.npy"))
    mean, std = scaler[0], scaler[1]
    test_features_scaled = (test_features - mean) / std
    print("Loading training labels to create parameter grid...")
    train_labels = np.load(os.path.join(OUTPUT_DIR, "wst_labels.npy"))
    np.random.seed(42)
    random_indices = np.random.choice(train_labels.shape[0], 5, replace=False)
    param_grid = train_labels[random_indices]
    print("Parameter grid for conditioning (5 samples):")
    for idx, p in enumerate(param_grid):
        print("  Sample " + str(idx + 1) + ": " + str(np.round(p, 4)))
    print("Loading trained NSF model...")
    n_features_dim = test_features_scaled.shape[1]
    n_context_dim = train_labels.shape[1]
    flow = zuko.flows.NSF(features=n_features_dim, context=n_context_dim, transforms=8, hidden_features=[256, 256])
    flow.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "nsf_weights.pth"), map_location=device, weights_only=True))
    flow = flow.to(device)
    flow.eval()
    print("Computing NLL scores for test maps...")
    all_nlls = []
    test_dataset = TensorDataset(torch.tensor(test_features_scaled, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for params in param_grid:
            c = torch.tensor(params, dtype=torch.float32).to(device)
            c = c.unsqueeze(0)
            nlls_for_param = []
            for (x,) in test_loader:
                x = x.to(device)
                c_batch = c.expand(x.size(0), -1)
                loss = -flow(c_batch).log_prob(x)
                nlls_for_param.extend(loss.cpu().numpy())
            all_nlls.append(nlls_for_param)
    all_nlls = np.array(all_nlls)
    if np.isnan(all_nlls).any():
        print("Warning: NaNs detected in NLL scores. Replacing with max value.")
        max_nll = np.nanmax(all_nlls)
        all_nlls = np.nan_to_num(all_nlls, nan=max_nll)
    mean_nlls = np.mean(all_nlls, axis=0)
    std_nlls = np.std(all_nlls, axis=0)
    print("Mean NLL scores summary:")
    print("  Min: " + str(round(np.min(mean_nlls), 4)))
    print("  Max: " + str(round(np.max(mean_nlls), 4)))
    print("  Mean: " + str(round(np.mean(mean_nlls), 4)))
    print("  Std: " + str(round(np.std(mean_nlls), 4)))
    print("Uncertainty (errorbars) summary:")
    print("  Min: " + str(round(np.min(std_nlls), 4)))
    print("  Max: " + str(round(np.max(std_nlls), 4)))
    print("  Mean: " + str(round(np.mean(std_nlls), 4)))
    print("Saving submission...")
    zip_path = save_submission(mean_nlls, std_nlls, save_dir=SAVE_DIR)
    print("Submission saved to " + zip_path)