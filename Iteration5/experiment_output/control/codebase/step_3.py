# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import zuko
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from kymatio.torch import Scattering2D
import torchvision.transforms.functional as TF
import os

DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
OUTPUT_DIR = "data/"

def add_noise(flat_map, mask, ng=30, pixel_size=2.0):
    noise_sigma = 0.4 / (2 * ng * pixel_size**2)**0.5
    return flat_map + np.random.randn(*flat_map.shape) * noise_sigma * mask[mask]

def to_2d(flat_map, mask):
    full = np.zeros(mask.shape, dtype=np.float32)
    full[mask] = flat_map.astype(np.float32)
    return full

def score_phase2(test_labels, ood_scores):
    fpr, tpr, _ = roc_curve(test_labels, ood_scores)
    fpr_grid = np.logspace(np.log10(0.001), np.log10(0.05), 100)
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    return np.mean(tpr_interp)

if __name__ == '__main__':
    print("Loading features and labels...")
    features = np.load(os.path.join(OUTPUT_DIR, "wst_features_scaled.npy"))
    labels = np.load(os.path.join(OUTPUT_DIR, "wst_labels.npy"))
    scaler = np.load(os.path.join(OUTPUT_DIR, "wst_scaler.npy"))
    mean, std = scaler[0], scaler[1]
    n_cosmo_train = 80
    n_sys = 256
    n_realizations = 3
    train_samples = n_cosmo_train * n_sys * n_realizations
    train_features = features[:train_samples]
    train_labels = labels[:train_samples]
    val_features_ind = features[train_samples:]
    val_labels_ind = labels[train_samples:]
    print("Train samples: " + str(len(train_features)))
    print("Validation InD samples: " + str(len(val_features_ind)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    print("Initializing NSF for LOCO validation...")
    n_features_dim = train_features.shape[1]
    n_context_dim = train_labels.shape[1]
    flow = zuko.flows.NSF(features=n_features_dim, context=n_context_dim, transforms=8, hidden_features=[256, 256])
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    epochs = 30
    print("Training NSF on 80 cosmologies...")
    start_time = time.time()
    for epoch in range(epochs):
        flow.train()
        train_loss = 0.0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            loss = -flow(c).log_prob(x).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("Epoch " + str(epoch + 1) + "/" + str(epochs) + " - Train Loss: " + str(round(train_loss, 4)))
    print("Training completed in " + str(round(time.time() - start_time, 2)) + " seconds.")
    print("Generating synthetic OoD proxies (Gaussian smoothing) for validation set...")
    mask = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_mask.npy"))
    kappa = np.load(os.path.join(DATA_DIR, "WIDE12H_bin2_2arcmin_kappa_newrealization.npy"), mmap_mode='r')
    label_orig = np.load(os.path.join(DATA_DIR, "label.npy"))
    J = 4
    L = 8
    shape = mask.shape
    scattering = Scattering2D(J=J, L=L, shape=shape)
    if torch.cuda.is_available():
        scattering = scattering.cuda()
    n_cosmo_total = kappa.shape[0]
    val_cosmo_start = 80
    ood_features_list = []
    ood_labels_list = []
    batch_maps = []
    batch_labels = []
    wst_batch_size = 64
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    if torch.cuda.is_available():
        mask_tensor = mask_tensor.cuda()
    for c in range(val_cosmo_start, n_cosmo_total):
        for s in range(n_sys):
            flat_map = kappa[c, s]
            lbl = label_orig[c, s]
            for r in range(n_realizations):
                noisy_flat = add_noise(flat_map, mask)
                map_2d = to_2d(noisy_flat, mask)
                batch_maps.append(map_2d)
                batch_labels.append(lbl)
                if len(batch_maps) == wst_batch_size:
                    batch_tensor = torch.tensor(np.stack(batch_maps), dtype=torch.float32)
                    if torch.cuda.is_available():
                        batch_tensor = batch_tensor.cuda()
                    batch_tensor = batch_tensor.unsqueeze(1)
                    batch_tensor = TF.gaussian_blur(batch_tensor, kernel_size=5, sigma=1.0)
                    batch_tensor = batch_tensor.squeeze(1)
                    batch_tensor = batch_tensor * mask_tensor
                    with torch.no_grad():
                        wst_out = scattering(batch_tensor)
                        wst_features = wst_out.mean(dim=(2, 3)).cpu().numpy()
                    ood_features_list.append(wst_features)
                    ood_labels_list.append(np.array(batch_labels))
                    batch_maps = []
                    batch_labels = []
    if len(batch_maps) > 0:
        batch_tensor = torch.tensor(np.stack(batch_maps), dtype=torch.float32)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        batch_tensor = batch_tensor.unsqueeze(1)
        batch_tensor = TF.gaussian_blur(batch_tensor, kernel_size=5, sigma=1.0)
        batch_tensor = batch_tensor.squeeze(1)
        batch_tensor = batch_tensor * mask_tensor
        with torch.no_grad():
            wst_out = scattering(batch_tensor)
            wst_features = wst_out.mean(dim=(2, 3)).cpu().numpy()
        ood_features_list.append(wst_features)
        ood_labels_list.append(np.array(batch_labels))
    val_features_ood = np.vstack(ood_features_list)
    val_labels_ood = np.vstack(ood_labels_list)
    val_features_ood_scaled = (val_features_ood - mean) / std
    print("Validation OoD samples: " + str(len(val_features_ood_scaled)))
    print("Computing NLL scores...")
    flow.eval()
    def compute_nll(features_array, labels_array):
        dataset = TensorDataset(torch.tensor(features_array, dtype=torch.float32), torch.tensor(labels_array, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        nlls = []
        with torch.no_grad():
            for x, c in loader:
                x, c = x.to(device), c.to(device)
                loss = -flow(c).log_prob(x)
                nlls.extend(loss.cpu().numpy())
        return np.array(nlls)
    nll_ind = compute_nll(val_features_ind, val_labels_ind)
    nll_ood = compute_nll(val_features_ood_scaled, val_labels_ood)
    if np.isnan(nll_ind).any() or np.isnan(nll_ood).any():
        print("Warning: NaNs detected in NLL scores. Replacing with max value.")
        max_nll = np.nanmax(np.concatenate([nll_ind, nll_ood]))
        nll_ind = np.nan_to_num(nll_ind, nan=max_nll)
        nll_ood = np.nan_to_num(nll_ood, nan=max_nll)
    print("InD NLL - Mean: " + str(round(np.mean(nll_ind), 4)) + ", Std: " + str(round(np.std(nll_ind), 4)))
    print("InD NLL - Min: " + str(round(np.min(nll_ind), 4)) + ", Max: " + str(round(np.max(nll_ind), 4)))
    print("OoD NLL - Mean: " + str(round(np.mean(nll_ood), 4)) + ", Std: " + str(round(np.std(nll_ood), 4)))
    print("OoD NLL - Min: " + str(round(np.min(nll_ood), 4)) + ", Max: " + str(round(np.max(nll_ood), 4)))
    y_true = np.concatenate([np.zeros(len(nll_ind)), np.ones(len(nll_ood))])
    y_scores = np.concatenate([nll_ind, nll_ood])
    p_auc = score_phase2(y_true, y_scores)
    print("Validation Partial AUC (FPR 0.001-0.05): " + str(round(p_auc, 4)))
    min_val = min(np.percentile(nll_ind, 1), np.percentile(nll_ood, 1))
    max_val = max(np.percentile(nll_ind, 99), np.percentile(nll_ood, 99))
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(10, 6))
    plt.hist(nll_ind, bins=100, range=(min_val, max_val), alpha=0.5, density=True, label='InD (Validation Cosmologies)')
    plt.hist(nll_ood, bins=100, range=(min_val, max_val), alpha=0.5, density=True, label='OoD (Gaussian Smoothed)')
    plt.xlabel('Negative Log-Likelihood (NLL)')
    plt.ylabel('Density')
    plt.title('NLL Distribution: InD vs Synthetic OoD (LOCO Validation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = os.path.join(OUTPUT_DIR, "nll_distribution_1_" + str(timestamp) + ".png")
    plt.savefig(plot_path, dpi=300)
    print("NLL distribution plot saved to " + plot_path)