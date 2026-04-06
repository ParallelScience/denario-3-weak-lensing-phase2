# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.mixture import GaussianMixture

DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
OUTPUT_DIR = 'data'

def main():
    print('Starting Step 2 & 3: Train MLP Regressor and GMM')
    features_path = os.path.join(OUTPUT_DIR, 'wst_features.npy')
    top_100_idx_path = os.path.join(OUTPUT_DIR, 'top_100_wst_indices.npy')
    labels_path = os.path.join(DATA_DIR, 'label.npy')
    print('Loading features from ' + features_path)
    features = np.load(features_path)
    top_100_idx = np.load(top_100_idx_path)
    labels = np.load(labels_path)
    n_cosmo, n_sys, _ = features.shape
    features = features[:, :, top_100_idx]
    n_train_cosmo = int(n_cosmo * 0.8)
    train_cosmo_idx = np.arange(n_train_cosmo)
    val_cosmo_idx = np.arange(n_train_cosmo, n_cosmo)
    np.save(os.path.join(OUTPUT_DIR, 'train_cosmo_idx.npy'), train_cosmo_idx)
    np.save(os.path.join(OUTPUT_DIR, 'val_cosmo_idx.npy'), val_cosmo_idx)
    print('Saved train/val cosmology indices.')
    print('Train cosmologies: ' + str(n_train_cosmo) + ', Val cosmologies: ' + str(n_cosmo - n_train_cosmo))
    train_features = features[train_cosmo_idx].reshape(-1, 100)
    train_labels = labels[train_cosmo_idx].reshape(-1, 5)
    f_mean = np.mean(train_features, axis=0)
    f_std = np.std(train_features, axis=0) + 1e-8
    train_features_norm = (train_features - f_mean) / f_std
    l_mean = np.mean(train_labels, axis=0)
    l_std = np.std(train_labels, axis=0) + 1e-8
    train_labels_norm = (train_labels - l_mean) / l_std
    np.save(os.path.join(OUTPUT_DIR, 'feature_scaler.npy'), np.stack([f_mean, f_std]))
    np.save(os.path.join(OUTPUT_DIR, 'label_scaler.npy'), np.stack([l_mean, l_std]))
    print('Saved feature and label scalers.')
    class MLPRegressor(nn.Module):
        def __init__(self, input_dim=100, output_dim=5):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())
            self.mean_head = nn.Linear(64, output_dim)
            self.logvar_head = nn.Linear(64, output_dim)
        def forward(self, x):
            h = self.shared(x)
            mean = self.mean_head(h)
            logvar = self.logvar_head(h)
            return mean, logvar
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    model = MLPRegressor(input_dim=100, output_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(torch.tensor(train_features_norm, dtype=torch.float32), torch.tensor(train_labels_norm, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print('Training MLP Regressor...')
    model.train()
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mean, logvar = model(x)
            loss_mean = F.huber_loss(mean, y, reduction='mean')
            loss_var = F.mse_loss(torch.exp(logvar), (y - mean.detach())**2, reduction='mean')
            loss = loss_mean + loss_var
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ' - Loss: ' + str(round(total_loss / len(loader), 4)))
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'mlp_regressor.pth'))
    print('Saved MLP Regressor to ' + os.path.join(OUTPUT_DIR, 'mlp_regressor.pth'))
    print('Obtaining theta_hat for the training set...')
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(train_features_norm, dtype=torch.float32).to(device)
        theta_hat_norm, _ = model(x_tensor)
        theta_hat_norm = theta_hat_norm.cpu().numpy()
    print('Training GMM on joint space...')
    Z = np.concatenate([train_features_norm, theta_hat_norm], axis=1)
    Z_mean = np.mean(Z, axis=0)
    Z_std = np.std(Z, axis=0) + 1e-8
    Z_norm = (Z - Z_mean) / Z_std
    np.save(os.path.join(OUTPUT_DIR, 'z_scaler.npy'), np.stack([Z_mean, Z_std]))
    print('Saved Z scaler.')
    n_components = 10
    print('Fitting GMM with ' + str(n_components) + ' components...')
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42, max_iter=200)
    gmm.fit(Z_norm)
    joblib.dump(gmm, os.path.join(OUTPUT_DIR, 'gmm_model.joblib'))
    print('Saved GMM model to ' + os.path.join(OUTPUT_DIR, 'gmm_model.joblib'))
    print('Calculating calibration statistics on training set...')
    nll = -gmm.score_samples(Z_norm)
    mean_nll = np.mean(nll)
    std_nll = np.std(nll)
    np.save(os.path.join(OUTPUT_DIR, 'calibration_stats.npy'), np.array([mean_nll, std_nll]))
    print('Calibration stats - Mean NLL: ' + str(round(mean_nll, 4)) + ', Std NLL: ' + str(round(std_nll, 4)))
    print('Saved calibration stats to ' + os.path.join(OUTPUT_DIR, 'calibration_stats.npy'))
    print('Step 2 & 3 completed successfully.')

if __name__ == '__main__':
    main()