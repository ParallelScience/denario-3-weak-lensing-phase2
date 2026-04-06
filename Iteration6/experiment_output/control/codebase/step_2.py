# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_dim * 2))
    def forward(self, x):
        out = self.net(x)
        mean, log_var = torch.chunk(out, 2, dim=-1)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mean, log_var
def gaussian_nll_loss(mean, log_var, target):
    var = torch.exp(log_var)
    loss = 0.5 * ((target - mean) ** 2 / var + log_var)
    return loss.mean()
if __name__ == '__main__':
    start_time = time.time()
    data_dir = 'data/'
    X = np.load(os.path.join(data_dir, 'wst_whitened_features.npy'))
    Y = np.load(os.path.join(data_dir, 'label_flat.npy'))
    num_train = 80 * 256
    X_train, X_val = X[:num_train], X[num_train:]
    Y_train, Y_val = Y[:num_train], Y[num_train:]
    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_std[Y_std == 0] = 1.0
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std
    np.save(os.path.join(data_dir, 'y_scaler_mean.npy'), Y_mean)
    np.save(os.path.join(data_dir, 'y_scaler_std.npy'), Y_std)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train_norm, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val_norm, dtype=torch.float32))
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    model = MLPRegressor(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            mean, log_var = model(x_batch)
            loss = gaussian_nll_loss(mean, log_var, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                mean, log_var = model(x_batch)
                loss = gaussian_nll_loss(mean, log_var, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(data_dir, 'mlp_regressor.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    model.load_state_dict(torch.load(os.path.join(data_dir, 'mlp_regressor.pth')))
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            mean, _ = model(x_batch)
            val_preds.append(mean.cpu().numpy())
            val_targets.append(y_batch.numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_preds_unnorm = val_preds * Y_std + Y_mean
    val_targets_unnorm = val_targets * Y_std + Y_mean
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    param_names = ['Omega_m', 'S_8', 'T_AGN', 'f_0', 'Delta_z']
    for i in range(5):
        axes[i].scatter(val_targets_unnorm[:, i], val_preds_unnorm[:, i], alpha=0.1, s=1)
        min_val = min(val_targets_unnorm[:, i].min(), val_preds_unnorm[:, i].min())
        max_val = max(val_targets_unnorm[:, i].max(), val_preds_unnorm[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[i].set_xlabel('True ' + param_names[i])
        axes[i].set_ylabel('Predicted ' + param_names[i])
        axes[i].set_title(param_names[i])
        axes[i].grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(data_dir, 'true_vs_predicted_1_' + timestamp + '.png')
    fig.savefig(plot_path, dpi=300)