# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class ProbabilisticRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

def gaussian_nll_loss(mean, logvar, target):
    loss = 0.5 * (logvar + (target - mean)**2 * torch.exp(-logvar))
    return loss.mean()

def train_regressor():
    start_time = time.time()
    OUTPUT_DIR = "data/"
    print("Loading features and labels...")
    features = np.load(OUTPUT_DIR + "wst_pca_features.npy")
    labels = np.load(OUTPUT_DIR + "labels_flat.npy")
    print("Standardizing features and labels...")
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    label_scaler = StandardScaler()
    labels_scaled = label_scaler.fit_transform(labels)
    joblib.dump({'feature_scaler': feature_scaler, 'label_scaler': label_scaler}, OUTPUT_DIR + "regressor_scalers.pkl")
    print("Saved normalization statistics to " + OUTPUT_DIR + "regressor_scalers.pkl")
    X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_scaled, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    input_dim = features.shape[1]
    output_dim = labels.shape[1]
    model = ProbabilisticRegressor(input_dim=input_dim, output_dim=output_dim, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    print("Training Probabilistic Regressor...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            mean, logvar = model(X_batch)
            loss = gaussian_nll_loss(mean, logvar, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                mean, logvar = model(X_batch)
                loss = gaussian_nll_loss(mean, logvar, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR + "regressor_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        if (epoch + 1) % 10 == 0:
            print("Epoch " + str(epoch + 1) + "/" + str(epochs) + " - Train Loss: " + str(round(train_loss, 4)) + " - Val Loss: " + str(round(val_loss, 4)))
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered at epoch " + str(epoch + 1))
            break
    print("Training completed. Best Validation Loss: " + str(round(best_val_loss, 4)))
    model.load_state_dict(torch.load(OUTPUT_DIR + "regressor_model.pth", weights_only=True))
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            mean, _ = model(X_batch)
            val_preds.append(mean.cpu().numpy())
            val_targets.append(y_batch.numpy())
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_preds_orig = label_scaler.inverse_transform(val_preds)
    val_targets_orig = label_scaler.inverse_transform(val_targets)
    rmse = np.sqrt(np.mean((val_preds_orig - val_targets_orig)**2, axis=0))
    print("\nParameter Prediction RMSE (Original Scale):")
    param_names = ["Omega_m", "S_8", "T_AGN", "f_0", "Delta_z"]
    for name, r in zip(param_names, rmse):
        print(name + ": " + str(round(r, 6)))
    print("\nStep 2 completed in " + str(round(time.time() - start_time, 2)) + " seconds.")

if __name__ == "__main__":
    train_regressor()