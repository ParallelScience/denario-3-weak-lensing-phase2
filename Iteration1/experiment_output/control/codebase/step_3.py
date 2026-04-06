# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import joblib
import zuko

def train_cnf():
    start_time = time.time()
    OUTPUT_DIR = "data/"
    print("Loading features, labels, and scalers...")
    features = np.load(OUTPUT_DIR + "wst_pca_features.npy")
    labels = np.load(OUTPUT_DIR + "labels_flat.npy")
    scalers = joblib.load(OUTPUT_DIR + "regressor_scalers.pkl")
    feature_scaler = scalers['feature_scaler']
    label_scaler = scalers['label_scaler']
    features_scaled = feature_scaler.transform(features)
    labels_scaled = label_scaler.transform(labels)
    X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels_scaled, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    input_dim = features.shape[1]
    context_dim = labels.shape[1]
    print("Initializing Masked Autoregressive Flow (MAF)...")
    print("Input dimensions: " + str(input_dim) + ", Context dimensions: " + str(context_dim))
    flow = zuko.flows.MAF(features=input_dim, context=context_dim, transforms=4, hidden_features=[128, 128]).to(device)
    optimizer = optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    print("Training CNF...")
    for epoch in range(epochs):
        flow.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = -flow(y_batch).log_prob(X_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        flow.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss = -flow(y_batch).log_prob(X_batch).mean()
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(flow.state_dict(), OUTPUT_DIR + "cnf_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        if (epoch + 1) % 10 == 0:
            print("Epoch " + str(epoch + 1) + "/" + str(epochs) + " - Train NLL: " + str(round(train_loss, 4)) + " - Val NLL: " + str(round(val_loss, 4)))
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered at epoch " + str(epoch + 1))
            break
    print("Training completed. Best Validation NLL: " + str(round(best_val_loss, 4)))
    print("Step 3 completed in " + str(round(time.time() - start_time, 2)) + " seconds.")

if __name__ == '__main__':
    train_cnf()