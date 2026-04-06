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
import matplotlib.pyplot as plt
import zuko
import os

def main():
    DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
    OUTPUT_DIR = "data"
    print("Loading data...")
    wst_features = np.load(os.path.join(OUTPUT_DIR, "wst_features.npy"))
    train_indices = np.load(os.path.join(OUTPUT_DIR, "train_indices.npy"))
    val_indices = np.load(os.path.join(OUTPUT_DIR, "val_indices.npy"))
    labels = np.load(os.path.join(DATA_DIR, "label.npy"))
    labels = labels.reshape(-1, 5)
    train_labels = labels[train_indices]
    label_mean = np.mean(train_labels, axis=0)
    label_std = np.std(train_labels, axis=0)
    label_std[label_std == 0] = 1.0
    np.save(os.path.join(OUTPUT_DIR, "label_mean.npy"), label_mean)
    np.save(os.path.join(OUTPUT_DIR, "label_std.npy"), label_std)
    train_wst = torch.tensor(wst_features[train_indices], dtype=torch.float32)
    train_cond = torch.tensor(labels[train_indices], dtype=torch.float32)
    val_wst = torch.tensor(wst_features[val_indices], dtype=torch.float32)
    val_cond = torch.tensor(labels[val_indices], dtype=torch.float32)
    batch_size = 256
    train_dataset = TensorDataset(train_wst, train_cond)
    val_dataset = TensorDataset(val_wst, val_cond)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    features_dim = wst_features.shape[1]
    context_dim = 5
    print("Initializing MAF model...")
    try:
        flow = zuko.flows.MAF(features=features_dim, context=context_dim, transforms=8, hidden_features=[256, 256], dropout=0.1)
        print("Successfully initialized MAF with dropout.")
    except TypeError:
        print("zuko.flows.MAF does not support dropout argument. Initializing without dropout.")
        flow = zuko.flows.MAF(features=features_dim, context=context_dim, transforms=8, hidden_features=[256, 256])
    flow = flow.to(device)
    num_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print("Number of trainable parameters: " + str(num_params))
    optimizer = optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    label_mean_t = torch.tensor(label_mean, dtype=torch.float32, device=device)
    label_std_t = torch.tensor(label_std, dtype=torch.float32, device=device)
    def augment_and_normalize(cond):
        B = cond.shape[0]
        new_cond = cond.clone()
        new_cond[:, 2] = torch.rand(B, device=device) * (8.50 - 7.20) + 7.20
        new_cond[:, 3] = torch.rand(B, device=device) * (0.0264 - 0.00004) + 0.00004
        new_cond[:, 4] = torch.randn(B, device=device) * 0.022
        return (new_cond - label_mean_t) / label_std_t
    def normalize(cond):
        return (cond - label_mean_t) / label_std_t
    print("Starting training...")
    start_time = time.time()
    for epoch in range(epochs):
        flow.train()
        train_loss = 0.0
        valid_batches = 0
        for batch_wst, batch_cond in train_loader:
            batch_wst = batch_wst.to(device)
            batch_cond = batch_cond.to(device)
            batch_cond_aug = augment_and_normalize(batch_cond)
            optimizer.zero_grad()
            loss = -flow(batch_cond_aug).log_prob(batch_wst).mean()
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * batch_wst.size(0)
            valid_batches += batch_wst.size(0)
        if valid_batches > 0:
            train_loss /= valid_batches
        else:
            train_loss = float('inf')
        train_losses.append(train_loss)
        flow.eval()
        val_loss = 0.0
        valid_val_batches = 0
        with torch.no_grad():
            for batch_wst, batch_cond in val_loader:
                batch_wst = batch_wst.to(device)
                batch_cond = batch_cond.to(device)
                batch_cond_norm = normalize(batch_cond)
                loss = -flow(batch_cond_norm).log_prob(batch_wst).mean()
                if torch.isnan(loss):
                    continue
                val_loss += loss.item() * batch_wst.size(0)
                valid_val_batches += batch_wst.size(0)
        if valid_val_batches > 0:
            val_loss /= valid_val_batches
        else:
            val_loss = float('inf')
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " - Train Loss: " + str(round(train_loss, 4)) + " - Val Loss: " + str(round(val_loss, 4)))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(flow.state_dict(), os.path.join(OUTPUT_DIR, "best_maf_model.pth"))
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered at epoch " + str(epoch+1))
            break
        if time.time() - start_time > 1700:
            print("Time limit approaching, stopping training.")
            break
    print("Training completed in " + str(round(time.time() - start_time, 2)) + " seconds.")
    print("Best Validation Loss: " + str(round(best_val_loss, 4)))
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('MAF Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_filename = os.path.join(OUTPUT_DIR, "maf_loss_curve_1_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("Loss curve saved to " + plot_filename)

if __name__ == '__main__':
    main()