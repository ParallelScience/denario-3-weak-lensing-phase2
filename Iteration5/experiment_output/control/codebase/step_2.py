# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import zuko
import matplotlib.pyplot as plt

def main():
    print('Loading standardized WST features and labels...')
    features = np.load('data/wst_features_scaled.npy')
    labels = np.load('data/wst_labels.npy')
    n_samples, n_features = features.shape
    _, n_context = labels.shape
    print('Features shape: ' + str(features.shape))
    print('Labels shape: ' + str(labels.shape))
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    split = int(0.9 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    train_features, val_features = features[train_idx], features[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    print('Train samples: ' + str(len(train_features)))
    print('Validation samples: ' + str(len(val_features)))
    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.float32))
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    print('Initializing Conditional Neural Spline Flow (NSF)...')
    flow = zuko.flows.NSF(features=n_features, context=n_context, transforms=8, hidden_features=[256, 256])
    flow = flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    epochs = 40
    train_losses = []
    val_losses = []
    print('Starting training...')
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
        train_losses.append(train_loss)
        flow.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, c in val_loader:
                x, c = x.to(device), c.to(device)
                loss = -flow(c).log_prob(x).mean()
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ' - Train Loss: ' + str(round(train_loss, 4)) + ' - Val Loss: ' + str(round(val_loss, 4)))
        if np.isnan(train_loss) or np.isnan(val_loss):
            print('Warning: NaN loss detected. Stopping training.')
            break
    end_time = time.time()
    print('Training completed in ' + str(round(end_time - start_time, 2)) + ' seconds.')
    print('Final Train Loss: ' + str(round(train_losses[-1], 4)))
    print('Final Validation Loss: ' + str(round(val_losses[-1], 4)))
    model_path = 'data/nsf_weights.pth'
    torch.save(flow.state_dict(), model_path)
    print('Model weights saved to ' + model_path)
    plt.rcParams['text.usetex'] = False
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('NSF Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    timestamp = int(time.time())
    plot_path = 'data/loss_plot_1_' + str(timestamp) + '.png'
    plt.savefig(plot_path, dpi=300)
    print('Loss plot saved to ' + plot_path)

if __name__ == '__main__':
    main()