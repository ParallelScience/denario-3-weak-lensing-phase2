# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import zuko

plt.rcParams['text.usetex'] = False

if __name__ == '__main__':
    start_time = time.time()
    data_dir = 'data/'
    
    print('Loading data...')
    X = np.load(os.path.join(data_dir, 'wst_whitened_features.npy'))
    Y = np.load(os.path.join(data_dir, 'label_flat.npy'))
    
    num_train = 80 * 256
    X_train, X_val = X[:num_train], X[num_train:]
    Y_train, Y_val = Y[:num_train], Y[num_train:]
    
    Y_mean = np.load(os.path.join(data_dir, 'y_scaler_mean.npy'))
    Y_std = np.load(os.path.join(data_dir, 'y_scaler_std.npy'))
    
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train_norm, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val_norm, dtype=torch.float32)
    )
    
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ' + str(device))
    
    features_dim = X.shape[1]
    context_dim = Y.shape[1]
    
    print('Initializing NSF with features_dim=' + str(features_dim) + ', context_dim=' + str(context_dim))
    
    flow = zuko.flows.NSF(
        features=features_dim,
        context=context_dim,
        transforms=5,
        hidden_features=[128, 128]
    ).to(device)
    
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    epochs = 200
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    
    print('Starting training...')
    for epoch in range(epochs):
        flow.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = -flow(y_batch).log_prob(x_batch).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        flow.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                loss = -flow(y_batch).log_prob(x_batch).mean()
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(flow.state_dict(), os.path.join(data_dir, 'nsf_flow.pth'))
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            
        if epoch % 5 == 0 or epochs_no_improve >= patience:
            print('Epoch ' + str(epoch) + ' | Train NLL: ' + str(round(train_loss, 4)) + ' | Val NLL: ' + str(round(val_loss, 4)))
            
        if epochs_no_improve >= patience:
            print('Early stopping triggered at epoch ' + str(epoch))
            break
            
    print('Training completed. Best Val NLL: ' + str(round(best_val_loss, 4)) + ' at epoch ' + str(best_epoch))
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_losses, label='Train NLL')
    ax.plot(val_losses, label='Validation NLL')
    ax.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.set_title('NSF Training and Validation NLL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    plot_path = os.path.join(data_dir, 'nsf_training_curve_1_' + timestamp + '.png')
    fig.savefig(plot_path, dpi=300)
    print('Training curve saved to ' + plot_path)
    print('Step 3 completed successfully in ' + str(round(time.time() - start_time, 2)) + ' seconds.')