# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MLPRegressor(nn.Module):
    def __init__(self, input_dim=217, output_dim=5):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    data_dir = 'data/'
    
    train_features_norm = np.load(os.path.join(data_dir, 'wst_features_train_normalized.npy'))
    labels_flat = np.load(os.path.join(data_dir, 'labels_flat.npy'))
    train_indices = np.load(os.path.join(data_dir, 'train_indices.npy'))
    val_indices = np.load(os.path.join(data_dir, 'val_indices.npy'))
    
    label_mean = np.load(os.path.join(data_dir, 'label_mean.npy'))
    label_std = np.load(os.path.join(data_dir, 'label_std.npy'))
    
    labels_norm = (labels_flat - label_mean) / label_std
    
    fallback = np.load(os.path.join(data_dir, 'cnf_fallback.npy'))[0]
    output_dim = 2 if fallback else 5
    
    if fallback:
        labels_norm = labels_norm[:, :2]
        
    X_train = torch.tensor(train_features_norm[train_indices], dtype=torch.float32)
    Y_train = torch.tensor(labels_norm[train_indices], dtype=torch.float32)
    X_val = torch.tensor(train_features_norm[val_indices], dtype=torch.float32)
    Y_val = torch.tensor(labels_norm[val_indices], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MLPRegressor(input_dim=217, output_dim=output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 100
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print('--- MLP Regressor Training ---')
    print('Training samples: ' + str(len(train_indices)) + ', Validation samples: ' + str(len(val_indices)))
    print('Input dimension: 217, Output dimension: ' + str(output_dim))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ' - Train MSE: ' + str(round(train_loss, 4)) + ' - Val MSE: ' + str(round(val_loss, 4)))
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(data_dir, 'mlp_weights.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print('Early stopping at epoch ' + str(epoch + 1))
            break
            
    print('Best Validation MSE: ' + str(round(best_val_loss, 4)))
    
    model.load_state_dict(torch.load(os.path.join(data_dir, 'mlp_weights.pth')))
    model.eval()
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    rmse = np.sqrt(np.mean((all_preds - all_targets)**2, axis=0))
    
    param_names = ['Omega_m', 'S_8', 'T_AGN', 'f_0', 'Delta_z']
    print('\nPer-parameter Normalized RMSE on Validation Set:')
    for i in range(output_dim):
        print('  ' + param_names[i] + ': ' + str(round(rmse[i], 4)))
        
    print('\nModel weights saved to data/mlp_weights.pth')