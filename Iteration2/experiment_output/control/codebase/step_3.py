# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import zuko
import time

def train_cnf():
    data_dir = 'data/'
    Z_path = os.path.join(data_dir, 'Z_latent.npy')
    y_path = os.path.join(data_dir, 'y_labels.npy')
    if not os.path.exists(Z_path) or not os.path.exists(y_path):
        raise FileNotFoundError('Z_latent.npy or y_labels.npy not found in data/ directory.')
    Z = np.load(Z_path)
    y_labels = np.load(y_path)
    if np.isnan(Z).any() or np.isinf(Z).any():
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    y_mean = np.mean(y_labels, axis=0)
    y_std = np.std(y_labels, axis=0)
    y_std[y_std == 0] = 1.0
    np.save(os.path.join(data_dir, 'y_mean.npy'), y_mean)
    np.save(os.path.join(data_dir, 'y_std.npy'), y_std)
    y_scaled = (y_labels - y_mean) / y_std
    Z_tensor = torch.tensor(Z, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    dataset = TensorDataset(Z_tensor, y_tensor)
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = Z.shape[1]
    context_dim = y_labels.shape[1]
    flow = zuko.flows.MAF(features=latent_dim, context=context_dim, transforms=5, hidden_features=[128, 128])
    flow = flow.to(device)
    optimizer = optim.Adam(flow.parameters(), lr=0.001, weight_decay=1e-05)
    epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_time = time.time()
    flow.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_Z, batch_y in dataloader:
            batch_Z = batch_Z.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.clone()
            swap_mask = torch.rand(batch_y.shape[0], device=device) < 0.2
            if swap_mask.any():
                num_swap = swap_mask.sum().item()
                random_indices = torch.randint(0, batch_y.shape[0], (num_swap,), device=device)
                batch_y[swap_mask, 2:] = batch_y[random_indices, 2:]
            optimizer.zero_grad()
            loss = -flow(batch_y).log_prob(batch_Z).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * batch_Z.shape[0]
        scheduler.step()
    model_path = os.path.join(data_dir, 'cnf_model.pth')
    torch.save(flow.state_dict(), model_path)
    flow.eval()
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        all_nll = []
        for batch_Z, batch_y in eval_loader:
            batch_Z = batch_Z.to(device)
            batch_y = batch_y.to(device)
            nll = -flow(batch_y).log_prob(batch_Z)
            all_nll.append(nll.cpu().numpy())
        all_nll = np.concatenate(all_nll)
    print('Training Set NLL Statistics:')
    print('Mean: ' + str(round(float(np.mean(all_nll)), 4)))
    print('Std: ' + str(round(float(np.std(all_nll)), 4)))

if __name__ == '__main__':
    train_cnf()