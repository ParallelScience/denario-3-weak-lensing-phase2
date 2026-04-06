# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import zuko

if __name__ == '__main__':
    data_dir = 'data/'
    train_features_norm = np.load(os.path.join(data_dir, 'wst_features_train_normalized.npy'))
    labels_flat = np.load(os.path.join(data_dir, 'labels_flat.npy'))
    train_indices = np.load(os.path.join(data_dir, 'train_indices.npy'))
    val_indices = np.load(os.path.join(data_dir, 'val_indices.npy'))
    train_labels = labels_flat[train_indices]
    label_mean = np.mean(train_labels, axis=0)
    label_std = np.std(train_labels, axis=0)
    label_std[label_std == 0] = 1e-8
    np.save(os.path.join(data_dir, 'label_mean.npy'), label_mean)
    np.save(os.path.join(data_dir, 'label_std.npy'), label_std)
    labels_norm = (labels_flat - label_mean) / label_std
    X_train = torch.tensor(train_features_norm[train_indices], dtype=torch.float32)
    Y_train = torch.tensor(labels_norm[train_indices], dtype=torch.float32)
    X_val = torch.tensor(train_features_norm[val_indices], dtype=torch.float32)
    Y_val = torch.tensor(labels_norm[val_indices], dtype=torch.float32)
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow = zuko.flows.MAF(features=217, context=5, transforms=5, hidden_features=[256, 256])
    flow = flow.to(device)
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    print('--- CNF Training ---')
    print('Training samples: ' + str(len(train_indices)) + ', Validation samples: ' + str(len(val_indices)))
    print('Feature dimension: 217, Context dimension: 5')
    for epoch in range(epochs):
        flow.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = -flow(y_batch).log_prob(x_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        flow.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                loss = -flow(y_batch).log_prob(x_batch).mean()
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ' - Train NLL: ' + str(round(train_loss, 4)) + ' - Val NLL: ' + str(round(val_loss, 4)))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(flow.state_dict(), os.path.join(data_dir, 'cnf_weights.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping at epoch ' + str(epoch + 1))
            break
    print('Best Validation NLL: ' + str(round(best_val_loss, 4)))
    fallback = False
    if np.isnan(best_val_loss) or best_val_loss > 1e6:
        fallback = True
        print('\nTraining failed or diverged. Falling back to 2 parameters (Omega_m, S_8)...')
        flow = zuko.flows.MAF(features=217, context=2, transforms=5, hidden_features=[256, 256])
        flow = flow.to(device)
        optimizer = optim.Adam(flow.parameters(), lr=1e-3)
        Y_train_2 = Y_train[:, :2]
        Y_val_2 = Y_val[:, :2]
        train_dataset_2 = TensorDataset(X_train, Y_train_2)
        val_dataset_2 = TensorDataset(X_val, Y_val_2)
        train_loader_2 = DataLoader(train_dataset_2, batch_size=512, shuffle=True)
        val_loader_2 = DataLoader(val_dataset_2, batch_size=512, shuffle=False)
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(epochs):
            flow.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader_2:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = -flow(y_batch).log_prob(x_batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            train_loss /= len(train_loader_2.dataset)
            flow.eval()
            with torch.no_grad():
                val_loss = 0.0
                for x_batch, y_batch in val_loader_2:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    loss = -flow(y_batch).log_prob(x_batch).mean()
                    val_loss += loss.item() * x_batch.size(0)
            val_loss /= len(val_loader_2.dataset)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print('Fallback Epoch ' + str(epoch + 1) + '/' + str(epochs) + ' - Train NLL: ' + str(round(train_loss, 4)) + ' - Val NLL: ' + str(round(val_loss, 4)))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(flow.state_dict(), os.path.join(data_dir, 'cnf_weights.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping at epoch ' + str(epoch + 1))
                break
        print('Fallback Best Validation NLL: ' + str(round(best_val_loss, 4)))
    print('Model weights saved to data/cnf_weights.pth')
    print('Final model uses ' + str(2 if fallback else 5) + ' parameters.')
    np.save(os.path.join(data_dir, 'cnf_fallback.npy'), np.array([fallback]))