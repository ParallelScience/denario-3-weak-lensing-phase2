# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import os

if __name__ == '__main__':
    data_dir = 'data/'
    public_data_dir = '/home/node/work/weak_lensing_phase2/data/public_data'
    train_features = np.load(os.path.join(data_dir, 'wst_features_train.npy'))
    test_features = np.load(os.path.join(data_dir, 'wst_features_test.npy'))
    labels = np.load(os.path.join(public_data_dir, 'label.npy'))
    labels_flat = labels.reshape(-1, 5)
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std[std == 0] = 1e-8
    np.save(os.path.join(data_dir, 'wst_mean.npy'), mean)
    np.save(os.path.join(data_dir, 'wst_std.npy'), std)
    train_features_norm = (train_features - mean) / std
    test_features_norm = (test_features - mean) / std
    np.save(os.path.join(data_dir, 'wst_features_train_normalized.npy'), train_features_norm)
    np.save(os.path.join(data_dir, 'wst_features_test_normalized.npy'), test_features_norm)
    np.save(os.path.join(data_dir, 'labels_flat.npy'), labels_flat)
    num_cosmo = 101
    num_sys = 256
    rng = np.random.RandomState(42)
    val_cosmo_indices = rng.choice(num_cosmo, 20, replace=False)
    train_cosmo_indices = np.setdiff1d(np.arange(num_cosmo), val_cosmo_indices)
    train_indices = []
    for c in train_cosmo_indices:
        train_indices.extend(range(c * num_sys, (c + 1) * num_sys))
    train_indices = np.array(train_indices)
    val_indices = []
    for c in val_cosmo_indices:
        val_indices.extend(range(c * num_sys, (c + 1) * num_sys))
    val_indices = np.array(val_indices)
    np.save(os.path.join(data_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(data_dir, 'val_indices.npy'), val_indices)
    print('--- Data Preparation and Normalization ---')
    print('Feature Normalization Statistics:')
    print('  Mean (min, max, mean): ' + str(mean.min()) + ', ' + str(mean.max()) + ', ' + str(mean.mean()))
    print('  Std  (min, max, mean): ' + str(std.min()) + ', ' + str(std.max()) + ', ' + str(std.mean()))
    print('  Train features normalized (mean, std): ' + str(train_features_norm.mean()) + ', ' + str(train_features_norm.std()))
    print('  Test features normalized (mean, std): ' + str(test_features_norm.mean()) + ', ' + str(test_features_norm.std()))
    print('\nData Shapes:')
    print('  Normalized training features shape: ' + str(train_features_norm.shape))
    print('  Normalized test features shape: ' + str(test_features_norm.shape))
    print('  Flattened labels shape: ' + str(labels_flat.shape))
    print('\nData Split:')
    print('  Training indices count: ' + str(len(train_indices)) + ' (from ' + str(len(train_cosmo_indices)) + ' cosmologies)')
    print('  Validation indices count: ' + str(len(val_indices)) + ' (from ' + str(len(val_cosmo_indices)) + ' cosmologies)')
    print('  Validation cosmologies: ' + str(val_cosmo_indices))
    print('\nLabel Statistics (min, max, mean):')
    param_names = ['Omega_m', 'S_8', 'T_AGN', 'f_0', 'Delta_z']
    for i in range(5):
        p_min = labels_flat[:, i].min()
        p_max = labels_flat[:, i].max()
        p_mean = labels_flat[:, i].mean()
        print('  ' + param_names[i] + ': min=' + str(p_min) + ', max=' + str(p_max) + ', mean=' + str(p_mean))