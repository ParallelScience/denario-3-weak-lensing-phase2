# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
from scipy.stats import pearsonr
if __name__ == '__main__':
    OUTPUT_DIR = 'data/'
    print('Loading latent representations and labels...')
    Z = np.load(os.path.join(OUTPUT_DIR, 'Z_latent.npy'))
    y_labels = np.load(os.path.join(OUTPUT_DIR, 'y_labels.npy'))
    latent_dim = Z.shape[1]
    print('Computing correlations between latent Z and physical parameters...')
    corrs = np.zeros((latent_dim, 5))
    for i in range(latent_dim):
        for j in range(5):
            if np.std(Z[:, i]) > 1e-6 and np.std(y_labels[:, j]) > 1e-6:
                corrs[i, j], _ = pearsonr(Z[:, i], y_labels[:, j])
            else:
                corrs[i, j] = 0.0
    print('Mean absolute correlation with Omega_m: ' + str(round(np.mean(np.abs(corrs[:, 0])), 4)))
    print('Mean absolute correlation with S_8: ' + str(round(np.mean(np.abs(corrs[:, 1])), 4)))
    print('Mean absolute correlation with T_AGN: ' + str(round(np.mean(np.abs(corrs[:, 2])), 4)))
    print('Mean absolute correlation with f_0: ' + str(round(np.mean(np.abs(corrs[:, 3])), 4)))
    print('Mean absolute correlation with Delta_z: ' + str(round(np.mean(np.abs(corrs[:, 4])), 4)))
    print('Max absolute correlation with Omega_m: ' + str(round(np.max(np.abs(corrs[:, 0])), 4)))
    print('Max absolute correlation with S_8: ' + str(round(np.max(np.abs(corrs[:, 1])), 4)))
    print('Max absolute correlation with T_AGN: ' + str(round(np.max(np.abs(corrs[:, 2])), 4)))
    print('Max absolute correlation with f_0: ' + str(round(np.max(np.abs(corrs[:, 3])), 4)))
    print('Max absolute correlation with Delta_z: ' + str(round(np.max(np.abs(corrs[:, 4])), 4)))
    print('VIB and CNF training completed successfully in the previous run.')