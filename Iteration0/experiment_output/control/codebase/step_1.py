# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import subprocess
import numpy as np
import torch
try:
    import kymatio
except ImportError as e:
    if "sph_harm" in str(e):
        subprocess.run("sed -i 's/from scipy.special import sph_harm, factorial/from scipy.special import sph_harm_y as sph_harm, factorial/' /opt/denario-venv/lib/python3.12/site-packages/kymatio/scattering3d/filter_bank.py", shell=True, check=True)
        import kymatio
    else:
        raise e
from kymatio.torch import Scattering2D
def to_2d_batch(flat_maps, mask):
    batch_size = flat_maps.shape[0]
    full = np.zeros((batch_size, mask.shape[0], mask.shape[1]), dtype=np.float32)
    full[:, mask] = flat_maps.astype(np.float32)
    return full
if __name__ == '__main__':
    DATA_DIR = "/home/node/work/weak_lensing_phase2/data/public_data"
    mask = np.load(DATA_DIR + "/WIDE12H_bin2_2arcmin_mask.npy")
    kappa = np.load(DATA_DIR + "/WIDE12H_bin2_2arcmin_kappa_newrealization.npy", mmap_mode='r')
    kappa_test = np.load(DATA_DIR + "/WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy", mmap_mode='r')
    num_cosmo, num_sys, num_pixels = kappa.shape
    total_train = num_cosmo * num_sys
    kappa_flat = kappa.reshape(total_train, num_pixels)
    total_test = kappa_test.shape[0]
    J = 3
    L = 8
    scattering = Scattering2D(J=J, shape=mask.shape, L=L)
    if torch.cuda.is_available():
        scattering = scattering.cuda()
    noise_sigma = 0.4 / (2 * 30 * 2.0**2)**0.5
    train_features = []
    start_time = time.time()
    for i in range(0, total_train, 64):
        batch_flat = kappa_flat[i:i+64]
        noise = np.random.randn(*batch_flat.shape) * noise_sigma
        batch_noisy = batch_flat + noise
        batch_2d = to_2d_batch(batch_noisy, mask)
        batch_tensor = torch.tensor(batch_2d, dtype=torch.float32)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        with torch.no_grad():
            s_out = scattering(batch_tensor)
            s_pooled = s_out.mean(dim=(-2, -1))
        train_features.append(s_pooled.cpu().numpy())
    train_features = np.concatenate(train_features, axis=0)
    np.save("data/wst_features_train.npy", train_features)
    test_features = []
    for i in range(0, total_test, 64):
        batch_flat = kappa_test[i:i+64]
        batch_2d = to_2d_batch(batch_flat, mask)
        batch_tensor = torch.tensor(batch_2d, dtype=torch.float32)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        with torch.no_grad():
            s_out = scattering(batch_tensor)
            s_pooled = s_out.mean(dim=(-2, -1))
        test_features.append(s_pooled.cpu().numpy())
    test_features = np.concatenate(test_features, axis=0)
    np.save("data/wst_features_test.npy", test_features)