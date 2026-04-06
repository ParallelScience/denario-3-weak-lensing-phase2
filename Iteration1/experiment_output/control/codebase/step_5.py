# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import time
import numpy as np
import torch
import torch.nn as nn
from kymatio.torch import Scattering2D
import joblib
import zuko
import json
import zipfile

class ProbabilisticRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar

def save_submission(ood_scores, errorbars=None, save_dir="/home/node/work/weak_lensing_phase2"):
    if errorbars is None:
        errorbars = [0.0] * len(ood_scores)
    data = {"means": list(map(float, ood_scores)), "errorbars": list(map(float, errorbars))}
    json_path = os.path.join(save_dir, "submission.json")
    zip_path = os.path.join(save_dir, "submission.zip")
    with open(json_path, 'w') as f:
        json.dump(data, f)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(json_path, "submission.json")
    return zip_path

def main():
    start_time = time.time()
    DATA_DIR = '/home/node/work/weak_lensing_phase2/data/public_data'
    OUTPUT_DIR = 'data/'
    SAVE_DIR = '/home/node/work/weak_lensing_phase2'
    
    print("Loading test data and mask...")
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    kappa_test = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_test_phase2_test.npy'), mmap_mode='r')
    
    print("Loading pipelines and models...")
    pipeline = joblib.load(os.path.join(OUTPUT_DIR, 'feature_pipeline.pkl'))
    selected_indices = pipeline['selected_indices']
    pca_scaler = pipeline['scaler']
    pca = pipeline['pca']
    
    scalers = joblib.load(os.path.join(OUTPUT_DIR, 'regressor_scalers.pkl'))
    feature_scaler = scalers['feature_scaler']
    
    with open(os.path.join(OUTPUT_DIR, 'calibration.json'), 'r') as f:
        calibration_params = json.load(f)
    shift = calibration_params['shift']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    
    input_dim = pca.n_components_
    output_dim = 5
    
    regressor = ProbabilisticRegressor(input_dim=input_dim, output_dim=output_dim, hidden_dim=256).to(device)
    regressor.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'regressor_model.pth'), weights_only=True))
    regressor.eval()
    
    cnf = zuko.flows.MAF(features=input_dim, context=output_dim, transforms=4, hidden_features=[128, 128]).to(device)
    cnf.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'cnf_model.pth'), weights_only=True))
    cnf.eval()
    
    J = 3
    L = 8
    shape = (mask.shape[0], mask.shape[1])
    scattering = Scattering2D(J=J, shape=shape, L=L).to(device)
    
    N_test = kappa_test.shape[0]
    batch_size = 50
    
    all_avg_nll = []
    all_var_nll = []
    
    print("Processing " + str(N_test) + " test maps in batches of " + str(batch_size) + "...")
    mask_tensor = torch.tensor(mask, device=device)
    
    for i in range(0, N_test, batch_size):
        end = min(i + batch_size, N_test)
        flat_maps_np = kappa_test[i:end].astype(np.float32)
        flat_maps = torch.tensor(flat_maps_np, device=device)
        
        full_maps = torch.zeros((end - i, shape[0], shape[1]), dtype=torch.float32, device=device)
        full_maps[:, mask] = flat_maps
        
        with torch.no_grad():
            wst = scattering(full_maps).mean(dim=(-1, -2))
        wst_np = wst.cpu().numpy()
        
        selected = wst_np[:, selected_indices]
        pca_features = pca.transform(pca_scaler.transform(selected))
        scaled_features = feature_scaler.transform(pca_features)
        
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            mean, logvar = regressor(features_tensor)
            std = torch.exp(0.5 * logvar)
            
            nll_list = []
            num_samples = 5
            for _ in range(num_samples):
                epsilon = torch.randn_like(std)
                theta_sample = mean + std * epsilon
                log_prob = cnf(theta_sample).log_prob(features_tensor)
                nll_list.append(-log_prob)
                
            nll_stack = torch.stack(nll_list, dim=0)
            avg_nll = nll_stack.mean(dim=0).cpu().numpy()
            var_nll = nll_stack.var(dim=0).cpu().numpy()
            
        all_avg_nll.append(avg_nll)
        all_var_nll.append(var_nll)
        
        if (i + batch_size) % 1000 == 0 or end == N_test:
            print("Processed " + str(end) + "/" + str(N_test) + " test maps...")
            
    all_avg_nll = np.concatenate(all_avg_nll, axis=0)
    all_var_nll = np.concatenate(all_var_nll, axis=0)
    
    print("Applying calibration shift...")
    calibrated_scores = all_avg_nll - shift
    
    print("Saving submission...")
    zip_path = save_submission(calibrated_scores, all_var_nll, save_dir=SAVE_DIR)
    print("Submission saved to " + zip_path)
    
    print("Step 5 completed in " + str(round(time.time() - start_time, 2)) + " seconds.")
    
    print("\n--- Submission Statistics ---")
    print("Total test samples: " + str(len(calibrated_scores)))
    print("Mean OoD Score: " + str(round(float(np.mean(calibrated_scores)), 4)))
    print("Std OoD Score: " + str(round(float(np.std(calibrated_scores)), 4)))
    print("Min OoD Score: " + str(round(float(np.min(calibrated_scores)), 4)))
    print("Max OoD Score: " + str(round(float(np.max(calibrated_scores)), 4)))
    print("Mean Errorbar (Variance): " + str(round(float(np.mean(all_var_nll)), 4)))

if __name__ == '__main__':
    main()