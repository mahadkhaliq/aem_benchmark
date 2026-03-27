import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

import config

CKPT = os.path.join(os.path.dirname(__file__), 'models', 'MLP', 'adm_mlp', 'best_model_forward.pt')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load(CKPT, map_location=device)
net.eval()

# Load test data
os.chdir(config.DATA_DIR)
test_x = pd.read_csv(os.path.join('ADM', 'testset', 'test_g.csv'), header=None).astype('float32').values
test_y = pd.read_csv(os.path.join('ADM', 'testset', 'test_s.csv'), header=None).astype('float32').values

# Normalize inputs (same as training)
from AEML.data.loader import normalize_np
train_x = pd.read_csv(os.path.join('ADM', 'data_g.csv'), header=None).astype('float32').values
_, x_max, x_min = normalize_np(train_x.copy())
test_x, _, _ = normalize_np(test_x, x_max, x_min)

# Run inference on full test set
with torch.no_grad():
    pred = net(torch.tensor(test_x).to(device)).cpu().numpy()

# MSE per sample
mse_per_sample = np.mean((pred - test_y) ** 2, axis=1)
overall_mse = np.mean(mse_per_sample)
print(f"Test MSE:    {overall_mse:.6f}")
print(f"Test RMSE:   {np.sqrt(overall_mse):.6f}")
print(f"Input shape:  {test_x.shape}")
print(f"Output shape: {pred.shape}")

# Plot 5 random sample predictions vs ground truth
np.random.seed(42)
indices = np.random.choice(len(test_y), 5, replace=False)
fig, axes = plt.subplots(5, 1, figsize=(12, 15))
for ax, idx in zip(axes, indices):
    ax.plot(test_y[idx], label='Ground truth', alpha=0.8)
    ax.plot(pred[idx], label='Predicted', alpha=0.8, linestyle='--')
    ax.set_title(f'Sample {idx} | MSE: {mse_per_sample[idx]:.5f}')
    ax.legend()
    ax.set_xlabel('Frequency index')
    ax.set_ylabel('Absorptivity')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sample_predictions.png'), dpi=150)
print(f"Plot saved to results/sample_predictions.png")

# MSE distribution
plt.figure(figsize=(8, 4))
plt.hist(mse_per_sample, bins=50, edgecolor='black')
plt.xlabel('MSE per sample')
plt.ylabel('Count')
plt.title('Test MSE Distribution')
plt.savefig(os.path.join(OUT_DIR, 'mse_distribution.png'), dpi=150)
print(f"MSE distribution saved to results/mse_distribution.png")
