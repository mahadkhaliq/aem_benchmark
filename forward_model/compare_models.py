"""
compare_models.py — Compare MSE between our trained MLP and the paper's pretrained MLP
on the ADM test set.
"""
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import config
from AEML.data.loader import normalize_np
from AEML.models.MLP import model_maker
import sys
# The paper's pretrained weights were saved with 'model_maker' as a top-level module.
# Remap it to AEML's model_maker so torch.load can unpickle it.
sys.modules['model_maker'] = model_maker

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'comparison')
os.makedirs(OUT_DIR, exist_ok=True)

BASE = os.path.dirname(__file__)

MODELS = {
    'Our MLP (Nautilus final)': os.path.join(BASE, 'models', 'MLP', 'adm_mlp_nautilus_final', 'best_model_forward.pt'),
    'Paper MLP (pretrained)':   os.path.join(BASE, 'pre_trained_models', 'MLP', 'ADM', 'best_model_forward.pt'),
}

# ── Load test data ─────────────────────────────────────────────────────────────
os.chdir(config.DATA_DIR)
test_x_raw = pd.read_csv(os.path.join('ADM', 'testset', 'test_g.csv'), header=None).astype('float32').values
test_y     = pd.read_csv(os.path.join('ADM', 'testset', 'test_s.csv'), header=None).astype('float32').values
train_x    = pd.read_csv(os.path.join('ADM', 'data_g.csv'),            header=None).astype('float32').values

# Normalize using training set statistics
_, x_max, x_min = normalize_np(train_x.copy())
test_x, _, _    = normalize_np(test_x_raw.copy(), x_max, x_min)

test_x_t = torch.tensor(test_x).to(DEVICE)

# ── Evaluate each model ────────────────────────────────────────────────────────
results = {}

from AEML.models.MLP.class_wrapper import Network

LINEAR = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]

for name, path in MODELS.items():
    print(f"\nEvaluating: {name}")
    model_dir = os.path.dirname(path)
    ntwk = Network(
        dim_g=14, dim_s=2001, linear=LINEAR,
        inference_mode=True, saved_model=os.path.basename(model_dir),
        ckpt_dir=os.path.dirname(model_dir),
    )
    ntwk.load_model(model_directory=model_dir)
    model = ntwk.model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(test_x_t).cpu().numpy()
    mse_per_sample = np.mean((pred - test_y) ** 2, axis=1)
    mse  = float(np.mean(mse_per_sample))
    rmse = float(np.sqrt(mse))
    results[name] = {'mse': mse, 'rmse': rmse, 'mse_per_sample': mse_per_sample, 'pred': pred}
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")

# ── Print summary table ────────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"{'Model':<30} {'MSE':>10} {'RMSE':>10}")
print("="*50)
for name, res in results.items():
    print(f"{name:<30} {res['mse']:>10.6f} {res['rmse']:>10.6f}")
print("="*50)

# ── Bar chart ─────────────────────────────────────────────────────────────────
names  = list(results.keys())
mses   = [results[n]['mse'] for n in names]
colors = ['#2196F3', '#FF9800']

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(names, mses, color=colors, edgecolor='black', width=0.4)
ax.bar_label(bars, fmt='%.6f', padding=3, fontsize=10)
ax.set_ylabel('Test MSE')
ax.set_title('ADM Test MSE — Our MLP vs Paper MLP')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=10, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mse_comparison_bar.png'), dpi=150)
print(f"\nBar chart saved to results/comparison/mse_comparison_bar.png")

# ── MSE distribution overlay ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for name, color in zip(names, colors):
    ax.hist(results[name]['mse_per_sample'], bins=60, alpha=0.6, label=name, color=color, edgecolor='none')
ax.set_xlabel('MSE per sample')
ax.set_ylabel('Count')
ax.set_title('ADM Test MSE Distribution — Our MLP vs Paper MLP')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mse_distribution_overlay.png'), dpi=150)
print(f"Distribution overlay saved to results/comparison/mse_distribution_overlay.png")
