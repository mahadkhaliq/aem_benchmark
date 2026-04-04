"""
train_transformer.py — Train DukeTransformer on the ADM dataset.

Three architecture variants to try (set VARIANT below):
  1 — Paper's pretrained config (baseline, val MSE ~0.00176)
  2 — Deeper head MLP + more encoder layers
  3 — Wider feature channels + longer sequence

Goal: match or beat paper's MLP MSE of ~0.00123 on the official test set.

Usage:
    cd forward_model
    python train_transformer.py --variant 1
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import pandas as pd

import config
from AEML.data import ADM
from AEML.models.Transformer import DukeTransformer
from AEML.data.loader import normalize_np

# ── Variant definitions ────────────────────────────────────────────────────────
VARIANTS = {
    # Paper's exact Transformer config (from flags.obj)
    1: dict(
        feature_channel_num=512,
        nhead_encoder=8,
        dim_fc_encoder=32,
        num_encoder_layer=6,
        head_linear=[14, 500, 500, 500, 500, 500, 500, 500, 500, 6144],
        tail_linear=[],
        sequence_length=12,
        lr=2e-4,
        weight_decay=5e-4,
        lr_decay_rate=0.2,
        epochs=300,
        eval_step=10,
        stop_threshold=-1,
        batch_size=256,
    ),
    # Deeper head MLP, more encoder layers, lower LR
    2: dict(
        feature_channel_num=512,
        nhead_encoder=8,
        dim_fc_encoder=64,
        num_encoder_layer=8,
        head_linear=[14, 1000, 1000, 1000, 1000, 1000, 1000, 6144],
        tail_linear=[],
        sequence_length=12,
        lr=1e-4,
        weight_decay=1e-4,
        lr_decay_rate=0.2,
        epochs=500,
        eval_step=10,
        stop_threshold=1e-7,
    ),
    # Wider feature channels, longer sequence, tail MLP
    3: dict(
        feature_channel_num=1024,
        nhead_encoder=8,
        dim_fc_encoder=64,
        num_encoder_layer=6,
        head_linear=[14, 1000, 1000, 1000, 1000, 12288],
        tail_linear=[2001],
        sequence_length=16,
        lr=1e-4,
        weight_decay=1e-4,
        lr_decay_rate=0.2,
        epochs=500,
        eval_step=10,
        stop_threshold=1e-7,
    ),
}

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--variant', type=int, default=1, choices=[1, 2, 3],
                    help='Architecture variant to train (1, 2, or 3)')
parser.add_argument('--batch-size', type=int, default=None,
                    help='Override batch size (default: use variant config)')
args = parser.parse_args()

V = VARIANTS[args.variant].copy()
if args.batch_size is not None:
    V['batch_size'] = args.batch_size
model_name = f'adm_transformer_v{args.variant}'
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'models', 'Transformer', model_name)
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"\n{'='*55}")
print(f"Training Transformer Variant {args.variant}: {model_name}")
print(f"{'='*55}")
for k, v in V.items():
    print(f"  {k}: {v}")
print(f"{'='*55}\n")

# ── Data ───────────────────────────────────────────────────────────────────────
os.chdir(config.DATA_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_loader, val_loader, test_x, test_y = ADM(
    normalize=config.NORMALIZE_INPUT,
    batch_size=V.get('batch_size', config.BATCH_SIZE),
)

# ── Model ──────────────────────────────────────────────────────────────────────
ntwk = DukeTransformer(
    dim_g=14,
    dim_s=2001,
    feature_channel_num=V['feature_channel_num'],
    nhead_encoder=V['nhead_encoder'],
    dim_fc_encoder=V['dim_fc_encoder'],
    num_encoder_layer=V['num_encoder_layer'],
    head_linear=V['head_linear'],
    tail_linear=V['tail_linear'] if V['tail_linear'] else None,
    sequence_length=V['sequence_length'],
    model_name=model_name,
    ckpt_dir=os.path.join(os.path.dirname(__file__), 'models', 'Transformer'),
)

# ── Training ───────────────────────────────────────────────────────────────────
ntwk.train_(
    train_loader=train_loader,
    test_loader=val_loader,
    save_model=True,
    epochs=V['epochs'],
    optm='Adam',
    weight_decay=V['weight_decay'],
    lr=V['lr'],
    lr_scheduler_name='reduce_plateau',
    lr_decay_rate=V['lr_decay_rate'],
    eval_step=V['eval_step'],
    stop_threshold=V['stop_threshold'],
)

# ── Final test evaluation on official test set ─────────────────────────────────
net = torch.load(os.path.join(CKPT_DIR, 'best_model_forward.pt'), map_location=device)
net.eval()

test_x_t = torch.tensor(test_x)
test_y_t = torch.tensor(test_y)

# Evaluate in batches to avoid OOM on large test sets
_batch = 256
_preds = []
with torch.no_grad():
    for i in range(0, len(test_x_t), _batch):
        _preds.append(net(test_x_t[i:i+_batch].to(device)).cpu())
pred = torch.cat(_preds, dim=0)
test_mse = nn.functional.mse_loss(pred, test_y_t).item()

results = {
    'variant': args.variant,
    'test_mse': test_mse,
    'best_val_mse': ntwk.best_validation_loss,
    'epochs': V['epochs'],
    'lr': V['lr'],
    'weight_decay': V['weight_decay'],
    'feature_channel_num': V['feature_channel_num'],
    'nhead_encoder': V['nhead_encoder'],
    'num_encoder_layer': V['num_encoder_layer'],
    'sequence_length': V['sequence_length'],
    'head_linear': V['head_linear'],
}
with open(os.path.join(CKPT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nTest MSE:     {test_mse:.6f}")
print(f"Best val MSE: {ntwk.best_validation_loss:.6f}")
print(f"Results saved to {CKPT_DIR}/results.json")

# ── Save checkpoint directly to results PVC (bypasses shell cp issues) ─────────
import shutil
PVC_RESULTS = '/develop/results/models/adm_transformer_v1'
if os.path.exists('/develop/results/'):
    os.makedirs(PVC_RESULTS, exist_ok=True)
    src = os.path.join(CKPT_DIR, 'best_model_forward.pt')
    dst = os.path.join(PVC_RESULTS, 'best_model_forward.pt')
    shutil.copy2(src, dst)
    print(f"Checkpoint saved to PVC: {dst}")
    with open(os.path.join(PVC_RESULTS, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"results.json saved to PVC: {PVC_RESULTS}")
