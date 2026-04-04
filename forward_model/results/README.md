# Results — ADM Forward Model

## Summary

This directory contains training logs, evaluation plots, and prediction outputs from multiple training runs of the MLP forward model on the ADM dataset.

## Results by Run

### 1. Local Run (RTX 3070 Laptop)

- **Directory:** `forward_model/results/` (root-level files)
- **Model checkpoint:** `models/MLP/adm_mlp/`
- **Test MSE:** 0.003022
- **Best val MSE:** 0.002705
- **Training time:** ~2 hours
- **Notes:** Used a custom training loop with TensorBoard logging. The custom loop stepped the LR scheduler on validation loss every 10 epochs instead of training loss every epoch, causing degraded performance compared to the paper's AEML loop.

**Files:**
- `Loss_train.csv` — Training loss per epoch
- `Loss_val.csv` — Validation loss per eval step
- `Loss_test_final.csv` — Final test loss
- `LR.csv` — Learning rate schedule
- `sample_predictions.csv` — 5 random test sample predictions vs ground truth
- `mse_distribution.png` — Histogram of per-sample MSE across all 5868 test samples
- `predict_0.png`, `predict_10.png`, `predict_42.png` — Single sample prediction plots

---

### 2. Nautilus Intermediate Run (GPU cluster)

- **Directory:** `forward_model/results/nautilus/`
- **Model checkpoint:** `models/MLP/adm_mlp_nautilus/`
- **Test MSE:** ~0.003022 (same custom loop issue)
- **Notes:** One of several Nautilus runs that used the custom training loop. CSV exports were copied from the cluster but reflect the same degraded performance.

**Files:** Same CSV and PNG structure as the local run.

---

### 3. Nautilus Final Run (GPU cluster) — Best Result

- **Directory:** `forward_model/results/nautilus_final/`
- **Model checkpoint:** `models/MLP/adm_mlp_nautilus_final/`
- **Test MSE: 0.001849**
- **Best val MSE: 0.001683**
- **Training time:** ~25 minutes
- **Notes:** Used the pure AEML `ntwk.train_()` loop, matching the paper's training procedure exactly. This is the best result and the one used for final evaluation.

**Files:**
- `Loss_train.csv`, `Loss_val.csv`, `LR.csv` — Training logs
- `Loss_test_final.csv` — Final test loss
- `mse_distribution.png` — Per-sample MSE histogram
- `sample_predictions.png` — 5 random sample predictions vs ground truth
- `predict_42.png` — Single sample prediction
- `training_summary.png` — Combined training curves (loss + LR)
- `results/` — Duplicate of CSVs and plots copied from the cluster PVC

**Important:** The CSV files in this directory were exported from TensorBoard logs generated during a prior run. The `results.json` in the model checkpoint directory (`models/MLP/adm_mlp_nautilus_final/results.json`) contains the correct final metrics for this run.

---

## Comparison with Paper

Source: Supplement Table 3 of Deng et al., "Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials," NeurIPS 2021.

| Model | ADM Test MSE | Notes |
|---|---|---|
| Paper — Baseline (Nadell et al.) | 1.20e-3 | Different architecture (12x1000 + conv layers) |
| Paper — MLP (optimized) | **1.23e-3** | Same architecture as ours, hyperparameter-tuned (96hr GPU budget) |
| Paper — Transformer | 1.47e-3 | |
| Paper — Mixer | 1.87e-3 | |
| **Ours — MLP (default hyperparams)** | **1.85e-3** | No hyperparameter tuning |

Our result (1.85e-3) uses the same architecture as the paper's optimized MLP (10 hidden layers of 2000 neurons, ~154M parameters) but with default hyperparameters. The paper's 1.23e-3 was achieved after greedy hyperparameter optimization with a 96-hour GPU budget.

Known hyperparameter differences from the paper's `parameters.py`:
- `EVAL_STEP`: paper uses 20, we used 10
- `STOP_THRESHOLD`: paper uses 1e-9, we used 1e-7 (100x more aggressive early stopping)

---

## Model Weight Backups

All model checkpoints are backed up in `models/MLP/backups/`:
- `best_model_forward_local.pt` — Local run (MSE 0.003022)
- `best_model_forward_nautilus_intermediate.pt` — Intermediate Nautilus run
- `best_model_forward_nautilus_final.pt` — Final Nautilus run (MSE 0.001849)
