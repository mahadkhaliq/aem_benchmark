#!/bin/bash
set -e

echo "=== Cloning repository ==="
cd /develop
git clone https://github.com/mahadkhaliq/aem_benchmark.git code

echo "=== Linking data PVC to expected path ==="
# config.py computes DATA_DIR as forward_model/../data
# The data PVC is mounted at /develop/data, so we symlink it.
ln -s /develop/data /develop/code/data

echo "=== Starting training ==="
cd /develop/code/forward_model
python train.py

echo "=== Saving results to results PVC ==="
mkdir -p /develop/results/models
cp -r models/MLP/adm_mlp /develop/results/models/
cp -r results /develop/results/train_results

echo "=== Done ==="
