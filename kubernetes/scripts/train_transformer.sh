#!/bin/bash
set -e

echo "=== Cloning repository ==="
cd /develop
git clone https://github.com/mahadkhaliq/aem_benchmark.git code

echo "=== Linking data PVC to expected path ==="
ln -s /develop/data /develop/code/data

echo "=== Starting Transformer training (Variant 1, batch_size=1024) ==="
cd /develop/code/forward_model
python train_transformer.py --variant 1 --batch-size 1024 || echo "WARNING: script exited with error, saving checkpoint anyway"

echo "=== Saving results to results PVC ==="
mkdir -p /develop/results/models
cp -r models/Transformer/adm_transformer_v1 /develop/results/models/ && echo "Checkpoint saved." || echo "ERROR: checkpoint copy failed"
echo "=== Done ==="
