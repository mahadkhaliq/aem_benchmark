# Forward Model All-Dielectric Metasurface

Replication of the MLP and Transformer forward models from:
> *Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials*, NeurIPS 2021

> *https://github.com/yangdeng-EML/ML_MM_Benchmark*

The forward model takes 14 geometric parameters of an all-dielectric metasurface as input and predicts a 2001-point electromagnetic absorptivity spectrum.

## Setup

### MLP (PyTorch 2.x)

**1. Create and activate a conda environment**
```bash
conda create -n adm_benchmark python=3.9
conda activate adm_benchmark
```

**2. Install dependencies**
```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> For CPU-only machines, replace `cu121` with `cpu` in the torch install command.

### Transformer (PyTorch 1.9.1)

The Transformer requires PyTorch 1.9.1 to match the paper's training environment and avoid numerical instability in newer PyTorch versions.

```bash
conda create -n adm_transformer python=3.8
conda activate adm_transformer
pip install --no-cache-dir setuptools==59.5.0
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 --index-url https://download.pytorch.org/whl/cu111
pip install AEML==0.0.1 "numpy<2" pandas scikit-learn matplotlib tensorboard tqdm einops seaborn
```

## Data

Download the ADM dataset from https://doi.org/10.7924/r4jm2bv29 and place it as follows:

```
data/
└── ADM/
    ├── data_g.csv        # training inputs  [52812 x 14]
    ├── data_s.csv        # training outputs [52812 x 2001]
    └── testset/
        ├── test_g.csv    # test inputs  [5868 x 14]
        └── test_s.csv    # test outputs [5868 x 2001]
```

## Training

**MLP:**
```bash
cd forward_model
python train.py
```

**Transformer:**
```bash
cd forward_model
python train_transformer.py --variant 1
```

Three architecture variants are available (`--variant 1/2/3`). Variant 1 matches the paper's exact config. Hyperparameters are defined at the top of [forward_model/train_transformer.py](forward_model/train_transformer.py).

## Monitoring

```bash
tensorboard --logdir forward_model/models/MLP/adm_mlp
```

Then open `http://localhost:6006`.

## Inference

**MLP — single sample:**
```bash
cd forward_model
python predict.py --idx 0      # sample index 0 to 5867
```

**Transformer — single sample:**
```bash
cd forward_model
python predict_transformer.py --idx 0
```

**Full test set (MLP):**
```bash
cd forward_model
python test_model.py
```

**Export training logs:**
```bash
cd forward_model
python export_logs.py
```

All outputs are saved to `forward_model/results/`.

## ONNX Export

Both models can be exported to ONNX for deployment without PyTorch or AEML:

```bash
# Transformer
python forward_model/utils/export_transformer_onnx.py

# MLP / Mixer
python forward_model/utils/pt_onx_conv.py
```

**Loading the exported model:**
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("forward_model/converted_models/transformer_v1.onnx")
x = np.random.randn(1, 14).astype(np.float32)  # normalized geometry input
output = session.run(['spectrum_output'], {'geometry_input': x})
# output[0] shape: (1, 2001)
```

## Results

| Model | Test MSE | Paper Target |
|---|---|---|
| MLP | 0.001813 | ~0.00123 |
| Transformer (Variant 1) | 0.001501 | 0.001470 |

The Transformer matches the paper's reported Transformer MSE within ~2%. Model checkpoints and results JSON are saved under `forward_model/models/`.

## Nautilus Cluster Deployment

Training can be run on the Nautilus HPC cluster using the Kubernetes job and Docker image in `kubernetes/`. See [kubernetes/GUIDE.md](kubernetes/GUIDE.md) for full instructions.

To monitor a running job and automatically copy results when done:
```bash
bash monitor_nautilus.sh
```

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train the MLP model |
| `train_transformer.py` | Train the Transformer model |
| `config.py` | Hyperparameters and paths |
| `predict.py` | Predict and plot a single test sample (MLP) |
| `predict_transformer.py` | Predict and plot a single test sample (Transformer) |
| `test_model.py` | Evaluate full test set |
| `export_logs.py` | Export TensorBoard logs to CSV |
| `utils/export_transformer_onnx.py` | Export Transformer to ONNX |
| `utils/pt_onx_conv.py` | Export MLP/Mixer to ONNX |
| `monitor_nautilus.sh` | Monitor Nautilus job and copy results on completion |
