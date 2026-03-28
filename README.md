# Forward Model All-Dielectric Metasurface

Replication of the MLP forward model from:
> *Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials*, NeurIPS 2021

> *https://github.com/yangdeng-EML/ML_MM_Benchmark*
>
> 
The forward model takes 14 geometric parameters of an all-dielectric metasurface as input and predicts a 2001-point electromagnetic absorptivity spectrum.

## Setup

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

```bash
cd forward_model
python train.py
```

Hyperparameters are in [forward_model/config.py](forward_model/config.py).

## Monitoring

```bash
tensorboard --logdir forward_model/models/MLP/adm_mlp
```

Then open `http://localhost:6006`. Logs include train loss, validation loss, and learning rate per epoch.

## Inference

**Single sample**  predict the spectrum for one test sample and plot it:
```bash
cd forward_model
python predict.py --idx 0      # sample index 0 to 5867
```

**Full test set**  evaluate all 5868 test samples and save summary plots:
```bash
cd forward_model
python test_model.py
```

**Export training logs**  save TensorBoard scalars to CSV:
```bash
cd forward_model
python export_logs.py
```

All outputs are saved to `forward_model/results/`.

## Results

| Metric | Value |
|---|---|
| Test MSE | 0.001813 |
| Best validation MSE | 0.00169 |

Model checkpoints are saved to `forward_model/models/MLP/adm_mlp/best_model_forward.pt`.
A full results summary is written to `forward_model/models/MLP/adm_mlp/results.json` after training.

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train the model |
| `config.py` | Hyperparameters and paths |
| `predict.py` | Predict and plot a single test sample |
| `test_model.py` | Evaluate full test set |
| `export_logs.py` | Export TensorBoard logs to CSV |
