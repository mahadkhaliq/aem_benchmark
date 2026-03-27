# Forward Model — All-Dielectric Metasurface

Replication of the MLP forward model from:
> *Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials*, NeurIPS 2021

The forward model takes 14 geometric parameters of an all-dielectric metasurface as input and predicts a 2001-point electromagnetic spectrum.

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

Then open `http://localhost:6006`.

## Results

Trained model checkpoints are saved to `forward_model/models/MLP/adm_mlp/best_model_forward.pt`.
