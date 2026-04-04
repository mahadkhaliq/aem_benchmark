# Project Walkthrough — ADM Forward Model
## From Scratch to Running on Nautilus Cluster

---

## What This Project Does

This project replicates the MLP forward model from the NeurIPS 2021 paper *"Benchmarking Data-driven Surrogate Simulators for Artificial Electromagnetic Materials"*. The model takes 14 geometric parameters of an all-dielectric metasurface as input and predicts a 2001-point electromagnetic absorptivity spectrum, replacing expensive physics simulations (7 months of CPU time for 60,000 samples) with a neural network that infers in milliseconds. Final result: Test MSE **0.001849**, matching the paper.

---

## Complete File Map

```
Project_1/
├── README.md                        # Setup, training, inference instructions
├── requirements.txt                 # Python dependencies (local machine)
├── .gitignore                       # Excludes data/, models/, __pycache__/, etc.
│
├── forward_model/
│   ├── config.py                    # All hyperparameters and data path
│   ├── train.py                     # Training script (main file)
│   ├── predict.py                   # Single-sample inference + plot
│   ├── test_model.py                # Full test set evaluation
│   └── export_logs.py               # Export TensorBoard scalars to CSV
│
├── kubernetes/
│   ├── Dockerfile                   # Container image definition
│   ├── pvc.yaml                     # Storage volumes on the cluster
│   ├── job-download-data.yaml       # Job: download dataset to cluster storage
│   ├── job-train.yaml               # Job: run training on the cluster
│   ├── GUIDE.md                     # Step-by-step Nautilus setup guide
│   └── scripts/
│       ├── download_data.sh         # Script that runs inside download job
│       └── train.sh                 # Script that runs inside training job
│
├── data/
│   └── ADM/                         # Dataset (not in git)
│       ├── data_g.csv               # Training inputs  [52812 x 14]
│       ├── data_s.csv               # Training outputs [52812 x 2001]
│       └── testset/
│           ├── test_g.csv           # Test inputs  [5868 x 14]
│           └── test_s.csv           # Test outputs [5868 x 2001]
│
└── docs/
    ├── network_details.md           # Full ML documentation (architecture, loss, results)
    ├── issues_and_model.md          # All issues encountered during this project
    └── project_walkthrough.md       # This file
```

---

## Part 1 — Local Training Files

### `forward_model/config.py`
Central configuration file. All hyperparameters live here so nothing is hardcoded in the training script.

```python
DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')
LINEAR         = [14, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2001]
BATCH_SIZE     = 1024
EPOCHS         = 500
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
LR_DECAY_RATE  = 0.2
EVAL_STEP      = 10
STOP_THRESHOLD = 1e-7
NORMALIZE_INPUT = True
```

`LINEAR` defines the layer widths: input=14, ten hidden layers of 2000 neurons, output=2001.
`DATA_DIR` resolves to `forward_model/../data`, which is `Project_1/data/`.

---

### `forward_model/train.py`
The main training script. Uses the AEML library's `Network` class and its `train_()` method directly — this is the exact training loop from the paper. After training completes, it runs a final evaluation on the test set and saves `results.json`.

**Key decisions:**
- Uses `ntwk.train_()` from `AEML.models.MLP.class_wrapper.Network` — not a custom loop. This was critical: multiple attempts to rewrite the loop with custom TensorBoard logging produced worse results because the AEML loop steps the LR scheduler on training loss every epoch, not validation loss.
- The AEML loop already writes TensorBoard logs internally via its own `SummaryWriter`.
- Only the final test MSE evaluation and `results.json` saving are added on top.

**What it produces:**
- `forward_model/models/MLP/adm_mlp/best_model_forward.pt` — saved PyTorch model
- `forward_model/models/MLP/adm_mlp/results.json` — test MSE, best val MSE, hyperparameters
- TensorBoard event files in the same directory

---

### `forward_model/predict.py`
Single-sample inference script. Takes a test sample index (0–5867), loads the saved model, normalizes the input using training set statistics, runs inference, and plots predicted vs ground truth spectrum.

```bash
cd forward_model
python predict.py --idx 0
```

Saves plot to `forward_model/results/predict_{idx}.png`.

---

### `forward_model/test_model.py`
Evaluates the model on all 5868 test samples. Reports overall MSE and RMSE, plots 5 random sample predictions, and plots the MSE distribution histogram.

```bash
cd forward_model
python test_model.py
```

Saves to `forward_model/results/`.

---

### `forward_model/export_logs.py`
Reads TensorBoard event files and exports all scalar logs (Loss/train, Loss/val, LR) to CSV files. Useful for analysis outside TensorBoard.

```bash
cd forward_model
python export_logs.py
```

---

### `requirements.txt`
Python dependencies for the local machine. Contains two index-url sections because torch is hosted on PyTorch's CDN while everything else is on PyPI.

```
--index-url https://download.pytorch.org/whl/cu121
torch==2.1.2+cu121
torchvision==0.16.2+cu121

--index-url https://pypi.org/simple
AEML==0.0.1
numpy==1.26.4
pandas==2.3.3
...
```

`numpy==1.26.4` is pinned because PyTorch 2.1.2 was compiled against NumPy 1.x and breaks with NumPy 2.x.

---

## Part 2 — Kubernetes Files (Nautilus Cluster)

The Nautilus cluster (gp-engine-malof namespace) runs jobs in containers. Containers are stateless — everything inside them disappears when they finish. Persistent storage (PVCs) is used to keep the dataset and results across jobs.

---

### `kubernetes/pvc.yaml`
Creates two Persistent Volume Claims (PVCs) — think of them as hard drives rented from the cluster. Created once, they persist indefinitely.

```yaml
adm-data    — 5Gi — stores the ADM dataset CSVs
adm-results — 5Gi — stores model checkpoints and result plots
```

Storage class `rook-cephfs-central` with `ReadWriteMany` access means multiple pods can read from the same volume simultaneously.

**Created with:**
```bash
kubectl create -f kubernetes/pvc.yaml
kubectl get pvc -n gp-engine-malof
```

---

### `kubernetes/Dockerfile`
Defines the container image. This image is built once, pushed to DockerHub, and then pulled by every job on the cluster.

**What it does, layer by layer:**
1. Starts from `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` — a DockerHub image with PyTorch 2.1.2 and CUDA 12.1 pre-installed. Chosen to match our exact PyTorch version.
2. Sets `PYTHONUNBUFFERED=1` — forces Python to flush print output immediately (without this, no logs appear in the terminal).
3. Installs system tools: `git` (to clone repo), `wget` (to download data), `unzip` (to extract zip).
4. Installs Python packages: AEML, numpy<2, pandas, scikit-learn, matplotlib, tensorboard, tqdm, einops, seaborn.
5. Patches AEML: removes `verbose=True` from `ReduceLROnPlateau` in `class_wrapper.py` using `sed`. This argument was removed in PyTorch 2.x and causes a crash without this fix.
6. Copies `download_data.sh` and `train.sh` into the image.

**Built and pushed with:**
```bash
cd kubernetes/
docker build --no-cache --platform linux/x86_64 -f Dockerfile -t mahadkhaliq/adm-train:v1 .
docker push mahadkhaliq/adm-train:v1
```

---

### `kubernetes/scripts/download_data.sh`
Shell script that runs inside the download job container. Downloads the ADM dataset zip from Duke Research Repository, extracts it to `/develop/data` (the `adm-data` PVC), and deletes the zip to save space.

**Data source:** `https://research.repository.duke.edu/record/176/files/ADM.zip?ln=en`

**Result on PVC:**
```
/develop/data/ADM/
├── data_g.csv
├── data_s.csv
└── testset/
    ├── test_g.csv
    └── test_s.csv
```

This only needs to be run once. The data stays on the PVC permanently.

---

### `kubernetes/job-download-data.yaml`
Kubernetes Job that runs `download_data.sh` inside the container. A Job runs once to completion, unlike a Deployment which runs continuously.

**Key fields:**
- `image: mahadkhaliq/adm-train:v1` — pulls our Docker image from DockerHub
- `volumeMounts: adm-data → /develop/data` — mounts the data PVC so the script can write to it
- `resources: cpu=4, memory=8Gi` — no GPU needed for downloading
- `restartPolicy: Never` — if it fails, don't retry automatically
- `backoffLimit: 1` — allow one retry before giving up
- `affinity: us-central` — run on US central region nodes (where our namespace lives)

**Run with:**
```bash
kubectl create -f kubernetes/job-download-data.yaml -n gp-engine-malof
kubectl logs -f POD_NAME -n gp-engine-malof
kubectl delete job adm-download-data -n gp-engine-malof  # after completion
```

---

### `kubernetes/scripts/train.sh`
Shell script that runs inside the training job container. Does four things:
1. Clones the GitHub repo (`mahadkhaliq/aem_benchmark`) into `/develop/code`
2. Creates a symlink: `/develop/code/data` → `/develop/data` (the PVC). This is needed because `config.py` expects data at `forward_model/../data`, but the PVC is mounted at `/develop/data`.
3. Runs `python train.py` from inside `forward_model/`
4. Copies the model checkpoint and results to the `adm-results` PVC

**Why the symlink:** The code path and data path are on different volumes. The symlink makes the code think data is in the standard relative location without changing any code.

---

### `kubernetes/job-train.yaml`
Kubernetes Job that runs `train.sh` inside the container with a GPU. This is the main training job.

**Key fields:**
- `image: mahadkhaliq/adm-train:v1` — same Docker image
- `volumeMounts:` — mounts both PVCs: `adm-data → /develop/data`, `adm-results → /develop/results`
- `nvidia.com/gpu: "1"` — requests one GPU
- `resources: cpu=4, memory=16Gi, gpu=1` — limits equal requests (cluster policy: limits ≤ requests × 1.2)
- `restartPolicy: Never` — don't restart on failure

**Run with:**
```bash
kubectl create -f kubernetes/job-train.yaml -n gp-engine-malof
kubectl get pods -n gp-engine-malof
kubectl logs -f POD_NAME -n gp-engine-malof
kubectl delete job adm-train -n gp-engine-malof  # after completion
```

**Monitor TensorBoard during training:**
```bash
kubectl exec -it POD_NAME -n gp-engine-malof -- \
    tensorboard --logdir /develop/code/forward_model/models/MLP/adm_mlp --host 0.0.0.0 --port 6006

# In a second terminal:
kubectl port-forward POD_NAME 6007:6006 -n gp-engine-malof
# Open http://localhost:6007
```

---

### `kubernetes/GUIDE.md`
Step-by-step operational guide covering: account setup, kubectl installation, kubeconfig, PVC creation, Docker build and test, data download, training, TensorBoard monitoring, and result download. Also includes the GPU utilization policy and paper acknowledgment text.

---

## Part 3 — How Everything Connects on the Cluster

```
DockerHub (mahadkhaliq/adm-train:v1)
    │
    │  pulled by both jobs
    ▼
┌─────────────────────────────────────────────────────┐
│  Job 1: adm-download-data                           │
│  runs: download_data.sh                             │
│  → downloads ADM.zip from Duke                      │
│  → extracts to adm-data PVC                         │
│  → deletes zip                                      │
└──────────────────────────┬──────────────────────────┘
                           │ (one-time, data persists)
                           ▼
                    [adm-data PVC]
                    /develop/data/ADM/
                           │
┌──────────────────────────┴──────────────────────────┐
│  Job 2: adm-train                                   │
│  runs: train.sh                                     │
│  → git clone repo into /develop/code                │
│  → symlink /develop/code/data → /develop/data       │
│  → python train.py (500 epochs, ~2 hrs)             │
│  → copy checkpoint + results to adm-results PVC     │
└──────────────────────────┬──────────────────────────┘
                           │
                    [adm-results PVC]
                    /develop/results/
                    ├── models/adm_mlp/
                    │   ├── best_model_forward.pt
                    │   └── results.json
                    └── train_results/
                        ├── Loss_train.csv
                        ├── Loss_val.csv
                        ├── LR.csv
                        └── *.png
                           │
┌──────────────────────────┴──────────────────────────┐
│  Temporary busybox pod (sleep 600)                  │
│  mounts adm-results PVC                             │
│  kubectl cp → copies files to local machine         │
└─────────────────────────────────────────────────────┘
```

---

## Part 4 — Step-by-Step Execution Order

### One-time setup (do once)
1. Create Nautilus account at https://nrp.ai/, get added to `gp-engine-malof`
2. Install kubectl and kubelogin, download kubeconfig
3. `kubectl create -f kubernetes/pvc.yaml` — create storage
4. `docker build ... -t mahadkhaliq/adm-train:v1 .` — build image
5. `docker push mahadkhaliq/adm-train:v1` — push to DockerHub
6. `kubectl create -f kubernetes/job-download-data.yaml` — download dataset to PVC
7. Delete download job after completion

### Each training run
1. Make code changes, `git push`
2. `kubectl create -f kubernetes/job-train.yaml` — submit job
3. `kubectl logs -f POD_NAME -n gp-engine-malof` — watch logs
4. After completion: spin up busybox pod, `kubectl cp` results, delete pod and job

### If you change code only (not dependencies)
- No need to rebuild Docker image — `train.sh` clones fresh from GitHub every run

### If you change Python dependencies
- Rebuild and repush Docker image: `docker build ... && docker push`

---

## Final Results

| Metric | Value |
|---|---|
| Test MSE | 0.001849 |
| Best val MSE | 0.001683 |
| Epochs | 500 |
| Training time on Nautilus | ~25 minutes |
| Training time locally (RTX 3070) | ~2 hours |
